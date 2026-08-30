"""Stage B: a small trainable crop-evidence adapter over a frozen CLIP -> MLP256 -> Qwen2 stack.

EXPLORATORY — SUPERVISED GQA ORACLE-REGION ADAPTER — NOT ZERO-SHOT — NOT DEPLOYABLE. The crop is
located from GQA's ground-truth scene graph, and the adapter is trained on the GQA train split.
Nothing at inference time knows that box, and unlike every preceding section of this project these
numbers are not zero-shot on GQA.

THE QUESTION PATHWAY IS A LIVE ALTERNATIVE EXPLANATION, WHICH IS WHY THE NULL-VISUAL ARM EXISTS.
`q_proj` is trainable and reads the question, so the adapter can learn GQA answer priors and emit
them through the evidence tokens WITHOUT USING THE CROP AT ALL. The null-visual condition zeroes
the crop features before `in_proj` while leaving the learned queries, the question conditioning,
the gate and all evidence-token positions active. It is the control that separates "the model used
the visual evidence" from "the model learned GQA from supervision". Training uses this same
condition for evidence dropout, so the arm is in distribution rather than a novelty at test time.

WHAT STAGE A ESTABLISHED, AND WHY THIS EXISTS. With everything frozen, the correct oracle crop
beat the same-label wrong-image crop (+7.00 [+2.00, +12.00] on CLIP, n=300) but did NOT beat the
global baseline (+0.00 [-4.00, +4.00]). The entire effect was the wrong crop HURTING, not the
right crop helping. So the frozen stack is sensitive to which image a crop comes from, and cannot
convert that sensitivity into accuracy. This module tests one specific hypothesis: whether a
small trained gate and resampler can.

THE GLOBAL PATHWAY IS UNTOUCHED. Frozen CLIP -> frozen MLP256 -> the same 256 global tokens the
existing checkpoint was trained to produce, in the same order, spliced into the prompt the same
way. The adapter only APPENDS. With zero evidence tokens the sequence is byte-for-byte the
baseline, which is what makes the zero-evidence arm a real control rather than a near-miss.

THE TOKEN BUDGET IS THE POINT. Stage A appended a second 256-token view, which put the prompt out
of distribution in sequence length and confounded crop content with prompt length. The adapter
compresses the crop into N_EVIDENCE tokens instead, so the treatment adds a small fixed span and
every evidence arm carries the identical span — the arms differ only in what the crop shows and
whether the gate is live.

THE GATE IS QUESTION-CONDITIONED because the hypothesis is SELECTIVE fusion: evidence should be
admitted when the question needs a region and suppressed when it does not, or when the region is
wrong. A gate that ignores the question could only learn a global "trust crops this much" scalar,
which is not the claim being tested.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# The adapter's output budget. The pilot's declared range is 4-8 tokens; anything outside it is a
# different experiment and is refused rather than silently accepted.
MIN_EVIDENCE_TOKENS, MAX_EVIDENCE_TOKENS = 4, 8

# How the gate may be driven at inference. `learned` is the trained behaviour; `off` forces every
# gate value to zero, which zeroes the evidence CONTENT while leaving its token positions in the
# sequence. That separation is deliberate: the zero-evidence arm removes the positions too, so the
# pair isolates "what the evidence says" from "that a small span was appended at all".
GATE_MODES = ("learned", "off")


class CropEvidenceAdapter(nn.Module):
    """Compress one crop's patch features into 4-8 question-gated evidence tokens.

    Every parameter here is trainable; nothing else in the stack is. The module never sees an
    answer, a label or a score — only crop features and the question's frozen embeddings.
    """

    def __init__(self, vis_dim: int, llm_dim: int, n_evidence: int = 8, width: int = 512,
                 n_heads: int = 8, ffn_mult: int = 4, gate_init: float = 0.0):
        super().__init__()
        if not MIN_EVIDENCE_TOKENS <= n_evidence <= MAX_EVIDENCE_TOKENS:
            raise ValueError(f"n_evidence must be in [{MIN_EVIDENCE_TOKENS}, "
                             f"{MAX_EVIDENCE_TOKENS}], got {n_evidence}")
        if width % n_heads:
            raise ValueError(f"width {width} is not divisible by n_heads {n_heads}")
        self.vis_dim, self.llm_dim, self.n_evidence, self.width = vis_dim, llm_dim, n_evidence, width

        # Learned evidence queries: the only thing that decides WHAT to read out of the crop.
        self.queries = nn.Parameter(torch.randn(n_evidence, width) * 0.02)
        self.in_proj = nn.Linear(vis_dim, width)
        self.ln_kv, self.ln_q, self.ln_ffn = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(width, width * ffn_mult), nn.GELU(),
                                 nn.Linear(width * ffn_mult, width))
        self.out_proj = nn.Linear(width, llm_dim)

        # Question conditioning. The question's frozen input embeddings are mean-pooled and
        # projected; the gate sees that beside a summary of the evidence it is about to admit.
        self.q_proj = nn.Linear(llm_dim, width)
        self.gate = nn.Sequential(nn.Linear(2 * width, width // 2), nn.GELU(),
                                  nn.Linear(width // 2, n_evidence))
        # Start the gate near 0.5 rather than saturated: a gate initialised at 0 or 1 has almost
        # no gradient and would stay there, which is one of the failure modes the pilot checks for.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_init)

    def forward(self, crop_feats: torch.Tensor, question_embeds: torch.Tensor,
                gate_mode: str = "learned"):
        """crop_feats [B, P, vis_dim], question_embeds [B, T, llm_dim] -> ([B, N, llm_dim], gate).

        Returns the evidence tokens already multiplied by the gate, and the raw gate values so a
        caller can log them. Nothing here is answer-conditioned.
        """
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {GATE_MODES}, got {gate_mode!r}")
        if crop_feats.dim() != 3 or crop_feats.size(2) != self.vis_dim:
            raise ValueError(f"crop_feats must be [B, P, {self.vis_dim}]; got {tuple(crop_feats.shape)}")
        if question_embeds.dim() != 3 or question_embeds.size(2) != self.llm_dim:
            raise ValueError(f"question_embeds must be [B, T, {self.llm_dim}]; "
                             f"got {tuple(question_embeds.shape)}")
        b = crop_feats.size(0)
        kv = self.ln_kv(self.in_proj(crop_feats.float()))
        q = self.ln_q(self.queries).unsqueeze(0).expand(b, -1, -1)
        h = q + self.attn(q, kv, kv, need_weights=False)[0]
        h = h + self.ffn(self.ln_ffn(h))

        cond = self.q_proj(question_embeds.float().mean(dim=1))          # [B, width]
        gate = torch.sigmoid(self.gate(torch.cat([h.mean(dim=1), cond], dim=-1)))  # [B, N]
        if gate_mode == "off":
            gate = torch.zeros_like(gate)
        return self.out_proj(h) * gate.unsqueeze(-1), gate

    def trainable_parameters(self) -> int:
        """Exact count of parameters this module will hand to the optimiser."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def config(self) -> dict:
        """The architecture, as recorded in every checkpoint and every results file."""
        return {"vis_dim": self.vis_dim, "llm_dim": self.llm_dim, "n_evidence": self.n_evidence,
                "width": self.width, "n_heads": self.attn.num_heads,
                "trainable_parameters": self.trainable_parameters(),
                "oracle": True,
                "not_deployable": "the crop is located from GQA ground-truth boxes"}


def question_embeddings(llm, question: str, device):
    """Frozen input embeddings of the question text, for gate conditioning only.

    Detached deliberately: the gate is conditioned ON the question, and no gradient may flow back
    into the frozen embedding table through this path.
    """
    ids = llm.tokenizer(question, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        return llm.model.get_input_embeddings()(ids.to(device)).detach()


def null_crop_features(reference: torch.Tensor) -> torch.Tensor:
    """Exactly-zero crop features shaped like `reference`.

    A separate named function so the null-visual arm is one testable thing rather than an inline
    `torch.zeros_like` that a later edit could quietly turn into noise or a learned constant.
    """
    return torch.zeros_like(reference)


def evidence_visual_tokens(bridge, adapter, global_feats, crop_feats, question_embeds,
                           gate_mode: str = "learned", null_visual: bool = False):
    """[global 256 tokens | gated evidence tokens] for one example, plus the gate values.

    THREE DISTINCT NO-CROP CASES, and they are not interchangeable:

      null_visual=True     the adapter RUNS on all-zero crop features. Learned queries, question
                           conditioning, gate and every evidence position stay active. This is the
                           question-prior control, and it is what evidence dropout uses in
                           training.
      crop_feats=None      NO ADAPTER SPAN: the adapter is not called, no positions are appended,
                           and the returned sequence is exactly what the frozen baseline produces.
      gate_mode="off"      the adapter runs and its positions remain, but the gate zeroes their
                           content.
    """
    g = bridge(global_feats.float())
    if null_visual:
        # Shaped from the crop when there is one, else from the global view: the same frozen
        # encoder produces both, so the patch count and width match either way.
        crop_feats = null_crop_features(global_feats if crop_feats is None else crop_feats)
    elif crop_feats is None:
        return g, None
    ev, gate = adapter(crop_feats, question_embeds, gate_mode=gate_mode)
    return torch.cat([g, ev.to(g.dtype)], dim=1), gate


def assert_only_adapter_trainable(adapter, *frozen_modules) -> None:
    """Raise unless every frozen module is frozen AND the adapter has live parameters.

    Both halves matter. Checking only that the encoders are frozen would pass on a run where the
    adapter was accidentally frozen too and nothing trained at all.
    """
    for m in frozen_modules:
        if m is None:
            continue
        live = [n for n, p in m.named_parameters() if p.requires_grad]
        if live:
            raise RuntimeError(f"{type(m).__name__} has {len(live)} trainable parameters: "
                               f"{live[:5]}")
    n = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    if n == 0:
        raise RuntimeError("the adapter has no trainable parameters; this run would train nothing")


def assert_gradients_only_in_adapter(adapter, *frozen_modules) -> None:
    """After a backward pass, raise if any gradient landed outside the adapter.

    requires_grad=False is checked before the step; this is checked after it. A module can be
    frozen and still accumulate a .grad if it was unfrozen mid-run, and only this catches that.
    """
    for m in frozen_modules:
        if m is None:
            continue
        dirty = [n for n, p in m.named_parameters() if p.grad is not None]
        if dirty:
            raise RuntimeError(f"{type(m).__name__} accumulated gradients on {len(dirty)} "
                               f"parameters: {dirty[:5]}")
    if not any(p.grad is not None for p in adapter.parameters()):
        raise RuntimeError("no adapter parameter received a gradient; the graph is disconnected")


def gate_stats(values) -> dict:
    """Summary of observed gate values, so a dead or saturated gate is visible in the log."""
    t = torch.cat([v.detach().flatten().float().cpu() for v in values]) if values else torch.empty(0)
    if not t.numel():
        return {"n": 0}
    return {"n": int(t.numel()), "mean": float(t.mean()), "std": float(t.std()),
            "min": float(t.min()), "max": float(t.max()),
            "frac_below_0.01": float((t < 0.01).float().mean()),
            "frac_above_0.99": float((t > 0.99).float().mean()),
            "all_finite": bool(torch.isfinite(t).all())}
