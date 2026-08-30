"""Single source of truth for every LLM prompt, label mask and generation setting.

Two independent levers, deliberately never coupled:

    prompt_format : "raw"       -> byte-identical to the 56 existing checkpoints
                    "chatml_v1" -> Qwen2 ChatML with an explicit system turn
    supervise_eos : False       -> legacy (the answer span ends with no stop token)
                    True        -> the assistant <|im_end|> is a supervised target

The defaults are ("raw", False), so importing this module changes nothing until a
caller opts in. Production combinations are ("raw", False) for legacy reproduction
and ("chatml_v1", True) for the corrected protocol; the other two exist for tests.

WHY A MODULE. The short-answer cue was previously written out four times in four
files with nothing keeping them equal, and the VLM path re-implemented prompt
assembly inline. A drift between any two of those is silent and shows up only as an
unexplained accuracy shift, so every literal and every token id now lives here.

WHY ASSEMBLY IS MANUAL. Visual embeddings cannot exist inside a string, so ChatML
sequences are built by splicing bridge output between separately tokenized segments
rather than by calling apply_chat_template(). That equivalence is only checkable
when the visual span is empty; at n_visual == 0 the manual stream reproduces
apply_chat_template(...)[:-1] exactly (the template's trailing newline is dropped
deliberately - see TRAILING NEWLINE below), and tests pin it.

TRAILING NEWLINE. Qwen2's template ends every turn with <|im_end|> then "\n" (198).
That newline is unreachable at inference because generation halts at <|im_end|>, so
it is never emitted and never supervised.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

MODULE_VERSION = "prompt/1.0.0"

PROMPT_FORMAT_RAW = "raw"
PROMPT_FORMAT_CHATML = "chatml_v1"
EVIDENCE_LEGACY = "legacy_raw_prompt_no_eos"
EVIDENCE_CORRECTED = "corrected_chatml_v1_eos"

# --- Qwen2-7B-Instruct token ids (verified, never assumed) --------------------
IM_START, IM_END, ENDOFTEXT, NEWLINE = 151644, 151645, 151643, 198

# --- Prompt literals: the ONLY copies in the repository -----------------------
SYSTEM_TEXT = "You are a visual question-answering assistant."
# Byte-identical to the four literals it replaces, so the direct arm stays
# comparable across the format change up to the ChatML scaffolding itself.
SHORT_CUE = " Answer in one word or a short phrase."
COT_INSTRUCTION = (
    " Think step by step about the image, then give the final answer. "
    "End your response with 'Answer:' followed by one word or a short phrase."
)
COD_INSTRUCTION = (
    " Think step by step, but keep each reasoning step to at most five words. "
    "Use at most three steps. Then give the final answer after 'Answer:' as one "
    "word or a short phrase."
)
FINALIZE_INSTRUCTION = (
    "Based on your reasoning above, what is the final answer? "
    "Answer in one word or a short phrase."
)
GRAPH_HEDGED = (
    "An automatic object detector found these objects in the image (the list may "
    "be incomplete or imperfect): {context}.\n\nThe image itself is the final "
    "authority; use the detections only as hints. {question}{cue}"
)
GRAPH_PLAIN = (
    "Here is a structured description of the image: {context}.\n\nUsing this "
    "description and the image, answer the question. {question}{cue}"
)

INPUT_MODES = ("image_only", "graph_only", "image_plus_graph", "text_only")
REASON_MODES = ("direct", "cot", "cod")
MAX_NEW_TOKENS = {"direct": 32, "cot": 128, "cod": 32, "finalize": 8}

# A body that STARTS with a line break merges with the user header's trailing
# newline into a different BPE token, so the manual splice would stop matching the
# template. An internal "\n\n" is safe: it is inside one tokenized string.
_LEADING_BREAK = re.compile(r"\s*[\r\n]")


def sha256_text(text: str) -> str:
    """Stable hash of a prompt literal, for checkpoint provenance."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptSpec:
    """Everything that determines a prompt's token layout and its supervised span."""

    prompt_format: str = PROMPT_FORMAT_RAW
    input_mode: str = "image_only"
    reason_mode: str = "direct"
    system_text: str | None = SYSTEM_TEXT   # None => omit the system turn entirely
    supervise_eos: bool = False
    # None = do not truncate. The legacy single-example path (vlm_loss) never
    # truncated while the batched path did; keeping None available preserves that
    # difference exactly rather than silently "fixing" it under raw format.
    max_answer_tokens: int | None = 64
    graph_framing: str = "hedged"           # hedged | plain

    def __post_init__(self):
        """Reject impossible combinations at construction, not at training time."""
        if self.prompt_format not in (PROMPT_FORMAT_RAW, PROMPT_FORMAT_CHATML):
            raise ValueError(f"unknown prompt_format {self.prompt_format!r}")
        if self.input_mode not in INPUT_MODES:
            raise ValueError(f"unknown input_mode {self.input_mode!r}")
        if self.reason_mode not in REASON_MODES:
            raise ValueError(f"unknown reason_mode {self.reason_mode!r}")
        if self.prompt_format == PROMPT_FORMAT_RAW and self.supervise_eos:
            raise ValueError(
                "supervise_eos requires chatml_v1: the raw format has no assistant "
                "turn to close, so a bare <|im_end|> would be an unanchored token"
            )

    @property
    def evidence_layer(self) -> str:
        """Which evidence layer a run under this spec belongs to."""
        return (EVIDENCE_CORRECTED
                if self.prompt_format == PROMPT_FORMAT_CHATML and self.supervise_eos
                else EVIDENCE_LEGACY)


@dataclass(frozen=True)
class BuiltPrompt:
    """A tokenized prompt split around the position the visual span occupies."""

    prefix_ids: list[int]          # before the visual span (empty in raw format)
    body_ids: list[int]            # user body, after the visual span
    mid_ids: list[int]             # user tail + assistant header (empty in raw)
    target_ids: list[int] | None   # supervised span, or None at inference
    n_visual: int
    meta: dict = field(default_factory=dict)

    @property
    def n_prefix(self) -> int:
        """Count of positions that must be masked with -100."""
        return len(self.prefix_ids) + self.n_visual + len(self.body_ids) + len(self.mid_ids)

    def prompt_ids(self) -> list[int]:
        """Every id in order, excluding the visual span (which has no ids)."""
        return [*self.prefix_ids, *self.body_ids, *self.mid_ids]


def assert_body_safe(body: str) -> None:
    """Reject a user body that would merge with the header's trailing newline."""
    if _LEADING_BREAK.match(body):
        raise ValueError(
            "user body must not start with a line break: it merges with the ChatML "
            f"user header's trailing newline into a different token. Got {body[:20]!r}"
        )


def user_body(spec: PromptSpec, question: str, context: str | None = None) -> str:
    """Compose the user-turn text. The only place prompt text is assembled."""
    cue = {"direct": SHORT_CUE, "cot": COT_INSTRUCTION, "cod": COD_INSTRUCTION}[spec.reason_mode]
    if spec.input_mode in ("graph_only", "image_plus_graph") and context:
        template = GRAPH_HEDGED if spec.graph_framing == "hedged" else GRAPH_PLAIN
        body = template.format(context=context, question=question, cue=cue)
    else:
        body = f"{question}{cue}"
    if spec.prompt_format == PROMPT_FORMAT_CHATML:
        assert_body_safe(body)
    return body


def _chatml_segments(tok, system_text: str | None) -> tuple[list[int], list[int], list[int]]:
    """Return (system+user-header, user-tail, assistant-header) id lists."""
    sys_ids: list[int] = []
    if system_text is not None:
        sys_ids = tok(f"<|im_start|>system\n{system_text}<|im_end|>\n",
                      add_special_tokens=False).input_ids
    user_hdr = tok("<|im_start|>user\n", add_special_tokens=False).input_ids
    user_tail = tok("<|im_end|>\n", add_special_tokens=False).input_ids
    assist_hdr = tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids
    return [*sys_ids, *user_hdr], list(user_tail), list(assist_hdr)


def build(spec: PromptSpec, text: str, tok, *, context: str | None = None,
          answer: str | None = None, n_visual: int = 0,
          compose: bool = True) -> BuiltPrompt:
    """Tokenize a prompt (and optionally its supervised target) under `spec`.

    compose=False treats `text` as the finished user body. The training mix builder
    composes deliberately asymmetric prompts — VQAv2 examples carry the short-answer
    cue, LLaVA instructions are passed bare — so re-composing here would both
    double-append the cue on VQAv2 and change what the bridge learns from LLaVA.
    """
    body = user_body(spec, text, context) if compose else text
    if not compose and spec.prompt_format == PROMPT_FORMAT_CHATML:
        assert_body_safe(body)

    if spec.prompt_format == PROMPT_FORMAT_RAW:
        # Legacy layout: [visual | question | answer]. Qwen2 adds no special tokens,
        # so this reproduces src/models/vlm.py's original tokenization exactly.
        prefix_ids: list[int] = []
        body_ids = tok(body).input_ids
        mid_ids: list[int] = []
    else:
        prefix_ids, user_tail, assist_hdr = _chatml_segments(tok, spec.system_text)
        body_ids = tok(body, add_special_tokens=False).input_ids
        mid_ids = [*user_tail, *assist_hdr]

    target_ids, truncated = None, False
    if answer is not None:
        # Tokenize WITHOUT truncation first so the fit can be tested. Appending the
        # EOS string before a truncating tokenize would silently delete it on long
        # answers; truncating first and appending unconditionally would supervise
        # "stop" at an arbitrary mid-sentence cut. Both bugs are avoided by testing
        # the untruncated length and only then deciding.
        full = tok(answer, add_special_tokens=False).input_ids
        cap = spec.max_answer_tokens
        truncated = cap is not None and len(full) > cap
        target_ids = list(full if cap is None else full[:cap])
        if spec.supervise_eos and not truncated:
            target_ids.append(IM_END)

    return BuiltPrompt(
        prefix_ids=list(prefix_ids), body_ids=list(body_ids), mid_ids=list(mid_ids),
        target_ids=target_ids, n_visual=n_visual,
        meta={
            "prompt_format": spec.prompt_format,
            "input_mode": spec.input_mode,
            "reason_mode": spec.reason_mode,
            "supervise_eos": spec.supervise_eos,
            "evidence_layer": spec.evidence_layer,
            "answer_truncated": truncated,
            "eos_appended": bool(target_ids) and target_ids[-1] == IM_END,
            "n_prefix": len(prefix_ids) + n_visual + len(body_ids) + len(mid_ids),
            "system_sha256": sha256_text(spec.system_text or ""),
            "short_cue_sha256": sha256_text(SHORT_CUE),
            "prompt_module_version": MODULE_VERSION,
        },
    )


def generation_kwargs(spec: PromptSpec, stage: str = "direct") -> dict:
    """Generation settings: restrict the stop set to the token actually supervised.

    generation_config.json ships eos_token_id as [151645, 151643]. Only 151645 is
    ever supervised, so allowing a stop on 151643 would let "the bridge learned to
    stop" be satisfied by a token it was never taught, making the diagnostic
    unfalsifiable. pad_token_id is the real pad token, not the EOS.
    """
    return {
        "max_new_tokens": MAX_NEW_TOKENS[stage],
        "do_sample": False,
        "eos_token_id": IM_END,
        "pad_token_id": ENDOFTEXT,
    }


def format_of(ckpt: dict) -> str:
    """Prompt format a checkpoint was trained under; absent means legacy raw."""
    return ckpt.get("prompt_format", PROMPT_FORMAT_RAW)


def evidence_layer_of(ckpt: dict) -> str:
    """Evidence layer a checkpoint belongs to; absent fields mean legacy."""
    return (EVIDENCE_CORRECTED
            if format_of(ckpt) == PROMPT_FORMAT_CHATML and ckpt.get("supervise_eos")
            else EVIDENCE_LEGACY)


def assert_compatible(ckpt: dict, spec: PromptSpec, *, allow_mismatch: bool = False) -> None:
    """Refuse to evaluate a checkpoint under a protocol it was not trained for."""
    stored_format, stored_eos = format_of(ckpt), bool(ckpt.get("supervise_eos", False))
    problems = []
    if stored_format != spec.prompt_format:
        problems.append(f"prompt_format {spec.prompt_format!r} != checkpoint {stored_format!r}")
    if stored_eos != spec.supervise_eos:
        problems.append(f"supervise_eos {spec.supervise_eos} != checkpoint {stored_eos}")
    stored_sys = ckpt.get("system_sha256")
    if stored_format == PROMPT_FORMAT_CHATML and stored_sys is not None:
        if stored_sys != sha256_text(spec.system_text or ""):
            problems.append("system_text hash differs from the checkpoint's")
    if problems and not allow_mismatch:
        raise SystemExit(
            "protocol mismatch: " + "; ".join(problems)
            + ". Evaluating a checkpoint outside its training protocol is an "
              "off-distribution probe, never that checkpoint's score. Pass "
              "--allow-format-mismatch with a distinct --stem-suffix to do it deliberately."
        )
