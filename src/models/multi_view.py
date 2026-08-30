"""Assemble a prompt from SEVERAL visual views through one frozen connector.

WHAT THIS ADDS, AND WHAT IT DELIBERATELY DOES NOT. Stage A appends a focused crop alongside the
global image. Each view is encoded by the frozen encoder and passed through the frozen connector
independently, and the resulting token sequences are concatenated along the token axis before the
existing prompt assembly splices them into the visual span. **No model is modified, loaded
differently, or trained.** The connector still sees exactly the input shape it was trained on,
one view at a time; only the number of visual tokens in the prompt changes.

THE SEQUENCE-LENGTH CONFOUND, STATED UP FRONT. An MLP256 connector emits 256 tokens per view, so
a global+crop prompt carries 512 visual tokens where the bridge and the LLM have only ever seen
256. That is out of distribution in sequence length even though every individual token is in
distribution. It follows that beating the global baseline is NOT evidence that the crop's content
helped — the extra span alone could move the number in either direction. This is precisely why
Stage A runs area-matched `irrelevant` and deranged `wrong` crop arms: they carry the identical
512-token structure and differ only in which region the crop shows. The grounded contrasts are
relevant-vs-wrong and relevant-vs-irrelevant, never relevant-vs-global alone.

WHY THIS IS A SEPARATE IMPLEMENTATION FROM vlm_generate. `global_only` is routed through THIS
path with a single view, and then compared on-GPU against the established `vlm_generate`. Had
this function simply delegated to `vlm_generate` when given one view, that comparison would be
true by construction and could never fail — a check that cannot fail is not a check. The
assembly is therefore rebuilt here, and the parity test verifies it reproduces the established
path exactly. The decoding function itself IS shared deliberately: reimplementing greedy decoding
would introduce a second way to diverge without testing anything worth testing.

VIEW ORDER IS FIXED: global first, then the crop. It is never varied by condition, so no contrast
can be explained by ordering.
"""
from __future__ import annotations

import torch

import src.prompt as P
from src.models.vlm import _generate, assemble


def visual_tokens(bridge, views: list) -> torch.Tensor:
    """Concatenate per-view connector outputs along the token axis -> [1, sum(tokens), hidden]."""
    if not views:
        raise ValueError("at least one view is required")
    outs = []
    for v in views:
        if v.dim() != 3 or v.size(0) != 1:
            raise ValueError(f"each view must be [1, tokens, dim]; got {tuple(v.shape)}")
        outs.append(bridge(v.float()))
    widths = {o.size(2) for o in outs}
    if len(widths) != 1:
        raise ValueError(f"connector emitted inconsistent hidden sizes: {widths}")
    return torch.cat(outs, dim=1)


@torch.no_grad()
def generate_multi_view(bridge, llm, views: list, prompt: str, device, max_new_tokens: int,
                        spec=None, return_stop: bool = False):
    """Greedy answer conditioned on one or more visual views, under the established decoding path.

    `prompt` must already be the finished user body — P.build is called with compose=False, so
    nothing is appended downstream and the caller owns P.SHORT_CUE parity.
    """
    spec = spec or P.PromptSpec()
    visual = visual_tokens(bridge, views)
    embed = llm.model.get_input_embeddings()
    built = P.build(spec, prompt, llm.tokenizer, n_visual=visual.size(1), compose=False)
    inputs_embeds, _ = assemble(built, visual, embed, device)
    return _generate(llm, inputs_embeds, max_new_tokens, device, spec, return_stop)


def assert_frozen(*modules) -> None:
    """Raise unless every parameter of every module is frozen and in eval mode."""
    for m in modules:
        if m is None:
            continue
        live = [n for n, p in m.named_parameters() if p.requires_grad]
        if live:
            raise RuntimeError(f"{type(m).__name__} has {len(live)} trainable parameters: "
                               f"{live[:5]}")
        if getattr(m, "training", False):
            raise RuntimeError(f"{type(m).__name__} is in training mode")
