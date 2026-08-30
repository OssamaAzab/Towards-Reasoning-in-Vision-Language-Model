"""End-to-end VLM forward: feed image features through the bridge into the frozen LLM.

Shared by the training loop and the sanity checks. The bridge turns encoder patch
features into visual tokens; these are placed in the sequence and fed to the frozen
LLM. For training, a target answer is appended and the language-modelling loss is
computed on the answer tokens only (everything before them is masked).

Layout, masking, the supervised span and the generation stop set all come from
`src.prompt`, so there is one definition of the protocol rather than one per file.
The default `PromptSpec()` is ("raw", supervise_eos=False), which reproduces the
construction every existing checkpoint was trained under, token for token.

Callers compose their own prompt text (the training mix gives VQAv2 the short-answer
cue and LLaVA a bare instruction), so this module passes text through verbatim with
`compose=False` and only decides layout.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src import prompt as P

# How the per-token cross-entropies are reduced to the one scalar the optimiser sees.
#
#   "token"    HuggingFace's own reduction: a mean over every supervised token in the
#              flattened batch. An example with a long answer therefore contributes
#              proportionally more gradient than a short one. Every checkpoint trained
#              before 2026-08-03 used this, because it was the only code path.
#   "sequence" Each sequence is reduced to its own mean first, then those are averaged,
#              so every example weighs the same regardless of answer length.
#
# The distinction is not cosmetic here: C37 measured LLaVA at 30% of training examples
# but 78.20% of supervised tokens, so the two reductions weight the mix very differently.
LOSS_NORMS = ("token", "sequence")


def _embed_ids(embed, ids, device):
    """Embed a list of token ids as a [1, n, d] tensor (empty-safe)."""
    if not ids:
        return torch.empty(1, 0, embed.weight.size(1), device=device, dtype=embed.weight.dtype)
    return embed(torch.tensor([ids], device=device))


def assemble(built, visual, embed, device):
    """Splice [prefix | visual | body | mid | target] into embeddings + labels."""
    dtype = embed.weight.dtype
    parts = [_embed_ids(embed, built.prefix_ids, device)]
    if built.n_visual:
        parts.append(visual.to(dtype))
    parts.append(_embed_ids(embed, [*built.body_ids, *built.mid_ids], device))

    labels = None
    if built.target_ids is not None:
        parts.append(_embed_ids(embed, built.target_ids, device))
        target = torch.tensor([built.target_ids], device=device, dtype=torch.long)
        labels = torch.cat(
            [torch.full((1, built.n_prefix), -100, device=device, dtype=torch.long), target],
            dim=1)
    inputs_embeds = torch.cat([p.to(dtype) for p in parts], dim=1)
    if labels is not None and labels.size(1) != inputs_embeds.size(1):
        raise RuntimeError(
            f"label/embedding length mismatch: {labels.size(1)} vs {inputs_embeds.size(1)}")
    return inputs_embeds, labels


def _build_inputs(bridge, llm, encoder_features, question, device, answer=None, spec=None):
    """Build inputs_embeds and matching labels for one example.

    Defaults to NO answer truncation: the historical single-example path never
    truncated (only the batched path did), and silently adding a cap here would
    change vlm_loss for long answers.
    """
    spec = spec or P.PromptSpec(max_answer_tokens=None)
    visual = bridge(encoder_features.float())
    embed = llm.model.get_input_embeddings()
    built = P.build(spec, question, llm.tokenizer, answer=answer,
                    n_visual=visual.size(1), compose=False)
    return assemble(built, visual, embed, device)


def vlm_loss(bridge, llm, encoder_features, question, answer, device, spec=None):
    """Language-modelling loss on the answer, given the image (as features) and question.

    No `loss_norm` knob: with a single sequence the token-mean and the sequence-mean are
    the same number (the mean of one per-sequence mean), so the choice cannot arise here.
    """
    inputs_embeds, labels = _build_inputs(bridge, llm, encoder_features, question,
                                          device, answer, spec)
    return llm.model(inputs_embeds=inputs_embeds, labels=labels, use_cache=False).loss


def sequence_normalised_loss(logits, labels):
    """Mean over sequences of each sequence's own mean CE — equal weight per example.

    The causal shift and -100 masking are HuggingFace's own convention, reproduced here
    rather than re-derived: `scripts/34_nll_decomposition.py` uses the identical shift and
    mask and matched HuggingFace's `.loss` to within 1.3e-08 across 16 checkpoints on real
    data (outputs/analysis/nll_decomp_2288749/, control block of each artifact).

    Sequences with no supervised token are skipped rather than counted as zero, which
    would silently drag the mean down.
    """
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
    per_seq = []
    for i in range(shift_labels.size(0)):
        keep = shift_labels[i] != -100
        if not keep.any():
            continue
        # Index first, upcast second: only the supervised rows reach float32, so this
        # costs a [n_supervised, vocab] tensor rather than the whole [B, L, vocab] batch.
        per_seq.append(F.cross_entropy(shift_logits[i][keep].float(),
                                       shift_labels[i][keep], reduction="mean"))
    if not per_seq:
        raise RuntimeError(
            "no supervised tokens anywhere in the batch: every label was -100, so a "
            "sequence-normalised loss is undefined")
    return torch.stack(per_seq).mean()


def vlm_loss_batch(bridge, llm, feats, questions, answers, device, max_answer_tokens=64,
                   spec=None, loss_norm="token"):
    """Batched language-modelling loss on the answer spans.

    feats: [B, P, D] stacked encoder features; questions/answers: lists of B strings.
    Sequences are right-padded to the batch maximum, with an attention mask over the
    padding and -100 labels everywhere except the supervised answer span. Answers are
    truncated to bound memory; when the protocol supervises EOS, the stop token is
    appended only to answers that were not truncated (see src.prompt.build).

    loss_norm selects the reduction (see LOSS_NORMS). It defaults to "token", which is
    the pre-existing path and is left byte-for-byte unchanged, so every checkpoint
    trained before this argument existed reproduces exactly.

    use_cache=False: a KV cache is built and discarded on every training step
    otherwise, which is pure allocation for a forward that never generates.
    """
    if loss_norm not in LOSS_NORMS:
        raise ValueError(f"loss_norm must be one of {LOSS_NORMS}, got {loss_norm!r}")
    spec = spec if spec is not None else P.PromptSpec(max_answer_tokens=max_answer_tokens)
    visual = bridge(feats.float())
    embed = llm.model.get_input_embeddings()
    dtype = embed.weight.dtype

    seqs, label_seqs, lengths = [], [], []
    for i in range(len(questions)):
        built = P.build(spec, questions[i], llm.tokenizer, answer=answers[i],
                        n_visual=visual.size(1), compose=False)
        emb, lab = assemble(built, visual[i:i + 1], embed, device)
        seqs.append(emb[0])
        label_seqs.append(lab[0])
        lengths.append(emb.size(1))

    bsz, max_len, dim = len(seqs), max(lengths), visual.size(2)
    inputs_embeds = torch.zeros(bsz, max_len, dim, device=device, dtype=dtype)
    attention_mask = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long, device=device)
    for i, length in enumerate(lengths):
        inputs_embeds[i, :length] = seqs[i]
        attention_mask[i, :length] = 1
        labels[i, :length] = label_seqs[i]

    # Under "sequence" the labels are withheld from the model so it does not also compute
    # a reduction we would throw away; the same `labels` tensor drives our own masking.
    out = llm.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    labels=(labels if loss_norm == "token" else None), use_cache=False)
    if loss_norm == "token":
        return out.loss
    return sequence_normalised_loss(out.logits, labels)


@torch.no_grad()
def vlm_generate(bridge, llm, encoder_features, question, device, max_new_tokens=5,
                 spec=None, return_stop=False):
    """Greedy answer to the question, conditioned on the image (as features)."""
    spec = spec or P.PromptSpec()
    inputs_embeds, _ = _build_inputs(bridge, llm, encoder_features, question, device,
                                     spec=spec)
    return _generate(llm, inputs_embeds, max_new_tokens, device, spec, return_stop)


@torch.no_grad()
def text_only_generate(llm, question, device, max_new_tokens=5, spec=None,
                       return_stop=False):
    """Greedy answer with NO image — a text-only floor using the same prompt structure.

    Identical to vlm_generate except the visual span is omitted, so a difference in
    accuracy is attributable to the image alone. Under ChatML the floor therefore
    carries the same system turn and headers as the bridge arm; a floor built with a
    different prompt structure would not be a matched control.
    """
    spec = spec or P.PromptSpec()
    embed = llm.model.get_input_embeddings()
    built = P.build(spec, question, llm.tokenizer, n_visual=0, compose=False)
    inputs_embeds, _ = assemble(built, None, embed, device)
    return _generate(llm, inputs_embeds, max_new_tokens, device, spec, return_stop)


def _generate(llm, inputs_embeds, max_new_tokens, device, spec=None, return_stop=False):
    """Greedy generation from precomputed input embeddings; decode the new tokens.

    With inputs_embeds the model returns only the newly generated ids, so the whole
    sequence is the completion and no prompt slicing is needed.
    """
    spec = spec or P.PromptSpec()
    attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    kwargs = P.generation_kwargs(spec, "direct")
    kwargs["max_new_tokens"] = max_new_tokens
    if not spec.supervise_eos:
        # Legacy behaviour: the answer span was never trained to stop, so the model
        # has no reachable EOS and generation runs to the cap. Keep the historical
        # pad_token_id so legacy reproduction is exact.
        kwargs["eos_token_id"] = llm.tokenizer.eos_token_id
        kwargs["pad_token_id"] = llm.tokenizer.eos_token_id
    out = llm.model.generate(inputs_embeds=inputs_embeds, attention_mask=attn, **kwargs)
    ids = out[0].tolist()
    text = llm.tokenizer.decode(ids, skip_special_tokens=True)
    if not return_stop:
        return text
    stopped = bool(ids) and ids[-1] == kwargs["eos_token_id"]
    return text, {
        "stop_reason": f"eos_{kwargs['eos_token_id']}" if stopped else "cap",
        "final_token_id": ids[-1] if ids else None,
        "n_generated": len(ids),
        "cap_hit": not stopped,
    }
