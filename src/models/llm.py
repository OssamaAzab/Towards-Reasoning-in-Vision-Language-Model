"""Frozen 4-bit language model: load a Qwen2-style causal LM and generate text.

The language model is frozen and quantized to 4-bit (bitsandbytes, NF4) so a 7B
model fits within roughly 20 GB of VRAM alongside the vision encoder and the
bridge. Only the bridge is trained; here the LLM is loaded for inference and
wrapped with a small chat-style generate() helper.

Unquantized bf16 is available as the control for the quantization ablation: the
4-bit path computes in bf16 already (config bnb_4bit_compute_dtype), so 4-bit vs
bf16 varies only the weight quantization and nothing else.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.models.revisions import revision_for


@dataclass
class LanguageModel:
    """A frozen, 4-bit causal LM bundled with its tokenizer and generation budget."""
    model_id: str
    model: torch.nn.Module
    tokenizer: object
    max_new_tokens: int
    revision: str | None


PRECISIONS = ("4bit", "8bit", "bf16", "fp16")


def load_llm(cfg: dict, log=print, eight_bit: bool = False,
             precision: str | None = None) -> LanguageModel:
    """Load the configured causal LM (4-bit NF4 default, 8-bit, bf16 or fp16), frozen, eval.

    `precision` wins over the legacy `eight_bit` flag. bf16 and fp16 load the model
    unquantized (~15 GB of weights, so a large-VRAM GPU); bf16 is the correct
    full-precision control because the 4-bit path already computes in bf16, so
    quantization is the only variable that changes between them.
    """
    model_id = cfg["models"]["llm"]
    revision = revision_for(model_id)
    lc = cfg["llm"]
    compute_dtype = getattr(torch, lc["bnb_4bit_compute_dtype"])
    precision = precision or ("8bit" if eight_bit else "4bit")
    if precision not in PRECISIONS:
        # Never fall through to 4-bit: precision arrives from ckpt["llm_precision"],
        # so a typo or an unknown value would silently load a DIFFERENT model than
        # the one the run claims to be measuring.
        raise ValueError(f"unknown llm precision {precision!r} (expected one of {PRECISIONS})")

    # 8-bit keeps more precision (better quality, more VRAM); 4-bit (NF4) is smaller.
    if precision == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        label = "8-bit"
    elif precision in ("bf16", "fp16"):
        quant_config = None
        compute_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("bf16 requested but this GPU does not support it")
        if precision == "fp16":
            # Qwen2 was pre-trained in bf16, whose exponent range fp16 does not cover;
            # activations can overflow to inf and the model emits garbage silently.
            log("WARNING: fp16 on a bf16-trained model can overflow to inf. Prefer bf16 "
                "for the unquantized control — it also matches bnb_4bit_compute_dtype, "
                "so quantization stays the only variable.")
        label = f"{precision} (unquantized, ~15 GB weights)"
    else:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=lc["load_in_4bit"],
            bnb_4bit_quant_type=lc["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=lc["bnb_4bit_use_double_quant"],
        )
        label = f"4-bit ({lc['bnb_4bit_quant_type']})"

    log(f"loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    log(f"loading model in {label}; first run downloads ~15 GB")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=quant_config,
        device_map={"": 0},  # place the whole model on GPU 0
        torch_dtype=compute_dtype,  # one dtype throughout
    )
    model.eval()  # frozen: inference only (no dropout)
    model.requires_grad_(False)  # never trained; gradients may still flow THROUGH it

    # Default to greedy, deterministic decoding and clear the sampling defaults
    # (temperature/top_p/top_k) that Qwen ships, so they don't warn when unused.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    return LanguageModel(model_id=model_id, model=model, tokenizer=tokenizer,
                         max_new_tokens=lc["max_new_tokens"], revision=revision)


@torch.no_grad()
def logit_health(llm: LanguageModel, probes: list[str] | None = None,
                 inputs_embeds=None) -> dict:
    """Forward each probe and report whether the logits stayed finite.

    A reduced precision can damage this model without ever producing an inf, so a
    finite verdict is NOT a clean bill of health — see the fp16 result below.

    PASS `inputs_embeds` WHENEVER A BRIDGE IS INVOLVED. Measured on job 2285306: fp16
    reported 3/3 finite logits and max|logit| 20.6 on text probes, yet cost 12 net
    questions on the visual arm while leaving the text-only floor untouched (498/500
    identical correctness). Text probes exercise the learned embedding table, whose
    dynamic range is bounded by training; the bridge emits an unbounded learned
    projection, and that is where the damage was. Probing text alone checks the arm
    that is least likely to be broken.
    """
    worst, n_bad, n = 0.0, 0, 0

    def _check(logits):
        nonlocal worst, n_bad, n
        n += 1
        if not torch.isfinite(logits).all():
            n_bad += 1
        finite = logits[torch.isfinite(logits)]
        worst = max(worst, float(finite.abs().max()) if finite.numel() else float("inf"))

    if inputs_embeds is not None:
        attn = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                          device=inputs_embeds.device)
        _check(llm.model(inputs_embeds=inputs_embeds, attention_mask=attn).logits)

    probes = probes or ["Is the sky blue?", "What colour is the bus to the left of the man?",
                        "Describe the image in detail."]
    for text in probes:
        ids = llm.tokenizer(text, return_tensors="pt").to(llm.model.device)
        _check(llm.model(**ids).logits)

    return {"n_probes": n, "n_nonfinite": n_bad, "max_abs_logit": worst,
            "probed_visual": inputs_embeds is not None, "healthy": n_bad == 0}


@torch.no_grad()
def generate(llm: LanguageModel, prompt: str, max_new_tokens: int | None = None,
             system: str | None = None) -> str:
    """Generate a reply to one user prompt with the model's chat template (greedy).

    An optional `system` message steers the answer style (e.g. brevity).
    """
    # Wrap the prompt in the model's chat format (system/user/assistant turns) so the
    # instruct-tuned model behaves as intended, then add the assistant cue to start.
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    text = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = llm.tokenizer(text, return_tensors="pt").to(llm.model.device)

    generated = llm.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens or llm.max_new_tokens,
        do_sample=False,  # greedy decoding -> deterministic, reproducible output
        pad_token_id=llm.tokenizer.eos_token_id,
    )
    # The output includes the prompt tokens; keep only the newly generated ones.
    new_tokens = generated[0, inputs["input_ids"].shape[1]:]
    return llm.tokenizer.decode(new_tokens, skip_special_tokens=True)
