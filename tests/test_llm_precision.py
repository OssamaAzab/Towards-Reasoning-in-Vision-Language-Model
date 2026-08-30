"""The frozen-LLM precision selector: no silent fallback, and a real bf16 control.

These run on CPU. Loading a 7B model is not the point — what matters is that the
precision string is validated and mapped to the right dtype/quantization, because
`precision` arrives from ckpt["llm_precision"] and a wrong mapping would silently
measure a different model than the run claims.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.llm import PRECISIONS, load_llm  # noqa: E402

CFG = {
    "models": {"llm": "Qwen/Qwen2-7B-Instruct"},
    "llm": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True,
            "max_new_tokens": 64},
}


def _load(precision, bf16_supported=True, **kw):
    """Call load_llm with the heavy HF loads stubbed; return the captured kwargs."""
    captured = {}

    def fake_from_pretrained(model_id, **kwargs):
        captured.update(kwargs)
        return mock.MagicMock()

    with mock.patch("src.models.llm.AutoModelForCausalLM.from_pretrained",
                    side_effect=fake_from_pretrained), \
         mock.patch("src.models.llm.AutoTokenizer.from_pretrained",
                    return_value=mock.MagicMock()), \
         mock.patch("torch.cuda.is_bf16_supported", return_value=bf16_supported):
        load_llm(CFG, log=lambda *a, **k: None, precision=precision, **kw)
    return captured


def test_unknown_precision_raises_instead_of_loading_4bit():
    """The old code fell through to 4-bit, silently measuring a different model."""
    with pytest.raises(ValueError, match="unknown llm precision"):
        _load("bfloat16")          # plausible typo for "bf16"
    with pytest.raises(ValueError, match="unknown llm precision"):
        _load("fp32")


def test_bf16_is_unquantized_and_bf16_dtype():
    """bf16 must load unquantized — otherwise the quantization control is not a control."""
    got = _load("bf16")
    assert got["quantization_config"] is None
    assert got["torch_dtype"] is torch.bfloat16


def test_fp16_is_unquantized_and_fp16_dtype():
    """fp16 stays available, but as a distinct dtype from bf16."""
    got = _load("fp16")
    assert got["quantization_config"] is None
    assert got["torch_dtype"] is torch.float16


def test_bf16_control_matches_the_4bit_compute_dtype():
    """4-bit vs bf16 must vary ONLY quantization; the compute dtype has to match."""
    quantized, control = _load("4bit"), _load("bf16")
    assert quantized["torch_dtype"] is control["torch_dtype"] is torch.bfloat16
    assert quantized["quantization_config"] is not None
    assert control["quantization_config"] is None


def test_bf16_refuses_on_a_gpu_that_cannot_do_it():
    """Better to fail loudly than to silently produce numbers from a fallback dtype."""
    with pytest.raises(RuntimeError, match="does not support"):
        _load("bf16", bf16_supported=False)


def test_fp16_warns_about_overflow_on_a_bf16_trained_model():
    """Qwen2 is bf16-native; fp16 can overflow to inf and emit garbage silently."""
    lines = []
    with mock.patch("src.models.llm.AutoModelForCausalLM.from_pretrained",
                    return_value=mock.MagicMock()), \
         mock.patch("src.models.llm.AutoTokenizer.from_pretrained",
                    return_value=mock.MagicMock()):
        load_llm(CFG, log=lines.append, precision="fp16")
    assert any("overflow" in ln for ln in lines)


def test_legacy_eight_bit_flag_still_selects_8bit():
    """Existing callers pass eight_bit=True; that path must not regress."""
    got = _load(None, eight_bit=True)
    assert got["quantization_config"].load_in_8bit is True


def test_default_is_4bit_nf4():
    """The default must stay 4-bit NF4 — every existing checkpoint was trained under it."""
    got = _load(None)
    qc = got["quantization_config"]
    assert qc.load_in_4bit is True
    assert qc.bnb_4bit_quant_type == "nf4"


def test_every_advertised_precision_is_loadable():
    """PRECISIONS is the CLI's choices list; each entry must actually work."""
    for p in PRECISIONS:
        assert _load(p)["torch_dtype"] is not None


def test_vram_report_names_the_actual_device_not_a_hardcoded_budget():
    """Five scripts printed a fixed '20 GB'; on the 97 GB cluster card that misleads."""
    from src.utils import vram_report
    text = vram_report()
    if "n/a" in text:                       # CPU-only box: still must not invent a budget
        return
    assert "GB /" in text
    assert "(" in text and ")" in text, "the device name must be in the report"
    # The budget must come from the device, so it cannot be the old constant unless
    # this machine genuinely has a 20 GB card *and* names itself.
    import torch
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    assert f"/ {total:.0f} GB" in text


def test_logit_health_probes_the_visual_path_when_given_embeddings():
    """Text probes gave fp16 a clean verdict while it was damaging the visual arm only.

    Job 2285306: fp16 reported 3/3 finite logits, yet cost 12 net questions on the
    bridge arm and left the text-only floor untouched (498/500 identical). A health
    check that never sees visual tokens checks the arm least likely to be broken.
    """
    from src.models.llm import logit_health

    seen = {}

    class FakeModel:
        device = "cpu"

        def __call__(self, **kw):
            seen.setdefault("calls", []).append("visual" if "inputs_embeds" in kw else "text")
            return mock.MagicMock(logits=torch.zeros(1, 3, 8))

    llm = mock.MagicMock(model=FakeModel())
    llm.tokenizer.return_value = mock.MagicMock(to=lambda d: {"input_ids": torch.zeros(1, 3).long()})

    text_only = logit_health(llm, probes=["a"])
    assert text_only["probed_visual"] is False
    assert seen["calls"] == ["text"]

    seen.clear()
    with_visual = logit_health(llm, probes=["a"], inputs_embeds=torch.zeros(1, 4, 8))
    assert with_visual["probed_visual"] is True
    assert "visual" in seen["calls"], "the spliced sequence was never forwarded"
    assert with_visual["n_probes"] == 2, "the visual probe must be counted"
