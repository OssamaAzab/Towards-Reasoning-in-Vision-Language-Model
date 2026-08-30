"""Step 6 / 8: train the bridge on LLaVA (+ optional VQAv2 short-answer mix).

Trains ONLY the bridge; the encoder and LLM stay frozen. Encoder features are read from
a disk cache (built here on demand). Supports real batches, a VQAv2 short-answer mix to
teach terse answers (with GQA-eval images excluded to prevent leakage), a held-out
validation set, periodic checkpointing, and ablation levers (8-bit LLM, bridge size).
The checkpoint records the exact config so the evaluator reproduces it automatically.

    python scripts/06c_train_bridge.py --limit 64 --epochs 1 --short-frac 0.5      # smoke
    python scripts/06c_train_bridge.py --limit 15000 --batch-size 8 --short-frac 0.5
"""
import argparse
import contextlib
import hashlib
import math
import os
import random
import statistics
import sys
import time
import zlib
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Make the repo root importable when run as `python scripts/06c_train_bridge.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.feature_cache import cache_dir_for, has_features, load_features, save_features  # noqa: E402
from src.data.llava import LLaVADataset  # noqa: E402
from src.data.vqa import VQADataset  # noqa: E402
from src.models.connectors import build_bridge  # noqa: E402
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.models.llm import PRECISIONS, load_llm  # noqa: E402
from src import prompt as P  # noqa: E402
from src.models.vlm import LOSS_NORMS, vlm_loss_batch  # noqa: E402
from src.utils import (  # noqa: E402
    StepProfiler,
    canonical_dir_map,
    ensure_dir,
    load_config,
    set_seed,
    split_key,
    vram_report,
)

# Short-answer examples get the same suffix used at evaluation, so "terse" is a learned
# response to this exact cue. Aliased, not copied: scripts/24_diag_token_mix.py reads
# trainer.SHORT_SUFFIX to tag the 70/30 mix, and a private copy could drift from the
# cue actually used at eval without any test noticing.
SHORT_SUFFIX = P.SHORT_CUE
MLP_REFERENCE_WIDTHS = {"clip": 5888, "ijepa": 5700, "dinov2": 5888,
                        # The IN22K checkpoint is geometrically
                        # identical to `ijepa` (256 tokens, width 1280, no CLS, 630,762,240
                        # params), so it takes the same connector width. Without this entry the
                        # width guard silently does not apply to the new arm and a typo of 5888
                        # would be accepted, producing a non-comparable connector.
                        "ijepa_in22k": 5700}

# Encoders that must be validated against ANOTHER encoder's 150K MLP reference checkpoint.
# `ijepa_in22k` differs from `ijepa` only in pretraining corpus, so its connector must match
# `ijepa`'s exactly — which is the entire premise of the H1 intervention. Aliasing here turns
# that premise into a run-time assertion instead of opting out with
# --allow-missing-mlp-reference. Encoders absent from this map are unaffected.
MLP_REFERENCE_ALIAS = {"ijepa_in22k": "ijepa"}
TRAJECTORY_FIELDS = (
    "lr",
    "weight_decay",
    "batch_size",
    "grad_accum",
    "max_answer_tokens",
    "warmup_frac",
    "seed",
    "loss_norm",
)

# Trajectory fields whose absence from a checkpoint is NOT missing provenance, because the
# value is recoverable with certainty from when the checkpoint was written.
#
# loss_norm was added on 2026-08-03. Before it, `vlm_loss_batch` had exactly one reduction —
# HuggingFace's token mean — so a checkpoint without the field was demonstrably trained at
# "token". Defaulting is therefore a recovered fact, not a guess, and it keeps the conflict
# check live: resuming a legacy checkpoint with --loss-norm sequence still refuses.
TRAJECTORY_LEGACY_DEFAULTS = {"loss_norm": "token"}

_T0 = time.perf_counter()

# Shared no-op span used when profiling is off. contextlib.nullcontext holds no state, so a
# single module-level instance is safe to reuse and costs no allocation on the hot path.
_NO_SPAN = contextlib.nullcontext()


def log(msg: str) -> None:
    """Print a progress line prefixed with elapsed seconds, flushed immediately."""
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def _span(prof, kind: str, name: str):
    """Return a profiling span ('cpu' | 'wait' | 'cuda'), or a shared no-op when off.

    'cpu' and 'wait' are serial wall-clock segments and are additive. 'cuda' is elapsed
    time between CUDA events — an UPPER BOUND on device activity, not GPU-busy time and
    not utilisation, because the stream can idle inside the span while the CPU submits.
    It overlaps the wall timeline; StepProfiler keeps the two apart.
    """
    if prof is None or not prof.active:
        return _NO_SPAN
    if kind == "cpu":
        return prof.cpu(name)
    if kind == "wait":
        return prof.gpu_wait(name)
    return prof.cuda_elapsed(name)


def file_sha256(path):
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_mlp_hidden(bridge_type, encoder_name, requested):
    """Resolve and validate the connector hidden width."""
    if bridge_type != "mlp":
        return requested or 4096
    expected = MLP_REFERENCE_WIDTHS.get(encoder_name)
    if requested is None:
        detail = f" (expected {expected})" if expected is not None else ""
        raise SystemExit(f"fresh MLP training requires explicit --mlp-hidden{detail}")
    if expected is not None and requested != expected:
        raise SystemExit(
            f"--mlp-hidden={requested} conflicts with the preregistered {encoder_name} "
            f"MLP reference width {expected}"
        )
    return requested


def validate_strip_cls(encoder_name, strip_cls):
    """Allow CLS stripping only for encoders whose token 0 is a CLS token."""
    if strip_cls and encoder_name not in ("clip", "dinov2"):
        raise SystemExit(
            f"--strip-cls is valid only for CLIP and DINOv2: '{encoder_name}' has "
            "no [CLS] token, so token 0 is a real patch and must not be dropped"
        )


def resolve_qformer_bridge_config(bridge_config, dropout, norm_first):
    """Return bridge_config with validated, explicit Q-Former training options."""
    resolved = dict(bridge_config)
    if resolved.get("type", "qformer") != "qformer":
        if dropout is not None or norm_first is not None:
            raise SystemExit("--qformer-dropout/--qformer-norm-first are Q-Former-only")
        return resolved
    resolved_dropout = 0.1 if dropout is None else dropout
    if not 0.0 <= resolved_dropout < 1.0:
        raise SystemExit("--qformer-dropout must be in [0, 1)")
    resolved["dropout"] = resolved_dropout
    resolved["norm_first"] = False if norm_first is None else norm_first
    return resolved


def validate_qformer_resume_options(
    bridge_config,
    requested_dropout,
    requested_norm_first,
):
    """Reject explicit Q-Former options that conflict with checkpoint provenance."""
    if bridge_config.get("type", "qformer") != "qformer":
        if requested_dropout is not None or requested_norm_first is not None:
            raise SystemExit("--qformer-dropout/--qformer-norm-first are Q-Former-only")
        return None, None
    stored_dropout = bridge_config.get("dropout", 0.1)
    stored_norm_first = bridge_config.get("norm_first", False)
    if requested_dropout is not None and requested_dropout != stored_dropout:
        raise SystemExit(
            f"--qformer-dropout={requested_dropout} conflicts with the checkpoint's "
            f"dropout={stored_dropout}; it cannot change on resume"
        )
    if requested_norm_first is not None and requested_norm_first != stored_norm_first:
        raise SystemExit(
            f"--qformer-norm-first={requested_norm_first} conflicts with the checkpoint's "
            f"norm_first={stored_norm_first}; it cannot change on resume"
        )
    return stored_dropout, stored_norm_first


def assert_mlp_reference_match(bridge, bridge_config, reference_ckpt):
    """Assert that a fresh MLP matches its 150K reference architecture and size."""
    reference_config = reference_ckpt.get("bridge_config", {})
    for key in ("type", "drop_cls", "mlp_hidden", "mlp_depth"):
        current = bridge_config.get(key)
        reference = reference_config.get(key)
        if current != reference:
            raise SystemExit(
                f"fresh MLP {key}={current} does not match 150K reference {reference}"
            )
    current_count = sum(parameter.numel() for parameter in bridge.parameters())
    reference_count = sum(tensor.numel() for tensor in reference_ckpt["state_dict"].values())
    if current_count != reference_count:
        raise SystemExit(
            f"fresh MLP parameter count {current_count} does not match "
            f"150K reference {reference_count}"
        )
    return reference_count


def validate_mlp_reference(bridge, bridge_config, reference_path, allow_missing):
    """Validate an existing MLP reference or record an explicit missing-reference opt-in."""
    reference_path = Path(reference_path)
    parameter_count = sum(parameter.numel() for parameter in bridge.parameters())
    if not reference_path.is_file():
        if not allow_missing:
            raise SystemExit(f"MLP reference checkpoint not found: {reference_path}")
        return {
            "status": "missing-opt-in",
            "path": str(reference_path),
            "sha256": None,
            "parameter_count": parameter_count,
        }

    reference_ckpt = torch.load(reference_path, map_location="cpu", weights_only=False)
    reference_count = assert_mlp_reference_match(bridge, bridge_config, reference_ckpt)
    del reference_ckpt
    return {
        "status": "matched",
        "path": str(reference_path),
        "sha256": file_sha256(reference_path),
        "parameter_count": reference_count,
    }


def validate_trajectory_provenance(current, checkpoint, acknowledge_missing):
    """Validate trajectory-defining fields against a parent checkpoint."""
    # The caller must supply every trajectory field. Reading with .get() instead would let a
    # newly added field be omitted and silently skip its own conflict check — the failure
    # mode this function exists to prevent — so an incomplete `current` is an error.
    absent = tuple(key for key in TRAJECTORY_FIELDS if key not in current)
    if absent:
        raise SystemExit(
            f"internal: trajectory dict is missing {', '.join(absent)}; every field in "
            "TRAJECTORY_FIELDS must be supplied or its conflict check would be skipped")
    missing = tuple(key for key in TRAJECTORY_FIELDS
                    if checkpoint.get(key) is None and key not in TRAJECTORY_LEGACY_DEFAULTS)
    if missing and not acknowledge_missing:
        flags = ", ".join(f"--{key.replace('_', '-')}" for key in missing)
        raise SystemExit(
            f"checkpoint has missing provenance for {flags}; pass "
            "--acknowledge-legacy-provenance only after verifying the legacy recipe"
        )
    for key in TRAJECTORY_FIELDS:
        stored = checkpoint.get(key, TRAJECTORY_LEGACY_DEFAULTS.get(key))
        if stored is None:
            stored = TRAJECTORY_LEGACY_DEFAULTS.get(key)
        if stored is not None and current[key] != stored:
            flag = key.replace("_", "-")
            raise SystemExit(
                f"--{flag}={current[key]} conflicts with checkpoint's {stored}; "
                "trajectory-defining fields cannot change"
            )
    return missing


def validate_epoch_operation(
    checkpoint,
    target_epochs,
    requested_tag,
    continuation,
    lr_schedule,
    acknowledge_missing,
):
    """Validate same-experiment resume or explicit new-stem continuation semantics."""
    parent_epoch = int(checkpoint.get("epoch", 0))
    parent_tag = checkpoint.get("tag")
    stored_target = checkpoint.get("target_epochs")
    missing = ()
    if stored_target is None:
        missing = ("target_epochs",)
        if not acknowledge_missing:
            raise SystemExit(
                "checkpoint has missing provenance for target_epochs; pass "
                "--acknowledge-legacy-provenance only after verifying the legacy recipe"
            )
        original_target = parent_epoch if continuation else target_epochs
    else:
        original_target = int(stored_target)

    if not continuation:
        if parent_tag is None:
            raise SystemExit(
                "same-experiment resume requires a recorded parent tag; use explicit "
                "--continue-from with a new --ckpt-tag for an untagged legacy parent"
            )
        if requested_tag is not None and requested_tag != parent_tag:
            raise SystemExit(
                f"--ckpt-tag={requested_tag} conflicts with parent tag {parent_tag}; "
                "use explicit --continue-from to create a new experiment stem"
            )
        if target_epochs != original_target:
            raise SystemExit(
                f"target --epochs={target_epochs} conflicts with checkpoint's "
                f"target_epochs={original_target}; use explicit --continue-from for an extension"
            )
        return {
            "original_target_epochs": original_target,
            "extended_target_epochs": target_epochs,
            "missing_provenance": missing,
        }

    if not requested_tag or requested_tag == parent_tag:
        raise SystemExit(
            "explicit continuation requires a new --ckpt-tag different from the parent tag"
        )
    if lr_schedule != "flat":
        raise SystemExit(
            "explicit continuation is restricted to flat LR; extending cosine would "
            "silently reshape its schedule"
        )
    if target_epochs <= max(parent_epoch, original_target):
        raise SystemExit(
            f"continuation target --epochs={target_epochs} must be greater than parent "
            f"epoch/target {max(parent_epoch, original_target)}"
        )
    return {
        "original_target_epochs": original_target,
        "extended_target_epochs": target_epochs,
        "missing_provenance": missing,
    }


def capture_rng_state(include_cuda=True):
    """Capture Python, Torch CPU, and optionally Torch CUDA RNG states."""
    cuda_state = None
    if include_cuda and torch.cuda.is_available():
        cuda_state = [state.clone() for state in torch.cuda.get_rng_state_all()]
    return {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": cuda_state,
    }


def _cpu_byte(tensor):
    """Coerce a saved RNG state back to the CPU uint8 tensor the RNG setters require.

    The checkpoint is loaded with `torch.load(..., map_location=device)`, and map_location moves
    EVERY tensor in it to that device — including these RNG states. torch.set_rng_state and
    torch.cuda.set_rng_state_all both require CPU byte tensors, so a CUDA-resident state fails
    with `TypeError: RNG state must be a torch.ByteTensor`. That is what killed job 2286251
    after its optimizer state had already been restored.

    It had never fired before because every earlier continuation resumed from a legacy
    checkpoint with no rng_state at all, so this branch was skipped.
    """
    return tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def restore_rng_state(state, include_cuda=True):
    """Restore Python, Torch CPU, and optionally Torch CUDA RNG states."""
    random.setstate(state["python"])
    torch.set_rng_state(_cpu_byte(state["torch_cpu"]))
    cuda_state = state.get("torch_cuda")
    if include_cuda and cuda_state is not None:
        if not torch.cuda.is_available():
            raise SystemExit("checkpoint has CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([_cpu_byte(s) for s in cuda_state])


def build_examples(cfg, n_short, n_llava, shuffle_pool_seed=None):
    """Return a mixed list of (image_path, prompt, answer): VQAv2 (short) + LLaVA (verbose).

    shuffle_pool_seed: None (default, the LOCKED behaviour) draws the file-order first-n from
    each source. LLaVA-Instruct-150K is concatenated by task type (conversation, then detail,
    then complex-reasoning), so a file-order first-n draw of n_llava < ~57K yields an almost
    purely conversational LLaVA slice (verified in outputs/diagnostics/llava_typemix.csv: the
    locked 150K run's 45K LLaVA slice is 99.15% conversation-type, mean 15.7 words, and excludes
    the 106,547 detail/complex-reasoning examples, mean 105.6 words). Passing a seed instead
    draws a REPRESENTATIVE seeded sample of each source's
    full pool, mixing all three task types. Opt-in only — None keeps every locked run
    bit-identical (unshuffled indices reduce to the original first-n draw).
    """
    out = []
    llava = LLaVADataset(cfg["llava"]["annotations"], cfg["llava"]["image_dir"],
                         cfg["llava"].get("exclude_ids"))
    log(f"LLaVA: {len(llava):,} usable conversations (excluded {llava.n_excluded:,} GQA-eval images)")
    llava_idx = list(range(len(llava)))
    if shuffle_pool_seed is not None:
        random.Random(shuffle_pool_seed).shuffle(llava_idx)
        log(f"LLaVA POOL SHUFFLE on (seed {shuffle_pool_seed}): representative task-type sample")
    for i in llava_idx[:n_llava]:          # unshuffled == the locked file-order first-n draw
        ex = llava.get(i)
        out.append((ex.image_path, ex.instruction, ex.answer))
    if n_short:
        vqa = VQADataset(cfg["vqa"]["questions"], cfg["vqa"]["annotations"],
                         cfg["vqa"]["image_dir"], cfg["vqa"]["exclude_ids"])
        log(f"VQAv2: {len(vqa):,} usable examples (excluded {vqa.n_excluded:,} GQA-eval images)")
        vqa_idx = list(range(len(vqa)))
        if shuffle_pool_seed is not None:
            random.Random(shuffle_pool_seed + 1).shuffle(vqa_idx)   # +1: stream independent of LLaVA
        for i in vqa_idx[:n_short]:
            ex = vqa.get(i)
            out.append((ex.image_path, ex.question + SHORT_SUFFIX, ex.answer))
    return out


@torch.no_grad()
def eval_loss(bridge, llm, cache_dir, examples, device, bs, max_ans, spec=None,
              loss_norm="token"):
    """Mean loss over a held-out set (no gradients); bridge put back in train mode after.

    Uses the SAME reduction as training, so val_loss is the held-out value of the
    objective actually optimised. That makes val_loss incomparable ACROSS loss_norm arms —
    compare accuracy, or recompute both reductions with scripts/34_nll_decomposition.py.
    """
    bridge.eval()
    total, count = 0.0, 0
    for start in tqdm(range(0, len(examples), bs), desc="val", unit="batch",
                      dynamic_ncols=True, leave=False):
        batch = examples[start:start + bs]
        feats = torch.cat([load_features(cache_dir, p, device) for p, _, _ in batch], dim=0)
        loss = vlm_loss_batch(bridge, llm, feats, [x[1] for x in batch],
                              [x[2] for x in batch], device, max_ans, spec=spec,
                              loss_norm=loss_norm)
        total += loss.item() * len(batch)
        count += len(batch)
    bridge.train()
    return total / max(1, count)


def main() -> None:
    """Build a mixed slice, cache features, and train the bridge with val-loss tracking."""
    # Config first: config/default.yaml's train: block supplies the argparse DEFAULTS,
    # so config is the source of truth and a CLI flag is an explicit override.
    cfg = load_config()
    tcfg = cfg["train"]

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=200, help="total training examples")
    ap.add_argument("--epochs", type=int, default=tcfg["num_epochs"])
    ap.add_argument("--lr", type=float, default=tcfg["lr"])
    ap.add_argument("--weight-decay", type=float, default=tcfg["weight_decay"])
    ap.add_argument("--batch-size", type=int, default=tcfg["batch_size"])
    ap.add_argument("--grad-accum", type=int, default=tcfg["grad_accum_steps"])
    ap.add_argument("--max-answer-tokens", type=int, default=tcfg["max_answer_tokens"])
    ap.add_argument("--loss-norm", choices=LOSS_NORMS, default="token",
                    help="reduction over per-token CE: 'token' (HuggingFace's batch-level "
                         "token mean, what every existing checkpoint used) or 'sequence' "
                         "(each example weighted equally regardless of answer length)")
    ap.add_argument("--short-frac", type=float, default=tcfg["short_frac"],
                    help="fraction from VQAv2 short answers")
    ap.add_argument("--val-frac", type=float, default=tcfg["val_frac"],
                    help="held-out fraction for val loss")
    ap.add_argument("--ckpt-every", type=int, default=1000, help="checkpoint every N batches (0=off)")
    ap.add_argument("--log-every", type=int, default=100, help="print a live loss/progress line every N batches")
    ap.add_argument("--8bit", dest="eight_bit", action="store_true")
    ap.add_argument(
        "--encoder",
        default=None,
        help="override active encoder: clip | ijepa | dinov2 | vjepa2",
    )
    ap.add_argument("--strip-cls", action="store_true",
                    help="bridge drops memory token 0 (CLIP/DINOv2 [CLS]), "
                         "so the connector sees 256 pure patch tokens like I-JEPA")
    ap.add_argument("--lr-schedule", choices=["flat", "cosine"], default="flat",
                    help="flat = the locked protocol (constant --lr); cosine = linear "
                         "warmup to --lr then cosine decay to 0 over the whole run "
                         "(LR-schedule ablation; peak and weight decay unchanged)")
    ap.add_argument("--warmup-frac", type=float, default=0.05,
                    help="fraction of total optimizer steps spent in linear warmup "
                         "(cosine schedule only)")
    ap.add_argument("--bridge-type", choices=["qformer", "mlp", "pool"], default=None,
                    help="connector ablation (B2): qformer = locked default; mlp = "
                         "LLaVA-style per-patch projector (all patch tokens to the "
                         "LLM); pool = average-pool to 32 tokens then MLP")
    ap.add_argument("--mlp-hidden", type=int, default=None,
                    help="hidden width (required for MLP; pool defaults to 4096)")
    ap.add_argument("--mlp-depth", type=int, default=None,
                    help="linear-layer count of the mlp/pool connector (default 3)")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config seed (B-seeds replication; data slice is "
                         "seed-independent, so replicas share data but differ in init "
                         "and batch order)")
    ap.add_argument("--llm-precision", choices=list(PRECISIONS), default=None,
                    help="frozen-LLM precision (B4 ablation; default = 4-bit, "
                         "--8bit legacy flag still honoured). bf16 is the correct "
                         "unquantized control: the 4-bit path already computes in "
                         "bf16, so quantization is the only variable that changes")
    ap.add_argument("--query-tokens", type=int, default=None)
    ap.add_argument("--bridge-layers", type=int, default=None)
    ap.add_argument(
        "--qformer-dropout",
        type=float,
        default=None,
        help="Q-Former decoder dropout (locked/default behaviour: 0.1)",
    )
    ap.add_argument(
        "--qformer-norm-first",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use pre-LN Q-Former layers plus the required final LayerNorm "
             "(locked/default behaviour: post-LN)",
    )
    ap.add_argument("--shuffle-pool", action="store_true",
                    help="draw a REPRESENTATIVE seeded sample from each source's FULL pool "
                         "instead of the locked file-order first-n draw. Fixes the LLaVA "
                         "task-type ordering bias (file-order first-n at 150K is ~all "
                         "conversation, excluding detail/complex-reasoning). NEW stem only — "
                         "never pass this on a locked run; the default (off) is bit-identical "
                         "to the locked data slice")
    ap.add_argument("--ckpt-tag", default=None,
                    help="tag for per-epoch checkpoints bridge_<enc>_<tag>_ep<N>.pt (default: <limit/1000>k)")
    resume_group = ap.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        help="resume the same experiment and target from a checkpoint",
    )
    resume_group.add_argument(
        "--continue-from",
        help="extend a completed flat-LR checkpoint under a required new --ckpt-tag",
    )
    ap.add_argument(
        "--acknowledge-legacy-provenance",
        action="store_true",
        help="allow an explicitly verified legacy checkpoint with missing provenance fields",
    )
    ap.add_argument(
        "--allow-missing-mlp-reference",
        action="store_true",
        help="explicitly allow a fresh MLP encoder family with no 150K reference yet; "
             "the opt-in is recorded in checkpoint provenance",
    )
    # --- Corrected protocol (two independent levers; defaults reproduce every locked run) ---
    ap.add_argument("--prompt-format", choices=["raw", "chatml_v1"], default="raw",
                    help="raw = the format all existing checkpoints were trained "
                         "under (default); chatml_v1 = Qwen2 ChatML with an explicit "
                         "system turn and the visual span inside the user turn")
    ap.add_argument("--supervise-eos", action="store_true",
                    help="supervise the assistant <|im_end|> so the bridge is "
                         "trained to STOP. Requires --prompt-format chatml_v1. "
                         "Appended only to answers that were not truncated")
    ap.add_argument("--system-text", default=None,
                    help="override the ChatML system prompt; the empty string omits "
                         "the system turn entirely (default: the module's versioned text)")
    # --- Throughput profiling (measurement only; 0 = OFF = the unmodified training path) ---
    ap.add_argument("--profile-steps", type=int, default=0,
                    help="record per-phase timings for N steps then stop recording "
                         "(0 = off). Measurement only: never changes weights, data, "
                         "RNG or the loss. Use a throwaway --ckpt-tag")
    ap.add_argument("--profile-warmup", type=int, default=20,
                    help="steps timed and discarded before recording (allocator growth "
                         "and kernel autotuning make early steps unrepresentative)")
    ap.add_argument("--profile-out", default=None,
                    help="write the profile JSON here (default: "
                         "outputs/diagnostics/profile_<encoder>_<tag>.json)")
    ap.add_argument("--profile-trace", default=None,
                    help="also capture a short torch.profiler window to this Chrome "
                         "trace path (kernel-level detail, e.g. bitsandbytes dequant)")
    args = ap.parse_args()

    # The corrected-protocol spec. Defaults ("raw", supervise_eos=False) reproduce
    # every locked run token-for-token; --prompt-format/--supervise-eos opt in.
    system_text = (None if args.system_text == "" else
                   (args.system_text if args.system_text is not None else P.SYSTEM_TEXT))
    spec = P.PromptSpec(prompt_format=args.prompt_format,
                        supervise_eos=args.supervise_eos,
                        system_text=system_text,
                        max_answer_tokens=args.max_answer_tokens)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)
    if args.seed is not None:
        print(f"SEED OVERRIDE: {seed} (B-seeds replication run)")
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()

    name = args.encoder or cfg["models"].get("active_encoder", "ijepa")
    validate_strip_cls(name, args.strip_cls)
    llm_precision = args.llm_precision or ("8bit" if args.eight_bit else "4bit")
    log(f"loading frozen encoder '{name}' and the LLM ({llm_precision}) ...")
    enc = load_vision_encoder(name, cfg, device)
    llm = load_llm(cfg, log=log, precision=llm_precision)

    bcfg = cfg["bridge"]
    bridge_config = {
        "type": args.bridge_type or bcfg.get("type", "qformer"),
        "num_query_tokens": args.query_tokens or bcfg["num_query_tokens"],
        "hidden_size": bcfg["hidden_size"],
        "num_hidden_layers": args.bridge_layers or bcfg["num_hidden_layers"],
        "num_attention_heads": bcfg["num_attention_heads"],
        "drop_cls": args.strip_cls,
    }
    bridge_config = resolve_qformer_bridge_config(
        bridge_config,
        dropout=args.qformer_dropout,
        norm_first=args.qformer_norm_first,
    )
    if bridge_config["type"] in ("mlp", "pool"):
        bridge_config["mlp_hidden"] = resolve_mlp_hidden(
            bridge_config["type"], name, args.mlp_hidden
        )
        bridge_config["mlp_depth"] = args.mlp_depth or 3
    if args.allow_missing_mlp_reference and bridge_config["type"] != "mlp":
        raise SystemExit("--allow-missing-mlp-reference is valid only for a fresh MLP")

    # Resume/continuation loads the parent architecture exactly. Same-experiment resume keeps
    # its original target and tag; continuation requires an explicit new stem and larger target.
    source_path = args.resume or args.continue_from
    is_continuation = args.continue_from is not None
    resume_ckpt, start_epoch = None, 0
    epoch_provenance = {
        "original_target_epochs": args.epochs,
        "extended_target_epochs": args.epochs,
        "missing_provenance": (),
    }
    missing_provenance = ()
    legacy_rng_reset = False
    rng_state_to_restore = None
    if source_path:
        resume_ckpt = torch.load(source_path, map_location=device, weights_only=False)
        epoch_provenance = validate_epoch_operation(
            resume_ckpt,
            target_epochs=args.epochs,
            requested_tag=args.ckpt_tag,
            continuation=is_continuation,
            lr_schedule=args.lr_schedule,
            acknowledge_missing=args.acknowledge_legacy_provenance,
        )
        bridge_config = resume_ckpt.get("bridge_config", bridge_config)
        stored_qformer_options = validate_qformer_resume_options(
            bridge_config,
            requested_dropout=args.qformer_dropout,
            requested_norm_first=args.qformer_norm_first,
        )
        bridge_config = resolve_qformer_bridge_config(
            bridge_config,
            dropout=stored_qformer_options[0],
            norm_first=stored_qformer_options[1],
        )
        if args.strip_cls != bridge_config.get("drop_cls", False):
            raise SystemExit(f"--strip-cls={args.strip_cls} conflicts with the checkpoint's "
                             f"drop_cls={bridge_config.get('drop_cls', False)}; the flag is part "
                             "of the trained architecture and cannot change on resume")
        stored_sched = resume_ckpt.get("lr_schedule") or "flat"
        if args.lr_schedule != stored_sched:
            raise SystemExit(f"--lr-schedule={args.lr_schedule} conflicts with the checkpoint's "
                             f"{stored_sched}; the schedule shapes the whole training "
                             "trajectory and cannot change on resume")
        stored_format = P.format_of(resume_ckpt)
        if args.prompt_format != stored_format:
            raise SystemExit(f"--prompt-format={args.prompt_format} conflicts with the "
                             f"checkpoint's {stored_format}; the prompt layout is what "
                             "the bridge was fit against and cannot change on resume")
        stored_supervise_eos = bool(resume_ckpt.get("supervise_eos", False))
        if args.supervise_eos != stored_supervise_eos:
            raise SystemExit(f"--supervise-eos={args.supervise_eos} conflicts with the "
                             f"checkpoint's {stored_supervise_eos}; whether the answer "
                             "span ends in a stop token defines the training objective")
        stored_shuffle = bool(resume_ckpt.get("shuffle_pool", False))
        if args.shuffle_pool != stored_shuffle:
            raise SystemExit(f"--shuffle-pool={args.shuffle_pool} conflicts with the checkpoint's "
                             f"shuffle_pool={stored_shuffle}; the LLaVA draw defines the training "
                             "data and cannot change on resume")
        if args.shuffle_pool:
            stored_seed = resume_ckpt.get("seed")
            if stored_seed is not None and stored_seed != seed:
                raise SystemExit(f"seed {seed} conflicts with the checkpoint's {stored_seed}; with "
                                 "--shuffle-pool the seed selects which LLaVA examples were drawn, "
                                 "so it cannot change on resume")
        if args.bridge_type and args.bridge_type != bridge_config.get("type", "qformer"):
            raise SystemExit(f"--bridge-type={args.bridge_type} conflicts with the checkpoint's "
                             f"{bridge_config.get('type', 'qformer')}; the connector is the "
                             "trained architecture and cannot change on resume")
        stored_prec = resume_ckpt.get("llm_precision") or ("8bit" if resume_ckpt.get("eight_bit") else "4bit")
        if llm_precision != stored_prec:
            raise SystemExit(f"llm precision {llm_precision} conflicts with the checkpoint's "
                             f"{stored_prec}; gradients flowed through that LLM, so precision "
                             "cannot change on resume")
        start_epoch = int(resume_ckpt.get("epoch", 0))
        if resume_ckpt.get("encoder") not in (None, enc.model_id):
            raise SystemExit(f"resume encoder {resume_ckpt.get('encoder')} != active {enc.model_id}")
        if resume_ckpt.get("encoder_revision") not in (None, enc.revision):
            raise SystemExit("resume encoder revision does not match the pinned upstream snapshot")
        if resume_ckpt.get("llm_revision") not in (None, llm.revision):
            raise SystemExit("resume LLM revision does not match the pinned upstream snapshot")
        trajectory = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_answer_tokens": args.max_answer_tokens,
            "warmup_frac": args.warmup_frac,
            "seed": seed,
            "loss_norm": args.loss_norm,
        }
        trajectory_missing = validate_trajectory_provenance(
            trajectory,
            resume_ckpt,
            args.acknowledge_legacy_provenance,
        )
        # Restore the data/split-defining args from the checkpoint so a resume continues on the
        # SAME dataset, mix, and validation split. If the user explicitly passed a *different*
        # (non-default) value, refuse rather than silently retrain on the wrong data. The
        # defaults now come from the config, so compare against the parser's resolved default
        # rather than a hardcoded literal (which would silently go stale if the config changed).
        data_keys = ["limit", "short_frac", "val_frac"]
        if not is_continuation:
            data_keys.append("ckpt_tag")
        data_missing = []
        for key in data_keys:
            stored = resume_ckpt.get("tag" if key == "ckpt_tag" else key)
            if stored is None:
                data_missing.append("tag" if key == "ckpt_tag" else key)
                continue
            cur = getattr(args, key)
            if cur != ap.get_default(key) and cur != stored:
                raise SystemExit(f"--{key.replace('_', '-')}={cur} conflicts with checkpoint's "
                                 f"{stored}; omit it to resume on the original data")
            setattr(args, key, stored)
        if data_missing and not args.acknowledge_legacy_provenance:
            raise SystemExit(
                f"checkpoint has missing provenance for {', '.join(data_missing)}; pass "
                "--acknowledge-legacy-provenance only after verifying the legacy recipe"
            )
        rng_state_to_restore = resume_ckpt.get("rng_state")
        legacy_rng_reset = rng_state_to_restore is None
        if legacy_rng_reset and not args.acknowledge_legacy_provenance:
            raise SystemExit(
                "checkpoint has missing provenance for rng_state; pass "
                "--acknowledge-legacy-provenance to record a known RNG reset"
            )
        missing_provenance = tuple(dict.fromkeys(
            epoch_provenance["missing_provenance"]
            + trajectory_missing
            + tuple(data_missing)
            + (("rng_state",) if legacy_rng_reset else ())
        ))
        if missing_provenance:
            log("LEGACY PROVENANCE ACKNOWLEDGED: missing " + ", ".join(missing_provenance))
        operation = "CONTINUE" if is_continuation else "RESUME"
        log(f"{operation} from {Path(source_path).name}: through epoch {start_epoch} "
            f"(limit={args.limit}, short_frac={args.short_frac}, val_frac={args.val_frac})")

    bridge = build_bridge(bridge_config, enc.hidden_dim, llm.model.config.hidden_size)
    mlp_reference_provenance = None
    if resume_ckpt is None and bridge_config["type"] == "mlp":
        ref_name = MLP_REFERENCE_ALIAS.get(name, name)
        reference_path = Path(cfg["paths"]["checkpoints"]) / f"bridge_{ref_name}_150k_mlp_ep3.pt"
        mlp_reference_provenance = validate_mlp_reference(
            bridge,
            bridge_config,
            reference_path,
            allow_missing=args.allow_missing_mlp_reference,
        )
        if mlp_reference_provenance["status"] == "matched":
            log(
                f"MLP REFERENCE MATCH: {reference_path.name}, "
                f"{mlp_reference_provenance['parameter_count']:,} parameters"
            )
        else:
            log(
                f"MLP REFERENCE MISSING — EXPLICIT OPT-IN: {reference_path.name}, "
                f"{mlp_reference_provenance['parameter_count']:,} parameters"
            )
    elif resume_ckpt is not None:
        if args.allow_missing_mlp_reference:
            raise SystemExit("--allow-missing-mlp-reference is valid only for a fresh MLP")
        mlp_reference_provenance = resume_ckpt.get("mlp_reference")
    bridge = bridge.to(device)
    if resume_ckpt is not None:
        bridge.load_state_dict(resume_ckpt["state_dict"])
    bridge.train()
    if bridge.drop_cls:
        log("[CLS] ABLATION ACTIVE: bridge drops memory token 0 — CLIP's 257-token "
            "memory becomes 256 pure patch tokens (matching I-JEPA's structure)")
    bridge_summary = (
        f"bridge[{bridge_config['type']}]: "
        f"{sum(p.numel() for p in bridge.parameters()) / 1e6:.1f}M params, "
        f"{bridge_config['num_query_tokens']} query tokens, "
        f"{bridge_config['num_hidden_layers']} layers (qformer only)"
    )
    if bridge_config["type"] == "qformer":
        bridge_summary += (
            f", dropout={bridge_config['dropout']}, "
            f"norm_first={bridge_config['norm_first']}"
        )
    log(bridge_summary)
    # Record what this run ACTUALLY used (config defaults after any CLI override / resume
    # restore), so the log is a self-contained record of the hyperparameters.
    log("resolved hyperparameters (config/default.yaml -> CLI override): "
        f"epochs={args.epochs} lr={args.lr} weight_decay={args.weight_decay} "
        f"batch_size={args.batch_size} grad_accum={args.grad_accum} "
        f"(effective batch {args.batch_size * args.grad_accum}) limit={args.limit} "
        f"short_frac={args.short_frac} val_frac={args.val_frac} "
        f"max_answer_tokens={args.max_answer_tokens} lr_schedule={args.lr_schedule}")

    # Build the mixed example list (VQAv2 short + LLaVA verbose).
    n_short = int(args.limit * args.short_frac)
    # --shuffle-pool draws a representative sample instead of the file-order first-n (fixes the
    # LLaVA task-type ordering bias). Off by default -> the locked data slice is unchanged.
    pool_seed = seed if args.shuffle_pool else None
    examples = build_examples(cfg, n_short, args.limit - n_short, shuffle_pool_seed=pool_seed)
    random.shuffle(examples)
    log(f"examples: {len(examples)} total ({n_short} VQAv2 short, {args.limit - n_short} LLaVA)"
        + (f" [POOL-SHUFFLED, seed {pool_seed}]" if pool_seed is not None else " [file-order first-n]"))

    # Cache features for every unique image (encode + store any missing), with a progress bar.
    cache_dir = cache_dir_for(cfg["paths"]["features"], name)
    unique = list(dict.fromkeys(p for p, _, _ in examples))
    to_cache = [p for p in unique if not has_features(cache_dir, p)]
    log(f"caching {len(to_cache):,} new image features (of {len(unique):,} unique) ...")
    bad = set()
    for path in tqdm(to_cache, desc="caching", unit="img", dynamic_ncols=True):
        try:
            save_features(cache_dir, path, encode_image(enc, Image.open(path).convert("RGB")))
        except Exception:
            bad.add(path)
    if bad:
        examples = [e for e in examples if e[0] not in bad]
        log(f"skipped {len(bad)} unreadable images")
    log(f"features ready ({len(to_cache) - len(bad):,} newly encoded, {len(unique):,} unique images)")

    # Hold out validation DETERMINISTICALLY by a hash of the image path, so the exact same
    # examples are validation on every (re)start — independent of shuffle order or of which
    # images happened to be (re)encoded this run. This keeps the per-epoch val-loss curve
    # comparable across resumes, and puts all questions about one image on the same side
    # (no intra-run image leakage between train and val).
    # The hash is taken over the CANONICAL image path, not the resolved one. Hashing a
    # host-specific mount path makes validation membership host-dependent. canonical_dir_map()
    # rewrites to the first-listed candidate, reproducing the frozen split across mounts.
    cut = int(round(args.val_frac * 100))
    dir_map = canonical_dir_map(cfg)
    is_val = lambda ex: zlib.crc32(split_key(ex[0], dir_map).encode()) % 100 < cut
    val_examples = [e for e in examples if is_val(e)]
    train_examples = [e for e in examples if not is_val(e)]
    log(f"split: {len(train_examples)} train, {len(val_examples)} val "
        f"(deterministic by CANONICAL image hash, val_frac={args.val_frac})")
    for resolved, canon in sorted(dir_map.items()):
        if resolved != canon:
            log(f"  split key canonicalised: {resolved} -> {canon}")

    opt = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_restored = False
    if resume_ckpt is not None and resume_ckpt.get("optimizer") is not None:
        opt.load_state_dict(resume_ckpt["optimizer"])
        optimizer_restored = True
        log("RESUME: optimizer state restored (AdamW moments continue, not reset)")
    if is_continuation and not optimizer_restored:
        raise SystemExit("explicit continuation requires a parent checkpoint with optimizer state")
    bs, accum = args.batch_size, args.grad_accum

    # Optional LR schedule (A1 ablation). The schedule is defined over OPTIMIZER steps
    # for the FULL target run (all --epochs), so a resume continues the same curve via
    # the restored scheduler state rather than restarting warmup.
    scheduler = None
    if args.lr_schedule == "cosine":
        steps_per_epoch = (len(train_examples) + bs - 1) // bs
        total_opt_steps = max(1, args.epochs * ((steps_per_epoch + accum - 1) // accum))
        warmup_steps = max(1, int(round(args.warmup_frac * total_opt_steps)))

        def _lr_lambda(step):
            """Linear warmup to 1x peak, then cosine decay to 0 over the remainder."""
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
        if resume_ckpt is not None and resume_ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            log("RESUME: LR-scheduler state restored (cosine position continues)")
        log(f"LR SCHEDULE: cosine — linear warmup {warmup_steps} of {total_opt_steps} "
            f"optimizer steps to peak {args.lr}, then cosine decay to 0")

    rng_restored = False
    if rng_state_to_restore is not None:
        restore_rng_state(rng_state_to_restore)
        rng_restored = True
        log("RESUME: Python, Torch CPU, and Torch CUDA RNG states restored")
    elif resume_ckpt is not None:
        log("LEGACY RNG RESET: parent has no RNG state; continuation permutation restarts")

    ckpt_dir = ensure_dir(cfg["paths"]["checkpoints"])
    tag = args.ckpt_tag or f"{args.limit // 1000}k"
    latest_path = ckpt_dir / f"bridge_{name}_{tag}_latest.pt"   # crash-recovery (overwritten each ckpt)
    continuation_provenance = resume_ckpt.get("continuation") if resume_ckpt else None
    if is_continuation:
        continuation_provenance = {
            "parent_checkpoint_path": str(Path(source_path).resolve()),
            "parent_checkpoint_sha256": file_sha256(source_path),
            "parent_tag": resume_ckpt.get("tag"),
            "parent_epoch": start_epoch,
            "optimizer_restored": optimizer_restored,
            "original_target_epochs": epoch_provenance["original_target_epochs"],
            "extended_target_epochs": epoch_provenance["extended_target_epochs"],
            "legacy_rng_reset": legacy_rng_reset,
        }

    def save_ckpt(path, epoch, val_loss):
        """Save bridge weights AND optimizer state (resumable) plus provenance, ATOMICALLY.

        Writes to a temp sibling then os.replace()s into place, so a kill mid-write never leaves
        a truncated checkpoint (critical for the unattended multi-day run's crash-recovery file).
        """
        tmp = path.with_name(path.name + ".tmp")
        torch.save({"encoder": enc.model_id, "encoder_revision": enc.revision,
                    "llm_model_id": llm.model_id, "llm_revision": llm.revision,
                    "bridge_config": bridge_config,
                    "eight_bit": args.eight_bit, "state_dict": bridge.state_dict(),
                    "optimizer": opt.state_dict(), "epoch": epoch, "val_loss": val_loss,
                    "lr": args.lr, "weight_decay": args.weight_decay,
                    "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                    "effective_batch_size": args.batch_size * args.grad_accum,
                    "max_answer_tokens": args.max_answer_tokens,
                    "warmup_frac": args.warmup_frac, "loss_norm": args.loss_norm,
                    "short_frac": args.short_frac, "val_frac": args.val_frac,
                    "limit": args.limit, "tag": tag, "lr_schedule": args.lr_schedule,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "llm_precision": llm_precision, "seed": seed,
                    "shuffle_pool": args.shuffle_pool, "pool_seed": pool_seed,
                    "target_epochs": args.epochs,
                    "original_target_epochs": epoch_provenance["original_target_epochs"],
                    "extended_target_epochs": epoch_provenance["extended_target_epochs"],
                    "continuation": continuation_provenance,
                    "optimizer_restored": optimizer_restored,
                    "rng_restored": rng_restored,
                    "legacy_rng_reset": legacy_rng_reset,
                    "missing_parent_provenance": missing_provenance,
                    "legacy_provenance_acknowledged": args.acknowledge_legacy_provenance,
                    "allow_missing_mlp_reference": args.allow_missing_mlp_reference,
                    "mlp_reference": mlp_reference_provenance,
                    "rng_state": capture_rng_state(),
                    # --- corrected-protocol provenance (absent => legacy raw) ---
                    "prompt_format": spec.prompt_format,
                    "supervise_eos": spec.supervise_eos,
                    "evidence_layer": spec.evidence_layer,
                    "eos_token_id": P.IM_END if spec.supervise_eos else None,
                    "eos_policy": "conditional_on_untruncated",
                    "emit_trailing_newline": False,
                    "system_text": spec.system_text,
                    "system_sha256": P.sha256_text(spec.system_text or ""),
                    "short_cue_sha256": P.sha256_text(P.SHORT_CUE),
                    "prompt_module_version": P.MODULE_VERSION}, tmp)
        os.replace(tmp, path)

    epochs_todo = list(range(start_epoch + 1, args.epochs + 1))
    if not epochs_todo:
        raise SystemExit(f"nothing to do: resume epoch {start_epoch} >= target --epochs {args.epochs}")
    if (resume_ckpt is None or is_continuation) and latest_path.exists():
        raise SystemExit(
            f"{latest_path.name} already exists — refusing to reuse a fresh experiment stem"
        )
    existing_epochs = [
        ckpt_dir / f"bridge_{name}_{tag}_ep{epoch}.pt"
        for epoch in epochs_todo
        if (ckpt_dir / f"bridge_{name}_{tag}_ep{epoch}.pt").exists()
    ]
    if existing_epochs:
        names = ", ".join(path.name for path in existing_epochs)
        raise SystemExit(f"checkpoint output already exists: {names}; resume the newest epoch")
    log(f"training: epochs {epochs_todo[0]}..{epochs_todo[-1]} (target {args.epochs}) x "
        f"{len(train_examples)} examples, batch={bs}, grad_accum={accum} "
        f"(effective {bs * accum}), lr={args.lr}, loss_norm={args.loss_norm}, tag='{tag}'")

    # Optional throughput profiling. Measurement only — nothing below changes what is
    # computed, so a profiled run and an unprofiled one produce identical losses and
    # checkpoints. The attention backend is recorded here because transformers picks it
    # implicitly (sdpa when unspecified) and it appears in no other artifact.
    prof = None
    if args.profile_steps:
        prof_out = args.profile_out or (
            Path(cfg["paths"]["outputs"]) / "diagnostics" / f"profile_{name}_{tag}.json")
        prof = StepProfiler(
            steps=args.profile_steps, warmup=args.profile_warmup, device=device,
            out_path=prof_out,
            meta={"encoder": enc.model_id, "bridge_type": bridge_config["type"],
                  "batch_size": bs, "grad_accum": accum, "llm_precision": llm_precision,
                  "attn_implementation": getattr(llm.model.config,
                                                 "_attn_implementation", "unknown"),
                  "torch": torch.__version__, "tag": tag, "limit": args.limit,
                  "gpu": torch.cuda.get_device_name(0),
                  # A profile is only valid for the forward it measured. The EOS/ChatML
                  # correction will change sequence composition and lengths, so stamp the
                  # sources: a differing hash means this profile is stale, not comparable.
                  "vlm_sha256": file_sha256(Path(__file__).resolve().parent.parent
                                            / "src" / "models" / "vlm.py"),
                  "trainer_sha256": file_sha256(Path(__file__).resolve())})
        log(f"PROFILING: {args.profile_warmup} warm-up + {args.profile_steps} measured "
            f"steps -> {prof_out} (attn={prof.meta['attn_implementation']})")

    # A short torch.profiler window for kernel-level attribution (which kernels dominate,
    # e.g. bitsandbytes dequantisation). with_stack stays off: it distorts short kernels.
    trace_prof = None
    if args.profile_trace:
        trace_path = Path(args.profile_trace)
        ensure_dir(trace_path.parent)
        trace_prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(
                wait=args.profile_warmup, warmup=3, active=4, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(str(trace_path)),
            record_shapes=True, with_stack=False)
        trace_prof.start()
        log(f"TRACE: torch.profiler wait={args.profile_warmup} warmup=3 active=4 "
            f"-> {trace_path}")
    step_times, train_t0, global_batch = [], time.perf_counter(), 0
    for epoch in epochs_todo:
        random.shuffle(train_examples)
        starts = list(range(0, len(train_examples), bs))
        running = 0.0
        opt.zero_grad()
        # Live tqdm bar with running + recent loss in the postfix.
        pbar = tqdm(starts, desc=f"epoch {epoch}/{args.epochs}", unit="batch", dynamic_ncols=True)
        for bi, start in enumerate(pbar, 1):
            t = time.perf_counter()
            if prof is not None and prof.active:
                prof.start_step()
            batch = train_examples[start:start + bs]
            # One synchronous np.load + one pageable H2D copy PER SAMPLE, so this span
            # covers both; splitting them needs src/data/feature_cache.py (out of scope).
            with _span(prof, "cpu", "feature_io_h2d"):
                feats = torch.cat([load_features(cache_dir, p, device) for p, _, _ in batch], dim=0)
            # Two nested spans: the GPU span is device time, the _wall span is elapsed time.
            # Their difference bounds the CPU-side cost inside vlm_loss_batch (per-example
            # tokenisation, embedding lookups, the two padding loops).
            with _span(prof, "cpu", "forward_wall"), _span(prof, "cuda", "forward"):
                loss = vlm_loss_batch(bridge, llm, feats, [x[1] for x in batch],
                                      [x[2] for x in batch], device, args.max_answer_tokens,
                                      spec=spec, loss_norm=args.loss_norm) / accum
            with _span(prof, "cpu", "backward_wall"), _span(prof, "cuda", "backward"):
                loss.backward()
            # The CPU blocks here until the queued forward+backward drain. That wait is real
            # serial wall time, but the SAME milliseconds are inside the forward/backward
            # CUDA-event spans, so it is categorised 'wait' and never added to them.
            with _span(prof, "wait", "gpu_drain_at_loss_item"):
                lv = loss.item() * accum
            running += lv
            if bi % accum == 0:
                with _span(prof, "cpu", "optimizer_wall"), _span(prof, "cuda", "optimizer"):
                    opt.step()
                    opt.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
            global_batch += 1
            step_times.append(time.perf_counter() - t)
            with _span(prof, "cpu", "logging"):
                pbar.set_postfix(loss=f"{lv:.3f}", avg=f"{running / bi:.3f}")
            # Mid-epoch crash-recovery: mark epoch-1 completed so a resume redoes this epoch.
            if args.ckpt_every and global_batch % args.ckpt_every == 0:
                with _span(prof, "cpu", "crash_ckpt"):
                    save_ckpt(latest_path, epoch - 1, float("nan"))
                tqdm.write(f"[{time.perf_counter() - _T0:7.1f}s]   crash-ckpt @ batch "
                           f"{global_batch} (avg train loss {running / bi:.4f})")
            if prof is not None and prof.active:
                prof.end_step()
                if not prof.active:
                    # Recording just finished — persist now so a later failure (validation,
                    # a 723 MB checkpoint write) cannot lose the profile.
                    log(f"profile written: {prof.save()}")
            if trace_prof is not None:
                trace_prof.step()
        if len(starts) % accum != 0:
            opt.step()
            opt.zero_grad()
            if scheduler is not None:
                scheduler.step()
        t_val = time.perf_counter()
        vloss = eval_loss(bridge, llm, cache_dir, val_examples, device, bs,
                          args.max_answer_tokens, spec=spec,
                          loss_norm=args.loss_norm) if val_examples else float("nan")
        # Per-epoch checkpoint (NEVER overwritten) + a rolling 'latest' pointing at the last full epoch.
        ep_path = ckpt_dir / f"bridge_{name}_{tag}_ep{epoch}.pt"
        if ep_path.exists():
            raise SystemExit(f"{ep_path.name} already exists — refusing to overwrite a saved epoch; "
                             f"resume from the latest epoch or use a fresh --ckpt-tag")
        t_ckpt = time.perf_counter()
        save_ckpt(ep_path, epoch, vloss)
        save_ckpt(latest_path, epoch, vloss)
        if prof is not None:
            # Per-epoch costs are recorded once, outside the per-step records: the val pass
            # and the two 723 MB checkpoint writes are amortised over a whole epoch.
            prof.meta.setdefault("epoch_costs", []).append(
                {"epoch": epoch, "val_s": t_ckpt - t_val,
                 "epoch_ckpt_s": time.perf_counter() - t_ckpt,
                 "n_val_examples": len(val_examples)})
        log(f"epoch {epoch}/{args.epochs}  train loss {running / len(starts):.4f}  "
            f"val loss {vloss:.4f}  -> {ep_path.name}  (elapsed {time.perf_counter() - train_t0:.1f}s)")

    if trace_prof is not None:
        trace_prof.stop()
    train_dt = time.perf_counter() - train_t0
    log(f"saved per-epoch checkpoints: bridge_{name}_{tag}_ep{epochs_todo[0]}..ep{epochs_todo[-1]}.pt")
    log(f"training wall-clock: {train_dt:.1f}s ({train_dt / len(epochs_todo):.1f}s/epoch)")
    log(f"per-batch: mean {statistics.mean(step_times) * 1000:.0f} ms")
    log(f"peak VRAM: {vram_report()}")
    if prof is not None:
        log(prof.format_table())
        log(f"profile written: {prof.save()}")


if __name__ == "__main__":
    main()
