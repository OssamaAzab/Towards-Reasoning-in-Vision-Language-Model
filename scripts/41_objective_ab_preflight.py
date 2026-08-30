"""Preflight for the training-objective A/B (token- vs sequence-normalised loss).

Runs ON THE NODE, before any model is loaded, and refuses the launch rather than discovering
a problem 11 hours into training. Every check is written so that it can fail: the mutation
harness in tests/test_objective_ab_launcher.py breaks each guarded property in turn and
requires the named check to reject.

WHAT THIS PROTECTS. The A arm of this experiment is not retrained — it is six checkpoints
that already exist. The comparison is therefore only valid if the B run differs from its
matched control in exactly one respect: the loss reduction. Most checks below exist to prove
that single-difference claim rather than assert it, by reading the control checkpoint's own
recorded recipe and comparing field by field.

    python scripts/41_objective_ab_preflight.py --encoder clip --seed 42 \
        --control-stem bridge_clip_w1_bf16_v1 --tag w1seq_s42_v1 --qids data/gqa/objective_4000_qids.json

Exit 0 = launch may proceed. Any non-zero exit = do not train.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The frozen endpoint for this experiment, pinned by content. A different slice, or an edited
# one, changes what the result means and must be a deliberate re-review, not a silent swap.
SLICE_SHA256 = "dbc0e49061092c1d738df942d8fcb25440568dfe5b2672104a7c9b6035081d46"
SLICE_N = 4000
SPENT_SLICES = ("eval_2000", "confirm_3000", "tune_500")

# Fields that MUST be identical between the control and the run we are about to start. Verified
# equal across all six controls before this file was written, so a mismatch means the recipe
# drifted, not that the controls disagree.
SHARED_RECIPE = {
    "llm_precision": "bf16", "batch_size": 8, "grad_accum": 1, "lr": 1e-4,
    "weight_decay": 0.01, "warmup_frac": 0.05, "lr_schedule": "flat",
    "max_answer_tokens": 64, "short_frac": 0.7, "val_frac": 0.05, "limit": 150000,
    "target_epochs": 5, "prompt_format": "chatml_v1", "supervise_eos": True,
    "evidence_layer": "corrected_chatml_v1_eos",
}
# Per-encoder geometry, measured from the encoders themselves and fixed by the locked protocol.
ENCODER_GEOMETRY = {
    "clip": {"mlp_hidden": 5888, "drop_cls": True, "model_id": "openai/clip-vit-large-patch14"},
    "ijepa": {"mlp_hidden": 5700, "drop_cls": False, "model_id": "facebook/ijepa_vith14_1k"},
}


def check(problems, ok, message):
    """Record one named check; print PASS or FAIL with the same text either way."""
    print(f"{'PASS' if ok else 'FAIL'}  {message}")
    if not ok:
        problems.append(message)
    return ok


def load_trainer():
    """Import the trainer so its own constants are the ones checked, not a copy."""
    path = ROOT / "scripts" / "06c_train_bridge.py"
    spec = importlib.util.spec_from_file_location("train_bridge_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_of(path: Path) -> str:
    """Content hash of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_slice(problems, qids_path: Path):
    """The endpoint must be the frozen fourth slice, by content and not by filename."""
    if not check(problems, qids_path.is_file(), f"eval slice exists at {qids_path}"):
        return
    got = sha256_of(qids_path)
    check(problems, got == SLICE_SHA256,
          f"eval slice content hash {got[:16]} matches the pinned {SLICE_SHA256[:16]}")
    qids = json.loads(qids_path.read_text())
    check(problems, len(qids) == SLICE_N and len(set(qids)) == SLICE_N,
          f"eval slice holds {SLICE_N} distinct question ids (got {len(qids)}, "
          f"{len(set(qids))} distinct)")
    named = [s for s in SPENT_SLICES if s in qids_path.name]
    check(problems, not named,
          f"eval slice is not a spent endpoint (name matches none of {SPENT_SLICES}; "
          f"matched {named})")


def check_loss_implementation(problems):
    """The sequence reduction must exist and be selectable, or the B arm is a no-op re-run."""
    try:
        from src.models import vlm
    except Exception as exc:                                     # pragma: no cover
        check(problems, False, f"src.models.vlm imports cleanly ({exc})")
        return
    check(problems, getattr(vlm, "LOSS_NORMS", None) == ("token", "sequence"),
          f"vlm.LOSS_NORMS is the two documented reductions (got "
          f"{getattr(vlm, 'LOSS_NORMS', None)})")
    check(problems, callable(getattr(vlm, "sequence_normalised_loss", None)),
          "vlm.sequence_normalised_loss is importable and callable")

    # A behavioural check, not a presence check: the two reductions must actually differ on a
    # batch with unequal supervised spans. A stub that returned the token mean would pass every
    # check above and silently make the B arm identical to its control.
    try:
        import torch
        import torch.nn.functional as F
        torch.manual_seed(0)
        logits = torch.randn(2, 8, 32)
        labels = torch.full((2, 8), -100)
        labels[0, 7:] = 3
        labels[1, 2:] = 5
        keep = labels[:, 1:] != -100
        token = float(F.cross_entropy(logits[:, :-1, :][keep].float(),
                                      labels[:, 1:][keep], reduction="mean"))
        seq = float(vlm.sequence_normalised_loss(logits, labels))
        check(problems, abs(seq - token) > 1e-6,
              f"the sequence reduction differs from the token mean on unequal spans "
              f"(sequence {seq:.6f} vs token {token:.6f})")
    except Exception as exc:                                     # pragma: no cover
        check(problems, False, f"the two reductions could be compared ({exc})")


def check_trainer_contract(problems, trainer):
    """The objective must be recorded in provenance, or a resume could switch it silently."""
    check(problems, "loss_norm" in trainer.TRAJECTORY_FIELDS,
          "loss_norm is a trajectory-defining field, so it cannot change across a resume")
    check(problems, trainer.TRAJECTORY_LEGACY_DEFAULTS.get("loss_norm") == "token",
          "a checkpoint predating the flag is read as token-normalised, which is what it was")


def check_control(problems, trainer, control_stem: str, encoder: str, seed: int):
    """Read the control's own recorded recipe and prove the single-difference claim."""
    import torch
    path = ROOT / "outputs" / "checkpoints" / "corrected" / f"{control_stem}_ep5.pt"
    if not check(problems, path.is_file(), f"control checkpoint present at {path.name}"):
        return
    ck = torch.load(path, map_location="cpu", weights_only=False)

    geom = ENCODER_GEOMETRY[encoder]
    check(problems, ck.get("encoder") == geom["model_id"],
          f"control was trained on {geom['model_id']} (recorded {ck.get('encoder')})")
    check(problems, ck.get("seed") == seed,
          f"control seed {ck.get('seed')} matches the seed this run will use ({seed})")

    bc = ck.get("bridge_config", {})
    check(problems, bc.get("type") == "mlp" and bc.get("mlp_depth") == 3,
          f"control connector is the 3-layer MLP (got {bc.get('type')}, depth {bc.get('mlp_depth')})")
    check(problems, bc.get("mlp_hidden") == geom["mlp_hidden"],
          f"control mlp_hidden {bc.get('mlp_hidden')} is the {encoder} reference "
          f"{geom['mlp_hidden']}")
    check(problems, bool(bc.get("drop_cls")) == geom["drop_cls"],
          f"control drop_cls {bool(bc.get('drop_cls'))} matches the {encoder} convention "
          f"{geom['drop_cls']}")

    mismatched = {k: (ck.get(k), v) for k, v in SHARED_RECIPE.items() if ck.get(k) != v}
    check(problems, not mismatched,
          f"every shared recipe field matches the control (mismatched: {mismatched})")

    # The point of the whole experiment: the control must be the TOKEN-normalised arm.
    control_norm = ck.get("loss_norm", trainer.TRAJECTORY_LEGACY_DEFAULTS["loss_norm"])
    check(problems, control_norm == "token",
          f"the control arm is token-normalised (recorded/recovered '{control_norm}'), so "
          f"the sequence run is a contrast and not a duplicate")


def check_no_overwrite(problems, encoder: str, tag: str):
    """A run whose outputs already exist would silently replace a reported artifact."""
    ckpt_dir = ROOT / "outputs" / "checkpoints"
    clash = sorted(p.name for p in ckpt_dir.glob(f"bridge_{encoder}_{tag}_ep*.pt"))
    check(problems, not clash,
          f"no checkpoint already exists for bridge_{encoder}_{tag} (found: {clash})")


def check_gpu(problems, require_gpu: bool):
    """One permitted card, unshared. Skipped only when explicitly running off-node."""
    if not require_gpu:
        print("SKIP  GPU checks (--no-gpu): this preflight is not running on the compute node")
        return
    try:
        names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True).stdout.split("\n")
        names = [n.strip() for n in names if n.strip()]
    except Exception as exc:
        check(problems, False, f"nvidia-smi is queryable ({exc})")
        return
    check(problems, len(names) == 1, f"exactly one GPU is visible (saw {len(names)}: {names})")
    check(problems, all("RTX PRO 6000" in n for n in names),
          f"the allocated card is an RTX PRO 6000 (saw {names})")
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    occupants = [line for line in apps.split("\n") if line.strip()]
    check(problems, not occupants,
          f"no other compute process is on this card (found {len(occupants)})")


def main() -> None:
    """Run every check and exit non-zero if any failed."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--encoder", required=True, choices=sorted(ENCODER_GEOMETRY))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--control-stem", required=True)
    ap.add_argument("--tag", required=True, help="ckpt tag for the new sequence-normalised run")
    ap.add_argument("--qids", default="data/gqa/objective_4000_qids.json")
    ap.add_argument("--no-gpu", action="store_true",
                    help="skip the GPU checks (for reviewing this file off the node)")
    args = ap.parse_args()

    print(f"=== objective A/B preflight: {args.encoder} seed {args.seed} "
          f"vs control {args.control_stem} ===")
    problems: list[str] = []
    trainer = load_trainer()

    check_slice(problems, ROOT / args.qids)
    check_loss_implementation(problems)
    check_trainer_contract(problems, trainer)
    check_control(problems, trainer, args.control_stem, args.encoder, args.seed)
    check_no_overwrite(problems, args.encoder, args.tag)
    check_gpu(problems, require_gpu=not args.no_gpu)

    print()
    if problems:
        print(f"PREFLIGHT REJECTED THE LAUNCH — {len(problems)} failed check(s):")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(92)
    print("PREFLIGHT PASSED — the sequence-normalised run differs from its control in the "
          "loss reduction alone.")


if __name__ == "__main__":
    main()
