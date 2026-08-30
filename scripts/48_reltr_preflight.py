"""Preflight for the RelTR evaluation job. Runs ON THE NODE, before any model is loaded.

    python scripts/48_reltr_preflight.py --manifest <snapshot> --qids <slice> --out-dir <dir> \
        --checkpoint-stem bridge_clip_w1_bf16_v1_ep5 --encoder clip [--require-committed-source]

WHY ON THE NODE. Hashing caches anywhere else cannot prove which files THIS allocation will read.
A cluster copy that silently diverged from the reviewed one is exactly the failure this catches,
and catching it before the LLM is in VRAM saves the allocation rather than wasting it.

WHAT IT REFUSES, AND WHY EACH ONE COST SOMETHING BEFORE.
  * a spent endpoint as the eval slice  — the locked 2,000, confirmatory 3,000 and objective
    4,000 are all spent; only the tuning surface may be used here
  * a leaked slice                      — the audit must show residual 0, or the generator was
    trained on images it is being evaluated on
  * a Visual Genome SGG checkpoint      — GQA is built on Visual Genome; only the Open Images
    geometry (289 entity / 31 relation classes) is admissible
  * an encoder that is not the checkpoint's — scripts/09 falls back to config's `active_encoder`
    (`ijepa`), which would load a 1280-dim encoder against a 1024-dim CLIP bridge and fail with
    the LLM already resident
  * records that already exist          — a re-run must never silently replace a record a
    reported number rests on
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPENT_SLICES = ("eval_2000", "locked", "confirm_3000", "objective_4000")
OI_HEADS = (290, 32)          # entity logits, relation logits — Open Images V6. VG is (152, 52).
EXPECTED_ARMS = ("reltr", "reltr_wrong_image", "oracle")


def sha256_file(p: Path) -> str:
    """Hash a file so the node verifies exactly what it will read."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--qids", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-stem", default="bridge_clip_w1_bf16_v1_ep5")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--audit", default="outputs/reltr_sgg/overlap_audit.json")
    ap.add_argument("--require-committed-source", action="store_true")
    args = ap.parse_args()

    fails: list[str] = []

    def check(ok: bool, msg: str):
        print(("PASS  " if ok else "FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    print("=== RelTR evaluation preflight ===")

    # ---- the eval surface ----
    check(not any(b in args.qids for b in SPENT_SLICES),
          f"eval slice {args.qids} is not a spent endpoint")
    qp = ROOT / args.qids
    check(qp.is_file(), f"eval slice exists at {args.qids}")

    man_p = Path(args.manifest)
    check(man_p.is_file(), f"manifest snapshot exists at {args.manifest}")
    if not man_p.is_file() or not qp.is_file():
        print("\nPREFLIGHT FAILED (cannot continue without the manifest and slice)")
        raise SystemExit(92)
    man = json.loads(man_p.read_text())

    check(man["development_surface"] == args.qids,
          f"manifest surface {man['development_surface']} == the slice passed to the evaluator")
    check(sha256_file(qp) == man["development_surface_sha256"],
          "eval slice hash matches the manifest")
    qids = json.loads(qp.read_text())
    check(man["expected_common_meta"]["n_questions"] == len(qids),
          f"manifest question count {man['expected_common_meta']['n_questions']} == "
          f"{len(qids)} in the slice")

    # ---- leakage: the audit must show residual zero for THIS slice ----
    ap_ = ROOT / args.audit
    check(ap_.is_file(), f"overlap audit exists at {args.audit}")
    if ap_.is_file():
        audit = json.loads(ap_.read_text())
        sl_man = ROOT / (str(qp).replace(".json", "_MANIFEST.json"))
        check(sl_man.is_file(), "the clean slice carries its derivation manifest")
        if sl_man.is_file():
            sm = json.loads(sl_man.read_text())
            check(sm.get("residual_leakage") == 0,
                  "the slice manifest asserts residual leakage 0")
            check(sm.get("audit_sha256") == sha256_file(ap_),
                  "the slice was derived from THIS audit file (hash matches)")
            excl = sm["excluded_images"]
            n_excl = len(excl["exact_matches"]) + len(excl["unresolved"])
            check(n_excl == audit["result"]["exact_matches"] + audit["result"]["unresolved"],
                  f"all {n_excl} matched-or-unresolved images are excluded from the slice")

    # ---- the graph generator: Open Images, never Visual Genome ----
    gm_path = man.get("sources", {}).get("graphs")
    if gm_path:
        gm = ROOT / str(gm_path).replace(".json", "_MANIFEST.json")
        check(gm.is_file(), "the graph cache carries its generation manifest")
        if gm.is_file():
            g = json.loads(gm.read_text())
            check(g["entity_classes"] == OI_HEADS[0] - 1 and g["predicates"] == OI_HEADS[1] - 1,
                  f"generator is Open Images geometry ({g['entity_classes']} entity / "
                  f"{g['predicates']} relation classes), not Visual Genome (151/51)")
            check("Visual Genome" not in g["training_data"].replace("NOT Visual Genome", ""),
                  "generator training data is not Visual Genome")
            check(g["mode"].startswith("inference only"), "generator was used inference-only")
            gc = ROOT / g["cache"]
            check(gc.is_file() and sha256_file(gc) == g["cache_sha256"],
                  "graph cache hash matches its generation manifest")

    # ---- the condition caches ----
    check(sorted(man["arms"]) == sorted(a for a in EXPECTED_ARMS if a != "oracle"),
          f"manifest defines the expected cached arms: {sorted(man['arms'])}")
    for a, spec in man["arms"].items():
        p = ROOT / spec["path"]
        check(p.is_file(), f"condition cache {a} exists at {spec['path']}")
        if p.is_file():
            check(sha256_file(p) == spec["sha256"], f"condition cache {a} hash matches manifest")
            cache = json.loads(p.read_text())
            check(len(cache) == len(qids),
                  f"condition cache {a} covers {len(cache)} questions == {len(qids)} in the slice")
            check(set(cache) == set(str(q) for q in qids),
                  f"condition cache {a} covers exactly the slice's question ids")

    # The control must not be the treatment. Identical caches would make the primary
    # contrast structurally zero and look like a clean null.
    if {"reltr", "reltr_wrong_image"} <= set(man["arms"]):
        check(man["arms"]["reltr"]["sha256"] != man["arms"]["reltr_wrong_image"]["sha256"],
              "the wrong-image control is not byte-identical to the treatment")
        t = json.loads((ROOT / man["arms"]["reltr"]["path"]).read_text())
        c = json.loads((ROOT / man["arms"]["reltr_wrong_image"]["path"]).read_text())
        same = sum(1 for q in t if json.dumps(t[q], sort_keys=True) == json.dumps(c.get(q), sort_keys=True))
        check(same < len(t) * 0.5,
              f"treatment and control differ on most questions ({same}/{len(t)} identical; "
              f"identical graphs are expected only where both are empty)")

    # ---- the checkpoint, and the encoder that must match it ----
    ck = ROOT / "outputs" / "checkpoints" / "corrected" / f"{args.checkpoint_stem}.pt"
    check(ck.is_file(), f"checkpoint {args.checkpoint_stem}.pt is present")
    if ck.is_file():
        check(sha256_file(ck) == man["checkpoint_sha256"], "checkpoint hash matches the manifest")
        import torch
        c = torch.load(ck, map_location="cpu", weights_only=False)
        enc = str(c.get("encoder", ""))
        check(args.encoder.lower() in enc.lower(),
              f"--encoder {args.encoder!r} matches the checkpoint's encoder {enc!r}")
        check(c.get("prompt_format") == "chatml_v1", "checkpoint is chatml_v1")
        check(bool(c.get("supervise_eos", False)), "checkpoint was trained with supervised EOS")
        check(c.get("llm_precision") == "bf16", "checkpoint records bf16 LLM precision")

    # ---- no-overwrite ----
    od = Path(args.out_dir)
    existing = sorted(od.glob("*records.json")) if od.is_dir() else []
    check(not existing, f"no records already exist in {args.out_dir} "
                        f"({len(existing)} found)" if existing else
                        f"no records already exist in {args.out_dir}")

    # ---- source provenance ----
    # The cluster project directory is a STAGED COPY, not a git checkout, so `git status` there
    # exits 128 with empty stdout. An earlier version tested `not dirty` on that empty string and
    # printed PASS unconditionally on every cluster run — a check that could not fail, which is
    # exactly the defect class this project keeps rediscovering. Where git cannot answer, the
    # provenance control is the SRC_COMMIT the launcher exports, so that is what gets checked.
    if args.require_committed_source:
        proc = subprocess.run(["git", "status", "--porcelain", "--", "scripts", "src"],
                              cwd=ROOT, capture_output=True, text=True)
        if proc.returncode == 0:
            dirty = proc.stdout.strip()
            check(not dirty, "scripts/ and src/ are committed (no uncommitted source)")
            if dirty:
                print("      " + dirty.replace("\n", "\n      ")[:600])
        else:
            src_commit = os.environ.get("SRC_COMMIT", "").strip()
            print("NOTE  this tree is not a git checkout (staged copy); provenance falls back "
                  "to the exported SRC_COMMIT")
            check(bool(re.fullmatch(r"[0-9a-f]{40}", src_commit)),
                  f"SRC_COMMIT is a full 40-char commit sha ({src_commit or '<unset>'})")

    n_arms = len(EXPECTED_ARMS)
    print()
    if fails:
        print(f"PREFLIGHT FAILED — {len(fails)} check(s) did not pass:")
        for f in fails:
            print(f"  - {f}")
        raise SystemExit(92)
    print(f"PREFLIGHT PASSED — {n_arms} arms, {n_arms} paired evaluator calls, "
          f"{len(qids)} questions each, residual leakage 0")


if __name__ == "__main__":
    main()
