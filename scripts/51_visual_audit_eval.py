"""Six-condition visual-information audit across three CLIP connectors. ONE GPU.

    python scripts/51_visual_audit_eval.py --coverage outputs/visual_audit/coverage.json \
        --out-dir outputs/visual_audit/eval_<jobid> [--limit 10]

WHAT IT ASKS. Whether GQA-relevant information is (a) absent from the CLIP encoder, (b) lost by
the connector, or (c) present but ignored by the frozen Qwen2. The six conditions separate those:

  1 correct          the image, unmodified                          — the reference
  2 wrong_image      a one-to-one deranged partner image            — is any gain image-specific?
  3 no_visual        no visual tokens at all (text-only)            — the floor
  4 shuffled         patch-token ORDER permuted, contents untouched — sensitivity to ordering of
                     already position-encoded patch tokens. NOT destruction of spatial info:
                     CLIP bakes position into each feature before the bridge ever sees it.
  5 mask_relevant    the question's own referent box greyed out     — does the right region matter?
  6 mask_irrelevant  a size-matched non-referent box greyed out     — control for "a box was hidden"

NOTHING IS TRAINED AND NO MODEL CODE IS TOUCHED. `vlm_generate` already takes encoder features as
an argument, so every condition is a transformation of its INPUT; condition 3 is the existing
`text_only_generate`. Encoder, connectors and LLM are loaded frozen and in eval mode.

ONE MODEL LOAD, ALL THREE CONNECTORS. The encoder and the 7B LLM dominate both memory and load
time and are identical across connectors, so the three bridges are loaded alongside them and every
condition is scored under one decoding path. Splitting this per connector would compare across
model loads.

WHAT MAKES THE COMPARISON PAIRED. Every condition is scored on the SAME question ids, with the
same prompt spec, the same max_new_tokens and the same `exact_full` scorer. Condition 3 involves
no bridge, so it is computed once per question and is byte-identical across connectors by
construction — the analysis asserts that as a harness check.

CLS IS NOT SHUFFLED. `drop_cls=True` makes the bridge slice token 0 AFTER receiving these
features, so permuting the whole axis would change which token is discarded and confound
condition 4 with a dropped random patch. See src/data/visual_audit.shuffle_patch_positions.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.prompt as P  # noqa: E402
from src.data.gqa import GQADataset  # noqa: E402
from src.data.visual_audit import mask_image_region, shuffle_patch_positions  # noqa: E402
from src.eval.metrics import METRIC_VERSION, exact_full  # noqa: E402
from src.models.connectors import build_bridge  # noqa: E402
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.models.llm import load_llm  # noqa: E402
from src.models.vlm import text_only_generate, vlm_generate  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

SEED = 42
CONNECTORS = {
    "mlp256": "bridge_clip_w1_bf16_v1_ep5",
    "pool32": "bridge_clip_w1pool_bf16_v1_ep5",
    "qformer32": "bridge_clip_w1qf_bf16_v1_ep5",
}
CONDITIONS = ("correct", "wrong_image", "no_visual", "shuffled",
              "mask_relevant", "mask_irrelevant")
SPENT = ("eval_2000", "locked", "confirm_3000", "objective_4000")


def sha256_file(p: Path) -> str:
    """Hash a file so every input and output is pinned."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_derangement(qids, pairs, rng_seed=SEED):
    """One-to-one wrong-image partner per question: every image donated exactly once.

    Picking a partner independently per question collapses onto popular donors and makes the
    control's image distribution unlike the treatment's. Sorting by a complexity profile and
    swapping ADJACENT pairs gives a perfect matching whose partners are the most similar
    available. Matched on object count then image id — never on the answer.
    """
    prof = {q: (pairs[q]["n_objects"], pairs[q]["image_id"]) for q in qids}
    pool = sorted(qids, key=lambda q: prof[q])
    donor = {}
    for i in range(0, len(pool) - 1, 2):
        a, b = pool[i], pool[i + 1]
        donor[a], donor[b] = b, a
    if len(pool) % 2:
        a, b, c = pool[-3], pool[-2], pool[-1]
        donor[a], donor[b], donor[c] = b, c, a

    def shares(q):
        return pairs[donor[q]]["image_id"] == pairs[q]["image_id"]
    for q in pool:
        if not shares(q):
            continue
        for r in pool:
            if r == q or shares(r):
                continue
            dq, dr = donor[q], donor[r]
            if (pairs[dr]["image_id"] != pairs[q]["image_id"]
                    and pairs[dq]["image_id"] != pairs[r]["image_id"] and dr != q and dq != r):
                donor[q], donor[dr] = dr, q
                donor[r], donor[dq] = dq, r
                break
    bad = [q for q in pool if pairs[donor[q]]["image_id"] == pairs[q]["image_id"]]
    if bad:
        raise SystemExit(f"FAIL {len(bad)} wrong-image donors share their own image")
    if sorted(donor.values()) != sorted(pool):
        raise SystemExit("FAIL the wrong-image donor map is not a permutation")
    return donor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coverage", default="outputs/visual_audit/coverage.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N questions")
    ap.add_argument("--connectors", default=",".join(CONNECTORS))
    ap.add_argument("--dump-probe-features", action="store_true",
                    help="also save pooled pre/post-connector representations for the probes")
    args = ap.parse_args()

    cov_p = ROOT / args.coverage
    if not cov_p.is_file():
        raise SystemExit(f"FAIL coverage audit {cov_p} absent; run scripts/50 first")
    cov = json.loads(cov_p.read_text())
    for bad in SPENT:
        if bad in cov["slice"]:
            raise SystemExit(f"FAIL coverage was built on {cov['slice']!r}, a spent endpoint")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "visual_audit_records.json"
    if rec_path.exists():
        raise SystemExit(f"FAIL refusing to overwrite {rec_path}; outputs/ is gitignored so an "
                         f"overwrite is unrecoverable")

    names = [n.strip() for n in args.connectors.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONNECTORS]
    if unknown:
        raise SystemExit(f"FAIL unknown connector(s) {unknown}; known: {sorted(CONNECTORS)}")

    qids = list(cov["usable_qids"])
    pairs = cov["mask_pairs"]
    if args.limit:
        qids = qids[: args.limit]
        pairs = {q: pairs[q] for q in qids}
    donor = build_derangement(qids, pairs)

    if not torch.cuda.is_available():
        raise SystemExit("FAIL a CUDA GPU is required")
    device = "cuda"
    set_seed(SEED)
    cfg = load_config()

    print("=== visual-information audit — six conditions x three CLIP connectors ===")
    print(f"questions : {len(qids)}{'  (SMOKE TEST)' if args.limit else ''}")
    print(f"connectors: {names}")
    print(f"conditions: {list(CONDITIONS)}")

    # ---- frozen models, loaded once ----
    enc = load_vision_encoder("clip", cfg, device)
    llm = load_llm(cfg, precision="bf16")
    bridges, specs, ckpt_meta = {}, {}, {}
    for n in names:
        cp = ROOT / "outputs" / "checkpoints" / "corrected" / f"{CONNECTORS[n]}.pt"
        if not cp.is_file():
            raise SystemExit(f"FAIL checkpoint {cp} absent")
        ck = torch.load(cp, map_location=device, weights_only=False)
        if "clip" not in str(ck.get("encoder", "")).lower():
            raise SystemExit(f"FAIL {n} was trained on {ck.get('encoder')!r}, not CLIP")
        b = build_bridge(ck["bridge_config"], enc.hidden_dim, llm.model.config.hidden_size)
        b.load_state_dict(ck["state_dict"])
        b.to(device).eval()
        b.requires_grad_(False)
        bridges[n] = b
        specs[n] = P.PromptSpec(prompt_format=P.format_of(ck),
                                supervise_eos=bool(ck.get("supervise_eos", False)),
                                system_text=ck.get("system_text", P.SYSTEM_TEXT),
                                max_answer_tokens=None)
        P.assert_compatible(ck, specs[n])
        ckpt_meta[n] = {"stem": CONNECTORS[n], "sha256": sha256_file(cp),
                        "bridge_type": ck["bridge_config"]["type"],
                        "num_query_tokens": ck["bridge_config"].get("num_query_tokens"),
                        "drop_cls": bool(ck["bridge_config"].get("drop_cls", False)),
                        "encoder": ck.get("encoder"), "epoch": ck.get("epoch"),
                        "seed": ck.get("seed"), "llm_precision": ck.get("llm_precision")}
        print(f"  {n:10s} {CONNECTORS[n]}  type={ck['bridge_config']['type']} "
              f"sha256={ckpt_meta[n]['sha256'][:12]}…")

    # All three must share one protocol, or the conditions are not comparable across connectors.
    protos = {(s.prompt_format, s.supervise_eos, s.evidence_layer) for s in specs.values()}
    if len(protos) != 1:
        raise SystemExit(f"FAIL connectors disagree on protocol: {protos}")
    spec = specs[names[0]]
    max_new = 32 if spec.supervise_eos else 10
    print(f"protocol  : {spec.prompt_format} supervise_eos={spec.supervise_eos} "
          f"layer={spec.evidence_layer} max_new_tokens={max_new} metric={METRIC_VERSION}")

    # The user body this script sends must be the SAME string scripts/07 sends, or the audit is
    # not comparable to any reported result. scripts/07 builds `ex.question + P.SHORT_CUE`; this
    # script routes through P.user_body. Assert the two agree on a probe question rather than
    # trusting that they do — job 2291478 ran every arm without the cue.
    _probe = "What colour is the dog?"
    if P.user_body(spec, _probe) != _probe + P.SHORT_CUE:
        raise SystemExit(
            f"FAIL prompt composition disagrees with scripts/07.\n"
            f"  this script : {P.user_body(spec, _probe)!r}\n"
            f"  scripts/07  : {_probe + P.SHORT_CUE!r}")
    print(f"prompt body: <question> + {P.SHORT_CUE!r}  (matches scripts/07)")

    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])

    # ---- pass 1: encoder features per question (correct + the two masked variants) ----
    from PIL import Image
    print("\npass 1: encoding images (correct, relevant-masked, irrelevant-masked) ...")
    feats, missing = {}, []
    t0 = time.time()
    for i, q in enumerate(qids, 1):
        ex = ds.get(q)
        if ex.image_path is None:
            missing.append(q)
            continue
        im = Image.open(ex.image_path).convert("RGB")
        mp = pairs[q]
        feats[q] = {
            "correct": encode_image(enc, im).cpu(),
            "mask_relevant": encode_image(enc, mask_image_region(im, mp["relevant_box"])).cpu(),
            "mask_irrelevant": encode_image(enc,
                                            mask_image_region(im, mp["irrelevant_box"])).cpu(),
        }
        if i % 25 == 0 or i == len(qids):
            print(f"  {i}/{len(qids)}  ({time.time() - t0:.0f}s)", flush=True)
    if missing:
        raise SystemExit(f"FAIL {len(missing)} questions have no image file: {missing[:5]}")

    # ---- pass 2: generate under every condition ----
    print("\npass 2: generating ...")
    gen = torch.Generator().manual_seed(SEED)
    records, t1 = [], time.time()
    for i, q in enumerate(qids, 1):
        ex = ds.get(q)
        f_correct = feats[q]["correct"].to(device)
        f_wrong = feats[donor[q]]["correct"].to(device)
        f_shuf = shuffle_patch_positions(f_correct, gen, has_cls=True)
        by_cond = {
            "correct": f_correct, "wrong_image": f_wrong, "shuffled": f_shuf,
            "mask_relevant": feats[q]["mask_relevant"].to(device),
            "mask_irrelevant": feats[q]["mask_irrelevant"].to(device),
        }
        rec = {"qid": q, "image_id": pairs[q]["image_id"], "question": ex.question,
               "gold": ex.answer, "category": ds.category_of(ex),
               "structural": ex.structural,
               "wrong_image_donor_qid": donor[q],
               "wrong_image_donor_image": pairs[donor[q]]["image_id"],
               "relevant_name": pairs[q]["relevant_name"],
               "irrelevant_name": pairs[q]["irrelevant_name"]}

        # vlm_generate and text_only_generate both call P.build(..., compose=False), which means
        # the string handed to them IS the finished user body — no cue is added downstream. The
        # bridge was TRAINED with the short-answer cue (scripts/06c) and every reported result
        # evaluates with it (scripts/07). Passing the bare question, as this script first did,
        # puts every arm out of distribution: job 2291478 answered "I'm sorry, but I can't answer
        # this question as I don't have any..." and truncated 10/10 text-only generations.
        # Composed through P.user_body so it is the same single source, not a fourth copy.
        prompt = P.user_body(spec, ex.question)

        # Condition 3 uses no bridge, so it is computed ONCE and shared. Recomputing it per
        # connector would introduce three chances for it to differ when it cannot.
        ans_to, stop_to = text_only_generate(llm, prompt, device, max_new_tokens=max_new,
                                             spec=spec, return_stop=True)
        rec["no_visual__ans"] = ans_to
        rec["no_visual__exact_full"] = bool(exact_full(ans_to, ex.answer))
        rec["no_visual__stop_reason"] = stop_to["stop_reason"]
        rec["no_visual__n_generated"] = stop_to["n_generated"]

        for n in names:
            for cond, f in by_cond.items():
                ans, stop = vlm_generate(bridges[n], llm, f, prompt, device,
                                         max_new_tokens=max_new, spec=spec, return_stop=True)
                rec[f"{n}__{cond}__ans"] = ans
                rec[f"{n}__{cond}__exact_full"] = bool(exact_full(ans, ex.answer))
                rec[f"{n}__{cond}__stop_reason"] = stop["stop_reason"]
                rec[f"{n}__{cond}__n_generated"] = stop["n_generated"]
        records.append(rec)
        if i % 10 == 0 or i == len(qids):
            print(f"  {i}/{len(qids)}  ({time.time() - t1:.0f}s)", flush=True)

    # ---- optional: pooled representations for the diagnostic probes ----
    probe_path = None
    if args.dump_probe_features:
        print("\ndumping pooled pre/post-connector representations for the probes ...")
        store = {"pre": {}, "post": {n: {} for n in names}}
        for q in qids:
            img = pairs[q]["image_id"]
            f = feats[q]["correct"].to(device)
            # Mean over patch tokens (CLS excluded) so pre- and post-connector vectors are
            # formed the same way and a probe difference cannot come from the pooling.
            store["pre"][img] = f[:, 1:, :].mean(1).squeeze(0).float().cpu().tolist()
            for n in names:
                v = bridges[n](f.float())
                store["post"][n][img] = v.mean(1).squeeze(0).float().cpu().tolist()
        probe_path = out_dir / "probe_features.json"
        probe_path.write_text(json.dumps(store))
        print(f"  wrote {probe_path.name} ({len(store['pre'])} images)")

    meta = {
        "audit": "visual-information audit — six conditions",
        "evidence_layer": spec.evidence_layer,
        "prompt_format": spec.prompt_format, "supervise_eos": spec.supervise_eos,
        "metric_version": METRIC_VERSION, "max_new_tokens": max_new,
        "seed": SEED, "n_questions": len(qids), "smoke_test": bool(args.limit),
        "conditions": list(CONDITIONS), "connectors": ckpt_meta,
        "coverage": args.coverage, "coverage_sha256": sha256_file(cov_p),
        "slice": cov["slice"], "slice_sha256": cov["slice_sha256"],
        "wrong_image_assignment": "one-to-one derangement, matched on object count then image id",
        "probe_note": "probe features are pooled frozen representations; a probe that recovers a "
                      "property shows DECODABILITY, never causal use by Qwen2",
        "models_frozen": True,
    }
    rec_path.write_text(json.dumps({"_meta": meta, "records": records}, indent=1))
    print(f"\nwrote {rec_path.relative_to(ROOT)}  sha256 {sha256_file(rec_path)[:16]}…")
    if probe_path:
        print(f"wrote {probe_path.relative_to(ROOT)}  sha256 {sha256_file(probe_path)[:16]}…")

    # ---- inline validity summary, so a smoke test is readable without the analysis ----
    n = len(records)
    print(f"\n=== accuracy (exact_full), n={n} ===")
    print(f"  {'connector':11s} " + " ".join(f"{c:>15s}" for c in CONDITIONS))
    for cn in names:
        row = []
        for c in CONDITIONS:
            key = "no_visual__exact_full" if c == "no_visual" else f"{cn}__{c}__exact_full"
            row.append(100 * sum(r[key] for r in records) / n)
        print(f"  {cn:11s} " + " ".join(f"{v:14.1f}%" for v in row))
    # src/models/vlm.py writes f"eos_{eos_token_id}" or "cap" — never the bare string "eos".
    # Comparing against "eos" counted every healthy arm as a failure (job 2291478 reported
    # 160/160 when the truth was 11), which is a check that can never pass.
    by_arm = collections.Counter(
        k[:-len("__stop_reason")] for r in records for k in r
        if k.endswith("__stop_reason") and not str(r[k]).startswith("eos_"))
    caps = sum(by_arm.values())
    total = sum(1 for r in records for k in r if k.endswith("__stop_reason"))
    print(f"\n  cap-truncated stops: {caps}/{total} arm-generations "
          f"(a truncated answer scores wrong on exact_full, so a condition that truncates "
          f"more is penalised by the metric rather than by the intervention)")
    for a, n in by_arm.most_common():
        print(f"    {a:34s} {n}")


if __name__ == "__main__":
    main()
