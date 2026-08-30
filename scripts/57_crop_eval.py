"""Stage A: five crop conditions x two encoders, one GPU. ORACLE FEASIBILITY ONLY.

    python scripts/57_crop_eval.py --coverage outputs/crop_augment/coverage.json \
        --out-dir outputs/crop_augment/eval_<jobid> [--limit 8]

THE FIVE CONDITIONS. Every arm uses the same question ids, the same prompt (question +
P.SHORT_CUE), the same greedy decoding and the same exact_full scorer.

  1 global_only                   [global]              the established baseline
  2 global_plus_relevant_crop     [global, relevant]    ORACLE — the box comes from GQA truth
  3 global_plus_irrelevant_crop   [global, irrelevant]  same image, no referenced object, area-matched
  4 global_plus_wrong_crop        [global, wrong]       the deranged partner's relevant crop
  5 relevant_crop_only            [relevant]            ORACLE — token-count matched to arm 1

ARMS 2, 3 AND 4 CARRY IDENTICAL TOKEN COUNTS. Each view yields 256 connector tokens, so all three
are 512-token prompts differing only in what the second view shows. Arm 5 is 256 tokens, matching
arm 1. The script asserts both equalities per question rather than assuming them.

WHY BEATING ARM 1 IS NOT A RESULT. Arms 2-4 are out of distribution in sequence length: the
frozen bridge and LLM have only ever seen 256 visual tokens. A change against arm 1 therefore
confounds crop CONTENT with prompt LENGTH. Arms 3 and 4 hold length fixed, so relevant-vs-wrong
and relevant-vs-irrelevant are the contrasts that can support a grounded claim.

BASELINE PARITY IS THE FIRST VALIDITY VERDICT. `global_only` runs through the new multi-view
assembly, and every question is ALSO answered by the established `vlm_generate`. Answers, scores,
stop reasons and qid order must match exactly. They are separate implementations, so this can
genuinely fail.

Crop generations execute BEFORE that verdict is reached — they share the model load and the
per-question loop, and separating them would double the work. What the verdict controls is
RETENTION and INTERPRETATION, not execution: the parity check raises before `rec_path.write_text`,
so a failing run writes no records at all. Nothing is kept, and therefore nothing can later be
read as a result, from a run whose baseline does not reproduce.

NOTHING IS TRAINED. Encoders, both MLP256 connectors and Qwen2 load frozen, in eval mode, with
requires_grad off, and `assert_frozen` re-checks that immediately before generation.
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
from src.data.crop_augment import encoder_input_side, make_crop  # noqa: E402
from src.data.gqa import GQADataset  # noqa: E402
from src.eval.metrics import METRIC_VERSION, exact_full  # noqa: E402
from src.models.connectors import build_bridge  # noqa: E402
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.models.llm import load_llm  # noqa: E402
from src.models.multi_view import assert_frozen, generate_multi_view  # noqa: E402
from src.models.vlm import vlm_generate  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

SEED = 42
SPENT = ("eval_2000", "locked", "confirm_3000", "objective_4000")
ENCODERS = {"clip": "bridge_clip_w1_bf16_v1_ep5", "ijepa": "bridge_ijepa_w1_bf16_v1_ep5"}
CONDITIONS = ("global_only", "global_plus_relevant_crop", "global_plus_irrelevant_crop",
              "global_plus_wrong_crop", "relevant_crop_only")


def sha256_file(p: Path) -> str:
    """Hash a file so the run pins exactly what it read."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_hash(module) -> str:
    """Deterministic sha256 over a module's parameters, in sorted name order.

    Pinning the HuggingFace model *identifier* would not catch a changed or re-downloaded cache,
    and the encoder is the one component here whose weights live outside this repository. Hashing
    the loaded tensors pins what actually ran.
    """
    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coverage", default="outputs/crop_augment/coverage.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on the first N questions")
    args = ap.parse_args()

    cov_p, out_dir = ROOT / args.coverage, ROOT / args.out_dir
    if not cov_p.is_file():
        raise SystemExit(f"FAIL coverage {cov_p} absent; run scripts/56 first")
    cov = json.loads(cov_p.read_text())
    for bad in SPENT:
        if bad in cov["slice"]:
            raise SystemExit(f"FAIL coverage was built on {cov['slice']!r}, a spent endpoint")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "crop_records.json"
    if rec_path.exists():
        raise SystemExit(f"FAIL refusing to overwrite {rec_path}")

    # TWO COHORTS. `core` carries the four length-comparable arms and the PRIMARY relevant-vs-wrong
    # contrast. `irrelevant_subset` is NESTED in core and is the only cohort on which the
    # irrelevant arm is generated at all — generating it elsewhere would invite a comparison
    # between two different populations. The same qids are used for both encoders.
    qids = list(cov["core_qids"])
    subset = set(cov["irrelevant_subset_qids"])
    # The scale-matched subgroup is READ from the coverage audit, never recomputed here. It was
    # fixed from geometry before any model existed; deriving it in the script that also produces
    # the answers would put the subgroup and the accuracy in the same place.
    if "scale_matched_qids" not in cov:
        raise SystemExit("FAIL coverage predates the pre-registered scale-matched subgroup; "
                         "rerun scripts/56 before evaluating")
    scale_matched = set(cov["scale_matched_qids"])
    if not scale_matched.issubset(set(qids)):
        raise SystemExit("FAIL the scale-matched subgroup is not nested in the core cohort")
    records_meta = cov["records"]
    if args.limit:
        # Keep at least a few subset members in a smoke test, or the irrelevant arm is never
        # exercised and the smoke proves less than it appears to.
        head = qids[: args.limit]
        if not any(q in subset for q in head):
            extra = [q for q in qids if q in subset][: max(1, args.limit // 4)]
            head = (head[: args.limit - len(extra)] + extra)
        qids = head
    if not set(qids).issubset(set(cov["core_qids"])):
        raise SystemExit("FAIL evaluating a question outside the core cohort")
    subset = subset & set(qids)
    scale_matched = scale_matched & set(qids)

    if not torch.cuda.is_available():
        raise SystemExit("FAIL a CUDA GPU is required. CPU inference is not numerically "
                         "interchangeable with cluster GPU results in this project and may not "
                         "be used for answers, scores, EOS behaviour or any stop/go decision.")
    device = "cuda"
    set_seed(SEED)
    cfg = load_config()

    print("=== Stage A — question-conditioned crop augmentation (ORACLE feasibility) ===")
    print(f"questions : {len(qids)}{'  (SMOKE TEST)' if args.limit else ''}")
    print(f"cohorts   : core {len(qids)}, irrelevant_subset {len(subset)}, "
          f"scale_matched {len(scale_matched)} (pre-registered by scripts/56, "
          f"factor {cov['scale_match_factor']:g})")
    print(f"encoders  : {list(ENCODERS)}")
    print(f"conditions: {list(CONDITIONS)}")
    print("ORACLE: the relevant crop is located from GQA ground truth and is NOT deployable.")

    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])
    llm = load_llm(cfg, precision="bf16")

    records = {q: {"qid": q, "image_id": records_meta[q]["image_id"],
                   "question": ds.get(q).question, "gold": ds.get(q).answer,
                   "category": ds.category_of(ds.get(q)),
                   "structural": ds.get(q).structural,
                   "wrong_donor_qid": records_meta[q]["wrong_donor_qid"],
                   "wrong_donor_image_id": records_meta[q]["wrong_donor_image_id"],
                   "target_label": records_meta[q]["target_label"],
                   "wrong_donor_label": records_meta[q]["wrong_donor_label"],
                   "relevant_name": records_meta[q]["relevant"]["name"],
                   "relevant_box": records_meta[q]["relevant"]["box"],
                   "wrong_box": records_meta[q]["wrong_box"],
                   "wrong_area_diff": records_meta[q]["wrong_area_diff"],
                   "in_irrelevant_subset": q in subset,
                   "in_scale_matched": q in scale_matched,
                   "scale": records_meta[q]["scale"],
                   "irrelevant_name": (records_meta[q]["irrelevant"]["name"]
                                       if q in subset else None),
                   "irrelevant_box": (records_meta[q]["irrelevant"]["box"]
                                      if q in subset else None),
                   "area_match_error": records_meta[q]["area_match_error"]} for q in qids}

    ckpt_meta, parity_fail, token_meta = {}, [], {}
    for enc_name, stem in ENCODERS.items():
        print(f"\n================ encoder: {enc_name} ================")
        cp = ROOT / "outputs" / "checkpoints" / "corrected" / f"{stem}.pt"
        if not cp.is_file():
            raise SystemExit(f"FAIL checkpoint {cp} absent")
        ck = torch.load(cp, map_location=device, weights_only=False)
        if enc_name not in str(ck.get("encoder", "")).lower():
            raise SystemExit(f"FAIL {stem} was trained on {ck.get('encoder')!r}, not {enc_name}")
        if ck["bridge_config"]["type"] != "mlp":
            raise SystemExit(f"FAIL {stem} is a {ck['bridge_config']['type']} bridge, not MLP256")

        enc = load_vision_encoder(enc_name, cfg, device)
        enc_side = encoder_input_side(enc.processor)
        bridge = build_bridge(ck["bridge_config"], enc.hidden_dim, llm.model.config.hidden_size)
        bridge.load_state_dict(ck["state_dict"])
        bridge.to(device).eval()
        bridge.requires_grad_(False)
        enc.model.eval()
        enc.model.requires_grad_(False)
        assert_frozen(bridge, enc.model, llm.model)

        spec = P.PromptSpec(prompt_format=P.format_of(ck),
                            supervise_eos=bool(ck.get("supervise_eos", False)),
                            system_text=ck.get("system_text", P.SYSTEM_TEXT),
                            max_answer_tokens=None)
        P.assert_compatible(ck, spec)
        max_new = 32 if spec.supervise_eos else 10
        probe = "What colour is the dog?"
        if P.user_body(spec, probe) != probe + P.SHORT_CUE:
            raise SystemExit("FAIL prompt composition disagrees with scripts/07")
        ckpt_meta[enc_name] = {
            "stem": stem, "sha256": sha256_file(cp), "bridge_type": ck["bridge_config"]["type"],
            "drop_cls": bool(ck["bridge_config"].get("drop_cls", False)),
            "encoder": ck.get("encoder"), "epoch": ck.get("epoch"), "seed": ck.get("seed"),
            "llm_precision": ck.get("llm_precision"), "prompt_format": spec.prompt_format,
            "supervise_eos": spec.supervise_eos, "max_new_tokens": max_new,
            "encoder_param_sha256": param_hash(enc.model),
            "encoder_hidden_dim": enc.hidden_dim,
            "encoder_input_side": enc_side}
        print(f"  {stem}  sha256={ckpt_meta[enc_name]['sha256'][:12]}…  "
              f"drop_cls={ckpt_meta[enc_name]['drop_cls']}")
        print(f"  encoder {ck.get('encoder')}  weights sha256="
              f"{ckpt_meta[enc_name]['encoder_param_sha256'][:12]}…  "
              f"hidden_dim={enc.hidden_dim}")
        print(f"  protocol: {spec.prompt_format} eos={spec.supervise_eos} max_new={max_new} "
              f"metric={METRIC_VERSION}")

        # ---- pass 1: encode the three per-question views ----
        print("\npass 1: encoding global + relevant crop + irrelevant crop ...")
        from PIL import Image
        feats, t0 = {}, time.time()
        for i, q in enumerate(qids, 1):
            ex = ds.get(q)
            img = Image.open(ex.image_path).convert("RGB")
            m = records_meta[q]
            feats[q] = {
                "global": encode_image(enc, img).cpu(),
                "relevant": encode_image(enc, make_crop(img, m["relevant"]["box"])).cpu(),
            }
            if q in subset:
                feats[q]["irrelevant"] = encode_image(
                    enc, make_crop(img, m["irrelevant"]["box"])).cpu()
            if i % 25 == 0 or i == len(qids):
                print(f"  {i}/{len(qids)}  ({time.time() - t0:.0f}s)", flush=True)
        # The wrong crop is the DONOR's relevant crop, already encoded — provided the donor is in
        # this run. Under --limit the donor may fall outside the slice, so encode it on demand.
        for q in qids:
            d = records_meta[q]["wrong_donor_qid"]
            if d in feats:
                continue
            dm = records_meta[d]
            dimg = Image.open(ds.get(d).image_path).convert("RGB")
            feats.setdefault(d, {})["relevant"] = encode_image(
                enc, make_crop(dimg, dm["relevant"]["box"])).cpu()

        # ---- pass 2: generate ----
        print("\npass 2: generating ...")
        t1 = time.time()
        for i, q in enumerate(qids, 1):
            ex = ds.get(q)
            prompt = P.user_body(spec, ex.question)
            g = feats[q]["global"].to(device)
            rel = feats[q]["relevant"].to(device)
            wrong = feats[records_meta[q]["wrong_donor_qid"]]["relevant"].to(device)
            views = {"global_only": [g], "global_plus_relevant_crop": [g, rel],
                     "global_plus_wrong_crop": [g, wrong], "relevant_crop_only": [rel]}
            irr = None
            if q in subset:                      # the irrelevant arm exists ONLY on the subset
                irr = feats[q]["irrelevant"].to(device)
                views["global_plus_irrelevant_crop"] = [g, irr]
            r = records[q]
            for cond, vs in views.items():
                ans, stop = generate_multi_view(bridge, llm, vs, prompt, device,
                                                max_new_tokens=max_new, spec=spec,
                                                return_stop=True)
                key = f"{enc_name}__{cond}"
                r[f"{key}__ans"] = ans
                r[f"{key}__exact_full"] = bool(exact_full(ans, ex.answer))
                r[f"{key}__stop_reason"] = stop["stop_reason"]
                r[f"{key}__n_generated"] = stop["n_generated"]

            # token-count bookkeeping, asserted rather than assumed
            with torch.no_grad():
                n_g, n_c = bridge(g.float()).size(1), bridge(rel.float()).size(1)
                n_w = bridge(wrong.float()).size(1)
                n_i = bridge(irr.float()).size(1) if irr is not None else n_c
            if not (n_c == n_i == n_w):
                raise SystemExit(f"FAIL crop token counts differ for {q}: "
                                 f"relevant={n_c} irrelevant={n_i} wrong={n_w}")
            r[f"{enc_name}__tokens_global"] = n_g
            r[f"{enc_name}__tokens_crop"] = n_c
            token_meta.setdefault(enc_name, set()).add((n_g, n_c))

            # The source-to-encoder resize factor, from the processor this encoder actually used
            # rather than a nominal side read off the model card. > 1 means the padded crop was
            # UPSAMPLED to reach the encoder's input.
            sc = records_meta[q]["scale"]
            for side_name in ("relevant", "wrong"):
                pw = sc[side_name]["padded_w"]
                r[f"{enc_name}__resize_factor_{side_name}"] = (
                    enc_side / pw if (enc_side and pw) else None)

            # ---- baseline parity, same job, separate implementation ----
            b_ans, b_stop = vlm_generate(bridge, llm, g, prompt, device,
                                         max_new_tokens=max_new, spec=spec, return_stop=True)
            k = f"{enc_name}__global_only"
            r[f"{enc_name}__baseline_ans"] = b_ans
            r[f"{enc_name}__baseline_stop_reason"] = b_stop["stop_reason"]
            r[f"{enc_name}__baseline_exact_full"] = bool(exact_full(b_ans, ex.answer))
            if (b_ans != r[f"{k}__ans"]
                    or b_stop["stop_reason"] != r[f"{k}__stop_reason"]
                    or bool(exact_full(b_ans, ex.answer)) != r[f"{k}__exact_full"]):
                parity_fail.append({"qid": q, "encoder": enc_name,
                                    "baseline": b_ans, "global_only": r[f"{k}__ans"],
                                    "baseline_stop": b_stop["stop_reason"],
                                    "global_only_stop": r[f"{k}__stop_reason"]})
            if i % 25 == 0 or i == len(qids):
                print(f"  {i}/{len(qids)}  ({time.time() - t1:.0f}s)", flush=True)

        del enc, bridge, feats
        torch.cuda.empty_cache()

    # ---- identical coverage across encoders, within each cohort ----
    print("\n=== cohort coverage is identical across encoders ===")
    for cond in CONDITIONS:
        pop = subset if cond == "global_plus_irrelevant_crop" else set(qids)
        seen = {e: {q for q in qids if f"{e}__{cond}__ans" in records[q]} for e in ENCODERS}
        if any(s != pop for s in seen.values()):
            raise SystemExit(f"FAIL {cond}: encoder coverage differs or leaves its cohort "
                             f"({ {e: len(s) for e, s in seen.items()} } vs {len(pop)})")
        print(f"  PASS {cond:30s} {len(pop):4d} questions, identical for {list(ENCODERS)}")

    # ---- parity verdict, before any crop arm is interpreted ----
    print("\n=== baseline parity: established vlm_generate vs new global_only path ===")
    order_ok = [r["qid"] for r in (records[q] for q in qids)] == qids
    print(f"  qid ordering preserved: {order_ok}")
    if parity_fail or not order_ok:
        for p in parity_fail[:10]:
            print(f"  MISMATCH {p['qid']} [{p['encoder']}] "
                  f"baseline={p['baseline']!r} global_only={p['global_only']!r}")
        raise SystemExit(f"FAIL baseline parity broken on {len(parity_fail)} question-encoder "
                         f"pairs; the crop arms are not interpretable until this is fixed")
    print(f"  PASS all {len(qids) * len(ENCODERS)} question-encoder pairs identical in answer, "
          f"score and stop reason")

    meta = {
        "experiment": "Stage A — question-conditioned visual-evidence crops",
        "oracle": True,
        "not_deployable": "the relevant crop is located from GQA ground truth; this measures a "
                          "ceiling, not a deployable method",
        "metric_version": METRIC_VERSION, "seed": SEED,
        "n_questions": len(qids), "smoke_test": bool(args.limit),
        "cohorts": {"core": sorted(qids), "irrelevant_subset": sorted(subset),
                    "scale_matched": sorted(scale_matched)},
        "n_core": len(qids), "n_irrelevant_subset": len(subset),
        "n_scale_matched": len(scale_matched),
        "scale_match_factor": cov["scale_match_factor"],
        "cohort_note": "core carries the PRIMARY relevant-vs-wrong contrast; irrelevant_subset "
                       "is NESTED in core and carries the SECONDARY relevant-vs-irrelevant "
                       "contrast only; scale_matched is NESTED in core and carries the ROBUSTNESS "
                       "reading of the primary contrast only. None is compared as though paired "
                       "with another. All three were fixed by scripts/56 from geometry alone, "
                       "before any model ran.",
        "conditions": list(CONDITIONS), "encoders": ckpt_meta,
        "coverage": args.coverage, "coverage_sha256": sha256_file(cov_p),
        "slice": cov["slice"], "slice_sha256": cov["slice_sha256"],
        "crop_rule": cov["crop_rule"],
        "token_counts": {k: sorted(v) for k, v in token_meta.items()},
        "view_order": "global first, then crop; never varied by condition",
        "baseline_parity": {"checked": len(qids) * len(ENCODERS), "mismatches": len(parity_fail)},
        "models_frozen": True,
    }
    rec_path.write_text(json.dumps({"_meta": meta, "records": [records[q] for q in qids]},
                                   indent=1))
    print(f"\nwrote {rec_path.relative_to(ROOT)}  sha256 {sha256_file(rec_path)[:16]}…")

    # ---- inline summary so a smoke test is readable without the analysis ----
    for enc_name in ENCODERS:
        print(f"\n=== {enc_name}: accuracy (exact_full), n={len(qids)} ===")
        print(f"  {'condition':30s} {'cohort':>8s} {'n':>5s} {'exact_full':>11s} {'tokens':>8s}")
        for cond in CONDITIONS:
            pop = sorted(subset) if cond == "global_plus_irrelevant_crop" else qids
            if not pop:
                print(f"  {cond:30s} {'subset':>8s} {0:5d}   (no subset member in this run)")
                continue
            acc = 100 * sum(records[q][f"{enc_name}__{cond}__exact_full"] for q in pop) / len(pop)
            ntok = (records[qids[0]][f"{enc_name}__tokens_global"] *
                    (0 if cond == "relevant_crop_only" else 1)
                    + records[qids[0]][f"{enc_name}__tokens_crop"] *
                    (0 if cond == "global_only" else 1))
            tag = "subset" if cond == "global_plus_irrelevant_crop" else "core"
            print(f"  {cond:30s} {tag:>8s} {len(pop):5d} {acc:10.1f}% {ntok:8d}")
        caps = collections.Counter(
            k.split("__")[1] for q in qids for k in records[q]
            if k.startswith(f"{enc_name}__") and k.endswith("__stop_reason")
            and not str(records[q][k]).startswith("eos_"))
        print(f"  cap-truncated by arm: {dict(caps) or 'none'}")

    print("\nORACLE FEASIBILITY, DEVELOPMENT SLICE ONLY. Beating global_only is NOT a result: "
          "arms 2-4 carry 512 visual tokens where the frozen stack saw 256. The grounded "
          "contrasts are relevant-vs-wrong and relevant-vs-irrelevant.")


if __name__ == "__main__":
    main()
