"""Verify the frozen text-only floor is byte-identical across a set of evaluations.

WHY THIS IS A SCRIPT AND NOT A HEREDOC. Launcher 21 carried this check inline as embedded
Python and it crashed on `TypeError: string indices must be integers` after all ten of its
evaluations had succeeded — it iterated the records payload directly, but corrected artifacts
wrap records as {"_meta": ..., "records": [...]}, so iterating yields the dict's KEYS. The
launcher's tests asserted that the string "byte-identical" appeared in the file; they could not
catch a bug in code they never executed. Embedded heredoc Python is untestable by construction,
so the logic lives here where tests can run it.

WHAT THE CHECK MEANS. The text-only floor uses no image, no encoder and no bridge, and decodes
greedily from frozen weights. Across evaluations on ONE GPU architecture it must therefore be
bit-identical. If it is not, decoding is nondeterministic within the node and no contrast drawn
across those files is safe. Across architectures it legitimately differs — 24 of 500
generations between RTX PRO 6000 and RTX A6000 (C18) — which is why the architecture, not just
the check, is part of every corrected claim.

    python scripts/32_verify_floor_identity.py outputs/eval/tuning/*_blackwell_records.json
    python scripts/32_verify_floor_identity.py --expect-n 3000 outputs/eval/confirm/*_records.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_records(path):
    """Records from either the corrected {"_meta", "records"} form or the legacy bare list."""
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        if "records" not in payload:
            raise SystemExit(f"FAIL {Path(path).name}: dict artifact without a 'records' key")
        return payload["records"]
    if not isinstance(payload, list):
        raise SystemExit(f"FAIL {Path(path).name}: unexpected artifact type {type(payload).__name__}")
    return payload


def floors(path, *, expect_n=None):
    """qid -> frozen text-only floor generation, with size and duplicate checks."""
    recs = load_records(path)
    if expect_n is not None and len(recs) != expect_n:
        raise SystemExit(f"FAIL {Path(path).name}: {len(recs)} records, expected {expect_n}")
    out = {}
    for r in recs:
        qid = str(r["qid"])
        if qid in out:
            raise SystemExit(f"FAIL {Path(path).name}: duplicate qid {qid!r}")
        out[qid] = r["floor"]
    return out


def compare(paths, *, expect_n=None):
    """Return (reference_name, n, [(name, n_differing, example)]) across all paths."""
    if len(paths) < 2:
        raise SystemExit("FAIL need at least two evaluations to compare floors")
    ref_name, ref = Path(paths[0]).name, floors(paths[0], expect_n=expect_n)
    report = []
    for p in paths[1:]:
        got = floors(p, expect_n=expect_n)
        missing = set(ref) - set(got)
        if missing:
            raise SystemExit(f"FAIL {Path(p).name}: missing {len(missing)} qids present in "
                             f"{ref_name}; these evaluations are not on the same question set")
        diff = [q for q in ref if ref[q] != got[q]]
        example = None if not diff else (diff[0], ref[diff[0]], got[diff[0]])
        report.append((Path(p).name, len(diff), example))
    return ref_name, len(ref), report


def main() -> None:
    """Fail loudly if any evaluation's floor differs from the first."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("records", nargs="+", help="records artifacts to compare")
    ap.add_argument("--expect-n", type=int, default=None, help="required record count per file")
    args = ap.parse_args()

    ref_name, n, report = compare(args.records, expect_n=args.expect_n)
    bad = [r for r in report if r[1]]
    for name, n_diff, example in report:
        if n_diff:
            qid, a, b = example
            print(f"FAIL {name}: {n_diff}/{n} floor generations differ from {ref_name}\n"
                  f"     qid={qid}\n     {ref_name}: {a!r}\n     {name}: {b!r}")
    if bad:
        sys.exit(f"FAIL {len(bad)} of {len(report)} evaluations have a divergent floor. "
                 "Either these span GPU architectures, or decoding is nondeterministic "
                 "within the node — in both cases contrasts across them are unsafe.")
    print(f"PASS text-only floor byte-identical across {len(args.records)} evaluations "
          f"({n} qids), reference {ref_name}")


if __name__ == "__main__":
    main()
