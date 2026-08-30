"""GQA loader: balanced-split questions + Visual Genome image-path resolution.

GQA ships its questions as one big JSON dict keyed by question id. It does NOT
ship images here — GQA image ids ARE Visual Genome image ids, and VG already
lives on the cluster, so we resolve each question's `imageId` to a `.jpg` in one
of the configured VG folders (an in-memory index is built once, lazily).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# The five GQA reasoning buckets reported for RQ1; everything else -> "other".
REASONING_CATEGORIES = ["relate", "compare", "count", "exist", "choose"]


@dataclass
class GQAExample:
    """One GQA question: answer, raw GQA type tags, and resolved image path."""
    qid: str
    image_id: str
    question: str
    answer: str
    full_answer: str
    structural: str            # GQA types.structural: verify|query|choose|logical|compare
    semantic: str              # GQA types.semantic: obj|attr|rel|cat|global
    detailed: str              # GQA types.detailed: fine-grained (e.g. existRel, queryAttr)
    image_path: Path | None    # None if no <image_id>.jpg found in any VG dir


def category_of(question: str, structural: str, semantic: str, detailed: str) -> str:
    """Map a GQA question into one of REASONING_CATEGORIES, else 'other'.

    Calibrated to GQA's type schema; order matters (count is checked first because
    counting questions are otherwise structurally 'query').
    """
    d = (detailed or "").lower()
    if d.startswith("count") or question.lower().startswith("how many"):
        return "count"
    if structural == "choose":
        return "choose"
    if structural == "compare":
        return "compare"
    if d.startswith("exist"):
        return "exist"
    if semantic == "rel":
        return "relate"
    return "other"


class GQADataset:
    """Loads a GQA questions JSON and resolves image paths against VG folders."""

    def __init__(self, questions_path: str, image_dirs: list[str]):
        """Read the questions JSON; image index is built on first image lookup."""
        self.questions_path = Path(questions_path)
        self.image_dirs = [Path(d) for d in image_dirs]
        with open(self.questions_path) as f:
            self._raw: dict = json.load(f)
        self.qids: list[str] = list(self._raw.keys())
        self._image_index: dict[str, Path] | None = None

    def __len__(self) -> int:
        """Number of questions in the split."""
        return len(self.qids)

    def _build_image_index(self) -> dict[str, Path]:
        """Scan the VG dirs once, mapping image-id (filename stem) -> Path.

        Skips 0-byte files: the cluster VG share holds many empty placeholder
        .jpgs, and an undecodable image is effectively a missing one. This keeps
        `image_path is None` meaning "no usable image" for all downstream code.
        Dirs are scanned in order, so a real image shadows an empty duplicate.
        """
        index: dict[str, Path] = {}
        for d in self.image_dirs:
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.suffix == ".jpg" and p.stat().st_size > 0:
                    index.setdefault(p.stem, p)
        return index

    def _resolve_image(self, image_id: str) -> Path | None:
        """Return the path to <image_id>.jpg, or None if absent (lazy index)."""
        if self._image_index is None:
            self._image_index = self._build_image_index()
        return self._image_index.get(str(image_id))

    def get(self, qid: str) -> GQAExample:
        """Build a GQAExample for one question id."""
        e = self._raw[qid]
        t = e.get("types", {})
        image_id = str(e["imageId"])
        return GQAExample(
            qid=qid,
            image_id=image_id,
            question=e["question"],
            answer=e.get("answer", ""),
            full_answer=e.get("fullAnswer", ""),
            structural=t.get("structural", ""),
            semantic=t.get("semantic", ""),
            detailed=t.get("detailed", ""),
            image_path=self._resolve_image(image_id),
        )

    def category_of(self, ex: GQAExample) -> str:
        """Reasoning bucket for an example (one of REASONING_CATEGORIES or 'other')."""
        return category_of(ex.question, ex.structural, ex.semantic, ex.detailed)

    def examples(self, n: int | None = None) -> Iterator[GQAExample]:
        """Yield up to n examples (all if n is None), in file order."""
        qids = self.qids if n is None else self.qids[:n]
        for qid in qids:
            yield self.get(qid)

    def filter_by_category(self, category: str, n: int | None = None) -> list[GQAExample]:
        """Return up to n examples whose reasoning bucket == `category`."""
        out: list[GQAExample] = []
        for ex in self.examples():
            if self.category_of(ex) == category:
                out.append(ex)
                if n is not None and len(out) >= n:
                    break
        return out
