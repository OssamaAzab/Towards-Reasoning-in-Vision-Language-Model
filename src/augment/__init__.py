"""Inference-time augmentations for RQ1 (no training): none | cot | scene_graph | scene_graph_pred.

Each augmenter decides three things: what per-example CONTEXT to retrieve, which
PromptSpec that context should be rendered under, and how to EXTRACT the final short
answer from the model's output. Augmentations are applied at inference only; the
trained bridge and the frozen LLM are unchanged.

Prompt TEXT is not assembled here. Every literal and every framing template lives in
src.prompt, and build_prompt() delegates to src.prompt.user_body(), so the augmented
and unaugmented arms cannot drift apart. tests/test_augment_prompt.py pins the output
byte-for-byte against frozen copies of the pre-port strings, so existing augmentation
results stay reproducible.

GENERATION BUDGET. `max_new_tokens` is the historical per-augmenter cap and is what
the legacy evidence layer used. Under the corrected protocol the cap is no longer the
terminator — the bridge is trained to emit <|im_end|> — so budget() returns the
protocol's budget for the reasoning mode instead. Read budget(), not the attribute,
unless you are deliberately reproducing a legacy run.
"""
from __future__ import annotations

import re

from src import prompt as P


class NoAugment:
    """Baseline: the standard short-answer prompt, no augmentation."""

    name = "none"
    label = "bridge (baseline)"
    max_new_tokens = 10          # legacy cap; see budget()
    input_mode = "image_only"
    reason_mode = "direct"
    graph_framing = "plain"      # unused when input_mode has no graph

    def spec(self, base: P.PromptSpec | None = None) -> P.PromptSpec:
        """This augmenter's PromptSpec, inheriting protocol levers from `base`."""
        base = base or P.PromptSpec()
        return P.PromptSpec(
            prompt_format=base.prompt_format, supervise_eos=base.supervise_eos,
            system_text=base.system_text, max_answer_tokens=base.max_answer_tokens,
            input_mode=self.input_mode, reason_mode=self.reason_mode,
            graph_framing=self.graph_framing)

    def budget(self, base: P.PromptSpec | None = None) -> int:
        """Generation cap: historical value for legacy, protocol budget for corrected."""
        spec = self.spec(base)
        if spec.evidence_layer == P.EVIDENCE_CORRECTED:
            return P.MAX_NEW_TOKENS[self.reason_mode]
        return self.max_new_tokens

    def context_for(self, ex) -> str | None:
        """No per-example context for the plain baseline."""
        return None

    def build_prompt(self, question: str, context: str | None = None,
                     spec: P.PromptSpec | None = None) -> str:
        """Render the user-turn text for this augmentation via src.prompt."""
        return P.user_body(self.spec(spec), question, context)

    def extract_answer(self, text: str) -> str:
        """The model already answers tersely; just strip whitespace."""
        return text.strip()


class CoTAugment(NoAugment):
    """Chain-of-thought: ask the LLM to reason, then parse the final 'Answer:' span."""

    name = "cot"
    label = "bridge + chain-of-thought"
    max_new_tokens = 128
    reason_mode = "cot"

    def extract_answer(self, text: str) -> str:
        """Pull the short answer after the last 'Answer:'; fall back to the last non-empty line."""
        matches = list(re.finditer(r"answer\s*[:\-]\s*", text, flags=re.IGNORECASE))
        if matches:
            tail = text[matches[-1].end():]
            if tail.strip():
                return _clean(tail.splitlines()[0])
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return _clean(lines[-1]) if lines else text.strip()


class CoDAugment(CoTAugment):
    """Chain-of-draft: the same parse, but the prompt caps each reasoning step at five words.

    Shares CoT's extraction because the output contract is identical ('Answer:' last);
    only the instruction and the token budget differ.
    """

    name = "cod"
    label = "bridge + chain-of-draft"
    max_new_tokens = 64
    reason_mode = "cod"


class SceneGraphAugment(NoAugment):
    """Inject a natural-language scene-graph description before the question, then ask it.

    Stage 1 is an ORACLE probe: the description comes from the eval image's own GROUND-TRUTH
    GQA scene graph (the very structure the question was generated from), so it is an upper
    bound, not a realistic result. The store is swapped for a predicted-graph generator in
    Stage 2 with no change here.
    """

    name = "scene_graph"
    label = "bridge + scene-graph (ORACLE)"
    max_new_tokens = 10
    input_mode = "image_plus_graph"
    graph_framing = "plain"
    oracle = True

    def __init__(self, store):
        """store: a SceneGraphStore exposing describe(image_id) -> str."""
        self.store = store
        self.n_asked = 0        # coverage counters: how often a graph was requested ...
        self.n_with_graph = 0   # ... and how often one existed (else plain-prompt fallback)

    def context_for(self, ex) -> str | None:
        """The serialised ground-truth graph for this example's image ('' if none)."""
        desc = self.store.describe(ex.image_id)
        self.n_asked += 1
        self.n_with_graph += bool(desc)
        return desc

    def coverage(self) -> str:
        """One-line coverage report: injected graphs vs silent plain-prompt fallbacks."""
        pct = 100 * self.n_with_graph / self.n_asked if self.n_asked else 0.0
        return (f"scene-graph coverage: {self.n_with_graph}/{self.n_asked} questions "
                f"({pct:.1f}%) had a graph; {self.n_asked - self.n_with_graph} silently "
                f"fell back to the plain prompt")

    def extract_answer(self, text: str) -> str:
        """Direct short answer; just strip."""
        return _clean(text)


class PredQPlainAugment(SceneGraphAugment):
    """Stage 2b: QUESTION-CONDITIONED predicted graphs, original prompt framing.

    The store is keyed by qid (each question got its own OWLv2 pass with the question's
    nouns added as queries), so context_for looks up ex.qid, not ex.image_id. Prompt framing
    is inherited unchanged — this variant isolates the question-conditioning effect in the
    tuning A/B.
    """

    name = "scene_graph_predq_plain"
    label = "bridge + scene-graph (PREDICTED, q-cond)"
    oracle = False

    def context_for(self, ex) -> str | None:
        """The per-QUESTION predicted graph ('' if none)."""
        desc = self.store.describe(ex.qid)
        self.n_asked += 1
        self.n_with_graph += bool(desc)
        return desc


class PredQAugment(PredQPlainAugment):
    """Stage 2b full package: question-conditioned graphs + HEDGED framing.

    The Stage-2a records showed the model reading the injected list as a closed world
    ("The man is not in a vehicle" when the detector missed the carriage). The hedge names
    the list as imperfect detector output and the image as the final authority, so absence
    from the list stops implying absence from the image.
    """

    name = "scene_graph_predq"
    label = "bridge + scene-graph (PREDICTED, q-cond, hedged)"
    graph_framing = "hedged"


class DegradedGraphAugment(SceneGraphAugment):
    """One Stage-A degradation arm, read from a frozen prompt cache keyed by qid.

    The corruption is NOT performed here. `scripts/82_build_degraded_graphs.py` renders every arm
    on CPU into a hashed cache, so the whole intervention is auditable before a card is requested
    and this class is a reader. It stays an ORACLE variant: the text is derived from the eval
    image's own ground-truth graph, degraded -- not from a detector.

    TWO THINGS IT REFUSES TO DO SILENTLY.

    A missing qid RAISES rather than falling back to the plain prompt. Every other graph augmenter
    here counts silent fallbacks precisely because they corrupt a comparison while looking like a
    result; for a degraded arm the cache was built from the same frozen slice, so a missing key
    means the wrong cache, not a missing graph.

    It also verifies the cache against the sibling manifest's sha256 before returning. A launcher
    that points at a stale or hand-edited cache would otherwise produce a perfectly clean run of
    the wrong intervention.
    """

    oracle = True

    def __init__(self, cache_path):
        import hashlib
        import json
        from pathlib import Path

        path = Path(cache_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"degraded-graph cache not found: {path} — run "
                f"scripts/82_build_degraded_graphs.py first")
        # cache_<ARM>_s<NNN>.json  ->  arm, severity tag
        stem = path.stem.split("_")
        if len(stem) != 3 or stem[0] != "cache":
            raise ValueError(f"cache filename is not cache_<ARM>_s<NNN>.json: {path.name}")
        self.arm, tag = stem[1], stem[2]

        manifest_path = path.parent / f"MANIFEST_{tag}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no manifest beside the cache: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        entry = manifest["arms"].get(self.arm)
        if entry is None:
            raise KeyError(f"{manifest_path.name} does not describe arm {self.arm}")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != entry["cache_sha256"]:
            raise ValueError(
                f"degraded cache {path.name} does not match its manifest\n"
                f"  manifest {entry['cache_sha256']}\n  actual   {got}\n"
                f"Rebuild with scripts/82_build_degraded_graphs.py rather than editing a cache.")

        self.store = None
        self.cache = json.loads(path.read_text())
        self.severity = manifest["severity"]
        self.fingerprint = entry["content_fingerprint"]
        self.name = f"scene_graph_degraded_{self.arm.lower()}"
        self.label = f"bridge + scene-graph ({self.arm} {entry['arm_name']})"
        self.n_asked = 0
        self.n_with_graph = 0

    def context_for(self, ex) -> str | None:
        """The frozen degraded description for this question. Absent qid is an error, not a null."""
        if ex.qid not in self.cache:
            raise KeyError(
                f"qid {ex.qid} is absent from the {self.arm} cache ({len(self.cache)} entries). "
                f"The eval slice and the cache disagree; a plain-prompt fallback here would "
                f"silently score an un-degraded question inside a degraded arm.")
        desc = self.cache[ex.qid]
        self.n_asked += 1
        self.n_with_graph += bool(desc)
        return desc

    def coverage(self) -> str:
        """Coverage plus the identity of the artifact, so the log names what actually ran."""
        empty = self.n_asked - self.n_with_graph
        return (f"{self.arm} ({self.label}): {self.n_with_graph}/{self.n_asked} questions carried "
                f"graph text, {empty} empty; severity {self.severity}; "
                f"content fingerprint {self.fingerprint[:16]}")


def _clean(answer: str) -> str:
    """Strip wrapping angle brackets / quotes / whitespace the model sometimes adds."""
    return answer.strip().strip("<>\"'` ").strip()


def make_augmenter(kind: str, cfg: dict | None = None):
    """Factory: 'none' | 'cot' | 'cod' | 'scene_graph' (oracle) | 'scene_graph_pred' (Stage 2)."""
    kind = (kind or "none").lower()
    if kind in ("none", ""):
        return NoAugment()
    if kind == "cot":
        return CoTAugment()
    if kind == "cod":
        return CoDAugment()
    if kind == "scene_graph":
        if cfg is None:
            raise ValueError("scene_graph augmentation needs cfg (for the scene-graph path)")
        from src.data.scene_graph import SceneGraphStore
        return SceneGraphAugment(SceneGraphStore(cfg["gqa"]["scene_graphs"]))
    if kind == "scene_graph_degraded":
        # Oracle-to-degraded diagnostic. The arm and severity come from the cache the
        # launcher points at via --pred-cache, so one code path serves all eight arms and the
        # launcher cannot select an arm the artifacts do not contain.
        if cfg is None:
            raise ValueError("scene_graph_degraded needs cfg (for the degraded cache path)")
        return DegradedGraphAugment(cfg["augmentation"]["pred_graphs"])
    if kind == "scene_graph_pred":
        # Stage 2: PREDICTED graphs (OWLv2 + box-geometry, colour off) — the realistic RQ1
        # condition. NOT an oracle: no `oracle` flag, so no upper-bound caveats are printed.
        if cfg is None:
            raise ValueError("scene_graph_pred augmentation needs cfg (for the graph cache path)")
        from pathlib import Path

        from src.data.predicted_graph import PredictedGraphStore
        path = cfg["augmentation"]["pred_graphs"]
        if not Path(path).exists():
            raise FileNotFoundError(
                f"predicted-graph cache not found: {path} — run "
                f"scripts/16_precompute_pred_graphs.py first (silent plain-prompt fallback "
                f"on every question would corrupt the Stage-2 comparison)")
        aug = SceneGraphAugment(PredictedGraphStore(path))
        aug.name = "scene_graph_pred"
        aug.label = "bridge + scene-graph (PREDICTED)"
        aug.oracle = False
        return aug
    if kind in ("scene_graph_predq", "scene_graph_predq_plain"):
        # Stage 2b: per-QUESTION predicted graphs (question-conditioned OWLv2). The cache is
        # keyed by qid; scripts/18 builds it. 'plain' keeps the original framing (A/B lever).
        if cfg is None:
            raise ValueError(f"{kind} augmentation needs cfg (for the per-question cache path)")
        from pathlib import Path

        from src.data.predicted_graph import PredictedGraphStore
        path = cfg["augmentation"]["pred_graphs_q"]
        if not Path(path).exists():
            raise FileNotFoundError(
                f"per-question predicted-graph cache not found: {path} — run "
                f"scripts/18_precompute_pred_graphs_q.py first (silent plain-prompt fallback "
                f"on every question would corrupt the Stage-2b comparison)")
        cls = PredQAugment if kind == "scene_graph_predq" else PredQPlainAugment
        return cls(PredictedGraphStore(path))
    raise ValueError(f"unknown augmentation '{kind}' (available: none, cot, cod, scene_graph, "
                     f"scene_graph_pred, scene_graph_predq, scene_graph_predq_plain)")
