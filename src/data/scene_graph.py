"""GQA ground-truth scene graphs -> compact natural-language descriptions.

Used ONLY for the RQ1 scene-graph ORACLE probe (Stage 1). GQA's questions were generated
from these graphs, so a description of an eval image's own graph is privileged information
and OVERSTATES any realistic gain — it answers "if the structure were perfect, would
injecting it help?". The realistic version (Stage 2) replaces this store with a predicted
scene-graph generator; the serialisation and injection stay the same.

The graph for an image is `{objects: {id: {name, attributes:[...], relations:[{name, object}]}}}`.
We turn it into spatial/attribute facts like "a red mug to the left of a white plate".
"""
from __future__ import annotations

import json
from pathlib import Path


class SceneGraphStore:
    """Loads GQA scene graphs and serialises an image's graph to a compact NL description."""

    def __init__(self, path: str, max_facts: int = 40, max_attrs: int = 3):
        """Read the scene-graph JSON (keyed by image id)."""
        with open(path) as f:
            self._graphs: dict = json.load(f)
        self.max_facts = max_facts
        self.max_attrs = max_attrs

    def __contains__(self, image_id) -> bool:
        return str(image_id) in self._graphs

    def _phrase(self, objects: dict, oid: str) -> str | None:
        """A noun phrase for one object: '<up to max_attrs attributes> <name>'."""
        o = objects.get(str(oid))
        if not o or not o.get("name"):
            return None
        attrs = " ".join(o.get("attributes", [])[: self.max_attrs])
        return (attrs + " " + o["name"]).strip()

    def describe(self, image_id) -> str:
        """Serialise the image's graph to '; '-joined facts; '' if the image has no graph.

        Relations become 'a <subject> <relation> a <object>' (e.g. 'a red mug to the left of
        a white plate'); objects in no relation are listed as standalone 'a <phrase>'. Capped
        at max_facts to bound prompt length.
        """
        g = self._graphs.get(str(image_id))
        if not g:
            return ""
        objects = g.get("objects", {})
        facts, related = [], set()
        for oid, o in objects.items():
            sp = self._phrase(objects, oid)
            if not sp:
                continue
            for rel in o.get("relations", []):
                tp = self._phrase(objects, rel.get("object"))
                if tp and rel.get("name"):
                    facts.append(f"a {sp} {rel['name']} a {tp}")
                    related.add(str(oid))
                    related.add(str(rel.get("object")))
            if len(facts) >= self.max_facts:
                break
        for oid, o in objects.items():                  # objects with no relation: list them too
            if str(oid) not in related and len(facts) < self.max_facts:
                p = self._phrase(objects, oid)
                if p:
                    facts.append(f"a {p}")
        return "; ".join(facts[: self.max_facts])

    def coverage(self, image_ids) -> tuple[int, int]:
        """(num with a graph, total) over the given image ids — for honest reporting."""
        ids = list(image_ids)
        have = sum(1 for i in ids if str(i) in self._graphs)
        return have, len(ids)
