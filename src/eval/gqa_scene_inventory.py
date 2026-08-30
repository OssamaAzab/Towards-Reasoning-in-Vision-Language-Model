"""Per-image object inventory assembled from GQA question programs.

WHY THIS EXISTS. GQA publishes no scene graphs for Test-Dev, so there is no direct source of image
truth for the 398 evaluation images. But every question program carries a presence marker per
selected object: `select: woman (7)` names a real scene-graph object, `select: glasses (-)` names
one the annotation states is absent. Pooling those markers over all 12,578 programs gives a partial
but annotation-grounded inventory of what each image contains.

WHAT IT IS NOT. The inventory is OPEN-WORLD. The median image has ~10 known-present and ~1
known-absent object, so a name missing from `present` is unknown, never absent. Callers may assert
"this object is in the image"; they may never assert "this object is not". A contradiction — the
same name marked present and absent for one image — is unresolvable evidence and aborts the build
rather than being broken by a rule.
"""
from __future__ import annotations

import hashlib
import json
import re

INVENTORY_VERSION = "gqa_scene_inventory/1.1.0"

# "woman (7)" -> name "woman", id "7";  "folding chair (-)" -> name "folding chair", id None.
_SELECT_ARG = re.compile(r"^(?P<name>.+?)\s*\((?P<ident>[^()]*)\)\s*$")


class InventoryContradiction(Exception):
    """One image marks the same object name both present and absent."""


def parse_select_argument(argument: str) -> tuple[str, str | None] | None:
    """Split a `select` argument into (lowercased name, object id or None if marked absent)."""
    m = _SELECT_ARG.match(argument.strip())
    if not m:
        return None
    ident = m.group("ident").strip()
    return m.group("name").strip().lower(), (None if ident == "-" else ident)


def build_inventory(questions: dict) -> dict:
    """Pool presence markers into {image_id: {present, absent, ids}}.

    `ids` maps each present name to the set of scene-graph object ids GQA gave it. It exists
    because names alone cannot establish that two answers refer to different things: GQA annotates
    one object as `snowboarder` and `person`, `houses` and `building`, `street sign` and `sign`.
    A caller claiming a wrong referent needs two non-empty, DISJOINT id sets, not two names.
    """
    present: dict[str, set[str]] = {}
    absent: dict[str, set[str]] = {}
    ids: dict[str, dict[str, set[str]]] = {}
    for q in questions.values():
        image_id = q["imageId"]
        for step in q.get("semantic", []):
            if step.get("operation") != "select":
                continue
            parsed = parse_select_argument(step.get("argument", ""))
            if parsed is None:
                continue
            name, ident = parsed
            if ident is None:
                absent.setdefault(image_id, set()).add(name)
                continue
            present.setdefault(image_id, set()).add(name)
            # GQA writes multi-object selections as "chairs (3,9)".
            ids.setdefault(image_id, {}).setdefault(name, set()).update(
                part.strip() for part in ident.split(",") if part.strip())
    inventory = {}
    for image_id in sorted(set(present) | set(absent)):
        p = present.get(image_id, set())
        a = absent.get(image_id, set())
        clash = p & a
        if clash:
            raise InventoryContradiction(
                f"image {image_id}: {sorted(clash)} marked both present and absent")
        inventory[image_id] = {
            "present": frozenset(p),
            "absent": frozenset(a),
            "ids": {n: frozenset(v) for n, v in sorted(ids.get(image_id, {}).items())},
        }
    return inventory


def inventory_sha256(inventory: dict[str, dict[str, frozenset[str]]]) -> str:
    """Order-independent digest of an inventory, for provenance."""
    canonical = {img: {"present": sorted(sets["present"]),
                       "absent": sorted(sets["absent"]),
                       "ids": {n: sorted(v) for n, v in sorted(sets["ids"].items())}}
                 for img, sets in sorted(inventory.items())}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
