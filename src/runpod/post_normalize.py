"""delivery_v1 strict compliance post-fixup.

Runs AFTER ``TagNormalizer.normalize_shot`` and BEFORE
``ShotLabel.model_validate`` in ``pod_runner.py``. Coerces VLM outputs
that occasionally slip past vLLM ``StructuredOutputsParams`` (which has
weak enforcement for deeply-nested ``required`` / enum fields) into
values that the Pydantic schema accepts.

Scope: only clean up well-defined, schema-mandated cases. Anything
semantic (e.g. emotion synonyms, action taxonomy) stays in
``normalize_tags.TagNormalizer``.

Fixes applied:
  A. ``persons[].body_analysis.interaction``:
     - ``count``    → enum ``solo|dyadic|triadic|crowd`` (clone shot on violation)
     - ``contact``  → enum ``none|incidental|sustained`` (self-contact → none)
     - ``relation`` → enum ``parallel|coordinated|opposing|hierarchical``
     - ``interacts_with_person_index`` → list[int ≥ 0]
  B. ``persons[].face_analysis.face_clearly_visible``:
     - inject when missing — ``true`` iff face_analysis has meaningful
       content, ``false`` otherwise (vLLM 0.19 structured_outputs fails
       to enforce this required nested bool; see Pod log 2026-04-20 04:51).
  C. ``persons[].face_analysis.facial_attributes.{glasses,mask,makeup_visible}``:
     - coerce string degree words (``subtle``/``light``/``heavy``) → bool.

The ``VALID_*`` sets mirror ``src/runpod/schemas.py:47-51`` — if the
Literal enums in schemas.py ever change, update this file too (no
import to avoid a circular dependency with the Pod runtime loader).
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = [
    "post_fix_compliance",
    "fix_interaction",
    "fix_face_clearly_visible",
    "fix_facial_attribute_bools",
]

VALID_COUNTS:    frozenset[str] = frozenset({"solo", "dyadic", "triadic", "crowd"})
VALID_CONTACTS:  frozenset[str] = frozenset({"none", "incidental", "sustained"})
VALID_RELATIONS: frozenset[str] = frozenset(
    {"parallel", "coordinated", "opposing", "hierarchical"}
)

RELATION_SYNONYMS: dict[str, str] = {
    "independent":    "parallel",
    "self-directed":  "parallel",
    "self_directed":  "parallel",
    "solo":           "parallel",
    "self":           "parallel",
    "parallel_play":  "parallel",
    "isolated":       "parallel",
    "confrontation":  "opposing",
    "conflict":       "opposing",
    "adversarial":    "opposing",
    "antagonistic":   "opposing",
    "opposition":     "opposing",
    "cooperation":    "coordinated",
    "collaborative":  "coordinated",
    "cooperative":    "coordinated",
    "joint":          "coordinated",
    "dialogue":       "coordinated",
    "conversational": "coordinated",
    "hierarchy":      "hierarchical",
    "authority":      "hierarchical",
    "dominant":       "hierarchical",
    "subordinate":    "hierarchical",
}

CONTACT_SELF_MARKERS: tuple[str, ...] = (
    "self-contact", "self_contact", "self contact",
)

STR_TO_BOOL: dict[str, bool] = {
    "true": True, "yes": True, "y": True, "on": True, "visible": True,
    "subtle": True, "light": True, "medium": True, "mid": True,
    "heavy": True, "strong": True, "moderate": True, "present": True,
    "false": False, "no": False, "n": False, "off": False,
    "none": False, "n/a": False, "na": False, "absent": False, "": False,
}


def _coerce_enum(
    value: Any,
    valid_set: frozenset[str],
    synonyms: dict[str, str],
    fallback: str,
) -> str:
    """Return ``value`` if already valid; else map via ``synonyms``; else ``fallback``."""
    if value in valid_set:
        return value  # type: ignore[return-value]
    if isinstance(value, str):
        v = value.strip().lower()
        if v in valid_set:
            return v
        if v in synonyms:
            return synonyms[v]
    return fallback


def _safe_shot_enum(value: Any, valid_set: frozenset[str], default: str) -> str:
    """Validate a shot-level enum or fall back to ``default``."""
    return value if value in valid_set else default


def fix_interaction(obj: dict[str, Any]) -> int:
    """Coerce ``persons[].body_analysis.interaction`` into strict enums.

    Returns the number of field-level fixes applied (for telemetry).
    """
    fixes = 0
    shot_inter = obj.get("interaction") or {}
    shot_count    = _safe_shot_enum(shot_inter.get("count"),    VALID_COUNTS,    "dyadic")
    shot_contact  = _safe_shot_enum(shot_inter.get("contact"),  VALID_CONTACTS,  "none")
    shot_relation = _safe_shot_enum(shot_inter.get("relation"), VALID_RELATIONS, "parallel")

    persons = obj.get("persons")
    if not isinstance(persons, list):
        return 0

    for p in persons:
        if not isinstance(p, dict):
            continue
        body = p.get("body_analysis")
        if not isinstance(body, dict):
            continue

        inter = body.get("interaction")
        if not isinstance(inter, dict):
            body["interaction"] = {
                "count":                       shot_count,
                "contact":                     shot_contact,
                "relation":                    shot_relation,
                "interacts_with_person_index": [],
            }
            fixes += 1
            continue

        if inter.get("count") not in VALID_COUNTS:
            inter["count"] = shot_count
            fixes += 1

        contact = inter.get("contact")
        if contact not in VALID_CONTACTS:
            if isinstance(contact, str) and any(
                m in contact.lower() for m in CONTACT_SELF_MARKERS
            ):
                inter["contact"] = "none"
            else:
                inter["contact"] = _coerce_enum(
                    contact, VALID_CONTACTS, {}, shot_contact
                )
            fixes += 1

        relation = inter.get("relation")
        if relation not in VALID_RELATIONS:
            inter["relation"] = _coerce_enum(
                relation, VALID_RELATIONS, RELATION_SYNONYMS, shot_relation
            )
            fixes += 1

        iwpi = inter.get("interacts_with_person_index")
        if not isinstance(iwpi, list):
            inter["interacts_with_person_index"] = []
            fixes += 1
        else:
            cleaned = [x for x in iwpi if isinstance(x, int) and x >= 0]
            if cleaned != iwpi:
                inter["interacts_with_person_index"] = cleaned
                fixes += 1

    return fixes


def fix_face_clearly_visible(obj: dict[str, Any]) -> int:
    """Inject ``face_clearly_visible`` when VLM omitted the required gate."""
    fixes = 0
    persons = obj.get("persons")
    if not isinstance(persons, list):
        return 0

    for p in persons:
        if not isinstance(p, dict):
            continue
        fa = p.get("face_analysis")
        if not isinstance(fa, dict):
            continue
        if "face_clearly_visible" in fa:
            continue
        has_content = bool(
            fa.get("primary_emotion")
            or fa.get("facial_components")
            or fa.get("expression_caption")
            or fa.get("valence") is not None
            or fa.get("arousal") is not None
        )
        fa["face_clearly_visible"] = has_content
        fixes += 1

    return fixes


def fix_facial_attribute_bools(obj: dict[str, Any]) -> int:
    """Coerce ``facial_attributes.{glasses,mask,makeup_visible}`` to bool."""
    fixes = 0
    persons = obj.get("persons")
    if not isinstance(persons, list):
        return 0

    BOOL_KEYS = ("glasses", "mask", "makeup_visible")
    for p in persons:
        if not isinstance(p, dict):
            continue
        fa = p.get("face_analysis")
        if not isinstance(fa, dict):
            continue
        attrs = fa.get("facial_attributes")
        if not isinstance(attrs, dict):
            continue
        for key in BOOL_KEYS:
            v = attrs.get(key)
            if v is None or isinstance(v, bool):
                continue
            if isinstance(v, str):
                attrs[key] = STR_TO_BOOL.get(v.strip().lower(), False)
            else:
                attrs[key] = bool(v)
            fixes += 1

    return fixes


def post_fix_compliance(
    obj: dict[str, Any],
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """Apply all delivery_v1 strict-compliance fixups in-place.

    Returns the same dict for chaining.
    """
    n_inter = fix_interaction(obj)
    n_gate  = fix_face_clearly_visible(obj)
    n_bool  = fix_facial_attribute_bools(obj)
    total = n_inter + n_gate + n_bool
    if log is not None and total > 0:
        shot_id = obj.get("shot_id", "?")
        log.info(
            f"[post-fix] {shot_id}: interaction={n_inter} "
            f"face_gate={n_gate} face_bools={n_bool} total={total}"
        )
    return obj
