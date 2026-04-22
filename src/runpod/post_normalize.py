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
  D. ``persons[].body_analysis.action_quality.{tempo,tone,intensity}``:
     - drop/null any value that contains a camera/cinematography term
       (static/tracking/pan/handheld/steady/etc.). Those terms describe
       the camera, not the subject's action. See Pod log 2026-04-20
       05:40 shot_0036 for ``tempo='static'`` → ``camera_term_in_action``
       hard error.
  E. ``strip_camera_terms_in_captions`` (C3, 2026-04-22):
     port from removed spec normalize_tags extension. spec
     ``normalize_tags.py`` (zip baseline) only strips camera terms from
     ``body_analysis.action_primary``. delivery_v1 § 5.3 expects camera
     terms also out of free-text captions / summaries. Pod-side here.
  F. ``enforce_altcap_null_consistency`` (C6, 2026-04-22):
     spec § 6.3: when ``face_clearly_visible=true`` (or
     ``body_clearly_visible=true``), the 4 ``alternative_captions``
     sub-fields cannot all be null — VLM either skipped the gate
     (= false, all null) or filled the captions (= true, ≥ 1 non-null).
     Inconsistent cases are coerced to consistency (gate kept, captions
     left as VLM provided; if all 4 are null but gate=true, downgrade
     gate to false).

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
    "fix_action_quality_camera_terms",
    "fix_primary_emotion",
]

VALID_EMOTIONS: frozenset[str] = frozenset({
    "anger", "sadness", "joy", "fear", "surprise",
    "disgust", "contempt", "neutral", "complex",
})

# VLM-observed non-enum emotion words → canonical Emotion enum.
# All keys lowercased. Motivation: Pod 2026-04-20 07:24 shot_0731 output
# primary_emotion='concern' which is semantically close to 'fear' but not
# in the 9-class taxonomy, causing Pydantic schema_validation_failed.
EMOTION_SYNONYMS: dict[str, str] = {
    # fear family
    "concern":        "fear",
    "concerned":      "fear",
    "worry":          "fear",
    "worried":        "fear",
    "anxious":        "fear",
    "anxiety":        "fear",
    "nervous":        "fear",
    "apprehension":   "fear",
    "apprehensive":   "fear",
    "uneasy":         "fear",
    "tense":          "fear",
    "alarmed":        "fear",
    "panic":          "fear",
    "dread":          "fear",
    # sadness family
    "sad":            "sadness",
    "sorrow":         "sadness",
    "sorrowful":      "sadness",
    "melancholy":     "sadness",
    "grief":          "sadness",
    "grieving":       "sadness",
    "despair":        "sadness",
    "despondent":     "sadness",
    "down":           "sadness",
    "blue":           "sadness",
    "gloomy":         "sadness",
    "heartbroken":    "sadness",
    # joy family
    "happy":          "joy",
    "happiness":      "joy",
    "joyful":         "joy",
    "cheerful":       "joy",
    "content":        "joy",
    "contentment":    "joy",
    "pleased":        "joy",
    "delighted":      "joy",
    "elated":         "joy",
    "excited":        "joy",
    "excitement":     "joy",
    "amused":         "joy",
    "amusement":      "joy",
    # anger family
    "angry":          "anger",
    "furious":        "anger",
    "rage":           "anger",
    "irritated":      "anger",
    "irritation":     "anger",
    "annoyed":        "anger",
    "annoyance":      "anger",
    "frustrated":     "anger",
    "frustration":    "anger",
    "hostile":        "anger",
    # surprise family
    "surprised":      "surprise",
    "shocked":        "surprise",
    "astonished":     "surprise",
    "astonishment":   "surprise",
    "stunned":        "surprise",
    "amazed":         "surprise",
    # disgust family
    "disgusted":      "disgust",
    "revolted":       "disgust",
    "repulsed":       "disgust",
    "distaste":       "disgust",
    # contempt family
    "contemptuous":   "contempt",
    "disdain":        "contempt",
    "scornful":       "contempt",
    "mocking":        "contempt",
    # neutral / calm
    "calm":           "neutral",
    "composed":       "neutral",
    "neutral-faced":  "neutral",
    "impassive":      "neutral",
    "stoic":          "neutral",
    "blank":          "neutral",
    "expressionless": "neutral",
    "thoughtful":     "neutral",
    "contemplative":  "neutral",
    "focused":        "neutral",
    "attentive":      "neutral",
    # complex / mixed
    "mixed":          "complex",
    "conflicted":     "complex",
    "bittersweet":    "complex",
    "ambivalent":     "complex",
    "torn":           "complex",
    "complicated":    "complex",
    "mixed emotion":  "complex",
    "mixed emotions": "complex",
    "mixed feelings": "complex",
}

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

# Camera/cinematography terms that must not appear in
# ``body_analysis.action_quality.{tempo,tone,intensity}`` because they
# describe the camera, not the subject. Mirrors ShotValidator's
# ``CAMERA_TERMS_FALLBACK`` (validate_body_analysis.py:103-118). Kept
# here as a lowercased set for O(1) membership checks.
ACTION_CAMERA_TERMS: frozenset[str] = frozenset({
    "close-up", "close up", "closeup", "medium shot", "wide shot",
    "medium-shot", "wide-shot", "medium close-up", "medium close up",
    "extreme close-up", "long shot", "establishing shot",
    "eye-level", "eye level", "low angle", "low-angle",
    "high angle", "high-angle", "dutch", "dutch angle",
    "overhead", "bird's eye", "frontal", "side view",
    "profile", "profile view",
    "steady", "steadicam", "handheld", "hand-held",
    "static", "tracking", "tracking shot",
    "pan", "panning", "tilt", "tilting",
    "zoom", "zooming", "dolly", "crane", "crane shot",
    "aerial", "pov", "point of view",
    "over the shoulder", "ots", "two-shot",
    "insert", "cutaway", "reverse", "reaction shot",
})

# Safe-substitution map for common VLM mistakes on action_quality.tempo:
# VLM often uses camera vocab to describe a motionless subject.
TEMPO_CAMERA_TO_BODY: dict[str, str | None] = {
    "static":    "minimal",   # motionless subject
    "tracking":  "sustained",
    "steady":    "steady-flow",
    "pan":       None,
    "handheld":  None,
    "hand-held": None,
    "tilt":      None,
    "tilting":   None,
    "zoom":      None,
    "zooming":   None,
    "dolly":     None,
    "crane":     None,
    "aerial":    None,
    "pov":       None,
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


def _contains_camera_term(value: str) -> bool:
    """Return True if the lowercased string contains any camera term."""
    v = value.strip().lower()
    if not v:
        return False
    if v in ACTION_CAMERA_TERMS:
        return True
    return any(term in v for term in ACTION_CAMERA_TERMS)


def fix_primary_emotion(obj: dict[str, Any]) -> int:
    """Coerce ``face_analysis.{primary,secondary}_emotion`` into the 9-class
    delivery_v1 Emotion enum.

    Motivation (Pod 2026-04-20 07:24 shot_0731):
      VLM emitted ``primary_emotion='concern'`` — semantically close to
      ``fear`` but outside the Literal enum, causing Pydantic
      ``schema_validation_failed`` and blocking the shot.

    Strategy:
      - If the value is already in VALID_EMOTIONS → keep as-is.
      - Else lowercase + look up EMOTION_SYNONYMS.
      - Non-matchable string or bad type → ``'complex'`` (catchall that
        delivery_v1 reserves for mixed / ambiguous affect). Prefer
        ``'complex'`` over ``'neutral'`` so downstream sees a deliberate
        ambiguity marker rather than an implicit "no emotion".

    ``secondary_emotion`` is Optional — only coerce when it's a
    non-null, non-valid string.

    Returns the number of field-level fixes applied.
    """
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

        for key in ("primary_emotion", "secondary_emotion"):
            v = fa.get(key)
            if v is None:
                continue
            if v in VALID_EMOTIONS:
                continue
            if isinstance(v, str):
                low = v.strip().lower()
                if low in VALID_EMOTIONS:
                    fa[key] = low
                elif low in EMOTION_SYNONYMS:
                    fa[key] = EMOTION_SYNONYMS[low]
                else:
                    fa[key] = "complex"
                fixes += 1
            else:
                # Non-string (int/bool) — coerce to 'complex' as safest default.
                fa[key] = "complex"
                fixes += 1

    return fixes


def fix_action_quality_camera_terms(obj: dict[str, Any]) -> int:
    """Scrub camera terms from ``body_analysis.action_quality.{tempo,tone}``.

    ShotValidator raises ``camera_term_in_action`` as a CRITICAL error
    when cinematography vocabulary appears in these body-describing
    fields. We first try a safe substitution (e.g. ``static`` →
    ``minimal`` for a motionless body), then fall back to ``None``.

    ``intensity`` is a Literal enum already, so we only check it for
    consistency with future expansion (currently it wouldn't match
    camera terms).

    Returns the number of field-level fixes applied.
    """
    fixes = 0
    persons = obj.get("persons")
    if not isinstance(persons, list):
        return 0

    for p in persons:
        if not isinstance(p, dict):
            continue
        body = p.get("body_analysis")
        if not isinstance(body, dict):
            continue
        aq = body.get("action_quality")
        if not isinstance(aq, dict):
            continue

        for key in ("tempo", "tone"):
            v = aq.get(key)
            if not isinstance(v, str) or not v.strip():
                continue
            if not _contains_camera_term(v):
                continue
            # Try substitution (tempo has the most common offenders).
            replaced = False
            if key == "tempo":
                v_low = v.strip().lower()
                if v_low in TEMPO_CAMERA_TO_BODY:
                    aq[key] = TEMPO_CAMERA_TO_BODY[v_low]
                    replaced = True
            if not replaced:
                aq[key] = None
            fixes += 1

    return fixes


# ── E. strip_camera_terms_in_captions (C3, 2026-04-22) ─────────────
# Loaded once on first call; key is yaml path so multiple calls with
# different spec snapshots keep their own list.
_CAMERA_TERMS_CACHE: dict[str, frozenset[str]] = {}


def _load_camera_terms(synonyms_yaml_path: str | None = None) -> frozenset[str]:
    """Return the spec's camera_terms_forbidden list as a frozenset."""
    if synonyms_yaml_path is None:
        from pathlib import Path
        synonyms_yaml_path = str(
            Path(__file__).resolve().parents[2]
            / "docs" / "labelingStandards" / "external_delivery_v1"
            / "docs" / "motion_synonyms.yaml"
        )
    if synonyms_yaml_path in _CAMERA_TERMS_CACHE:
        return _CAMERA_TERMS_CACHE[synonyms_yaml_path]
    try:
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path(synonyms_yaml_path).read_text(encoding="utf-8"))
        terms = data.get("camera_terms_forbidden") or []
        # Normalize to lowercase, strip
        out = frozenset(t.strip().lower() for t in terms if t and t.strip())
    except Exception:
        out = frozenset()
    _CAMERA_TERMS_CACHE[synonyms_yaml_path] = out
    return out


_CAPTION_FIELDS_FACE = ["expression_caption"]
_CAPTION_FIELDS_BODY = ["motion_caption", "gesture_detail"]
_CAPTION_ALT_KEYS    = ["direct", "literary", "direction", "situational"]
_UPPER_BODY_DETAIL_FIELDS = ["head", "neck", "shoulders", "arms", "hands", "torso"]
_SHOT_CONTEXT_FIELDS = ["shot_emotion_summary", "shot_motion_summary"]
_SCENE_CONTEXT_FIELDS = ["visible_setting", "narrative_situation"]


def _strip_camera_terms_from_text(text: str, terms: frozenset[str]) -> tuple[str, bool]:
    """Word-boundary strip of camera terms from one string. Returns (new, changed)."""
    if not text or not isinstance(text, str) or not terms:
        return text, False
    import re
    out = text
    changed = False
    for term in terms:
        pattern = re.compile(
            r"(?i)(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
        )
        new_out = pattern.sub("", out)
        if new_out != out:
            changed = True
            out = new_out
    if changed:
        out = re.sub(r"\s{2,}", " ", out)
        out = re.sub(r"\s*,\s*,", ",", out)
        out = re.sub(r"^[,\s]+|[,\s]+$", "", out)
    return out, changed


def strip_camera_terms_in_captions(obj: dict[str, Any]) -> int:
    """Strip spec camera terms from all free-text caption fields.

    Replaces the removed spec-side normalize_tags expansion (per RESTORE_LOG
    2026-04-22). Spec normalize_tags only handles body.action_primary.
    """
    terms = _load_camera_terms()
    if not terms:
        return 0
    fixes = 0

    def _strip_dict_text(d: Any, keys: list[str]) -> int:
        if not isinstance(d, dict):
            return 0
        c = 0
        for k in keys:
            v = d.get(k)
            if not isinstance(v, str):
                continue
            new_v, changed = _strip_camera_terms_from_text(v, terms)
            if changed:
                d[k] = new_v
                c += 1
        return c

    # shot_context summaries
    sc = obj.get("shot_context")
    if isinstance(sc, dict):
        fixes += _strip_dict_text(sc, _SHOT_CONTEXT_FIELDS)
        fixes += _strip_dict_text(sc.get("scene_context"), _SCENE_CONTEXT_FIELDS)

    for person in obj.get("persons") or []:
        if not isinstance(person, dict):
            continue
        fa = person.get("face_analysis")
        if isinstance(fa, dict):
            fixes += _strip_dict_text(fa, _CAPTION_FIELDS_FACE)
            fixes += _strip_dict_text(fa.get("alternative_captions"), _CAPTION_ALT_KEYS)
        ba = person.get("body_analysis")
        if isinstance(ba, dict):
            fixes += _strip_dict_text(ba, _CAPTION_FIELDS_BODY)
            fixes += _strip_dict_text(ba.get("alternative_captions"), _CAPTION_ALT_KEYS)
            fixes += _strip_dict_text(ba.get("upper_body_detail"), _UPPER_BODY_DETAIL_FIELDS)

    return fixes


# ── F. enforce_altcap_null_consistency (C6, 2026-04-22) ─────────────


def _all_alt_caps_null(alt: Any) -> bool:
    """Check if all 4 alt_captions sub-fields are null/missing."""
    if not isinstance(alt, dict):
        return True  # absent → treated as all-null
    return all(alt.get(k) is None for k in _CAPTION_ALT_KEYS)


def enforce_altcap_null_consistency(obj: dict[str, Any]) -> int:
    """Spec § 6.3: gate=true ⇔ at least one alt_caption non-null.

    If face_clearly_visible=true but all 4 alt_captions are null
    (= VLM contradicted itself), downgrade gate to false. Same for
    body_clearly_visible.
    """
    fixes = 0
    for person in obj.get("persons") or []:
        if not isinstance(person, dict):
            continue
        fa = person.get("face_analysis")
        if isinstance(fa, dict) and fa.get("face_clearly_visible") is True:
            if _all_alt_caps_null(fa.get("alternative_captions")):
                fa["face_clearly_visible"] = False
                fixes += 1
        ba = person.get("body_analysis")
        if isinstance(ba, dict) and ba.get("body_clearly_visible") is True:
            if _all_alt_caps_null(ba.get("alternative_captions")):
                ba["body_clearly_visible"] = False
                fixes += 1
    return fixes


def post_fix_compliance(
    obj: dict[str, Any],
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """Apply all delivery_v1 strict-compliance fixups in-place.

    Returns the same dict for chaining. Order matters:
      1. interaction enum coercion
      2. face gate injection
      3. facial attribute bool coercion
      4. action_quality camera term cleanup
      5. emotion synonym remap
      6. (C3) camera term stripping in free-text captions
      7. (C6) altcap-null consistency (run last; depends on prior fixes)
    """
    n_inter   = fix_interaction(obj)
    n_gate    = fix_face_clearly_visible(obj)
    n_bool    = fix_facial_attribute_bools(obj)
    n_cam     = fix_action_quality_camera_terms(obj)
    n_emotion = fix_primary_emotion(obj)
    n_capcam  = strip_camera_terms_in_captions(obj)
    n_altcons = enforce_altcap_null_consistency(obj)
    total = n_inter + n_gate + n_bool + n_cam + n_emotion + n_capcam + n_altcons
    if log is not None and total > 0:
        shot_id = obj.get("shot_id", "?")
        log.info(
            f"[post-fix] {shot_id}: interaction={n_inter} "
            f"face_gate={n_gate} face_bools={n_bool} "
            f"action_camera={n_cam} emotion={n_emotion} "
            f"caption_camera={n_capcam} altcap_consistency={n_altcons} "
            f"total={total}"
        )
    return obj


# ── alias for forward compatibility (Phase C plan named it fix_all) ─

fix_all = post_fix_compliance
