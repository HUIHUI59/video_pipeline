#!/usr/bin/env python3
"""
validate_body_analysis.py — VLM Inference Output Validator
Project: Video_DB_Face
Version: 1.0

Validates per-shot JSON files against the integrated schema
(face_analysis + body_analysis + cross-axis consistency).
Produces validation_report.json with errors, warnings, and distributions.

Usage:
    python scripts/validate_body_analysis.py \
      --input deliverable/movies/The_Dinner_2017/shots/ \
      --taxonomy docs/motion_taxonomy.yaml \
      --synonyms docs/motion_synonyms.yaml \
      --output deliverable/movies/The_Dinner_2017/validation_report.json
"""

import argparse
import json
import sys
import logging
import hashlib
import re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

# yaml is optional — fall back to built-in lists if not installed
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger("validate_body_analysis")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

FRAMING_MAX_PARTS: dict[str, set[str]] = {
    "close_face": {"head", "neck"},
    "bust": {"head", "neck", "shoulders"},
    "half_body": {
        "head", "neck", "shoulders", "upper_arms",
        "forearms", "hands", "torso",
    },
    "three_quarter": {
        "head", "neck", "shoulders", "upper_arms",
        "forearms", "hands", "torso", "hips", "thighs",
    },
    "full_body": {
        "head", "neck", "shoulders", "upper_arms",
        "forearms", "hands", "torso", "hips",
        "thighs", "shins", "feet",
    },
    "wide": {
        "head", "neck", "shoulders", "upper_arms",
        "forearms", "hands", "torso", "hips",
        "thighs", "shins", "feet",
    },
}

VALID_SHOT_FRAMES = set(FRAMING_MAX_PARTS.keys())

VALID_BODY_FOCUS = {
    "upper_body", "hands", "torso", "posture",
    "gesture", "full_body", "lower_body", "face_and_gaze",
}

VALID_INTENSITIES = {"low", "mid", "high"}
VALID_TONES = {"relaxed", "tense", "controlled", "contemplative"}
VALID_TEMPOS = {"sustained", "punctuated", "accelerating", "decelerating"}
VALID_TRAJECTORIES = {"linear", "arc", "circular", "erratic", "static"}
VALID_PERIODICITIES = {"periodic", "non_periodic"}
VALID_SYMMETRIES = {"bilateral_symmetric", "bilateral_asymmetric", "axial"}
VALID_DURATION_CLASSES = {"onset_only", "ongoing", "peak_then_release"}

VALID_EMOTIONS = {
    "anger", "sadness", "joy", "fear", "surprise",
    "disgust", "contempt", "neutral", "complex",
}

VALID_TEMPORAL_CHANGES = {
    "static", "building", "peak_then_release",
    "transition", "rapid_micro",
}

# Caption length ranges (word count)
CAPTION_RANGES = {
    "motion_caption": (50, 180),
    "expression_caption": (50, 120),
    "alt_direct": (20, 40),
    "alt_literary": (30, 60),
    "alt_direction": (30, 50),
    "alt_situational": (30, 60),
}

# Fallback camera terms if YAML not available
CAMERA_TERMS_FALLBACK: set[str] = {
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
}

# Vague gesture strings that indicate missing detail
VAGUE_GESTURE_PATTERNS = [
    r"^no gesture",
    r"^none$",
    r"^n/a$",
    r"^not applicable",
    r"^not visible",
    r"^no visible",
    r"^nothing",
    r"^unclear",
    r"^unknown",
]
VAGUE_GESTURE_RE = re.compile(
    "|".join(VAGUE_GESTURE_PATTERNS), re.IGNORECASE
)


# ──────────────────────────────────────────────────────────────
# Taxonomy & Synonym Loaders
# ──────────────────────────────────────────────────────────────

class TaxonomyLoader:
    """Load and validate against motion_taxonomy.yaml."""

    def __init__(self, path: Optional[Path] = None):
        self.leaves: set[str] = set()
        self._loaded = False
        if path and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        if not HAS_YAML:
            logger.warning(
                "pyyaml not installed — taxonomy validation will use "
                "built-in fallback leaf set."
            )
            self._use_fallback()
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            ap = data.get("action_primary", {})
            for category_key, category_val in ap.items():
                if isinstance(category_val, dict) and "leaves" in category_val:
                    for leaf in category_val["leaves"]:
                        self.leaves.add(str(leaf).strip())
            self._loaded = True
            logger.info(
                "Taxonomy loaded: %d leaves from %s", len(self.leaves), path
            )
        except Exception as e:
            logger.error("Failed to load taxonomy: %s — using fallback", e)
            self._use_fallback()

    def _use_fallback(self) -> None:
        # Minimal built-in fallback covering the taxonomy in the spec
        self.leaves = {
            # locomotion
            "walking", "running", "moving", "positioning", "leaning",
            "turning", "stepping", "stumbling",
            # manipulation
            "reaching", "carrying", "holding", "grasping", "clasping",
            "releasing", "lifting", "placing", "pushing", "pulling",
            "writing", "pouring", "opening", "closing",
            # posture
            "standing", "sitting", "kneeling", "lying", "crouching",
            "reclining", "bending", "straightening",
            # communication
            "talking", "gesturing", "looking", "focusing", "interacting",
            "engaging", "petting", "nodding", "shaking_head", "shrugging",
            "pointing", "waving", "beckoning", "hugging", "clapping",
            # impact
            "kicking", "spiking", "striking", "throwing", "catching",
            "punching", "slapping", "blocking", "jumping", "landing",
            "falling", "diving",
            # self-directed
            "adjusting", "scratching", "rubbing", "wiping", "stretching",
            "yawning", "breathing_heavy", "fidgeting", "smoking",
            "drinking", "eating",
        }
        self._loaded = True

    def is_valid_action(self, value: str) -> bool:
        """Check if action_primary is a valid leaf or other/<word> format."""
        if not self._loaded:
            self._use_fallback()
        if value in self.leaves:
            return True
        if value.startswith("other/") and len(value) > 6:
            word = value[6:]
            return bool(re.match(r"^[a-z_]+$", word))
        return False

    def is_other_word(self, value: str) -> Optional[str]:
        """Return the custom word if value is other/<word>, else None."""
        if value.startswith("other/") and len(value) > 6:
            return value[6:]
        return None


class SynonymLoader:
    """Load motion_synonyms.yaml for camera term and remap checks."""

    def __init__(self, path: Optional[Path] = None):
        self.camera_terms: set[str] = set()
        self._loaded = False
        if path and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        if not HAS_YAML:
            logger.warning(
                "pyyaml not installed — using built-in camera terms."
            )
            self._use_fallback()
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            ct = data.get("camera_terms_forbidden", [])
            for term in ct:
                self.camera_terms.add(str(term).strip().lower())
            self._loaded = True
            logger.info(
                "Synonyms loaded: %d camera terms from %s",
                len(self.camera_terms), path,
            )
        except Exception as e:
            logger.error("Failed to load synonyms: %s — using fallback", e)
            self._use_fallback()

    def _use_fallback(self) -> None:
        self.camera_terms = {t.lower() for t in CAMERA_TERMS_FALLBACK}
        self._loaded = True

    def contains_camera_term(self, text: str) -> Optional[str]:
        """Return the first camera term found in text, or None."""
        if not self._loaded:
            self._use_fallback()
        text_lower = text.lower().strip()
        # Check exact match first
        if text_lower in self.camera_terms:
            return text_lower
        # Check if any camera term appears as a substring
        for term in sorted(self.camera_terms, key=len, reverse=True):
            if term in text_lower:
                return term
        return None


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def word_count(text: Any) -> int:
    """Count words in a string."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def safe_get(obj: Any, dotpath: str, default: Any = None) -> Any:
    """Navigate a nested dict via dot-path (e.g. 'action_quality.intensity')."""
    parts = dotpath.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part, default)
        else:
            return default
    return current


def caption_hash(text: str) -> str:
    """SHA-256 hash of lowercased, whitespace-normalized caption."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def make_issue(
    shot_id: str,
    person_index: Optional[int],
    check: str,
    field: str,
    message: str,
    **extra: Any,
) -> dict:
    """Build a standardized issue dict."""
    issue: dict[str, Any] = {
        "shot_id": shot_id,
        "person_index": person_index,
        "check": check,
        "field": field,
        "message": message,
    }
    issue.update(extra)
    return issue


# ──────────────────────────────────────────────────────────────
# ShotValidator — validates a single shot JSON
# ──────────────────────────────────────────────────────────────

class ShotValidator:
    """Validate a single shot JSON against the integrated schema."""

    def __init__(
        self,
        taxonomy: TaxonomyLoader,
        synonyms: SynonymLoader,
    ):
        self.taxonomy = taxonomy
        self.synonyms = synonyms

    def validate(
        self, shot: dict
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Returns (errors, warnings, infos) for one shot.
        """
        errors: list[dict] = []
        warnings: list[dict] = []
        infos: list[dict] = []

        shot_id = shot.get("shot_id", "<unknown>")

        # ── Schema-level checks ──
        self._check_schema(shot, shot_id, errors)

        # ── Per-person checks ──
        persons = shot.get("persons")
        if not isinstance(persons, list):
            errors.append(make_issue(
                shot_id, None, "schema_validation_failure",
                "persons", "'persons' is missing or not a list",
            ))
            return errors, warnings, infos

        for person in persons:
            pidx = person.get("person_index")
            self._check_face(person, shot_id, pidx, errors, warnings)
            self._check_body(person, shot_id, pidx, errors, warnings, infos)
            self._check_cross_axis(person, shot_id, pidx, errors, warnings)

        # ── Shot-level interaction checks (CHECK 15 / 16) ──
        # delivery_v1 § 7.2：顶层 interaction 的 count / contact 必须自洽。
        # delivery_v1 § 5.4：persons[].body_analysis.interaction.
        # interacts_with_person_index 应当双向（对称）。
        self._check_top_interaction_consistency(shot, shot_id, errors)
        self._check_interaction_symmetry(shot, shot_id, warnings)

        return errors, warnings, infos

    # ── Schema-level ──

    def _check_schema(
        self, shot: dict, shot_id: str, errors: list[dict]
    ) -> None:
        """CHECK 7: Basic structural validation."""
        required_top = ["shot_id", "persons", "quality_flags"]
        for key in required_top:
            if key not in shot:
                errors.append(make_issue(
                    shot_id, None, "schema_validation_failure",
                    key, f"Required top-level key '{key}' is missing",
                ))

    # ── Shot-level interaction checks ──

    def _check_top_interaction_consistency(
        self, shot: dict, shot_id: str, errors: list[dict]
    ) -> None:
        """CHECK 15: 顶层 interaction 的 count / contact 必须自洽。
        delivery_v1 § 7.2：count='solo' 时 contact 必须是 'none'。
        """
        inter = shot.get("interaction")
        if not isinstance(inter, dict):
            return
        count = inter.get("count")
        contact = inter.get("contact")
        if count == "solo" and contact not in (None, "none"):
            errors.append(make_issue(
                shot_id, None, "interaction_solo_contact_mismatch",
                "interaction.contact",
                f"interaction.count='solo' but contact='{contact}' "
                f"(delivery_v1 § 7.2 requires 'none')",
                value=contact,
            ))

    def _check_interaction_symmetry(
        self, shot: dict, shot_id: str, warnings: list[dict]
    ) -> None:
        """CHECK 16: persons[].body_analysis.interaction.
        interacts_with_person_index 应当双向（对称）。
        - 若 A 引用了不存在的 peer index → WARNING
        - 若 A 引用 B 但 B 没有引用 A → WARNING
        先用 WARNING 不用 ERROR：避免历史数据被阻塞合并。
        """
        persons = shot.get("persons")
        if not isinstance(persons, list):
            return
        # 构建 index → list position 的映射（person_index 可能跳号）
        idx_to_pos: dict = {}
        for pos, p in enumerate(persons):
            if not isinstance(p, dict):
                continue
            idx = p.get("person_index", pos)
            idx_to_pos[idx] = pos

        def _peers_of(p):
            if not isinstance(p, dict):
                return []
            ba = p.get("body_analysis")
            if not isinstance(ba, dict):
                return []
            inter = ba.get("interaction")
            if not isinstance(inter, dict):
                return []
            raw = inter.get("interacts_with_person_index") or []
            return [x for x in raw if isinstance(x, int)]

        for pos, p in enumerate(persons):
            if not isinstance(p, dict):
                continue
            own_idx = p.get("person_index", pos)
            for peer in _peers_of(p):
                if peer not in idx_to_pos:
                    warnings.append(make_issue(
                        shot_id, own_idx,
                        "interaction_references_missing_person",
                        "body_analysis.interaction.interacts_with_person_index",
                        f"person_index={own_idx} references peer={peer} "
                        f"that is not in persons[]",
                        value=peer,
                    ))
                    continue
                peer_p = persons[idx_to_pos[peer]]
                peer_peers = _peers_of(peer_p)
                if own_idx not in peer_peers:
                    warnings.append(make_issue(
                        shot_id, own_idx,
                        "interaction_asymmetric",
                        "body_analysis.interaction.interacts_with_person_index",
                        f"person_index={own_idx} references peer={peer} but "
                        f"peer does not reference {own_idx} back",
                        value=peer,
                    ))

    # ── Face checks ──

    def _check_face(
        self,
        person: dict,
        shot_id: str,
        pidx: Optional[int],
        errors: list[dict],
        warnings: list[dict],
    ) -> None:
        """Basic face_analysis validation."""
        face = person.get("face_analysis")
        if face is None:
            return

        # CHECK 6: micro_expression vs temporal_change
        micro = face.get("micro_expression")
        temporal = face.get("temporal_change")
        if micro is True and temporal != "rapid_micro":
            errors.append(make_issue(
                shot_id, pidx, "micro_temporal_mismatch",
                "face_analysis.temporal_change",
                f"micro_expression=true but temporal_change='{temporal}' "
                f"(expected 'rapid_micro')",
                value=temporal,
            ))

        # Validate primary_emotion is in the allowed set
        prim_em = face.get("primary_emotion")
        if prim_em and prim_em not in VALID_EMOTIONS:
            warnings.append(make_issue(
                shot_id, pidx, "schema_validation_failure",
                "face_analysis.primary_emotion",
                f"Unknown primary_emotion: '{prim_em}'",
                value=prim_em,
            ))

        # Validate temporal_change
        if temporal and temporal not in VALID_TEMPORAL_CHANGES:
            warnings.append(make_issue(
                shot_id, pidx, "schema_validation_failure",
                "face_analysis.temporal_change",
                f"Unknown temporal_change: '{temporal}'",
                value=temporal,
            ))

        # expression_caption length
        expr_cap = face.get("expression_caption", "")
        wc = word_count(expr_cap)
        lo, hi = CAPTION_RANGES["expression_caption"]
        if expr_cap and (wc < lo or wc > hi):
            warnings.append(make_issue(
                shot_id, pidx, "caption_length_out_of_range",
                "face_analysis.expression_caption",
                f"expression_caption length {wc} words (expected {lo}-{hi})",
                value_length=wc,
            ))

    # ── Body checks ──

    def _check_body(
        self,
        person: dict,
        shot_id: str,
        pidx: Optional[int],
        errors: list[dict],
        warnings: list[dict],
        infos: list[dict],
    ) -> None:
        """Body-specific validation."""
        body = person.get("body_analysis")
        quality = person.get("face_analysis", {})

        # CHECK 5: null block without flag
        if body is None:
            # body_analysis is null — check body_clearly_visible
            bcv = safe_get(person, "body_analysis")
            # Need to check the shot-level quality_flags or person-level
            # The flag might be at quality_flags level or inside body_analysis
            # Since body_analysis is null, check if quality_flags says
            # body_clearly_visible != false
            # We cannot determine this here reliably — the flag should have
            # been set to false when body_analysis is null.
            # This will be checked at the batch level if needed.
            return

        bcv = body.get("body_clearly_visible")

        # ── CHECK 1: Camera terms in action fields ──
        action_fields_to_check = {
            "action_primary": body.get("action_primary", ""),
            "body_focus": body.get("body_focus", ""),
        }

        # Add action_quality sub-fields
        aq = body.get("action_quality", {})
        if isinstance(aq, dict):
            for sub_key in ("intensity", "tone", "tempo"):
                val = aq.get(sub_key, "")
                if val:
                    action_fields_to_check[
                        f"action_quality.{sub_key}"
                    ] = str(val)

        # Add kinematics_hint sub-fields
        kh = body.get("kinematics_hint", {})
        if isinstance(kh, dict):
            for sub_key in (
                "trajectory", "periodicity", "symmetry", "duration_class"
            ):
                val = kh.get(sub_key, "")
                if val:
                    # EXCEPTION: trajectory=static is allowed (body static,
                    # not camera static)
                    if sub_key == "trajectory" and str(val).lower() == "static":
                        continue
                    action_fields_to_check[
                        f"kinematics_hint.{sub_key}"
                    ] = str(val)

        for field_name, field_val in action_fields_to_check.items():
            if not field_val:
                continue
            found_term = self.synonyms.contains_camera_term(str(field_val))
            if found_term:
                errors.append(make_issue(
                    shot_id, pidx, "camera_term_in_action",
                    field_name,
                    f"Camera term found in {field_name}: '{found_term}' "
                    f"(value: '{field_val}')",
                    value=str(field_val),
                ))

        # ── CHECK 2: action_primary in taxonomy ──
        action_primary = body.get("action_primary") or ""
        if action_primary and not self.taxonomy.is_valid_action(action_primary):
            errors.append(make_issue(
                shot_id, pidx, "action_primary_not_in_taxonomy",
                "body_analysis.action_primary",
                f"action_primary '{action_primary}' is not in taxonomy "
                f"leaves and is not in other/<word> format",
                value=action_primary,
            ))

        # Collect other/<word> usages (INFO)
        other_word = self.taxonomy.is_other_word(action_primary) if action_primary else None
        if other_word:
            infos.append(make_issue(
                shot_id, pidx, "other_word_usage",
                "body_analysis.action_primary",
                f"other/<word> used: '{other_word}'",
                word=other_word,
            ))

        # ── CHECK 3: visible_body_parts vs shot_frame_of_body ──
        frame = body.get("shot_frame_of_body", "")
        visible_parts = body.get("visible_body_parts", [])
        if isinstance(visible_parts, list) and frame in FRAMING_MAX_PARTS:
            allowed = FRAMING_MAX_PARTS[frame]
            for part in visible_parts:
                if part not in allowed:
                    errors.append(make_issue(
                        shot_id, pidx,
                        "visible_body_parts_inconsistency",
                        "body_analysis.visible_body_parts",
                        f"Part '{part}' is not possible with "
                        f"shot_frame_of_body='{frame}' "
                        f"(allowed: {sorted(allowed)})",
                        value=part,
                        shot_frame=frame,
                    ))

        # ── CHECK 4: body_focus vs framing ──
        body_focus = body.get("body_focus", "")
        if body_focus == "full_body" and frame in ("close_face", "bust"):
            errors.append(make_issue(
                shot_id, pidx, "body_focus_framing_mismatch",
                "body_analysis.body_focus",
                f"body_focus='full_body' but shot_frame_of_body='{frame}'",
                body_focus=body_focus,
                shot_frame=frame,
            ))
        if body_focus == "lower_body" and frame in ("close_face", "bust"):
            errors.append(make_issue(
                shot_id, pidx, "body_focus_framing_mismatch",
                "body_analysis.body_focus",
                f"body_focus='lower_body' but shot_frame_of_body='{frame}'",
                body_focus=body_focus,
                shot_frame=frame,
            ))

        # ── CHECK 8: motion_caption length ──
        motion_cap = body.get("motion_caption", "")
        if motion_cap:
            wc = word_count(motion_cap)
            lo, hi = CAPTION_RANGES["motion_caption"]
            if wc < lo or wc > hi:
                warnings.append(make_issue(
                    shot_id, pidx, "caption_length_out_of_range",
                    "body_analysis.motion_caption",
                    f"motion_caption length {wc} words (expected {lo}-{hi})",
                    value_length=wc,
                ))

        # ── CHECK 9: gesture_detail quality ──
        gesture_detail = body.get("gesture_detail", "")
        if not gesture_detail or not gesture_detail.strip():
            warnings.append(make_issue(
                shot_id, pidx, "missing_gesture_detail",
                "body_analysis.gesture_detail",
                "gesture_detail is empty",
            ))
        elif VAGUE_GESTURE_RE.search(gesture_detail.strip()):
            # Vague but not necessarily wrong — only warn if hands are visible
            hands_visible = body.get("hands_visible", False)
            if hands_visible:
                warnings.append(make_issue(
                    shot_id, pidx, "missing_gesture_detail",
                    "body_analysis.gesture_detail",
                    f"gesture_detail is vague ('{gesture_detail[:60]}...') "
                    f"but hands_visible=true",
                ))

        # ── CHECK 10: hands_visible consistency ──
        hands_visible = body.get("hands_visible", False)
        if hands_visible:
            ubd = body.get("upper_body_detail", {})
            hands_text = ubd.get("hands", "") if isinstance(ubd, dict) else ""
            if not hands_text or hands_text.strip().lower() in (
                "", "not visible", "n/a", "none", "below frame",
                "not visible -- below frame line",
                "not visible — below frame line",
            ):
                warnings.append(make_issue(
                    shot_id, pidx,
                    "hands_visible_without_detail",
                    "body_analysis.upper_body_detail.hands",
                    f"hands_visible=true but upper_body_detail.hands "
                    f"is empty or says 'not visible'",
                    value=hands_text,
                ))

        # ── CHECK 11: alternative_captions completeness ──
        alt_caps = body.get("alternative_captions", {})
        if isinstance(alt_caps, dict):
            alt_keys = {
                "direct": "alt_direct",
                "literary": "alt_literary",
                "direction": "alt_direction",
                "situational": "alt_situational",
            }
            for cap_key, range_key in alt_keys.items():
                cap_val = alt_caps.get(cap_key)
                if cap_val is None:
                    warnings.append(make_issue(
                        shot_id, pidx,
                        "alternative_captions_incomplete",
                        f"body_analysis.alternative_captions.{cap_key}",
                        f"alternative caption '{cap_key}' is null",
                    ))
                elif isinstance(cap_val, str):
                    wc_val = word_count(cap_val)
                    lo, hi = CAPTION_RANGES[range_key]
                    if wc_val < lo:
                        warnings.append(make_issue(
                            shot_id, pidx,
                            "alternative_captions_incomplete",
                            f"body_analysis.alternative_captions.{cap_key}",
                            f"'{cap_key}' has {wc_val} words "
                            f"(minimum {lo} expected)",
                            value_length=wc_val,
                        ))

        # ── CHECK 14: low confidence ──
        motion_conf = body.get("motion_confidence")
        if isinstance(motion_conf, (int, float)) and motion_conf < 0.3:
            warnings.append(make_issue(
                shot_id, pidx, "low_confidence",
                "body_analysis.motion_confidence",
                f"motion_confidence={motion_conf:.2f} (< 0.3 threshold)",
                value=motion_conf,
            ))

    # ── Cross-axis checks ──

    def _check_cross_axis(
        self,
        person: dict,
        shot_id: str,
        pidx: Optional[int],
        errors: list[dict],
        warnings: list[dict],
    ) -> None:
        """CHECK 12 & 13: cross-axis contradictions."""
        body = person.get("body_analysis")
        face = person.get("face_analysis")

        if body is None:
            return

        action_primary = body.get("action_primary", "")
        frame = body.get("shot_frame_of_body", "")
        body_focus = body.get("body_focus", "")
        intensity = safe_get(body, "action_quality.intensity", "")
        tone = safe_get(body, "action_quality.tone", "")
        trajectory = safe_get(body, "kinematics_hint.trajectory", "")

        # CHECK 12: Cross-axis contradictions (body-body)
        # Walking but only face/bust visible
        if action_primary == "walking" and frame in ("close_face", "bust"):
            warnings.append(make_issue(
                shot_id, pidx, "cross_axis_contradiction",
                "action_primary vs shot_frame_of_body",
                f"action_primary='walking' with "
                f"shot_frame_of_body='{frame}' — walking is difficult "
                f"to observe in this framing",
                action_primary=action_primary,
                shot_frame=frame,
            ))

        # Running but only face/bust visible
        if action_primary == "running" and frame in ("close_face", "bust"):
            warnings.append(make_issue(
                shot_id, pidx, "cross_axis_contradiction",
                "action_primary vs shot_frame_of_body",
                f"action_primary='running' with "
                f"shot_frame_of_body='{frame}' — running is difficult "
                f"to observe in this framing",
                action_primary=action_primary,
                shot_frame=frame,
            ))

        # High intensity + static trajectory
        if intensity == "high" and trajectory == "static":
            warnings.append(make_issue(
                shot_id, pidx, "cross_axis_contradiction",
                "action_quality.intensity vs kinematics_hint.trajectory",
                "intensity='high' but trajectory='static' — "
                "high intensity typically involves motion",
                intensity=intensity,
                trajectory=trajectory,
            ))

        # body_focus=full_body with close framing (already ERROR above)
        # body_focus=lower_body with close framing (already ERROR above)

        # CHECK 13: Face-body emotion mismatch
        if face is not None:
            primary_emotion = face.get("primary_emotion", "")

            # Fear + relaxed body
            if primary_emotion == "fear" and tone == "relaxed":
                warnings.append(make_issue(
                    shot_id, pidx, "face_body_emotion_mismatch",
                    "primary_emotion vs action_quality.tone",
                    f"primary_emotion='fear' but action_quality.tone="
                    f"'relaxed' — unusual combination",
                    emotion=primary_emotion,
                    tone=tone,
                ))

            # Anger + relaxed body
            if primary_emotion == "anger" and tone == "relaxed":
                warnings.append(make_issue(
                    shot_id, pidx, "face_body_emotion_mismatch",
                    "primary_emotion vs action_quality.tone",
                    f"primary_emotion='anger' but action_quality.tone="
                    f"'relaxed' — unusual combination",
                    emotion=primary_emotion,
                    tone=tone,
                ))

            # Joy + tense body with high intensity
            if (
                primary_emotion == "joy"
                and tone == "tense"
                and intensity == "high"
            ):
                warnings.append(make_issue(
                    shot_id, pidx, "face_body_emotion_mismatch",
                    "primary_emotion vs action_quality",
                    f"primary_emotion='joy' with tone='tense' and "
                    f"intensity='high' — unusual combination",
                    emotion=primary_emotion,
                    tone=tone,
                    intensity=intensity,
                ))

            # Neutral emotion + high intensity action
            if primary_emotion == "neutral" and intensity == "high":
                warnings.append(make_issue(
                    shot_id, pidx, "face_body_emotion_mismatch",
                    "primary_emotion vs action_quality.intensity",
                    f"primary_emotion='neutral' but intensity='high' — "
                    f"unusual combination",
                    emotion=primary_emotion,
                    intensity=intensity,
                ))


# ──────────────────────────────────────────────────────────────
# BatchValidator — validates a directory of shot JSONs
# ──────────────────────────────────────────────────────────────

class BatchValidator:
    """Validate a directory of shot JSONs and produce a report."""

    def __init__(self, shot_validator: ShotValidator):
        self.shot_validator = shot_validator
        self.all_errors: list[dict] = []
        self.all_warnings: list[dict] = []
        self.all_infos: list[dict] = []

        # Distribution counters
        self.action_primary_dist: Counter = Counter()
        self.shot_frame_dist: Counter = Counter()
        self.intensity_dist: Counter = Counter()
        self.emotion_dist: Counter = Counter()
        self.body_focus_dist: Counter = Counter()
        self.tone_dist: Counter = Counter()
        self.tempo_dist: Counter = Counter()
        self.trajectory_dist: Counter = Counter()
        self.temporal_change_dist: Counter = Counter()

        # Other word tracking
        self.other_words: Counter = Counter()

        # Caption hash tracking
        self.motion_caption_hashes: defaultdict[str, list[str]] = defaultdict(
            list
        )

        # Null rate tracking
        self.total_persons = 0
        self.face_null_count = 0
        self.body_null_count = 0
        self.motion_caption_null_count = 0
        self.gesture_detail_empty_count = 0

        # Shot count
        self.total_shots = 0
        self.passed_shots = 0

    def validate_directory(self, input_dir: Path) -> dict:
        """Process all JSON files, return report dict."""
        json_files = sorted(input_dir.rglob("*.json"))

        if not json_files:
            logger.error("No JSON files found in %s", input_dir)
            return self._build_report(input_dir, 0)

        logger.info("Found %d JSON files in %s", len(json_files), input_dir)

        for i, json_file in enumerate(json_files, 1):
            if i % 500 == 0 or i == len(json_files):
                logger.info("Processing %d / %d ...", i, len(json_files))

            try:
                with open(json_file, encoding="utf-8") as f:
                    shot = json.load(f)
            except json.JSONDecodeError as e:
                shot_id = json_file.stem
                self.all_errors.append(make_issue(
                    shot_id, None, "schema_validation_failure",
                    "file",
                    f"Invalid JSON: {e}",
                    file=str(json_file),
                ))
                self.total_shots += 1
                continue
            except Exception as e:
                shot_id = json_file.stem
                self.all_errors.append(make_issue(
                    shot_id, None, "schema_validation_failure",
                    "file",
                    f"Cannot read file: {e}",
                    file=str(json_file),
                ))
                self.total_shots += 1
                continue

            self.total_shots += 1
            shot_errors, shot_warnings, shot_infos = (
                self.shot_validator.validate(shot)
            )

            self.all_errors.extend(shot_errors)
            self.all_warnings.extend(shot_warnings)
            self.all_infos.extend(shot_infos)

            if not shot_errors:
                self.passed_shots += 1

            # Collect distributions and stats
            self._collect_stats(shot)

        # Post-processing: duplicate caption detection
        self._detect_duplicates()

        return self._build_report(input_dir, len(json_files))

    def _collect_stats(self, shot: dict) -> None:
        """Aggregate distributions and null rates from one shot."""
        persons = shot.get("persons", [])
        shot_id = shot.get("shot_id", "")

        for person in persons:
            if not isinstance(person, dict):
                continue
            self.total_persons += 1

            # Face stats
            face = person.get("face_analysis")
            if face is None:
                self.face_null_count += 1
            else:
                prim_em = face.get("primary_emotion", "")
                if prim_em:
                    self.emotion_dist[prim_em] += 1
                tc = face.get("temporal_change", "")
                if tc:
                    self.temporal_change_dist[tc] += 1

            # Body stats
            body = person.get("body_analysis")
            if body is None:
                self.body_null_count += 1
                self.motion_caption_null_count += 1
                self.gesture_detail_empty_count += 1
                continue

            # action_primary
            ap = body.get("action_primary") or ""
            if ap:
                self.action_primary_dist[ap] += 1

            # other/<word> collection
            if ap and ap.startswith("other/") and len(ap) > 6:
                self.other_words[ap[6:]] += 1

            # shot_frame_of_body
            sf = body.get("shot_frame_of_body", "")
            if sf:
                self.shot_frame_dist[sf] += 1

            # action_quality
            aq = body.get("action_quality", {})
            if isinstance(aq, dict):
                if aq.get("intensity"):
                    self.intensity_dist[aq["intensity"]] += 1
                if aq.get("tone"):
                    self.tone_dist[aq["tone"]] += 1
                if aq.get("tempo"):
                    self.tempo_dist[aq["tempo"]] += 1

            # body_focus
            bf = body.get("body_focus", "")
            if bf:
                self.body_focus_dist[bf] += 1

            # kinematics trajectory
            traj = safe_get(body, "kinematics_hint.trajectory", "")
            if traj:
                self.trajectory_dist[traj] += 1

            # motion_caption null check
            mc = body.get("motion_caption", "")
            if not mc or not mc.strip():
                self.motion_caption_null_count += 1
            else:
                h = caption_hash(mc)
                self.motion_caption_hashes[h].append(shot_id)

            # gesture_detail empty check
            gd = body.get("gesture_detail", "")
            if not gd or not gd.strip():
                self.gesture_detail_empty_count += 1

    def _detect_duplicates(self) -> None:
        """CHECK 15: Hash-based motion_caption dedup detection."""
        dup_count = 0
        for h, shot_ids in self.motion_caption_hashes.items():
            if len(shot_ids) > 1:
                dup_count += len(shot_ids) - 1

        total_captions = sum(
            len(ids) for ids in self.motion_caption_hashes.values()
        )
        dup_rate = (
            (dup_count / total_captions * 100) if total_captions > 0 else 0.0
        )

        if dup_rate > 5.0:
            self.all_warnings.append(make_issue(
                "<global>", None, "duplicate_caption_hash",
                "motion_caption",
                f"Duplicate motion_caption rate {dup_rate:.1f}% "
                f"exceeds 5% threshold ({dup_count} duplicates "
                f"out of {total_captions} captions)",
                duplicate_count=dup_count,
                total_captions=total_captions,
                duplicate_rate_pct=round(dup_rate, 2),
            ))

        self._dup_count = dup_count
        self._total_captions = total_captions
        self._dup_rate = dup_rate

    def _build_report(self, input_dir: Path, file_count: int) -> dict:
        """Assemble the final validation report."""
        error_count = len(self.all_errors)
        warning_count = len(self.all_warnings)

        report = {
            "report_version": "1.0",
            "report_date": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "input_dir": str(input_dir),
            "taxonomy_file": "",  # filled by caller
            "synonyms_file": "",  # filled by caller

            "summary": {
                "total_shots": self.total_shots,
                "total_persons": self.total_persons,
                "json_files_scanned": file_count,
                "passed": self.passed_shots,
                "errors": error_count,
                "warnings": warning_count,
                "error_rate_pct": round(
                    error_count / max(self.total_shots, 1) * 100, 2
                ),
                "warning_rate_pct": round(
                    warning_count / max(self.total_shots, 1) * 100, 2
                ),
            },

            "errors": self.all_errors,
            "warnings": self.all_warnings,

            "distributions": {
                "action_primary": dict(
                    self.action_primary_dist.most_common()
                ),
                "shot_frame_of_body": dict(
                    self.shot_frame_dist.most_common()
                ),
                "action_quality_intensity": dict(
                    self.intensity_dist.most_common()
                ),
                "action_quality_tone": dict(
                    self.tone_dist.most_common()
                ),
                "action_quality_tempo": dict(
                    self.tempo_dist.most_common()
                ),
                "primary_emotion": dict(
                    self.emotion_dist.most_common()
                ),
                "body_focus": dict(
                    self.body_focus_dist.most_common()
                ),
                "trajectory": dict(
                    self.trajectory_dist.most_common()
                ),
                "temporal_change": dict(
                    self.temporal_change_dist.most_common()
                ),
            },

            "other_words": [
                {"word": word, "count": count}
                for word, count in self.other_words.most_common()
            ],

            "duplicate_captions": {
                "motion_caption_hash_duplicates": getattr(
                    self, "_dup_count", 0
                ),
                "total_captions": getattr(self, "_total_captions", 0),
                "duplicate_rate_pct": round(
                    getattr(self, "_dup_rate", 0.0), 2
                ),
            },

            "null_rates": {
                "face_analysis_null_pct": round(
                    self.face_null_count / max(self.total_persons, 1) * 100, 1
                ),
                "body_analysis_null_pct": round(
                    self.body_null_count / max(self.total_persons, 1) * 100, 1
                ),
                "motion_caption_null_pct": round(
                    self.motion_caption_null_count
                    / max(self.total_persons, 1) * 100, 1
                ),
                "gesture_detail_empty_pct": round(
                    self.gesture_detail_empty_count
                    / max(self.total_persons, 1) * 100, 1
                ),
            },
        }

        # Error breakdown by check type
        error_breakdown: Counter = Counter()
        for e in self.all_errors:
            error_breakdown[e.get("check", "unknown")] += 1
        report["error_breakdown"] = dict(error_breakdown.most_common())

        warning_breakdown: Counter = Counter()
        for w in self.all_warnings:
            warning_breakdown[w.get("check", "unknown")] += 1
        report["warning_breakdown"] = dict(warning_breakdown.most_common())

        return report


# ──────────────────────────────────────────────────────────────
# NULL-block validation (shot-level quality_flags check)
# ──────────────────────────────────────────────────────────────

def check_null_block_flags(
    shot: dict,
    errors: list[dict],
) -> None:
    """
    CHECK 5: body_analysis is null but quality_flags.body_clearly_visible
    is not false.
    """
    shot_id = shot.get("shot_id", "<unknown>")
    quality_flags = shot.get("quality_flags", {})
    bcv_flag = quality_flags.get("body_clearly_visible")

    persons = shot.get("persons", [])
    for person in persons:
        if not isinstance(person, dict):
            continue
        pidx = person.get("person_index")
        body = person.get("body_analysis")

        if body is None:
            # Check person-level or shot-level flag
            person_bcv = None
            if isinstance(body, dict):
                person_bcv = body.get("body_clearly_visible")

            # body_analysis is None so person_bcv check is not possible
            # Fall back to shot-level quality_flags
            if bcv_flag is not False and bcv_flag is not None:
                # Only flag if body_clearly_visible is explicitly true
                if bcv_flag is True:
                    errors.append(make_issue(
                        shot_id, pidx,
                        "null_block_without_flag",
                        "quality_flags.body_clearly_visible",
                        "body_analysis is null but "
                        "quality_flags.body_clearly_visible=true",
                    ))


# ──────────────────────────────────────────────────────────────
# Extended BatchValidator with null-block check
# ──────────────────────────────────────────────────────────────

class ExtendedBatchValidator(BatchValidator):
    """Extends BatchValidator with shot-level cross-checks."""

    def validate_directory(self, input_dir: Path) -> dict:
        json_files = sorted(input_dir.rglob("*.json"))

        if not json_files:
            logger.error("No JSON files found in %s", input_dir)
            return self._build_report(input_dir, 0)

        logger.info("Found %d JSON files in %s", len(json_files), input_dir)

        for i, json_file in enumerate(json_files, 1):
            if i % 500 == 0 or i == len(json_files):
                logger.info("Processing %d / %d ...", i, len(json_files))

            try:
                with open(json_file, encoding="utf-8") as f:
                    shot = json.load(f)
            except json.JSONDecodeError as e:
                shot_id = json_file.stem
                self.all_errors.append(make_issue(
                    shot_id, None, "schema_validation_failure",
                    "file",
                    f"Invalid JSON: {e}",
                    file=str(json_file),
                ))
                self.total_shots += 1
                continue
            except Exception as e:
                shot_id = json_file.stem
                self.all_errors.append(make_issue(
                    shot_id, None, "schema_validation_failure",
                    "file",
                    f"Cannot read file: {e}",
                    file=str(json_file),
                ))
                self.total_shots += 1
                continue

            self.total_shots += 1

            # Per-person validation
            shot_errors, shot_warnings, shot_infos = (
                self.shot_validator.validate(shot)
            )

            # Shot-level null-block check (CHECK 5)
            check_null_block_flags(shot, shot_errors)

            self.all_errors.extend(shot_errors)
            self.all_warnings.extend(shot_warnings)
            self.all_infos.extend(shot_infos)

            if not shot_errors:
                self.passed_shots += 1

            self._collect_stats(shot)

        # Post-processing
        self._detect_duplicates()

        return self._build_report(input_dir, len(json_files))


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate VLM inference output JSONs for the Video_DB_Face project. "
            "Checks body_analysis fields against motion_taxonomy.yaml, "
            "detects camera terms, cross-axis contradictions, and produces "
            "a validation report with distributions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/validate_body_analysis.py \\\n"
            "    --input deliverable/movies/The_Dinner_2017/shots/ \\\n"
            "    --taxonomy docs/motion_taxonomy.yaml \\\n"
            "    --synonyms docs/motion_synonyms.yaml \\\n"
            "    --output validation_report.json\n"
        ),
    )

    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Directory containing per-shot JSON files (searched recursively).",
    )
    parser.add_argument(
        "--taxonomy", "-t",
        type=Path,
        default=None,
        help="Path to motion_taxonomy.yaml (default: built-in fallback).",
    )
    parser.add_argument(
        "--synonyms", "-s",
        type=Path,
        default=None,
        help="Path to motion_synonyms.yaml (default: built-in fallback).",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help=(
            "Output path for validation_report.json "
            "(default: <input>/validation_report.json)."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging.",
    )

    args = parser.parse_args()

    # ── Logging setup ──
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # ── Input validation ──
    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(2)

    # ── Load taxonomy and synonyms ──
    taxonomy_path = args.taxonomy.resolve() if args.taxonomy else None
    synonyms_path = args.synonyms.resolve() if args.synonyms else None

    if taxonomy_path and not taxonomy_path.exists():
        logger.warning("Taxonomy file not found: %s — using fallback", taxonomy_path)
        taxonomy_path = None
    if synonyms_path and not synonyms_path.exists():
        logger.warning("Synonyms file not found: %s — using fallback", synonyms_path)
        synonyms_path = None

    if not HAS_YAML and (taxonomy_path or synonyms_path):
        logger.warning(
            "pyyaml is not installed. YAML files will not be loaded. "
            "Install with: pip install pyyaml"
        )

    taxonomy = TaxonomyLoader(taxonomy_path)
    synonyms = SynonymLoader(synonyms_path)

    # ── Run validation ──
    shot_validator = ShotValidator(taxonomy, synonyms)
    batch_validator = ExtendedBatchValidator(shot_validator)

    logger.info("Starting validation of %s", input_dir)
    report = batch_validator.validate_directory(input_dir)

    # Fill in file paths
    report["taxonomy_file"] = str(taxonomy_path) if taxonomy_path else "built-in fallback"
    report["synonyms_file"] = str(synonyms_path) if synonyms_path else "built-in fallback"

    # ── Output ──
    output_path = args.output
    if output_path is None:
        output_path = input_dir / "validation_report.json"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Report written to %s", output_path)

    # ── Summary to stderr ──
    summary = report["summary"]
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total shots:   %d", summary["total_shots"])
    logger.info("  Total persons: %d", summary["total_persons"])
    logger.info("  Passed:        %d", summary["passed"])
    logger.info("  Errors:        %d  (%.2f%%)",
                summary["errors"], summary["error_rate_pct"])
    logger.info("  Warnings:      %d  (%.2f%%)",
                summary["warnings"], summary["warning_rate_pct"])
    logger.info("=" * 60)

    if report.get("error_breakdown"):
        logger.info("Error breakdown:")
        for check, count in report["error_breakdown"].items():
            logger.info("  %-40s %d", check, count)

    if report.get("other_words"):
        logger.info("other/<word> usages:")
        for entry in report["other_words"][:20]:
            logger.info("  other/%-30s %d", entry["word"], entry["count"])

    dup_info = report.get("duplicate_captions", {})
    dup_rate = dup_info.get("duplicate_rate_pct", 0.0)
    if dup_rate > 0:
        logger.info(
            "Caption duplication: %d duplicates (%.2f%%)",
            dup_info.get("motion_caption_hash_duplicates", 0),
            dup_rate,
        )

    # ── Exit code ──
    if summary["errors"] > 0:
        logger.info("RESULT: FAIL (%d errors found)", summary["errors"])
        sys.exit(1)
    else:
        logger.info("RESULT: PASS (0 errors)")
        sys.exit(0)


if __name__ == "__main__":
    main()
