"""Tests for ``src/runpod/post_normalize.py`` — delivery_v1 compliance fixup.

Fixture data is derived from real VLM errors observed in Pod log
``2026-04-20 04:51`` for shot_0035 (4-person dominant shot).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.runpod.post_normalize import (  # noqa: E402
    fix_action_quality_camera_terms,
    fix_face_clearly_visible,
    fix_facial_attribute_bools,
    fix_interaction,
    fix_primary_emotion,
    post_fix_compliance,
)


def _shot(interaction: dict, persons: list[dict]) -> dict:
    return {
        "shot_id":     "test/shot_0035",
        "interaction": interaction,
        "persons":     persons,
    }


class TestFixInteraction(unittest.TestCase):

    def test_count_int_zero_cloned_from_shot(self):
        obj = _shot(
            {"count": "solo", "contact": "none", "relation": "parallel"},
            [{"body_analysis": {"interaction": {
                "count": 0, "contact": "none", "relation": "parallel",
                "interacts_with_person_index": [],
            }}}],
        )
        fixes = fix_interaction(obj)
        self.assertEqual(fixes, 1)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"]["count"],
            "solo",
        )

    def test_count_int_one_in_dyadic_shot(self):
        obj = _shot(
            {"count": "dyadic", "contact": "none", "relation": "coordinated"},
            [{"body_analysis": {"interaction": {
                "count": 1, "contact": "none", "relation": "coordinated",
                "interacts_with_person_index": [1],
            }}}],
        )
        fix_interaction(obj)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"]["count"],
            "dyadic",
        )

    def test_contact_false_bool_to_none(self):
        obj = _shot(
            {"count": "solo", "contact": "none", "relation": "parallel"},
            [{"body_analysis": {"interaction": {
                "count": "solo", "contact": False, "relation": "parallel",
                "interacts_with_person_index": [],
            }}}],
        )
        fix_interaction(obj)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"]["contact"],
            "none",
        )

    def test_contact_self_contact_marker_to_none(self):
        obj = _shot(
            {"count": "dyadic", "contact": "none", "relation": "coordinated"},
            [{"body_analysis": {"interaction": {
                "count": "dyadic",
                "contact": "self-contact (hand to face)",
                "relation": "coordinated",
                "interacts_with_person_index": [],
            }}}],
        )
        fix_interaction(obj)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"]["contact"],
            "none",
        )

    def test_contact_long_self_string_to_none(self):
        obj = _shot(
            {"count": "triadic", "contact": "incidental", "relation": "opposing"},
            [{"body_analysis": {"interaction": {
                "count": "triadic",
                "contact": "self-contact with right hand on face",
                "relation": "coordinated",
                "interacts_with_person_index": [],
            }}}],
        )
        fix_interaction(obj)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"]["contact"],
            "none",
        )

    def test_relation_independent_to_parallel(self):
        obj = _shot(
            {"count": "solo", "contact": "none", "relation": "parallel"},
            [{"body_analysis": {"interaction": {
                "count": "solo", "contact": "none", "relation": "independent",
                "interacts_with_person_index": [],
            }}}],
        )
        fix_interaction(obj)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"]["relation"],
            "parallel",
        )

    def test_relation_self_directed_to_parallel(self):
        obj = _shot(
            {"count": "dyadic", "contact": "none", "relation": "coordinated"},
            [{"body_analysis": {"interaction": {
                "count": "dyadic", "contact": "none",
                "relation": "self-directed",
                "interacts_with_person_index": [],
            }}}],
        )
        fix_interaction(obj)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"]["relation"],
            "parallel",
        )

    def test_relation_free_form_falls_back_to_shot(self):
        obj = _shot(
            {"count": "dyadic", "contact": "none", "relation": "opposing"},
            [{"body_analysis": {"interaction": {
                "count": "dyadic", "contact": "none",
                "relation": "self-directed emotional gesture",
                "interacts_with_person_index": [],
            }}}],
        )
        fix_interaction(obj)
        # Non-synonym free-form string → fall back to shot-level relation.
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"]["relation"],
            "opposing",
        )

    def test_interacts_with_bool_to_empty_list(self):
        obj = _shot(
            {"count": "solo", "contact": "none", "relation": "parallel"},
            [{"body_analysis": {"interaction": {
                "count": "solo", "contact": "none", "relation": "parallel",
                "interacts_with_person_index": False,
            }}}],
        )
        fix_interaction(obj)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"][
                "interacts_with_person_index"
            ],
            [],
        )

    def test_interacts_with_mixed_values_cleaned(self):
        obj = _shot(
            {"count": "triadic", "contact": "none", "relation": "parallel"},
            [{"body_analysis": {"interaction": {
                "count": "triadic", "contact": "none", "relation": "parallel",
                "interacts_with_person_index": [0, "x", -1, 2],
            }}}],
        )
        fix_interaction(obj)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"][
                "interacts_with_person_index"
            ],
            [0, 2],
        )

    def test_valid_values_untouched(self):
        obj = _shot(
            {"count": "dyadic", "contact": "none", "relation": "coordinated"},
            [{"body_analysis": {"interaction": {
                "count": "dyadic", "contact": "none", "relation": "coordinated",
                "interacts_with_person_index": [1],
            }}}],
        )
        fixes = fix_interaction(obj)
        self.assertEqual(fixes, 0)

    def test_per_person_contact_none_preserved_when_valid(self):
        # multi_action_confrontation: Person 2 has contact=none while
        # shot-level is incidental. Per-person valid enum must NOT be
        # overwritten.
        obj = _shot(
            {"count": "triadic", "contact": "incidental", "relation": "opposing"},
            [{"body_analysis": {"interaction": {
                "count": "triadic", "contact": "none", "relation": "parallel",
                "interacts_with_person_index": [],
            }}}],
        )
        fix_interaction(obj)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"]["contact"],
            "none",
        )
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"]["relation"],
            "parallel",
        )

    def test_null_interaction_block_rebuilt_from_shot(self):
        obj = _shot(
            {"count": "dyadic", "contact": "none", "relation": "coordinated"},
            [{"body_analysis": {"interaction": None}}],
        )
        fixes = fix_interaction(obj)
        self.assertEqual(fixes, 1)
        rebuilt = obj["persons"][0]["body_analysis"]["interaction"]
        self.assertEqual(rebuilt["count"], "dyadic")
        self.assertEqual(rebuilt["contact"], "none")
        self.assertEqual(rebuilt["relation"], "coordinated")
        self.assertEqual(rebuilt["interacts_with_person_index"], [])

    def test_shot_has_invalid_count_falls_back_to_dyadic(self):
        obj = _shot(
            {"count": "weird_value", "contact": "none", "relation": "parallel"},
            [{"body_analysis": {"interaction": {
                "count": 0, "contact": "none", "relation": "parallel",
                "interacts_with_person_index": [],
            }}}],
        )
        fix_interaction(obj)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["interaction"]["count"],
            "dyadic",
        )


class TestFixFaceClearlyVisible(unittest.TestCase):

    def test_missing_with_content_injects_true(self):
        obj = {"persons": [{"face_analysis": {
            "primary_emotion": "sadness",
            "expression_confidence": 0.9,
        }}]}
        fixes = fix_face_clearly_visible(obj)
        self.assertEqual(fixes, 1)
        self.assertIs(
            obj["persons"][0]["face_analysis"]["face_clearly_visible"],
            True,
        )

    def test_missing_with_empty_fields_injects_false(self):
        obj = {"persons": [{"face_analysis": {"distinctive_notes": ""}}]}
        fixes = fix_face_clearly_visible(obj)
        self.assertEqual(fixes, 1)
        self.assertIs(
            obj["persons"][0]["face_analysis"]["face_clearly_visible"],
            False,
        )

    def test_already_present_untouched(self):
        obj = {"persons": [{"face_analysis": {"face_clearly_visible": False}}]}
        self.assertEqual(fix_face_clearly_visible(obj), 0)
        self.assertIs(
            obj["persons"][0]["face_analysis"]["face_clearly_visible"],
            False,
        )

    def test_null_face_analysis_skipped(self):
        obj = {"persons": [{"face_analysis": None}]}
        self.assertEqual(fix_face_clearly_visible(obj), 0)

    def test_valence_zero_counts_as_content(self):
        obj = {"persons": [{"face_analysis": {"valence": 0.0}}]}
        fix_face_clearly_visible(obj)
        self.assertIs(
            obj["persons"][0]["face_analysis"]["face_clearly_visible"],
            True,
        )


class TestFixFacialAttributeBools(unittest.TestCase):

    def test_makeup_subtle_to_true(self):
        obj = {"persons": [{"face_analysis": {"facial_attributes": {
            "makeup_visible": "subtle",
        }}}]}
        fix_facial_attribute_bools(obj)
        self.assertIs(
            obj["persons"][0]["face_analysis"]["facial_attributes"][
                "makeup_visible"
            ],
            True,
        )

    def test_makeup_none_to_false(self):
        obj = {"persons": [{"face_analysis": {"facial_attributes": {
            "makeup_visible": "none",
        }}}]}
        fix_facial_attribute_bools(obj)
        self.assertIs(
            obj["persons"][0]["face_analysis"]["facial_attributes"][
                "makeup_visible"
            ],
            False,
        )

    def test_glasses_string_true_coerced(self):
        obj = {"persons": [{"face_analysis": {"facial_attributes": {
            "glasses": "true",
        }}}]}
        fix_facial_attribute_bools(obj)
        self.assertIs(
            obj["persons"][0]["face_analysis"]["facial_attributes"]["glasses"],
            True,
        )

    def test_already_bool_untouched(self):
        obj = {"persons": [{"face_analysis": {"facial_attributes": {
            "mask": False, "glasses": True, "makeup_visible": True,
        }}}]}
        self.assertEqual(fix_facial_attribute_bools(obj), 0)

    def test_facial_hair_not_coerced(self):
        obj = {"persons": [{"face_analysis": {"facial_attributes": {
            "facial_hair": "stubble", "makeup_visible": "heavy",
        }}}]}
        fixes = fix_facial_attribute_bools(obj)
        self.assertEqual(fixes, 1)
        self.assertEqual(
            obj["persons"][0]["face_analysis"]["facial_attributes"][
                "facial_hair"
            ],
            "stubble",
        )


class TestFixActionQualityCameraTerms(unittest.TestCase):

    def _body(self, action_quality: dict) -> dict:
        return {
            "persons": [{"body_analysis": {"action_quality": action_quality}}]
        }

    def test_tempo_static_substituted_with_minimal(self):
        # Pod 2026-04-20 05:40 shot_0036 root cause: VLM used 'static'
        # for a motionless subject, triggering camera_term_in_action.
        obj = self._body({"intensity": "low", "tempo": "static", "tone": "tense"})
        fixes = fix_action_quality_camera_terms(obj)
        self.assertEqual(fixes, 1)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["action_quality"]["tempo"],
            "minimal",
        )

    def test_tempo_tracking_substituted_with_sustained(self):
        obj = self._body({"tempo": "tracking"})
        fix_action_quality_camera_terms(obj)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["action_quality"]["tempo"],
            "sustained",
        )

    def test_tempo_unknown_camera_term_to_none(self):
        obj = self._body({"tempo": "pan"})
        fix_action_quality_camera_terms(obj)
        self.assertIsNone(
            obj["persons"][0]["body_analysis"]["action_quality"]["tempo"],
        )

    def test_tone_with_camera_term_to_none(self):
        # 'tone' has no safe substitution map.
        obj = self._body({"tone": "static"})
        fix_action_quality_camera_terms(obj)
        self.assertIsNone(
            obj["persons"][0]["body_analysis"]["action_quality"]["tone"],
        )

    def test_tempo_embedded_camera_term(self):
        # VLM sometimes writes "static hold" — also banned (contains a term).
        obj = self._body({"tempo": "static hold"})
        fixes = fix_action_quality_camera_terms(obj)
        self.assertEqual(fixes, 1)
        # Full-string not in substitution map → None
        self.assertIsNone(
            obj["persons"][0]["body_analysis"]["action_quality"]["tempo"],
        )

    def test_clean_tempo_untouched(self):
        obj = self._body({"tempo": "fast", "tone": "urgent"})
        self.assertEqual(fix_action_quality_camera_terms(obj), 0)
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["action_quality"]["tempo"],
            "fast",
        )

    def test_none_tempo_untouched(self):
        obj = self._body({"tempo": None, "tone": None})
        self.assertEqual(fix_action_quality_camera_terms(obj), 0)

    def test_missing_action_quality_no_crash(self):
        obj = {"persons": [{"body_analysis": {}}]}
        self.assertEqual(fix_action_quality_camera_terms(obj), 0)

    def test_empty_string_tempo_untouched(self):
        obj = self._body({"tempo": ""})
        self.assertEqual(fix_action_quality_camera_terms(obj), 0)


class TestFixPrimaryEmotion(unittest.TestCase):

    def _face(self, **kwargs) -> dict:
        return {"persons": [{"face_analysis": dict(kwargs)}]}

    def test_concern_maps_to_fear(self):
        # Real Pod 2026-04-20 07:24 shot_0731 root cause.
        obj = self._face(primary_emotion="concern")
        fixes = fix_primary_emotion(obj)
        self.assertEqual(fixes, 1)
        self.assertEqual(
            obj["persons"][0]["face_analysis"]["primary_emotion"],
            "fear",
        )

    def test_worried_anxious_map_to_fear(self):
        for synonym in ("worried", "anxious", "nervous", "alarmed"):
            obj = self._face(primary_emotion=synonym)
            fix_primary_emotion(obj)
            self.assertEqual(
                obj["persons"][0]["face_analysis"]["primary_emotion"],
                "fear",
                f"synonym {synonym!r} should map to 'fear'",
            )

    def test_happy_maps_to_joy(self):
        obj = self._face(primary_emotion="happy")
        fix_primary_emotion(obj)
        self.assertEqual(
            obj["persons"][0]["face_analysis"]["primary_emotion"], "joy",
        )

    def test_sad_maps_to_sadness(self):
        obj = self._face(primary_emotion="sad")
        fix_primary_emotion(obj)
        self.assertEqual(
            obj["persons"][0]["face_analysis"]["primary_emotion"], "sadness",
        )

    def test_mixed_maps_to_complex(self):
        obj = self._face(primary_emotion="mixed")
        fix_primary_emotion(obj)
        self.assertEqual(
            obj["persons"][0]["face_analysis"]["primary_emotion"], "complex",
        )

    def test_valid_enum_untouched(self):
        obj = self._face(primary_emotion="fear")
        fixes = fix_primary_emotion(obj)
        self.assertEqual(fixes, 0)
        self.assertEqual(
            obj["persons"][0]["face_analysis"]["primary_emotion"], "fear",
        )

    def test_unknown_string_falls_back_to_complex(self):
        obj = self._face(primary_emotion="bewildered-and-lost")
        fix_primary_emotion(obj)
        self.assertEqual(
            obj["persons"][0]["face_analysis"]["primary_emotion"], "complex",
        )

    def test_case_insensitive_match(self):
        obj = self._face(primary_emotion="Concern")
        fix_primary_emotion(obj)
        self.assertEqual(
            obj["persons"][0]["face_analysis"]["primary_emotion"], "fear",
        )

    def test_secondary_emotion_also_coerced(self):
        obj = self._face(primary_emotion="fear", secondary_emotion="worried")
        fix_primary_emotion(obj)
        self.assertEqual(
            obj["persons"][0]["face_analysis"]["secondary_emotion"], "fear",
        )

    def test_null_emotion_untouched(self):
        obj = self._face(primary_emotion="fear", secondary_emotion=None)
        fixes = fix_primary_emotion(obj)
        self.assertEqual(fixes, 0)
        self.assertIsNone(
            obj["persons"][0]["face_analysis"]["secondary_emotion"],
        )

    def test_non_string_emotion_to_complex(self):
        obj = self._face(primary_emotion=3)
        fix_primary_emotion(obj)
        self.assertEqual(
            obj["persons"][0]["face_analysis"]["primary_emotion"], "complex",
        )

    def test_missing_face_analysis_no_crash(self):
        obj = {"persons": [{"body_analysis": {}}]}
        self.assertEqual(fix_primary_emotion(obj), 0)


class TestPostFixCompliance(unittest.TestCase):

    def test_full_shot_fixture_passes(self):
        # Replay of real Pod 2026-04-20 04:51 error pattern.
        obj = {
            "shot_id":     "movie/shot_0035",
            "interaction": {
                "count": "dyadic", "contact": "none", "relation": "coordinated",
            },
            "persons": [
                {
                    "person_index": 0,
                    "spatial_position": "left",
                    "face_analysis": {
                        "primary_emotion": "sadness",
                        "expression_confidence": 0.9,
                        "facial_attributes": {"makeup_visible": "subtle"},
                    },
                    "body_analysis": {
                        "action_quality": {"intensity": "low", "tempo": "static"},
                        "interaction": {
                            "count": 0, "contact": False, "relation": "independent",
                            "interacts_with_person_index": [],
                        },
                    },
                },
                {
                    "person_index": 1,
                    "spatial_position": "center",
                    "face_analysis": {
                        "primary_emotion": "sadness",
                        "expression_confidence": 0.95,
                    },
                    "body_analysis": {
                        "action_quality": {"intensity": "mid", "tempo": "pan"},
                        "interaction": {
                            "count": 1,
                            "contact": "self-contact (hand to face)",
                            "relation": "self-directed",
                            "interacts_with_person_index": [],
                        },
                    },
                },
            ],
        }
        post_fix_compliance(obj)

        for p in obj["persons"]:
            self.assertIn("face_clearly_visible", p["face_analysis"])
            inter = p["body_analysis"]["interaction"]
            self.assertIn(
                inter["count"], {"solo", "dyadic", "triadic", "crowd"},
            )
            self.assertIn(
                inter["contact"], {"none", "incidental", "sustained"},
            )
            self.assertIn(
                inter["relation"],
                {"parallel", "coordinated", "opposing", "hierarchical"},
            )
            self.assertIsInstance(inter["interacts_with_person_index"], list)

        self.assertIs(
            obj["persons"][0]["face_analysis"]["facial_attributes"][
                "makeup_visible"
            ],
            True,
        )

        # action_quality.tempo: 'static' → 'minimal' (safe substitution),
        # 'pan' → None (no substitution available).
        self.assertEqual(
            obj["persons"][0]["body_analysis"]["action_quality"]["tempo"],
            "minimal",
        )
        self.assertIsNone(
            obj["persons"][1]["body_analysis"]["action_quality"]["tempo"],
        )

    def test_empty_persons_list_no_crash(self):
        obj = {"shot_id": "x", "interaction": {}, "persons": []}
        result = post_fix_compliance(obj)
        self.assertIs(result, obj)

    def test_missing_persons_key_no_crash(self):
        obj = {"shot_id": "x", "interaction": {}}
        result = post_fix_compliance(obj)
        self.assertIs(result, obj)

    def test_returns_same_ref_for_chaining(self):
        obj = {"shot_id": "x", "interaction": {}, "persons": []}
        self.assertIs(post_fix_compliance(obj), obj)


class TestStripCameraTermsInCaptions(unittest.TestCase):
    """C3 (2026-04-22): port of removed spec normalize_tags caption stripping."""

    def test_strips_close_up_from_face_caption(self):
        from src.runpod.post_normalize import strip_camera_terms_in_captions
        obj = {"persons": [{
            "face_analysis": {
                "expression_caption": "In a close-up, his eyes widen.",
            },
        }]}
        n = strip_camera_terms_in_captions(obj)
        self.assertGreater(n, 0)
        cap = obj["persons"][0]["face_analysis"]["expression_caption"]
        self.assertNotIn("close-up", cap.lower())

    def test_strips_alt_captions(self):
        from src.runpod.post_normalize import strip_camera_terms_in_captions
        obj = {"persons": [{
            "face_analysis": {
                "alternative_captions": {
                    "direct":      "A wide shot of his face",
                    "literary":    "the wide-angle held on him",
                    "direction":   "hold the close-up steady",
                    "situational": "as the camera does a slow pan",
                },
            },
        }]}
        n = strip_camera_terms_in_captions(obj)
        self.assertGreater(n, 0)
        for v in obj["persons"][0]["face_analysis"]["alternative_captions"].values():
            self.assertNotIn("close-up", v.lower())

    def test_strips_body_motion_caption(self):
        from src.runpod.post_normalize import strip_camera_terms_in_captions
        obj = {"persons": [{
            "body_analysis": {
                "motion_caption": "He leans in during the close-up.",
            },
        }]}
        n = strip_camera_terms_in_captions(obj)
        self.assertGreater(n, 0)
        self.assertNotIn(
            "close-up",
            obj["persons"][0]["body_analysis"]["motion_caption"].lower(),
        )

    def test_strips_shot_context_summaries(self):
        from src.runpod.post_normalize import strip_camera_terms_in_captions
        obj = {"shot_context": {
            "shot_emotion_summary": "Tense atmosphere held in a close-up.",
            "shot_motion_summary":  "Static frame with a slow pan toward end.",
        }}
        n = strip_camera_terms_in_captions(obj)
        self.assertGreater(n, 0)

    def test_no_camera_term_no_change(self):
        from src.runpod.post_normalize import strip_camera_terms_in_captions
        obj = {"persons": [{
            "face_analysis": {
                "expression_caption": "He looks at her with quiet sadness.",
            },
        }]}
        n = strip_camera_terms_in_captions(obj)
        self.assertEqual(n, 0)

    def test_word_boundary_does_not_strip_substrings(self):
        """'closeness' must not be touched by 'close-up' rule."""
        from src.runpod.post_normalize import strip_camera_terms_in_captions
        obj = {"persons": [{
            "face_analysis": {
                "expression_caption": "Their closeness is palpable.",
            },
        }]}
        strip_camera_terms_in_captions(obj)
        self.assertIn(
            "closeness",
            obj["persons"][0]["face_analysis"]["expression_caption"],
        )

    def test_missing_persons_no_crash(self):
        from src.runpod.post_normalize import strip_camera_terms_in_captions
        self.assertEqual(strip_camera_terms_in_captions({}), 0)


class TestEnforceAltcapNullConsistency(unittest.TestCase):
    """C6 (2026-04-22): face/body_clearly_visible vs all-null alt_captions."""

    def test_face_visible_but_all_alt_caps_null_downgrades_gate(self):
        from src.runpod.post_normalize import enforce_altcap_null_consistency
        obj = {"persons": [{
            "face_analysis": {
                "face_clearly_visible": True,
                "alternative_captions": {
                    "direct": None, "literary": None,
                    "direction": None, "situational": None,
                },
            },
        }]}
        n = enforce_altcap_null_consistency(obj)
        self.assertEqual(n, 1)
        self.assertFalse(obj["persons"][0]["face_analysis"]["face_clearly_visible"])

    def test_face_visible_with_one_caption_kept(self):
        from src.runpod.post_normalize import enforce_altcap_null_consistency
        obj = {"persons": [{
            "face_analysis": {
                "face_clearly_visible": True,
                "alternative_captions": {
                    "direct": "He smiles.",
                    "literary": None, "direction": None, "situational": None,
                },
            },
        }]}
        n = enforce_altcap_null_consistency(obj)
        self.assertEqual(n, 0)
        self.assertTrue(obj["persons"][0]["face_analysis"]["face_clearly_visible"])

    def test_face_not_visible_unaffected(self):
        from src.runpod.post_normalize import enforce_altcap_null_consistency
        obj = {"persons": [{
            "face_analysis": {
                "face_clearly_visible": False,
                "alternative_captions": None,
            },
        }]}
        self.assertEqual(enforce_altcap_null_consistency(obj), 0)

    def test_body_visible_but_all_alt_caps_null_downgrades_gate(self):
        from src.runpod.post_normalize import enforce_altcap_null_consistency
        obj = {"persons": [{
            "body_analysis": {
                "body_clearly_visible": True,
                "alternative_captions": {
                    "direct": None, "literary": None,
                    "direction": None, "situational": None,
                },
            },
        }]}
        n = enforce_altcap_null_consistency(obj)
        self.assertEqual(n, 1)
        self.assertFalse(obj["persons"][0]["body_analysis"]["body_clearly_visible"])

    def test_no_alt_captions_dict_at_all_treated_as_all_null(self):
        from src.runpod.post_normalize import enforce_altcap_null_consistency
        obj = {"persons": [{
            "face_analysis": {"face_clearly_visible": True},
        }]}
        n = enforce_altcap_null_consistency(obj)
        self.assertEqual(n, 1)
        self.assertFalse(obj["persons"][0]["face_analysis"]["face_clearly_visible"])

    def test_missing_persons_no_crash(self):
        from src.runpod.post_normalize import enforce_altcap_null_consistency
        self.assertEqual(enforce_altcap_null_consistency({}), 0)


class TestFixAllAlias(unittest.TestCase):
    """fix_all is alias of post_fix_compliance for forward compatibility."""

    def test_fix_all_is_post_fix_compliance(self):
        from src.runpod.post_normalize import fix_all, post_fix_compliance
        self.assertIs(fix_all, post_fix_compliance)


if __name__ == "__main__":
    unittest.main()
