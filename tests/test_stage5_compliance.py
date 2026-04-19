"""Stage 5 delivery_v1 合规 + 模型切换单元测试。

覆盖：
  - pod_runner._build_llm_kwargs 的 5 条路径（bf16 / fp8 / awq / awq-missing-quant / tp>1）
  - ShotValidator 新增 CHECK_15（solo → contact=none）和 CHECK_16（跨人对称）
  - TagNormalizer.normalize_shot 的 camera_terms_forbidden 扩散行为
  - schemas.ShotLabel 对 delivery_v1 9 个官方示例 JSON 全部通过

运行：
  pytest tests/test_stage5_compliance.py
  或  python tests/test_stage5_compliance.py  （无 pytest 时的回退入口）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (
    str(_ROOT / "src"),
    str(_ROOT / "src" / "runpod"),
    str(_ROOT / "src" / "common"),
    str(_ROOT / "docs" / "labelingStandards" / "external_delivery_v1" / "scripts"),
):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── _build_llm_kwargs ─────────────────────────────────────────────

def test_llm_kwargs_bf16():
    import pod_runner
    k = pod_runner._build_llm_kwargs({
        "precision": "bf16",
        "max_model_len": 16384,
        "gpu_memory_utilization": 0.9,
    })
    assert k["dtype"] == "bfloat16"
    assert "quantization" not in k
    assert "tensor_parallel_size" not in k
    assert k["limit_mm_per_prompt"] == {"image": 16}


def test_llm_kwargs_fp8():
    import pod_runner
    k = pod_runner._build_llm_kwargs({"precision": "fp8"})
    assert k["dtype"] == "auto"


def test_llm_kwargs_awq_with_quantization():
    import pod_runner
    k = pod_runner._build_llm_kwargs({
        "precision": "awq",
        "quantization": "awq_marlin",
        "limit_mm_per_prompt": {"image": 8},
    })
    assert "dtype" not in k
    assert k["quantization"] == "awq_marlin"
    assert k["limit_mm_per_prompt"] == {"image": 8}


def test_llm_kwargs_awq_missing_quantization_raises():
    import pod_runner
    try:
        pod_runner._build_llm_kwargs({"precision": "awq"})
    except ValueError:
        return
    raise AssertionError("awq without quantization should raise ValueError")


def test_llm_kwargs_tp_passes_through():
    import pod_runner
    k = pod_runner._build_llm_kwargs({"precision": "bf16", "tensor_parallel_size": 2})
    assert k["tensor_parallel_size"] == 2


# ── Validator CHECK_15 / CHECK_16 ─────────────────────────────────

def _fresh_validator():
    from validate_body_analysis import ShotValidator, TaxonomyLoader, SynonymLoader
    base = _ROOT / "docs" / "labelingStandards" / "external_delivery_v1" / "docs"
    return ShotValidator(
        TaxonomyLoader(base / "motion_taxonomy.yaml"),
        SynonymLoader(base / "motion_synonyms.yaml"),
    )


def test_check15_solo_contact_mismatch_is_error():
    val = _fresh_validator()
    shot = {
        "shot_id": "t_solo_mismatch",
        "persons": [],
        "quality_flags": {},
        "interaction": {"count": "solo", "contact": "sustained", "relation": "parallel"},
    }
    errors, _, _ = val.validate(shot)
    checks = [e["check"] for e in errors]
    assert "interaction_solo_contact_mismatch" in checks


def test_check15_solo_none_is_ok():
    val = _fresh_validator()
    shot = {
        "shot_id": "t_solo_ok",
        "persons": [],
        "quality_flags": {},
        "interaction": {"count": "solo", "contact": "none", "relation": "parallel"},
    }
    errors, _, _ = val.validate(shot)
    checks = [e["check"] for e in errors]
    assert "interaction_solo_contact_mismatch" not in checks


def test_check16_asymmetric_is_warning():
    val = _fresh_validator()
    shot = {
        "shot_id": "t_asym", "quality_flags": {},
        "persons": [
            {"person_index": 0, "body_analysis": {"interaction": {
                "count": "dyadic", "contact": "sustained", "relation": "parallel",
                "interacts_with_person_index": [1]}}},
            {"person_index": 1, "body_analysis": {"interaction": {
                "count": "dyadic", "contact": "sustained", "relation": "parallel",
                "interacts_with_person_index": []}}},
        ],
    }
    _, warnings, _ = val.validate(shot)
    checks = [w["check"] for w in warnings]
    assert "interaction_asymmetric" in checks


def test_check16_missing_peer_is_warning():
    val = _fresh_validator()
    shot = {
        "shot_id": "t_missing", "quality_flags": {},
        "persons": [{"person_index": 0, "body_analysis": {"interaction": {
            "count": "dyadic", "contact": "none", "relation": "parallel",
            "interacts_with_person_index": [99]}}}],
    }
    _, warnings, _ = val.validate(shot)
    checks = [w["check"] for w in warnings]
    assert "interaction_references_missing_person" in checks


# ── Normalizer: camera_terms 扩散 ─────────────────────────────────

def test_normalize_shot_strips_camera_terms_in_captions():
    from normalize_tags import TagNormalizer
    base = _ROOT / "docs" / "labelingStandards" / "external_delivery_v1" / "docs"
    norm = TagNormalizer(base / "motion_synonyms.yaml", base / "motion_taxonomy.yaml")

    shot = {
        "shot_id": "t_cam_strip",
        "persons": [{
            "person_index": 0,
            "face_analysis": {
                "expression_caption":
                    "In a close-up, his eyes widen slightly",
                "alternative_captions": {
                    "direct":      "A close-up of his widened eyes",
                    "literary":    "The close-up held on his face",
                    "direction":   "hold the close-up steady",
                    "situational": "as the camera comes to a close-up",
                },
            },
            "body_analysis": {
                "motion_caption": "He leans forward in a wide shot frame",
            },
        }],
        "shot_context": {
            "shot_motion_summary": "Slow dolly zoom into a medium shot",
        },
    }
    out, changes = norm.normalize_shot(shot)
    assert changes >= 4, f"expected multiple camera-term strips, got {changes}"
    fa = out["persons"][0]["face_analysis"]
    assert "close-up" not in fa["expression_caption"].lower()
    for v in fa["alternative_captions"].values():
        assert "close-up" not in v.lower()
    assert "wide shot" not in out["persons"][0]["body_analysis"]["motion_caption"].lower()


# ── ShotLabel self-test：delivery_v1 9 个官方示例 ─────────────────

def test_shotlabel_validates_all_delivery_examples():
    from schemas import ShotLabel
    root = _ROOT / "docs" / "labelingStandards" / "examples"
    if not root.exists():
        return  # 示例不在本机时跳过
    files = sorted(root.glob("*.json"))
    assert files, "delivery_v1 examples missing"
    for f in files:
        ShotLabel.model_validate(json.loads(f.read_text(encoding="utf-8")))


# ── 无 pytest 时的独立运行入口 ────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_llm_kwargs_bf16,
        test_llm_kwargs_fp8,
        test_llm_kwargs_awq_with_quantization,
        test_llm_kwargs_awq_missing_quantization_raises,
        test_llm_kwargs_tp_passes_through,
        test_check15_solo_contact_mismatch_is_error,
        test_check15_solo_none_is_ok,
        test_check16_asymmetric_is_warning,
        test_check16_missing_peer_is_warning,
        test_normalize_shot_strips_camera_terms_in_captions,
        test_shotlabel_validates_all_delivery_examples,
    ]
    ok = 0
    for t in tests:
        try:
            t()
            print(f"  OK  {t.__name__}")
            ok += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n  {ok} pass / {len(tests) - ok} fail")
    sys.exit(0 if ok == len(tests) else 1)
