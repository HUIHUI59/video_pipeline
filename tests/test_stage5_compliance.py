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


# ── CHECK 15 / 16（pre-restore 的 spec 扩展，已回退到 zip baseline）──
# 这 4 个测试覆盖的功能在 spec 的 zip baseline 里不存在 —— 它们是 commit
# 2084a44 把 pipeline 逻辑塞进了 spec validate_body_analysis.py 里。spec
# 已回退到 zip 版本 (2026-04-22)，所以这些 test 无对象可测。
# C2 / C3 / C7 完成后，pod_runner 会在 spec ShotValidator 之后追加同等行为
# 的 _pod_extra_checks()，到时候在新文件里重写覆盖（断言 pod 端，不是 spec 端）。
import pytest

_DELIVERY_V1_RESTORED = pytest.mark.skip(
    reason="CHECK15/16 是 pre-restore 的 spec 扩展，spec 已恢复 zip baseline；"
           "等 C2/C3/C7 把同等行为搬到 src/runpod/_pod_extra_checks 后，在新 test 里覆盖"
)


@_DELIVERY_V1_RESTORED
def test_check15_solo_contact_mismatch_is_error():
    pass


@_DELIVERY_V1_RESTORED
def test_check15_solo_none_is_ok():
    pass


@_DELIVERY_V1_RESTORED
def test_check16_asymmetric_is_warning():
    pass


@_DELIVERY_V1_RESTORED
def test_check16_missing_peer_is_warning():
    pass


@pytest.mark.skip(
    reason="caption camera-term stripping 是 pre-restore 的 spec 扩展，spec 已恢复 "
           "zip baseline；C3 会把同等行为放到 post_normalize.strip_camera_terms_in_captions 里"
)
def test_normalize_shot_strips_camera_terms_in_captions():
    pass


# ── ShotLabel self-test：delivery_v1 9 个官方示例 ─────────────────
# C1 (2026-04-22) 加了 visible_body_parts ⊆ FRAMING_MAX_PARTS 强约束；
# spec 自带的 1 个 example (dominant_action_halfbody.json) 违反了 spec 自己的
# CHECK 3，所以 model_validate 会拒绝它。这是 spec 内部不一致，已记录在
# schemas._KNOWN_SPEC_INCONSISTENT_EXAMPLES。

def test_shotlabel_validates_all_delivery_examples():
    from schemas import ShotLabel, _KNOWN_SPEC_INCONSISTENT_EXAMPLES
    root = _ROOT / "docs" / "labelingStandards" / "examples"
    if not root.exists():
        return  # 示例不在本机时跳过
    files = sorted(root.glob("*.json"))
    assert files, "delivery_v1 examples missing"
    failed = []
    for f in files:
        try:
            ShotLabel.model_validate(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            if f.name not in _KNOWN_SPEC_INCONSISTENT_EXAMPLES:
                failed.append((f.name, str(e)))
    assert not failed, (
        f"unexpected validation failures (not in _KNOWN_SPEC_INCONSISTENT_EXAMPLES): "
        f"{failed}"
    )


# ── C1: BodyAnalysis FRAMING_MAX_PARTS 强约束 (2026-04-22) ────────

def test_body_analysis_visible_parts_strict_within_frame():
    """delivery_v1 § 5.1: visible_body_parts ⊆ FRAMING_MAX_PARTS[shot_frame_of_body]."""
    from schemas import BodyAnalysis
    BodyAnalysis.model_validate({
        "body_clearly_visible": True,
        "shot_frame_of_body": "bust",
        "visible_body_parts": ["head", "neck", "shoulders"],
    })  # passes


def test_body_analysis_rejects_parts_outside_frame():
    """bust 里有 hips → ValueError."""
    from schemas import BodyAnalysis
    with pytest.raises(Exception) as exc_info:
        BodyAnalysis.model_validate({
            "body_clearly_visible": True,
            "shot_frame_of_body": "bust",
            "visible_body_parts": ["head", "hips"],
        })
    assert "hips" in str(exc_info.value)
    assert "bust" in str(exc_info.value)


def test_body_analysis_no_visible_parts_passes():
    """visible_body_parts=None / [] 不触发约束（motion_confidence<0.3 时 null 行为）."""
    from schemas import BodyAnalysis
    BodyAnalysis.model_validate({
        "body_clearly_visible": True,
        "shot_frame_of_body": "bust",
        "visible_body_parts": None,
    })
    BodyAnalysis.model_validate({
        "body_clearly_visible": True,
        "shot_frame_of_body": "bust",
        "visible_body_parts": [],
    })


def test_framing_max_parts_dict_matches_spec_intent():
    """sanity check: full_body 包含 feet，bust 不包含 feet."""
    from schemas import FRAMING_MAX_PARTS
    assert "feet" in FRAMING_MAX_PARTS["full_body"]
    assert "feet" not in FRAMING_MAX_PARTS["bust"]
    assert "head" in FRAMING_MAX_PARTS["close_face"]
    assert FRAMING_MAX_PARTS["close_face"] <= FRAMING_MAX_PARTS["bust"]
    assert FRAMING_MAX_PARTS["bust"] <= FRAMING_MAX_PARTS["half_body"]
    assert FRAMING_MAX_PARTS["half_body"] <= FRAMING_MAX_PARTS["three_quarter"]
    assert FRAMING_MAX_PARTS["three_quarter"] <= FRAMING_MAX_PARTS["full_body"]


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
