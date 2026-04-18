"""
src/runpod/pod_runner.py
════════════════════════════════════════════════════════════════
Stage 5 Pod 侧：在 Runpod H100 Pod 里跑。

依赖（由 tools/pod_setup.sh 预装）：
  vllm>=0.6.3, torch>=2.4, decord, pillow, pydantic>=2, pyyaml, rich

流程：
  1. 读 runpod.yaml
  2. 读 manifest.jsonl，过滤掉 checkpoint 里已完成的 shot_id
  3. 加载 Qwen3-VL-32B-Instruct + vLLM（guided_json = ShotLabel.model_json_schema()）
  4. 每个 shot：按 category 采 4 或 8 帧 → resize 448px shortest → VLM 推理
     → ShotLabel.model_validate 校验 → 写 output/<movie>/<shot_stem>.json
  5. 每完成一个 shot 追加 .checkpoint.jsonl
  6. 信号处理：SIGTERM/SIGINT 下写 checkpoint 后退出（再次启动自动续跑）

用法：
  python src/runpod/pod_runner.py --config runpod.yaml
  python src/runpod/pod_runner.py --config runpod.yaml --max-shots 10
  python src/runpod/pod_runner.py --config runpod.yaml --dry-run
"""

from __future__ import annotations
import argparse, base64, io, json, logging, os, signal, sys, time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.runpod.schemas import (  # noqa
    ManifestEntry, ShotLabel,
    Round1Label, Round2Label,
)

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pod_runner")


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _frame_count_for_category(category: str, cfg_sampling: dict[str, Any]) -> int:
    if category in ("single", "dominant"):
        return int(cfg_sampling.get("frames_single_dominant", 8))
    return int(cfg_sampling.get("frames_multi_wide", 4))


def _pil_to_data_url(img, fmt: str = "JPEG", quality: int = 90) -> str:
    """PIL Image -> data:image/jpeg;base64,... (vLLM chat API 要的 OpenAI 格式)"""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"


def _sample_frames(video_path: str, n_frames: int, resize_shortest: int):
    import decord
    from PIL import Image

    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    if total < 1:
        raise RuntimeError(f"空视频 {video_path}")
    if total < n_frames:
        idxs = list(range(total)) + [total - 1] * (n_frames - total)
    else:
        lo = int(total * 0.05)
        hi = int(total * 0.95)
        if hi - lo < n_frames:
            lo, hi = 0, total - 1
        idxs = [lo + int((hi - lo) * i / max(n_frames - 1, 1)) for i in range(n_frames)]

    frames = vr.get_batch(idxs).asnumpy()     # (n, h, w, 3), RGB
    out = []
    for fr in frames:
        img = Image.fromarray(fr)
        w, h = img.size
        short = min(w, h)
        if short > resize_shortest:
            scale = resize_shortest / short
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        out.append(img)
    return out


def _load_checkpoint(out_root: Path) -> set[str]:
    cp = out_root / ".checkpoint.jsonl"
    if not cp.exists():
        return set()
    done: set[str] = set()
    with open(cp, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                done.add(json.loads(line)["shot_id"])
            except Exception:
                pass
    return done


def _append_checkpoint(out_root: Path, shot_id: str) -> None:
    cp = out_root / ".checkpoint.jsonl"
    with open(cp, "a", encoding="utf-8") as f:
        f.write(json.dumps({"shot_id": shot_id, "completed_at": time.time()}) + "\n")


# SYSTEM_PROMPT 和 user prompt 模板改由 delivery_v1/scripts/build_vlm_prompt.py
# 在 main() 启动时动态构造（含 taxonomy 67 条 leaves + 30 条 forbidden 镜头术语
# + 每个 category 2 条 few-shot 示例）。不再在此硬编码，确保始终贴合官方交付规则。


def _category_for_prompt(shot_category: str) -> str:
    """官方 prompt 只有 single/dominant/multi 三类。wide 用 multi 的 prompt 兜底，
    landscape 在 upload 已被过滤不会到这里。"""
    if shot_category in ("single", "dominant", "multi"):
        return shot_category
    return "multi"


def _fill_user_template(template: str, entry: ManifestEntry) -> str:
    """build_user_prompt() 返回的模板含 few-shot 示例 JSON（有大量 {} 括号），
    用 str.format() 会炸。直接 str.replace() 替换三个占位符最安全。"""
    return (template
            .replace("{shot_id}",      str(entry.shot_id))
            .replace("{source_movie}", str(entry.source_movie))
            .replace("{num_people}",   str(entry.num_people)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5 pod-side runner")
    ap.add_argument("--config",    required=True)
    ap.add_argument("--max-shots", type=int, default=None, help="只跑前 N 个 shot（调试用）")
    ap.add_argument("--dry-run",   action="store_true", help="只加载模型、不跑推理，用于冒烟测试")
    args = ap.parse_args()

    cfg       = _load_config(args.config)
    workspace = Path(args.config).resolve().parent
    paths     = cfg["paths"]
    model_cfg = cfg["model"]
    samp      = cfg.get("sampling", {}) or {}

    manifest_path = workspace / "manifest.jsonl"
    clips_root    = workspace / "clips"
    out_root      = workspace / "output"
    out_root.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out_root / "pod_runner.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    log.addHandler(fh)

    log.info(f"workspace={workspace}")
    log.info(f"manifest={manifest_path}")
    log.info(f"output={out_root}")
    log.info(f"model={model_cfg.get('name')}  precision={model_cfg.get('precision')}")

    if not manifest_path.exists():
        log.error(f"manifest 不存在: {manifest_path}")
        return 1

    entries: list[ManifestEntry] = []
    with open(manifest_path, encoding="utf-8") as f:
        for ln_no, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                entries.append(ManifestEntry.model_validate(json.loads(line)))
            except Exception as e:
                log.warning(f"manifest:{ln_no} 校验失败，跳过 ({e})")
    log.info(f"manifest 总条目: {len(entries)}")

    done = _load_checkpoint(out_root)
    log.info(f"checkpoint 已完成: {len(done)}")

    todo = [e for e in entries if e.shot_id not in done]
    if args.max_shots:
        todo = todo[:args.max_shots]
    log.info(f"本次要跑: {len(todo)}")

    if not todo:
        log.info("没有待跑任务，退出。")
        return 0

    stop = {"flag": False}
    def _term(sig, _frame):
        log.warning(f"收到信号 {sig}，标记停止...")
        stop["flag"] = True
    signal.signal(signal.SIGINT, _term)
    signal.signal(signal.SIGTERM, _term)

    if args.dry_run:
        log.info("[DRY RUN] 跳过模型加载。")
        return 0

    # ── 集成 external_delivery_v1 ─────────────────────────────────
    # 依赖 upload.py 把 docs/labelingStandards/external_delivery_v1/ rsync 到
    # <workspace>/delivery_v1/。scripts/*.py 动态 import；docs/*.yaml 按路径加载。
    delivery_root = workspace / "delivery_v1"
    tax_path = delivery_root / "docs" / "motion_taxonomy.yaml"
    syn_path = delivery_root / "docs" / "motion_synonyms.yaml"
    exa_dir  = delivery_root / "docs" / "vlm_prompts" / "examples"
    scripts_dir = delivery_root / "scripts"

    if not (tax_path.exists() and syn_path.exists() and scripts_dir.exists()):
        log.error(
            f"delivery_v1 bundle 不完整: "
            f"{delivery_root}/{{docs,scripts}}。请在本地重新跑 01_push.sh。")
        return 4

    sys.path.insert(0, str(scripts_dir))
    try:
        from build_vlm_prompt import (
            load_taxonomy_leaves, load_forbidden_terms,
            load_examples, build_system_prompt, build_user_prompt,
        )
        from normalize_tags import TagNormalizer
        from validate_body_analysis import (
            ShotValidator, TaxonomyLoader, SynonymLoader,
            FRAMING_MAX_PARTS,
        )
    except ImportError as e:
        log.error(f"delivery_v1 scripts import 失败: {e}")
        return 4

    taxonomy_leaves = load_taxonomy_leaves(tax_path)
    forbidden_terms = load_forbidden_terms(tax_path)
    n_leaves_total = sum(len(v) for v in taxonomy_leaves.values())
    log.info(f"Loaded taxonomy: {len(taxonomy_leaves)} categories, "
             f"{n_leaves_total} leaves; {len(forbidden_terms)} forbidden terms")

    system_prompt = build_system_prompt(taxonomy_leaves, forbidden_terms)

    user_templates: dict[str, str] = {}
    for cat in ("single", "dominant", "multi"):
        examples = load_examples(exa_dir, cat)
        user_templates[cat] = build_user_prompt(cat, examples)
        log.info(f"Loaded few-shot: {cat}={len(examples)}")

    normalizer = TagNormalizer(syn_path, tax_path)
    validator  = ShotValidator(TaxonomyLoader(tax_path), SynonymLoader(syn_path))
    log.info("delivery_v1 集成就绪（system_prompt + normalize + validate）")

    # vLLM 版本间 GuidedDecodingParams 的位置几经变动：
    #   - 老版本：vllm.sampling_params.GuidedDecodingParams
    #   - 较新：  vllm.GuidedDecodingParams
    #   - 更新：  vllm.model_executor.guided_decoding 里或直接废弃，
    #             改成 SamplingParams(guided_decoding=dict(...))
    # 多路 import 回退；都找不到就让 vLLM 自己处理 dict 形式。
    try:
        from vllm import LLM, SamplingParams
    except ImportError as e:
        log.error(f"vLLM 未安装：{e}。请先跑 tools/pod_setup.sh")
        return 2

    GuidedDecodingParams = None
    for _modname in ("vllm.sampling_params", "vllm",
                     "vllm.model_executor.guided_decoding"):
        try:
            _mod = __import__(_modname, fromlist=["GuidedDecodingParams"])
            GuidedDecodingParams = getattr(_mod, "GuidedDecodingParams", None)
            if GuidedDecodingParams is not None:
                log.info(f"GuidedDecodingParams 来自 {_modname}")
                break
        except Exception:
            pass
    if GuidedDecodingParams is None:
        log.info("GuidedDecodingParams 类未找到，改用 dict 形式 guided_decoding=dict(json=schema)")

    model_name = model_cfg["name"]
    precision  = model_cfg.get("precision", "bf16")
    dtype      = {"bf16": "bfloat16", "fp8": "auto"}.get(precision, "auto")

    # 优先用本地 pre-download 的模型路径，避免 vLLM 去 HF hub 再下一份 67GB
    # 顺序很重要：容器盘本地 NVMe（~3-5 GB/s 读） > Network Volume MooseFS（~50 MB/s）
    # 同样 67GB 模型：本地 30s 加载完 vs Network Volume 冷读 22 min
    model_path = model_cfg.get("path")
    if not model_path:
        slug = model_name.split("/")[-1].lower()  # Qwen3-VL-32B-Instruct → qwen3-vl-32b-instruct
        for candidate in (
            # ── 本地容器盘优先 ─────────────────────────────
            "/root/qwen3-vl-32b",
            "/root/models/qwen3-vl-32b",
            f"/root/models/{slug}",
            # ── Network Volume 兜底 ───────────────────────
            "/workspace/models/qwen3-vl-32b",
            f"/workspace/models/{slug}",
            f"/workspace/models/{model_name.split('/')[-1]}",
        ):
            if (Path(candidate) / "config.json").exists():
                model_path = candidate
                break

    model_load_src = model_path or model_name
    # Qwen3-VL-32B 默认 max_seq_len=262144（262K token），vLLM 要为此预留 KV cache
    # ≈ 64GB；加上 67GB 模型权重远超 80GB H100。
    # 我们单 shot 实际用量：system~800 + user~1600 + 8 frames×1500 + output 2048 ≈ 16K。
    # 设 max_model_len=16384 让 KV cache 降到可管理的大小。
    max_model_len = int(model_cfg.get("max_model_len", 16384))
    gpu_mem_util  = float(model_cfg.get("gpu_memory_utilization", 0.9))

    log.info(f"加载模型 {model_load_src} (dtype={dtype}, max_model_len={max_model_len}, "
             f"gpu_util={gpu_mem_util}) ...")
    if model_path:
        log.info(f"  使用本地路径，跳过 HF hub 下载")
    else:
        log.warning(f"  未找到本地 pre-download，vLLM 会从 HF hub 下载到 {os.environ.get('HF_HOME')}")
    t0 = time.time()
    llm = LLM(model=model_load_src, dtype=dtype,
              trust_remote_code=True,
              max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem_util,
              limit_mm_per_prompt={"image": 16})
    log.info(f"模型加载完成 ({time.time()-t0:.1f}s)")

    # 两轮推理的两份 schema（docs/problem/01_stage5_output_truncation.md 方案 E）
    schema_r1 = Round1Label.model_json_schema()
    schema_r2 = Round2Label.model_json_schema()

    def _build_sampling(schema: dict, max_tokens: int, round_label: str) -> SamplingParams:
        """结构化输出 API 按新→旧依次尝试；全失败降级无约束。"""
        base_kwargs = dict(
            temperature       = float(samp.get("temperature", 0.2)),
            top_p             = float(samp.get("top_p", 0.9)),
            max_tokens        = max_tokens,
            repetition_penalty= float(samp.get("repetition_penalty", 1.05)),
        )
        attempts_: list[tuple[str, dict]] = []
        if GuidedDecodingParams is not None:
            attempts_.append(
                (f"{round_label}: guided_decoding=GuidedDecodingParams(json=...)",
                 dict(**base_kwargs,
                      guided_decoding=GuidedDecodingParams(json=schema))))
        # 探测 vLLM 0.19 的新类
        SOP_ = None
        for _mp in ("vllm.sampling_params",
                    "vllm.structured_output",
                    "vllm.v1.structured_output",
                    "vllm.v1.structured_output.params"):
            try:
                _mod = __import__(_mp, fromlist=["StructuredOutputsParams"])
                SOP_ = getattr(_mod, "StructuredOutputsParams", None)
                if SOP_ is not None:
                    break
            except ImportError:
                continue
        if SOP_ is not None:
            try:
                attempts_.append(
                    (f"{round_label}: structured_outputs=StructuredOutputsParams(json=...)",
                     dict(**base_kwargs, structured_outputs=SOP_(json=schema))))
            except TypeError:
                pass
        for lbl, kw in attempts_:
            try:
                sp = SamplingParams(**kw)
                log.info(f"结构化输出启用：{lbl}")
                return sp
            except TypeError as ex:
                log.debug(f"结构化输出尝试失败 [{lbl}]: {ex}")
        log.warning(f"{round_label}: 未找到可用结构化输出 API，降级无约束。"
                    f" max_tokens={max_tokens} 给足余量。")
        return SamplingParams(**base_kwargs)

    # 两轮各自的 output 预算：
    #   - Round 1 （body + scene + meta，无 face）：实测 2 人 ~3500-4500 token，给 6144
    #   - Round 2 （face_analysis only，按 person 对齐）：2 人 ~5000 token，给 6144
    max_tokens_r1 = int(samp.get("max_tokens_round1", 6144))
    max_tokens_r2 = int(samp.get("max_tokens_round2", 6144))
    sampling_r1 = _build_sampling(schema_r1, max_tokens_r1, "Round1-body")
    sampling_r2 = _build_sampling(schema_r2, max_tokens_r2, "Round2-face")
    resize_shortest = int(samp.get("resize_shortest", 448))

    ok = bad = 0
    for e in todo:
        if stop["flag"]:
            log.info("收到停止信号，退出主循环。")
            break

        rel = e.path
        if rel.startswith("clips/"):
            video = clips_root / rel[len("clips/"):]
        else:
            video = clips_root / rel
        if not video.exists():
            log.warning(f"[skip] 本地视频不存在: {video}")
            continue

        n_frames = _frame_count_for_category(e.shot_category, samp)
        try:
            frames = _sample_frames(str(video), n_frames, resize_shortest)
        except Exception as ex:
            log.warning(f"[skip] 采帧失败 {e.shot_id}: {ex}")
            continue

        # 构造 prompt：system 通用 + user 按 category 取模板再填 shot 信息
        cat_key = _category_for_prompt(e.shot_category)
        user_text = _fill_user_template(user_templates[cat_key], e)

        image_parts = [{"type": "image_url",
                        "image_url": {"url": _pil_to_data_url(im)}}
                       for im in frames]

        # ── Round 1：body + scene + interaction + quality + usability ──
        # system_prompt 保持 delivery_v1 原版；user_text 末尾追加硬指令：本轮只出 body
        r1_hint = (
            "\n\nCRITICAL ROUND 1 INSTRUCTION: For this round, DO NOT output "
            "face_analysis for any person. The output schema for this round "
            "excludes face_analysis entirely — focus on body_analysis, "
            "shot_context, interaction, quality_flags, usability_score. "
            "Face analysis will be done in a separate focused round.\n\n"
            "TOKEN BUDGET IS STRICT. Output COMPACT JSON on minimal lines: "
            "NO pretty-printing, NO 4-space indentation, NO unnecessary "
            "whitespace. Keep every string description CONCISE — "
            "motion_caption under 80 words, each alternative_caption under "
            "40 words. If in doubt, write less, not more."
        )
        messages_r1 = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": [
                *image_parts,
                {"type": "text", "text": user_text + r1_hint},
            ]},
        ]

        # 失败保存助手（提前定义，下面要用）
        def _save_failed(slug: str, raw_text: str, reason: str,
                         errors: list | None = None,
                         warnings: list | None = None) -> None:
            fail_dir = out_root / "_failed"
            fail_dir.mkdir(exist_ok=True)
            (fail_dir / f"{slug}.raw.txt").write_text(raw_text, encoding="utf-8")
            (fail_dir / f"{slug}.errors.json").write_text(
                json.dumps({"reason": reason,
                            "errors":   errors or [],
                            "warnings": warnings or []},
                           ensure_ascii=False, indent=2),
                encoding="utf-8")

        slug = e.shot_id.replace("/", "__")

        try:
            t1 = time.time()
            res_r1 = llm.chat(messages_r1, sampling_r1, use_tqdm=False)
            raw_r1 = res_r1[0].outputs[0].text
            infer_ms_r1 = int((time.time() - t1) * 1000)
        except Exception as ex:
            log.error(f"[ERR] Round1 推理失败 {e.shot_id}: {ex}")
            bad += 1
            continue

        # 1a) Round 1 JSON 解析
        try:
            obj = json.loads(raw_r1)
        except Exception as ex:
            log.error(f"[ERR parse-r1] {e.shot_id}: {ex}")
            _save_failed(slug, raw_r1, f"json_parse_round1: {ex}")
            bad += 1
            continue

        # ── Round 2：face_analysis only，对 Round 1 识别到的每个 person ──
        persons_r1 = obj.get("persons", []) or []
        n_persons = len(persons_r1)
        if n_persons == 0:
            log.info(f"[skip-face] {e.shot_id} Round1 未识别到 person，跳过 Round2")
            infer_ms_r2 = 0
        else:
            # 构造 Round 2 prompt：告诉模型 person 数量和 index，只出 face
            pos_list = ", ".join(
                f"index {p.get('person_index', i)}="
                f"{p.get('spatial_position', 'unknown')}"
                for i, p in enumerate(persons_r1)
            )
            face_user_text = (
                f"You previously identified {n_persons} person(s) in this "
                f"video shot, at positions: {pos_list}.\n\n"
                "Now provide detailed face_analysis for EACH person, indexed "
                "by person_index exactly matching the previous round. "
                "Follow the delivery_v1 face_analysis spec (primary_emotion, "
                "valence/arousal/intensity, expression_caption, "
                "alternative_captions, facial_components, facial_attributes, "
                "temporal_change, observable_blendshape_hints, "
                "expression_confidence, etc.). Output ONLY the JSON "
                '{"persons": [{"person_index": i, "face_analysis": {...}}, ...]}'
                "\n\nTOKEN BUDGET IS STRICT. Output COMPACT JSON on minimal "
                "lines: NO pretty-printing, NO 4-space indentation, NO "
                "unnecessary whitespace. Keep every string description "
                "CONCISE — expression_caption under 60 words, each "
                "alternative_caption under 40 words. If in doubt, write less."
            )
            messages_r2 = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": [
                    *image_parts,
                    {"type": "text", "text": face_user_text},
                ]},
            ]
            try:
                t2 = time.time()
                res_r2 = llm.chat(messages_r2, sampling_r2, use_tqdm=False)
                raw_r2 = res_r2[0].outputs[0].text
                infer_ms_r2 = int((time.time() - t2) * 1000)
            except Exception as ex:
                log.warning(f"[warn] Round2 face 推理失败 {e.shot_id}: {ex}"
                            f"（保留 Round1 结果，face_analysis=null）")
                raw_r2 = ""
                infer_ms_r2 = 0

            # 1b) Round 2 JSON 解析 + 合并
            if raw_r2:
                try:
                    face_obj = json.loads(raw_r2)
                    face_by_idx = {
                        fp.get("person_index"): fp.get("face_analysis")
                        for fp in face_obj.get("persons", [])
                        if isinstance(fp, dict)
                    }
                    for i, p in enumerate(persons_r1):
                        idx = p.get("person_index", i)
                        fa = face_by_idx.get(idx)
                        if fa is not None:
                            p["face_analysis"] = fa
                except Exception as ex:
                    log.warning(f"[warn] Round2 parse 失败 {e.shot_id}: {ex}"
                                f"（face_analysis 留空）")
                    _save_failed(slug + ".round2", raw_r2,
                                 f"json_parse_round2: {ex}")

        infer_ms = infer_ms_r1 + infer_ms_r2

        # 2) normalize_tags（动词归一、同义词、forbidden 镜头术语、intensity/tone 归轴）
        try:
            normalized = normalizer.normalize_shot(obj)
            obj = normalized[0] if isinstance(normalized, tuple) else normalized
        except Exception as ex:
            log.warning(f"[warn normalize] {e.shot_id}: {ex}（继续用原始 obj）")

        # 2.5) clamp visible_body_parts 到 shot_frame_of_body 白名单
        # delivery_v1 validator 的硬规则：bust 只能 head/neck/shoulders，
        # 模型经常自作主张多列 arms/hands/torso 导致 validate 挂掉。
        # 这里直接按白名单 filter，不合法的部件删除，模型意图保留。
        try:
            for _p in obj.get("persons", []):
                _ba = _p.get("body_analysis") or {}
                _frame = _ba.get("shot_frame_of_body")
                _parts = _ba.get("visible_body_parts")
                if (_frame in FRAMING_MAX_PARTS
                        and isinstance(_parts, list)):
                    _allowed = FRAMING_MAX_PARTS[_frame]
                    _ba["visible_body_parts"] = [
                        p for p in _parts if p in _allowed
                    ]
        except Exception as ex:
            log.warning(f"[warn clamp body-parts] {e.shot_id}: {ex}（继续）")

        # 3) 注入 meta
        obj.setdefault("meta", {})
        obj["meta"].setdefault("vlm_model",   model_name)
        obj["meta"].setdefault("vlm_version", "2026-04")
        obj["meta"]["frames_used"]   = n_frames
        obj["meta"]["infer_time_ms"] = infer_ms

        # 4) Pydantic 结构校验 —— 已关闭。
        #    理由：权威规范在 docs/labelingStandards/external_delivery_v1，
        #    model 按 delivery_v1 system_prompt 输出，和 schemas.py:ShotLabel
        #    已不同步（后者是旧 json_schema_integrated.md 的镜像）。
        #    业务合规改由第 5 步 ShotValidator 把关。
        #    如果需要重新启用，见 src/runpod/schemas.py 的过期提醒。

        # 5) 业务校验（14 项 ShotValidator）— errors=0 才算合格
        try:
            errors, warnings, _infos = validator.validate(obj)
        except Exception as ex:
            log.error(f"[ERR validator] {e.shot_id}: {ex}")
            _save_failed(slug, raw, f"validator_crash: {ex}")
            bad += 1
            continue

        if errors:
            log.error(f"[ERR validate] {e.shot_id}: {len(errors)} errors "
                      f"({len(warnings)} warnings)")
            _save_failed(slug, raw, "business_validation_failed",
                         errors=errors, warnings=warnings)
            bad += 1
            continue

        if warnings:
            log.info(f"[warn] {e.shot_id}: {len(warnings)} warnings")

        # 6) 合格，落盘 + checkpoint
        out_file = out_root / e.source_movie / f"{Path(e.shot_id).name}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
        _append_checkpoint(out_root, e.shot_id)
        ok += 1
        if ok % 10 == 0:
            log.info(f"进度 ok={ok} bad={bad}  最新 {e.shot_id} ({infer_ms}ms)")

    log.info(f"\n✅ 完成 ok={ok}  ❌ 失败={bad}")
    return 0 if bad == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
