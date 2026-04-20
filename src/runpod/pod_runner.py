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
import argparse, base64, io, json, logging, os, signal, sys, threading, time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.runpod.schemas import (  # noqa
    ManifestEntry, ShotLabel,
    Round2Label, Round3ShotLabel,
    BodyAnalysis,
)
from src.runpod.face_crops import detect_and_crop_faces  # noqa

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pod_runner")


# ── 心跳：长耗时阶段每 N 秒打一行日志，避免 tail 静默焦虑 ──────────
_HB = {"phase": None, "start": 0.0, "detail": ""}
_HB_LOCK = threading.Lock()
_HB_STOP = threading.Event()


def _hb_set(phase: str | None, detail: str = "") -> None:
    """设置当前心跳阶段名。phase=None 停止打印。"""
    with _HB_LOCK:
        _HB["phase"] = phase
        _HB["start"] = time.time() if phase else 0.0
        _HB["detail"] = detail


def _hb_loop(interval: float = 30.0) -> None:
    """守护线程：阶段非空就每 interval 秒打一行 "[heartbeat] phase...Xs"。"""
    while not _HB_STOP.wait(interval):
        with _HB_LOCK:
            phase = _HB["phase"]
            start = _HB["start"]
            detail = _HB["detail"]
        if phase and start > 0:
            elapsed = time.time() - start
            msg = f"[heartbeat] {phase} 进行中... {elapsed:.0f}s"
            if detail:
                msg += f" ({detail})"
            log.info(msg)


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"runpod config 根对象必须是 dict，实际 type={type(cfg).__name__} "
            f"path={path}"
        )
    return cfg


def _require_cfg(cfg: dict[str, Any], *keys: str) -> Any:
    """按路径取配置，缺失任意一级都抛带完整路径的 ValueError。
    比裸 `cfg["paths"]["pod_workspace"]` 崩溃时可读。
    """
    cur: Any = cfg
    path_so_far: list[str] = []
    for k in keys:
        path_so_far.append(k)
        if not isinstance(cur, dict):
            raise ValueError(
                f"runpod config: 期望 {'/'.join(path_so_far[:-1])} 是 dict，"
                f"实际 type={type(cur).__name__}"
            )
        if k not in cur:
            raise ValueError(
                f"runpod config 缺失字段: {'/'.join(path_so_far)}"
            )
        cur = cur[k]
    return cur


def _build_llm_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """根据 runpod.yaml 的 model 段组装 vLLM LLM(...) 的 kwargs。

    行为：
      - precision=bf16/fp8 → dtype=bfloat16/auto；不传 quantization
      - precision=awq/awq_marlin/gptq-int4 → 不传 dtype（vLLM 自动推），
        必须同时配置 model.quantization（例如 "awq_marlin"）
      - tensor_parallel_size>1 → 才传该 kwarg，让 vLLM 默认单卡路径不受影响
      - limit_mm_per_prompt 从 config 取，默认 {"image": 16}
    """
    precision = str(model_cfg.get("precision", "bf16")).lower()
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "max_model_len": int(model_cfg.get("max_model_len", 16384)),
        "gpu_memory_utilization": float(
            model_cfg.get("gpu_memory_utilization", 0.9)),
        "limit_mm_per_prompt":
            dict(model_cfg.get("limit_mm_per_prompt") or {"image": 16}),
    }

    if precision in ("bf16", "fp8"):
        kwargs["dtype"] = {"bf16": "bfloat16", "fp8": "auto"}[precision]
    else:
        # AWQ / GPTQ：必须显式指定 vLLM 的量化 kernel
        qt = model_cfg.get("quantization")
        if not qt:
            raise ValueError(
                f"model.precision={precision} 需要同时配置 "
                f"model.quantization（例如 awq_marlin / gptq_marlin）。"
                f"参考 configs/runpod.122b.yaml.example。"
            )
        kwargs["quantization"] = qt

    # vLLM 默认拒绝 chat message 里的 file:// URL（防 SSRF）；video_mode=true
    # 需要在 LLM 构造时显式白名单一个目录前缀。main() 会在 video_mode=true 时
    # 自动把 str(workspace) 塞进 model_cfg；用户也可以手动指定。
    allowed = model_cfg.get("allowed_local_media_path")
    if allowed:
        kwargs["allowed_local_media_path"] = str(allowed)

    tp = int(model_cfg.get("tensor_parallel_size", 1))
    if tp > 1:
        kwargs["tensor_parallel_size"] = tp
        # TP>1 时 vLLM 需要 CUDA_VISIBLE_DEVICES 至少 tp 张卡；这里只警告不 raise
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        seen = len([x for x in visible.split(",") if x.strip()]) if visible else 0
        if seen and seen < tp:
            log.warning(
                f"tensor_parallel_size={tp} 但 CUDA_VISIBLE_DEVICES 只看到 "
                f"{seen} 张卡；vLLM 很可能会报错。"
            )

    # 省显存开关（用于 H100 80GB 跑 122B 这种极限情况）
    # enforce_eager=True：关闭 CUDA graph 捕获，省 2-3 GB，推理慢 ~10-15%
    # max_num_seqs：限制并发请求数，直接影响 KV 池大小（默认 256 通常过大）
    if bool(model_cfg.get("enforce_eager", False)):
        kwargs["enforce_eager"] = True
    max_num_seqs = model_cfg.get("max_num_seqs")
    if max_num_seqs:
        kwargs["max_num_seqs"] = int(max_num_seqs)
    return kwargs


def _build_video_message(video_path: Path, fps: int,
                          max_pixels: int, min_pixels: int) -> dict[str, Any]:
    """Build vLLM OpenAI-chat API content block for a local video file.

    vLLM 0.19+ 支持在 chat messages 里发 video_url（`file://` 协议 + 绝对路径），
    Qwen3-VL preprocessor 会按 fps 均匀采样、在 embedding 里插时间戳 token
    (<t=0.0s>, <t=0.2s>...)，让模型做真正的时序推理而不是把多帧当同等重要
    的孤立图。max_pixels/min_pixels 控制单帧 patch 数 → 单 shot 总 token。
    """
    return {
        "type": "video_url",
        "video_url": {"url": f"file://{video_path.resolve()}"},
        # 下列 kwargs 由 vLLM 的 Qwen3-VL 多模态处理器识别
        "fps": fps,
        "max_pixels": max_pixels,
        "min_pixels": min_pixels,
    }


def _save_failed(out_root: Path, slug: str, raw_text: str, reason: str,
                 errors: list | None = None,
                 warnings: list | None = None) -> None:
    """把一条失败推理的原始文本 + 错误原因落盘到 output/_failed/。
    模块顶层函数避免在 for 循环内每次迭代重新定义导致的闭包陷阱。
    """
    fail_dir = out_root / "_failed"
    fail_dir.mkdir(exist_ok=True)
    try:
        (fail_dir / f"{slug}.raw.txt").write_text(raw_text, encoding="utf-8")
        (fail_dir / f"{slug}.errors.json").write_text(
            json.dumps({"reason": reason,
                        "errors":   errors or [],
                        "warnings": warnings or []},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
    except Exception as e:
        log.warning(f"_save_failed 写入 {fail_dir}/{slug} 失败: {e}")


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
    ap.add_argument("--dry-run",   action="store_true", help="只校验 config、不加载模型，最快冒烟")
    ap.add_argument("--dry-run-model-load", action="store_true",
                    help="加载模型后立刻退出并打印显存占用（租 Pod 后验证 GPU 容量）")
    args = ap.parse_args()

    try:
        cfg = _load_config(args.config)
    except Exception as e:
        log.error(f"无法加载 config: {e}")
        return 1
    workspace = Path(args.config).resolve().parent
    try:
        paths     = _require_cfg(cfg, "paths")
        model_cfg = _require_cfg(cfg, "model")
    except ValueError as e:
        log.error(str(e))
        return 1
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

    # 启动心跳线程：长耗时阶段每 30s 往 log 打一行，tail 永不静默
    threading.Thread(target=_hb_loop, args=(30.0,),
                     name="pod_runner_heartbeat", daemon=True).start()

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

    try:
        taxonomy_leaves = load_taxonomy_leaves(tax_path)
        forbidden_terms = load_forbidden_terms(tax_path)
    except Exception as e:
        log.error(f"读取 taxonomy 失败（{tax_path}）: {e}")
        return 4
    n_leaves_total = sum(len(v) for v in taxonomy_leaves.values())
    log.info(f"Loaded taxonomy: {len(taxonomy_leaves)} categories, "
             f"{n_leaves_total} leaves; {len(forbidden_terms)} forbidden terms")

    try:
        system_prompt = build_system_prompt(taxonomy_leaves, forbidden_terms)

        user_templates: dict[str, str] = {}
        for cat in ("single", "dominant", "multi"):
            examples = load_examples(exa_dir, cat)
            user_templates[cat] = build_user_prompt(cat, examples)
            log.info(f"Loaded few-shot: {cat}={len(examples)}")
    except Exception as e:
        log.error(f"构建 prompt 模板失败: {e}")
        return 4

    try:
        normalizer = TagNormalizer(syn_path, tax_path)
        validator  = ShotValidator(TaxonomyLoader(tax_path), SynonymLoader(syn_path))
    except FileNotFoundError as e:
        log.error(f"delivery_v1 YAML 文件缺失: {e}")
        return 4
    except Exception as e:
        log.error(f"初始化 normalizer/validator 失败: {e}")
        return 4
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

    # 优先用本地 pre-download 的模型路径，避免 vLLM 去 HF hub 再下一份 67GB
    # 顺序很重要：容器盘本地 NVMe（~3-5 GB/s 读） > Network Volume MooseFS（~50 MB/s）
    # 同样 67GB 模型：本地 30s 加载完 vs Network Volume 冷读 22 min
    model_path = model_cfg.get("path")
    if not model_path:
        slug = model_name.split("/")[-1].lower()   # Qwen/Qwen3-VL-32B-Instruct → qwen3-vl-32b-instruct
        for candidate in (
            # ── 本地容器盘优先 ─────────────────────────────
            f"/root/{slug}",
            f"/root/models/{slug}",
            # ── Network Volume 兜底 ───────────────────────
            f"/workspace/models/{slug}",
            f"/workspace/models/{model_name.split('/')[-1]}",
            # ── 兼容老路径（硬编码 qwen3-vl-32b）──────────
            "/root/qwen3-vl-32b",
            "/root/models/qwen3-vl-32b",
            "/workspace/models/qwen3-vl-32b",
        ):
            if (Path(candidate) / "config.json").exists():
                model_path = candidate
                break

    model_load_src = model_path or model_name

    # video_mode=true 需要 vLLM 白名单本地文件前缀，否则 file:// video_url 报
    # "Cannot load local files without --allowed-local-media-path"。默认用
    # workspace 目录（clips/ 在里面）。用户可在 yaml 显式设 model.allowed_local_media_path
    # 覆盖（例如只允许 clips/ 子目录）。
    if bool(samp.get("video_mode", False)) and not model_cfg.get("allowed_local_media_path"):
        model_cfg["allowed_local_media_path"] = str(workspace)
        log.info(f"video_mode=true → 自动注入 "
                 f"allowed_local_media_path={workspace}")

    # 按 model_cfg 组装 vLLM LLM() kwargs（支持 bf16/fp8/awq/awq_marlin/gptq-int4）
    try:
        llm_kwargs = _build_llm_kwargs(model_cfg)
    except ValueError as ex:
        log.error(str(ex))
        return 1

    log.info(f"加载模型 {model_load_src}")
    log.info(f"  vLLM kwargs: {llm_kwargs}")
    if model_path:
        log.info(f"  使用本地路径，跳过 HF hub 下载")
    else:
        log.warning(
            f"  未找到本地 pre-download，vLLM 会从 HF hub 下载到 "
            f"{os.environ.get('HF_HOME')}"
        )
    log.info("  （vLLM 详细 shard/kernel 日志在 pod_runner.stdout；"
             "log 这边每 30s 打心跳。Qwen3.5 首次加载需要 ~5 min 冷读 NV）")
    t0 = time.time()
    _hb_set("加载模型权重 (vLLM LLM init)", detail="看 stdout 有 shard 进度条")
    llm = LLM(model=model_load_src, **llm_kwargs)
    _hb_set(None)
    log.info(f"模型加载完成 ({time.time()-t0:.1f}s)")

    if args.dry_run_model_load:
        # Phase 3.3 早退：用于租到 Pod 后快速确认显存够不够
        try:
            import torch
            mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            log.info(f"[dry-run-model-load] 加载完成后显存 "
                     f"{mem_gb:.2f} GB（model weights + KV 初始分配）")
        except Exception as ex:
            log.warning(f"[dry-run-model-load] 无法读取 cuda 显存：{ex}")
        return 0

    # 三轮推理的 schema（2026-04-20 per-person 重构版）：
    #   R3：scene/interaction/quality/usability （shot 级元数据）
    #   R2：persons = [{person_index, spatial_position, face_analysis}]（人发现+脸）
    #   R1：每人独立一次调用，输出单个 BodyAnalysis dict（per-person body）
    schema_r1_person = BodyAnalysis.model_json_schema()
    schema_r2        = Round2Label.model_json_schema()
    schema_r3        = Round3ShotLabel.model_json_schema()

    def _build_sampling(schema: dict, max_tokens: int, round_label: str) -> SamplingParams:
        """结构化输出 API 按新→旧依次尝试；全失败降级无约束。
        逃生口：sampling.disable_structured_output=true 时直接无约束（用于
        vLLM 0.19 + MoE 量化 kernel 的 structured outputs 运行时崩溃）。
        """
        base_kwargs = dict(
            temperature       = float(samp.get("temperature", 0.2)),
            top_p             = float(samp.get("top_p", 0.9)),
            max_tokens        = max_tokens,
            repetition_penalty= float(samp.get("repetition_penalty", 1.05)),
        )
        if bool(samp.get("disable_structured_output", False)):
            log.warning(
                f"{round_label}: disable_structured_output=true，跳过 guided "
                f"JSON，改无约束生成 + 事后 json.loads。有 JSON 解析失败的风险。"
            )
            return SamplingParams(**base_kwargs)

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

    # 三轮各自的 output 预算：
    #   - Round 1 （body_analysis per person）：3 人 shot ~5-6K token
    #   - Round 2 （face_analysis only，按 person 对齐）：3 人 ~3-4K token
    #   - Round 3 （shot_context + interaction + quality + usability）：≤2K token
    max_tokens_r1 = int(samp.get("max_tokens_round1", 8192))
    max_tokens_r2 = int(samp.get("max_tokens_round2", 6144))
    max_tokens_r3 = int(samp.get("max_tokens_round3", 4096))
    # per-person 每人 body_analysis 约 3K tok，所以 max_tokens_r1 用户设 10240 也
    # 够余量。R2 现在同时做人发现 + face_analysis，schema 和 prompt 略胖。
    sampling_r1_person = _build_sampling(
        schema_r1_person, max_tokens_r1, "Round1-body-per-person")
    sampling_r2 = _build_sampling(schema_r2, max_tokens_r2, "Round2-persons+face")
    sampling_r3 = _build_sampling(schema_r3, max_tokens_r3, "Round3-scene")
    resize_shortest = int(samp.get("resize_shortest", 448))

    # 视频输入模式：video_mode=true 时直接发 video_url 给 Qwen3-VL，preprocessor
    # 按 sample_fps 自动采样 + 插时间戳。false 时走老路径用 decord 采帧发图片列表。
    video_mode     = bool(samp.get("video_mode", False))
    video_fps      = int(samp.get("video_fps", 5))
    video_max_pix  = int(samp.get("video_max_pixels", 112896))
    video_min_pix  = int(samp.get("video_min_pixels", 50176))
    if video_mode:
        log.info(f"video_mode=true：发 video_url 给 vLLM，fps={video_fps}, "
                 f"max_pixels={video_max_pix}, min_pixels={video_min_pix}")
    else:
        log.info(f"video_mode=false：走老路径（decord 采固定帧数发图片列表）")

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

        # 构造 prompt：system 通用 + user 按 category 取模板再填 shot 信息
        cat_key = _category_for_prompt(e.shot_category)
        user_text = _fill_user_template(user_templates[cat_key], e)

        if video_mode:
            # 原生 video 模式：vLLM preprocessor 按 video_fps 自动采样 + 插时间戳
            # n_frames 只是给 meta.frames_used 记账用；实际帧数 = duration × fps
            n_frames = max(1, int(round((e.duration_sec or 1.0) * video_fps)))
            try:
                video_part = _build_video_message(
                    video, fps=video_fps,
                    max_pixels=video_max_pix, min_pixels=video_min_pix)
            except Exception as ex:
                log.warning(f"[skip] 构建 video_message 失败 {e.shot_id}: {ex}")
                continue
            # 三轮共享 video content（vLLM prefix caching 能省重复 embed）
            content_base = [video_part]
        else:
            # 老路径：自己采帧发图片列表
            n_frames = _frame_count_for_category(e.shot_category, samp)
            try:
                frames = _sample_frames(str(video), n_frames, resize_shortest)
            except Exception as ex:
                log.warning(f"[skip] 采帧失败 {e.shot_id}: {ex}")
                continue
            image_parts = [{"type": "image_url",
                            "image_url": {"url": _pil_to_data_url(im)}}
                           for im in frames]
            content_base = image_parts

        # ═══════════════════════════════════════════════════════════════
        # 2026-04-20 重构：R3 → R2 → R1-per-person
        #   R3：shot 级元数据（最快 + 不依赖人发现）
        #   R2：人发现 + face_analysis（带 face crops）
        #   R1：对每个 person 独立一次 body_analysis 推理（per-person，
        #       避免单轮输出随 N 膨胀被 max_model_len 截断）
        # ═══════════════════════════════════════════════════════════════

        slug = e.shot_id.replace("/", "__")

        # ── Round 3：scene + interaction + quality + usability ──
        r3_user_text = (
            f"Analyze this {e.duration_sec:.1f}s video shot. Output ONLY "
            f"shot-level metadata (no persons[] array):\n"
            "  - shot_context: {shot_type, shot_emotion_summary, "
            "shot_motion_summary, scene_context: {visible_setting, "
            "narrative_situation, narrative_confidence}}\n"
            "  - interaction: {count (solo/dyadic/triadic/crowd), "
            "contact (none/incidental/sustained), relation}\n"
            "  - quality_flags: all 8 fields (face_clearly_visible, "
            "body_clearly_visible, motion_blur, occlusion, lighting, "
            "camera_stable, frame_sampling_ok, vlm_confidence)\n"
            "  - usability_score: {face, motion}, both numbers in [0,1]\n\n"
            "STRICT: do NOT output persons[] or any face_analysis / "
            "body_analysis — those come from Round 2/1. Output COMPACT JSON, "
            "minimal whitespace. shot_emotion_summary and shot_motion_summary "
            "each 1-2 sentences; narrative_situation null if "
            "narrative_confidence<0.3."
        )
        messages_r3 = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": [
                *content_base,
                {"type": "text", "text": r3_user_text},
            ]},
        ]
        try:
            if ok == 0 and bad == 0:
                log.info(
                    f"[info] {e.shot_id} 开始 Round 3 scene/interaction 推理；"
                    f"首个 shot 会触发 CUDA graph 捕获 + kernel JIT，"
                    f"2-4 min；心跳每 30s 打一行"
                )
            else:
                log.info(f"[info] {e.shot_id} 开始 Round 3 scene/interaction 推理")
            t3 = time.time()
            _hb_set(f"{e.shot_id} Round 3 推理", detail="scene+interaction+quality")
            res_r3 = llm.chat(messages_r3, sampling_r3, use_tqdm=False)
            _hb_set(None)
            raw_r3 = res_r3[0].outputs[0].text
            infer_ms_r3 = int((time.time() - t3) * 1000)
            log.info(f"[info] {e.shot_id} Round 3 完成 ({infer_ms_r3}ms, "
                     f"{len(raw_r3)} chars)")
        except Exception as ex:
            _hb_set(None)
            log.error(f"[ERR] Round3 推理失败 {e.shot_id}: {ex}")
            _save_failed(out_root, slug, "", f"round3_inference_crash: {ex}")
            bad += 1
            continue

        try:
            r3_obj = json.loads(raw_r3)
        except Exception as ex:
            log.error(f"[ERR parse-r3] {e.shot_id}: {ex}")
            _save_failed(out_root, slug, raw_r3, f"json_parse_round3: {ex}")
            bad += 1
            continue

        # ── Round 2：人发现 + face_analysis（附 face crops）──
        # 附加 face crops（可选，帮 VLM 看清面部细节）
        face_crop_parts: list[dict] = []
        face_crop_order: list[str] = []
        if bool(samp.get("face_crop_enabled", False)):
            try:
                fc_fps    = int(samp.get("face_crop_sample_fps", 2))
                fc_size   = int(samp.get("face_crop_size", 224))
                fc_max    = int(samp.get("face_crop_per_frame_max", 4))
                fc_back   = str(samp.get("face_detector", "haar"))
                fc_yolo_w = samp.get("yolo_face_weights")
                crops = detect_and_crop_faces(
                    str(video), fps=fc_fps, crop_size=fc_size,
                    max_per_frame=fc_max, backend=fc_back,
                    yolo_weights=fc_yolo_w)
                for ts_sec, slot, crop_img in crops:
                    face_crop_parts.append({
                        "type": "image_url",
                        "image_url": {"url": _pil_to_data_url(crop_img)},
                    })
                    face_crop_order.append(f"t={ts_sec:.1f}s-slot{slot}")
                log.info(f"[info] {e.shot_id} 附加 {len(face_crop_parts)} "
                         f"张 face crop (fps={fc_fps}, size={fc_size})")
            except Exception as ex:
                log.warning(f"[warn] face_crops 生成失败 {e.shot_id}: {ex}")
                face_crop_parts = []
                face_crop_order = []

        r2_user_text = (
            "Identify ALL visible persons in this video shot and output "
            "their face_analysis. For each person, provide:\n"
            "  - person_index: 0-based integer (0 = largest face or "
            "leftmost)\n"
            "  - spatial_position: one of 'center', 'left', 'right', "
            "'background'\n"
            "  - face_analysis: full dict per delivery_v1 face spec\n\n"
            'Output ONLY: {"persons": [{"person_index": i, '
            '"spatial_position": "...", "face_analysis": {...}}, ...]}. '
            "If a person's face is not clearly visible (occluded or <3% "
            "of frame), set face_clearly_visible=false and null the "
            "other face_analysis fields for that person.\n\n"
            "STRICT face_analysis STRUCTURE (do NOT flatten):\n"
            "  - 9-class primary_emotion: anger|sadness|joy|fear|surprise|"
            "disgust|contempt|neutral|complex\n"
            "  - valence/arousal/intensity: NUMBERS in [-1,1] / [0,1] / "
            "[0,1]\n"
            "  - alternative_captions: OBJECT {direct, literary, "
            "direction, situational}, NOT an array\n"
            "  - facial_components: OBJECT with 6 keys (eyes, eyebrows, "
            "mouth, jaw, gaze_direction, head_pose)\n"
            "  - facial_attributes: OBJECT with 8 keys (apparent_gender, "
            "apparent_age_range, glasses, facial_hair, head_covering, "
            "mask, makeup_visible, distinctive_notes)\n"
            "  - observable_blendshape_hints: OBJECT with 15 enum keys, "
            "NOT a flat string\n"
            "  - temporal_change enum: static|building|peak_then_release|"
            "transition|rapid_micro\n"
            "  - expression_confidence: number [0,1]\n\n"
            "COMPACT JSON, no pretty-print. expression_caption <60 words, "
            "each alt_caption <40 words."
        )
        if face_crop_parts:
            r2_user_text += (
                "\n\n=== FACE CROPS BELOW ===\n"
                f"The next {len(face_crop_parts)} images are face "
                f"close-ups (224×224 each) auto-detected from the same "
                "video, ordered by (timestamp, slot). Use them for "
                "detailed facial_components and observable_blendshape_hints. "
                "Crop order: " + ", ".join(face_crop_order)
            )

        messages_r2 = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": [
                *content_base,
                *face_crop_parts,
                {"type": "text", "text": r2_user_text},
            ]},
        ]
        try:
            log.info(f"[info] {e.shot_id} 开始 Round 2 推理（人发现 + face）")
            t2 = time.time()
            _hb_set(f"{e.shot_id} Round 2 推理")
            res_r2 = llm.chat(messages_r2, sampling_r2, use_tqdm=False)
            _hb_set(None)
            raw_r2 = res_r2[0].outputs[0].text
            infer_ms_r2 = int((time.time() - t2) * 1000)
            log.info(f"[info] {e.shot_id} Round 2 完成 ({infer_ms_r2}ms, "
                     f"{len(raw_r2)} chars)")
        except Exception as ex:
            _hb_set(None)
            log.error(f"[ERR] Round2 推理失败 {e.shot_id}: {ex}")
            _save_failed(out_root, slug, "", f"round2_inference_crash: {ex}")
            bad += 1
            continue

        try:
            r2_obj = json.loads(raw_r2)
        except Exception as ex:
            log.error(f"[ERR parse-r2] {e.shot_id}: {ex}")
            _save_failed(out_root, slug, raw_r2, f"json_parse_round2: {ex}")
            bad += 1
            continue

        r2_persons = r2_obj.get("persons", []) or []
        n_persons = len(r2_persons)
        log.info(f"[info] {e.shot_id} R2 发现 {n_persons} 个 person")

        # ── Round 1 per-person：每个 person 独立一次 body_analysis 推理 ──
        # 关键：per-person 调用 → 每次只出 1 人 body (~3K tok)，不会被
        # max_model_len 截断。vLLM prefix caching 让 video embed 复用。
        body_by_index: dict[int, dict | None] = {}
        infer_ms_r1 = 0

        for r2_p in r2_persons:
            if not isinstance(r2_p, dict):
                continue
            pidx = r2_p.get("person_index")
            if not isinstance(pidx, int):
                log.warning(f"[warn] {e.shot_id}: R2 person 无有效 person_index")
                continue
            spatial = r2_p.get("spatial_position", "unknown")
            r1_person_text = (
                f"This video shot contains {n_persons} person(s). Focus "
                f"ONLY on person_index={pidx} at spatial_position={spatial}. "
                f"Output the body_analysis dict for THIS ONE PERSON (do NOT "
                f"output a wrapping persons[] array or other people's data). "
                f"Follow delivery_v1 body_analysis spec:\n"
                "  - body_clearly_visible: bool\n"
                "  - shot_frame_of_body: close_face|bust|half_body|"
                "three_quarter|full_body|wide\n"
                "  - visible_body_parts: list[str]\n"
                "  - motion_caption: 50-180 words (BODY only, no face)\n"
                "  - alternative_captions: OBJECT with 4 keys (direct, "
                "literary, direction, situational), NOT a list\n"
                "  - action_primary: taxonomy leaf (walking, sitting, etc.)\n"
                "  - action_quality: OBJECT {intensity (low/mid/high), "
                "tone, tempo}, NOT a string\n"
                "  - body_focus: enum (upper_body, hands, etc.)\n"
                "  - kinematics_hint: OBJECT {trajectory, periodicity "
                "(periodic/non_periodic), symmetry "
                "(bilateral_symmetric/bilateral_asymmetric/axial), "
                "duration_class}, NOT a single word\n"
                "  - upper_body_detail: OBJECT with 7 string fields "
                "(head, neck, shoulders, arms, hands, torso, posture)\n"
                "  - gesture_detail: specific string (left/right hand, "
                "direction), not 'none'/'n/a'\n"
                "  - hands_visible: bool\n"
                "  - interaction: OBJECT {count, contact, relation, "
                "interacts_with_person_index: list[int]}\n"
                "  - motion_confidence: number [0,1]\n\n"
                "Output ONLY the body_analysis dict, no wrapping. COMPACT "
                "JSON. motion_caption <80 words, each alt_caption <40 words."
            )
            messages_r1 = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": [
                    *content_base,
                    {"type": "text", "text": r1_person_text},
                ]},
            ]
            try:
                log.info(f"[info] {e.shot_id} 开始 R1 推理 person_index={pidx}")
                t1 = time.time()
                _hb_set(f"{e.shot_id} R1 person={pidx}",
                        detail=f"spatial={spatial}")
                res_r1 = llm.chat(messages_r1, sampling_r1_person,
                                  use_tqdm=False)
                _hb_set(None)
                raw_r1 = res_r1[0].outputs[0].text
                ms = int((time.time() - t1) * 1000)
                infer_ms_r1 += ms
                log.info(f"[info] {e.shot_id} R1 person={pidx} 完成 "
                         f"({ms}ms, {len(raw_r1)} chars)")
            except Exception as ex:
                _hb_set(None)
                log.warning(f"[warn] R1 person={pidx} 推理失败 {e.shot_id}: "
                            f"{ex}（body_analysis=null）")
                body_by_index[pidx] = None
                continue

            try:
                body_by_index[pidx] = json.loads(raw_r1)
            except Exception as ex:
                log.warning(f"[warn] R1 person={pidx} parse 失败 "
                            f"{e.shot_id}: {ex}")
                _save_failed(out_root, f"{slug}.r1.p{pidx}", raw_r1,
                             f"json_parse_round1_person_{pidx}: {ex}")
                body_by_index[pidx] = None

        # ── 合并 R3 + R2 + R1-per-person 到 obj ──
        obj: dict[str, Any] = {
            "shot_id":         e.shot_id,
            "source_movie":    e.source_movie,
            "shot_context":    r3_obj.get("shot_context"),
            "interaction":     r3_obj.get("interaction"),
            "quality_flags":   r3_obj.get("quality_flags"),
            "usability_score": r3_obj.get("usability_score"),
            "persons":         [],
        }
        if "exclusion_reason" in r3_obj:
            obj["exclusion_reason"] = r3_obj["exclusion_reason"]

        for r2_p in r2_persons:
            if not isinstance(r2_p, dict):
                continue
            pidx = r2_p.get("person_index")
            obj["persons"].append({
                "person_index":     pidx,
                "spatial_position": r2_p.get("spatial_position"),
                "face_analysis":    r2_p.get("face_analysis"),
                "body_analysis":    body_by_index.get(pidx),
            })

        if n_persons == 0:
            log.info(f"[info] {e.shot_id} R2 未识别到 person，persons[]=空")

        infer_ms = infer_ms_r1 + infer_ms_r2 + infer_ms_r3

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

        # 3) 注入 meta —— 规范要求 vlm_model/vlm_version/frames_used/infer_time_ms；
        #    其余是推理复现参数，Meta 类 extra="allow" 容许，便于下游消融分析。
        obj.setdefault("meta", {})
        obj["meta"].setdefault("vlm_model",   model_name)
        obj["meta"].setdefault("vlm_version", "2026-04")
        obj["meta"]["frames_used"]   = n_frames
        obj["meta"]["infer_time_ms"] = infer_ms
        # 推理复现参数（采样 + 模型加载）
        obj["meta"]["temperature"]         = float(samp.get("temperature", 0.2))
        obj["meta"]["top_p"]               = float(samp.get("top_p", 0.9))
        obj["meta"]["repetition_penalty"]  = float(samp.get("repetition_penalty", 1.05))
        obj["meta"]["max_tokens_round1"]   = int(samp.get("max_tokens_round1", max_tokens_r1))
        obj["meta"]["max_tokens_round2"]   = int(samp.get("max_tokens_round2", max_tokens_r2))
        obj["meta"]["max_tokens_round3"]   = int(samp.get("max_tokens_round3", max_tokens_r3))
        obj["meta"]["precision"]           = model_cfg.get("precision", "bf16")
        obj["meta"]["quantization"]        = model_cfg.get("quantization")
        obj["meta"]["tensor_parallel_size"] = int(model_cfg.get("tensor_parallel_size", 1))
        obj["meta"]["max_model_len"]       = int(model_cfg.get("max_model_len", 16384))
        obj["meta"]["video_mode"]          = video_mode
        if video_mode:
            obj["meta"]["video_fps"]       = video_fps

        # 4) Pydantic 结构校验：ShotLabel 强制 root schema 完整性 + enum 合法。
        #    schemas.py 用 _Base(extra="allow") 宽容未来字段演进，但对 enum /
        #    必需 gate 字段依然严格；这层是 delivery_v1 合规的结构防线，业务
        #    语义由第 5 步 ShotValidator 把关，两者互补。
        #
        #    逃生口：vLLM 0.19 + MoE 量化 kernel (moe_wna16 等) 上 structured
        #    outputs 无法真正约束 JSON schema，VLM 会自由发挥输出格式（list vs
        #    dict、string vs nested）。此时开 sampling.disable_schema_validation
        #    跳过这层，让下游 normalize/validator 尽力消化；合规性由
        #    ShotValidator 的 16 条业务规则兜底。
        if not bool(samp.get("disable_schema_validation", False)):
            try:
                ShotLabel.model_validate(obj)
            except Exception as ex:
                log.error(f"[ERR schema] {e.shot_id}: {ex}")
                _save_failed(out_root, slug, raw_r1,
                             f"schema_validation_failed: {ex}")
                bad += 1
                continue
        else:
            log.info(f"[info] {e.shot_id}: 跳过 Pydantic 结构校验 "
                     f"(disable_schema_validation=true)")

        # 5) 业务校验（16 项 ShotValidator）— errors=0 才算合格
        try:
            errors, warnings, _infos = validator.validate(obj)
        except Exception as ex:
            log.error(f"[ERR validator] {e.shot_id}: {ex}")
            _save_failed(out_root, slug, raw_r1, f"validator_crash: {ex}")
            bad += 1
            continue

        if errors:
            log.error(f"[ERR validate] {e.shot_id}: {len(errors)} errors "
                      f"({len(warnings)} warnings)")
            _save_failed(out_root, slug, raw_r1, "business_validation_failed",
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
