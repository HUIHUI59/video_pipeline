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
import argparse, json, logging, os, signal, sys, time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.runpod.schemas import ManifestEntry, ShotLabel  # noqa

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


SYSTEM_PROMPT = (
    "You are a rigorous video shot annotator. Produce ONE JSON object "
    "conforming exactly to the provided JSON Schema. Describe only what is "
    "visible in the frames. Use English. Do not invent camera terms in body/"
    "action fields — camera only belongs in shot_context.shot_type. If a field "
    "is not observable, use the 'unknown' value where allowed, or null where "
    "the schema permits. Never emit empty strings for string-typed fields."
)


def _user_prompt(entry: ManifestEntry, n_frames: int) -> str:
    lines = [
        f"Shot ID: {entry.shot_id}",
        f"Source movie: {entry.source_movie}",
        f"Shot category (from manifest): {entry.shot_category}",
        f"Num people (from manifest, trust this): {entry.num_people}",
    ]
    # Stage 4 v2 additional hints（v1 旧 manifest 没有这些字段）
    if entry.num_faces is not None:
        lines.append(
            f"Num faces visible (trust this, order persons by face size): "
            f"{entry.num_faces}")
    if entry.largest_face_ratio is not None:
        lines.append(
            f"Largest face occupies approximately "
            f"{entry.largest_face_ratio*100:.1f}% of the frame")
    lines += [
        f"Duration: {entry.duration_sec:.2f}s  "
        f"Resolution: {entry.width}x{entry.height}  FPS: {entry.fps}",
        f"Frames sampled: {n_frames} (uniformly across the shot)",
        "",
        "Produce the ShotLabel JSON.",
    ]
    return "\n".join(lines)


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

    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
    except ImportError as e:
        log.error(f"vLLM 未安装：{e}。请先跑 tools/pod_setup.sh")
        return 2

    model_name = model_cfg["name"]
    precision  = model_cfg.get("precision", "bf16")
    dtype      = {"bf16": "bfloat16", "fp8": "auto"}.get(precision, "auto")
    log.info(f"加载模型 {model_name} (dtype={dtype}) ...")
    t0 = time.time()
    llm = LLM(model=model_name, dtype=dtype,
              trust_remote_code=True,
              limit_mm_per_prompt={"image": 16})
    log.info(f"模型加载完成 ({time.time()-t0:.1f}s)")

    schema = ShotLabel.model_json_schema()
    guided = GuidedDecodingParams(json=schema)
    sampling = SamplingParams(
        temperature       = float(samp.get("temperature", 0.2)),
        top_p             = float(samp.get("top_p", 0.9)),
        max_tokens        = int(samp.get("max_tokens", 2048)),
        repetition_penalty= float(samp.get("repetition_penalty", 1.05)),
        guided_decoding   = guided,
    )
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

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": [
                *[{"type": "image", "image": im} for im in frames],
                {"type": "text", "text": _user_prompt(e, n_frames)},
            ]},
        ]

        try:
            t1 = time.time()
            res = llm.chat(messages, sampling, use_tqdm=False)
            raw = res[0].outputs[0].text
            infer_ms = int((time.time() - t1) * 1000)
        except Exception as ex:
            log.error(f"[ERR] 推理失败 {e.shot_id}: {ex}")
            bad += 1
            continue

        try:
            obj = json.loads(raw)
            obj.setdefault("meta", {})
            obj["meta"].setdefault("vlm_model",   model_name)
            obj["meta"].setdefault("vlm_version", "2026-04")
            obj["meta"]["frames_used"]   = n_frames
            obj["meta"]["infer_time_ms"] = infer_ms
            ShotLabel.model_validate(obj)
        except Exception as ex:
            log.error(f"[ERR] 校验失败 {e.shot_id}: {ex}")
            bad += 1
            fail_dir = out_root / "_failed"
            fail_dir.mkdir(exist_ok=True)
            (fail_dir / f"{e.shot_id.replace('/', '__')}.raw.txt").write_text(raw)
            continue

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
