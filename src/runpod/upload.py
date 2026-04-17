"""
src/runpod/upload.py
════════════════════════════════════════════════════════════════
Stage 5 本地侧：把筛选后的 clips + manifest rsync 到 Runpod Pod。

用法：
  python -m src.runpod.upload --config configs/runpod.yaml
  python -m src.runpod.upload --config configs/runpod.yaml --dry-run
  python -m src.runpod.upload --config configs/runpod.yaml --shot-category single,dominant --movies MovieA,MovieB

流程：
  1. 读 configs/runpod.yaml，确认 Pod SSH、路径
  2. 扫 <local_manifest_dir>/*.jsonl，逐行用 ManifestEntry 校验
  3. 按 filters.shot_categories + filters.movies + filters.max_shots 裁剪
  4. 生成筛后的 manifest 到 /tmp/runpod_manifest_<timestamp>.jsonl
  5. 计算本次要推的 clip 文件列表（paths 去重）
  6. rsync -avz --progress 把 clips + 筛后 manifest + src/runpod/ + tools/pod_setup.sh 推到 Pod
"""

from __future__ import annotations
import argparse, json, os, shlex, subprocess, sys, time
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    from src.runpod.schemas import ManifestEntry
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.runpod.schemas import ManifestEntry  # noqa


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ssh_opts(pod: dict[str, Any]) -> list[str]:
    return ["-i", os.path.expanduser(pod["ssh_key"]),
            "-p", str(pod["port"]),
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10"]


def _iter_manifest_lines(manifest_dir: str,
                         movies_filter: set[str] | None
                         ) -> Iterable[tuple[str, ManifestEntry]]:
    for p in sorted(Path(manifest_dir).glob("*.jsonl")):
        movie = p.stem
        if movies_filter and movie not in movies_filter:
            continue
        with open(p, encoding="utf-8") as f:
            for ln_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = ManifestEntry.model_validate(json.loads(line))
                except Exception as e:
                    print(f"[WARN] {p.name}:{ln_no} 校验失败，跳过 ({e})", file=sys.stderr)
                    continue
                yield movie, entry


def _filter_entries(entries: list[tuple[str, ManifestEntry]],
                    categories: list[str] | None,
                    max_shots: int | None
                    ) -> list[tuple[str, ManifestEntry]]:
    out = []
    cat_set = set(categories) if categories else None
    for movie, e in entries:
        if cat_set and e.shot_category not in cat_set:
            continue
        out.append((movie, e))
        if max_shots and len(out) >= max_shots:
            break
    return out


def _write_filtered_manifest(entries: list[tuple[str, ManifestEntry]]) -> str:
    ts = int(time.time())
    out_path = f"/tmp/runpod_manifest_{ts}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for _movie, e in entries:
            f.write(e.model_dump_json() + "\n")
    return out_path


def _rsync_files(files: list[tuple[str, str]],
                 pod: dict[str, Any], dry_run: bool) -> None:
    if not files:
        print("  [skip] 没有文件要传")
        return
    ssh_cmd = "ssh " + " ".join(shlex.quote(x) for x in _ssh_opts(pod))
    print(f"\n  SSH 选项: {ssh_cmd}")
    for local, remote in files:
        cmd = ["rsync", "-avz", "--progress", "-e", ssh_cmd,
               local, f"{pod['user']}@{pod['host']}:{remote}"]
        print("  $", " ".join(shlex.quote(c) for c in cmd))
        if not dry_run:
            r = subprocess.run(cmd, check=False)
            if r.returncode != 0:
                print(f"  [ERR] rsync 失败 rc={r.returncode}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5: 推送 clips + manifest 到 Runpod Pod")
    ap.add_argument("--config",        required=True, help="configs/runpod.yaml 路径")
    ap.add_argument("--shot-category", default="",    help="覆盖 filters.shot_categories（逗号分隔）")
    ap.add_argument("--movies",        default="",    help="覆盖 filters.movies（逗号分隔）")
    ap.add_argument("--max-shots",     type=int, default=None, help="覆盖 filters.max_shots")
    ap.add_argument("--dry-run",       action="store_true", help="只打印 rsync 命令，不执行")
    ap.add_argument("--include-bad-quality", action="store_true",
                    help="即使 quality_ok=False 也上传（默认过滤掉太黑/太亮/模糊的镜头）")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    pod     = cfg["pod"]
    paths   = cfg["paths"]
    filters = cfg.get("filters", {}) or {}

    categories = (args.shot_category.split(",") if args.shot_category
                  else filters.get("shot_categories") or None)
    movies     = (args.movies.split(",") if args.movies
                  else filters.get("movies") or None)
    max_shots  = args.max_shots if args.max_shots is not None else filters.get("max_shots")

    manifest_dir  = os.path.expanduser(paths["local_manifest_dir"])
    clips_root    = os.path.expanduser(paths["local_clips_root"])
    pod_workspace = paths["pod_workspace"]

    if not Path(manifest_dir).exists():
        print(f"[ERR] manifest 目录不存在: {manifest_dir}", file=sys.stderr)
        return 1

    print(f"  扫描 manifest: {manifest_dir}")
    raw = list(_iter_manifest_lines(manifest_dir,
                                    set(movies) if movies else None))
    print(f"  校验通过: {len(raw)} 条")
    filtered = _filter_entries(raw, categories, max_shots,
                               skip_bad_quality=not args.include_bad_quality)
    print(f"  筛选后:   {len(filtered)} 条  "
          f"(categories={categories or 'all'}, movies={movies or 'all'}, "
          f"max={max_shots or '∞'}, skip_bad_quality={not args.include_bad_quality})")

    if not filtered:
        print("[WARN] 没有符合条件的 shot，退出。")
        return 0

    filt_path = _write_filtered_manifest(filtered)
    print(f"  筛后 manifest → {filt_path}")

    # 组 rsync 列表：clips + 筛后 manifest + src/runpod/ 代码 + pod_setup + 配置
    file_ops: list[tuple[str, str]] = []
    for _movie, e in filtered:
        rel = e.path
        if rel.startswith("clips/"):
            local = str(Path(clips_root) / rel[len("clips/"):])
        else:
            local = rel if os.path.isabs(rel) else str(Path(clips_root) / rel)
        remote = f"{pod_workspace}/{rel}"
        if not Path(local).exists():
            print(f"  [miss] 本地文件不存在，跳过: {local}")
            continue
        file_ops.append((local, remote))

    print(f"  clips 待传: {len(file_ops)} 个")

    file_ops.append((filt_path, f"{pod_workspace}/manifest.jsonl"))

    project_root = Path(__file__).resolve().parents[2]
    for fn in ("__init__.py", "schemas.py", "pod_runner.py"):
        src = project_root / "src" / "runpod" / fn
        if src.exists():
            file_ops.append((str(src), f"{pod_workspace}/src/runpod/{fn}"))

    pod_setup = project_root / "tools" / "pod_setup.sh"
    if pod_setup.exists():
        file_ops.append((str(pod_setup), f"{pod_workspace}/tools/pod_setup.sh"))
    file_ops.append((args.config, f"{pod_workspace}/runpod.yaml"))

    print(f"\n  目标 Pod: {pod['user']}@{pod['host']}:{pod['port']}  workspace={pod_workspace}")
    mkdir_cmd = (["ssh"] + _ssh_opts(pod) +
                 [f"{pod['user']}@{pod['host']}",
                  f"mkdir -p {pod_workspace}/clips {pod_workspace}/src/runpod {pod_workspace}/tools"])
    print("  $", " ".join(shlex.quote(c) for c in mkdir_cmd))
    if not args.dry_run:
        subprocess.run(mkdir_cmd, check=False)

    print(f"\n  rsync 共 {len(file_ops)} 个文件 (dry_run={args.dry_run})")
    _rsync_files(file_ops, pod, args.dry_run)

    print("\n  ✅ 上传完成。")
    print(f"  下一步：bash scripts/runpod/02_run.sh  （ssh 进 Pod 跑 pod_runner）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
