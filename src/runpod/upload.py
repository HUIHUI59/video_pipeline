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

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    from src.runpod.schemas import ManifestEntry
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.runpod.schemas import ManifestEntry  # noqa

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("upload")


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _known_hosts_path() -> str:
    """项目专用 known_hosts，避免污染用户 ~/.ssh/known_hosts。首次 ensure 存在。"""
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(ssh_dir, 0o700)
    except OSError:
        pass
    kh = ssh_dir / "video_pipeline_known_hosts"
    if not kh.exists():
        kh.touch(mode=0o600)
    return str(kh)


def _ssh_opts(pod: dict[str, Any]) -> list[str]:
    """SSH 选项：首次连接 TOFU（accept-new），后续严格 pinning。
    比 StrictHostKeyChecking=no 更安全：host key 变化时会被拒，可抵御 MITM。
    """
    return ["-i", os.path.expanduser(pod["ssh_key"]),
            "-p", str(pod["port"]),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={_known_hosts_path()}",
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
                    log.warning(f"{p.name}:{ln_no} 校验失败，跳过 ({e})")
                    continue
                yield movie, entry


def _filter_entries(entries: list[tuple[str, ManifestEntry]],
                    categories: list[str] | None,
                    max_shots: int | None,
                    skip_bad_quality: bool = True,
                    skip_landscape: bool = True,
                    ) -> list[tuple[str, ManifestEntry]]:
    """
    依次过滤：
      1. shot_category 不在 categories 白名单里 → 跳（categories 为空则不过滤）
      2. shot_category == "landscape" 且 skip_landscape=True → 跳（无人标注无意义）
      3. quality_ok == False 且 skip_bad_quality=True → 跳
      4. 达到 max_shots 上限 → 停止
    """
    out = []
    skipped_quality   = 0
    skipped_landscape = 0
    cat_set = set(categories) if categories else None
    for movie, e in entries:
        if cat_set and e.shot_category not in cat_set:
            continue
        if skip_landscape and e.shot_category == "landscape":
            skipped_landscape += 1
            continue
        # quality_ok == None 表示旧版 manifest（没有该字段），不过滤
        if skip_bad_quality and e.quality_ok is False:
            skipped_quality += 1
            continue
        out.append((movie, e))
        if max_shots and len(out) >= max_shots:
            break
    if skipped_landscape:
        log.info(f"[landscape] 跳过无人镜头 {skipped_landscape} 个"
                 f"（--include-landscape 可关闭此过滤）")
    if skipped_quality:
        log.info(f"[quality] 跳过画质不合格镜头 {skipped_quality} 个"
                 f"（--include-bad-quality 可关闭此过滤）")
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
        log.info("[skip] 没有文件要传")
        return
    ssh_cmd = "ssh " + " ".join(shlex.quote(x) for x in _ssh_opts(pod))
    log.info(f"SSH 选项: {ssh_cmd}")
    for local, remote in files:
        # --mkpath (rsync 3.2.3+) 自动创建 remote 的父目录链
        # --no-owner --no-group 跳过 owner/group 同步：WSL uid ≠ Pod uid 会
        #   触发 "chown Operation not permitted" 让 rsync 退出 rc=23（文件其实
        #   已经传完），我们不需要保留属主信息
        cmd = ["rsync", "-avz", "--progress", "--mkpath",
               "--no-owner", "--no-group",
               "-e", ssh_cmd,
               local, f"{pod['user']}@{pod['host']}:{remote}"]
        log.info("$ " + " ".join(shlex.quote(c) for c in cmd))
        if not dry_run:
            r = subprocess.run(cmd, check=False)
            if r.returncode != 0:
                log.error(f"rsync 失败 rc={r.returncode} local={local} remote={remote}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5: 推送 clips + manifest 到 Runpod Pod")
    ap.add_argument("--config",        required=True, help="configs/runpod.yaml 路径")
    ap.add_argument("--shot-category", default="",    help="覆盖 filters.shot_categories（逗号分隔）")
    ap.add_argument("--movies",        default="",    help="覆盖 filters.movies（逗号分隔）")
    ap.add_argument("--max-shots",     type=int, default=None, help="覆盖 filters.max_shots")
    ap.add_argument("--dry-run",       action="store_true", help="只打印 rsync 命令，不执行")
    ap.add_argument("--include-bad-quality", action="store_true",
                    help="即使 quality_ok=False 也上传（默认过滤掉太黑/太亮/模糊的镜头）")
    ap.add_argument("--include-landscape", action="store_true",
                    help="即使 shot_category=landscape 也上传（默认无人的镜头不标注）")
    ap.add_argument("--code-only", action="store_true",
                    help="只推代码 + 配置 + delivery_v1（跳过 clips 和 manifest）；"
                         "用于迭代 pod_runner 代码时快速部署。相当于跑 00_push_code.sh。")
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

    file_ops: list[tuple[str, str]] = []

    if args.code_only:
        log.info("[code-only] 跳过 clips + manifest，只推代码 + 配置 + delivery_v1")
    else:
        if not Path(manifest_dir).exists():
            log.error(f"manifest 目录不存在: {manifest_dir}")
            return 1

        log.info(f"扫描 manifest: {manifest_dir}")
        raw = list(_iter_manifest_lines(manifest_dir,
                                        set(movies) if movies else None))
        log.info(f"校验通过: {len(raw)} 条")
        filtered = _filter_entries(raw, categories, max_shots,
                                   skip_bad_quality=not args.include_bad_quality,
                                   skip_landscape=not args.include_landscape)
        log.info(f"筛选后:   {len(filtered)} 条  "
                 f"(categories={categories or 'all'}, movies={movies or 'all'}, "
                 f"max={max_shots or '∞'}, skip_bad_quality={not args.include_bad_quality}, "
                 f"skip_landscape={not args.include_landscape})")

        if not filtered:
            log.warning("没有符合条件的 shot，退出。")
            return 0

        filt_path = _write_filtered_manifest(filtered)
        log.info(f"筛后 manifest → {filt_path}")

        # clips 按筛选后逐个加进 rsync 列表
        for _movie, e in filtered:
            rel = e.path
            if rel.startswith("clips/"):
                local = str(Path(clips_root) / rel[len("clips/"):])
            else:
                local = rel if os.path.isabs(rel) else str(Path(clips_root) / rel)
            remote = f"{pod_workspace}/{rel}"
            if not Path(local).exists():
                log.warning(f"[miss] 本地文件不存在，跳过: {local}")
                continue
            file_ops.append((local, remote))

        log.info(f"clips 待传: {len(file_ops)} 个")

        file_ops.append((filt_path, f"{pod_workspace}/manifest.jsonl"))

    project_root = Path(__file__).resolve().parents[2]
    for fn in ("__init__.py", "schemas.py", "pod_runner.py"):
        src = project_root / "src" / "runpod" / fn
        if src.exists():
            file_ops.append((str(src), f"{pod_workspace}/src/runpod/{fn}"))

    # 推 tools/ 下所有 .sh（pod_setup.sh、kill_gpu.sh 等）
    tools_dir = project_root / "tools"
    if tools_dir.exists():
        for sh in sorted(tools_dir.glob("*.sh")):
            file_ops.append((str(sh), f"{pod_workspace}/tools/{sh.name}"))
    file_ops.append((args.config, f"{pod_workspace}/runpod.yaml"))

    # 推送 external_delivery_v1 bundle（官方 prompt 构造器 + normalize + validator
    # + taxonomy/synonyms YAML + 9 个 few-shot 示例 JSON）到 Pod。
    # pod_runner 运行时会 sys.path.insert 并 import 其中的脚本。
    delivery_root = project_root / "docs" / "labelingStandards" / "external_delivery_v1"
    if delivery_root.exists():
        for p in delivery_root.rglob("*"):
            if not p.is_file():
                continue
            name = p.name
            # 跳过 Zone.Identifier（Windows 拷过来的元数据）
            if name.endswith(":Zone.Identifier") or "Zone.Identifier" in name:
                continue
            # 只推白名单后缀（docs + scripts 需要的）
            if p.suffix not in (".py", ".yaml", ".yml", ".md", ".json", ".txt"):
                continue
            rel = p.relative_to(delivery_root)
            dst = f"{pod_workspace}/delivery_v1/{rel.as_posix()}"
            file_ops.append((str(p), dst))

    log.info(f"目标 Pod: {pod['user']}@{pod['host']}:{pod['port']}  workspace={pod_workspace}")

    # 远端命令里所有来自 config 的路径必须 shlex.quote —— 即便来源可信，
    # 防御性转义能彻底消除 path 里空格/分号/反引号引发的 shell 注入。
    _ws_q = shlex.quote(pod_workspace)
    remote_mkdir = (
        f"mkdir -p {_ws_q}/clips {_ws_q}/src/runpod "
        f"{_ws_q}/tools {_ws_q}/delivery_v1"
    )
    mkdir_cmd = (["ssh"] + _ssh_opts(pod) +
                 [f"{pod['user']}@{pod['host']}", remote_mkdir])
    log.info("$ " + " ".join(shlex.quote(c) for c in mkdir_cmd))
    if not args.dry_run:
        subprocess.run(mkdir_cmd, check=False)

    # 确保 Pod 端有 rsync（Runpod 默认 PyTorch 镜像不带，rsync 必须两端都有）
    # 这条命令没有用户输入参与，保留字面字符串即可。
    ensure_rsync_cmd = (["ssh"] + _ssh_opts(pod) +
                        [f"{pod['user']}@{pod['host']}",
                         "command -v rsync >/dev/null 2>&1 || "
                         "(apt-get update -qq && apt-get install -y -qq rsync)"])
    log.info("$ " + " ".join(shlex.quote(c) for c in ensure_rsync_cmd))
    if not args.dry_run:
        r = subprocess.run(ensure_rsync_cmd, check=False)
        if r.returncode != 0:
            log.warning(
                f"无法在 Pod 安装 rsync (rc={r.returncode})，"
                f"后续 rsync 可能失败。请手动在 Pod 跑 apt-get install rsync"
            )

    log.info(f"rsync 共 {len(file_ops)} 个文件 (dry_run={args.dry_run})")
    _rsync_files(file_ops, pod, args.dry_run)

    log.info("✅ 上传完成。")
    log.info("下一步：bash scripts/runpod/02_run.sh  （ssh 进 Pod 跑 pod_runner）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
