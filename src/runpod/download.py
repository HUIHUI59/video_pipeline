"""
src/runpod/download.py
════════════════════════════════════════════════════════════════
Stage 5 本地侧：从 Runpod Pod 把标注 JSON 拉回本地并做 schema 校验。

用法：
  python -m src.runpod.download --config configs/runpod.yaml
  python -m src.runpod.download --config configs/runpod.yaml --dry-run
  python -m src.runpod.download --config configs/runpod.yaml --no-validate

流程：
  1. 读 configs/runpod.yaml，确认 Pod SSH、本地落地路径
  2. rsync -avz pod:<pod_workspace>/output/ → <local_labels_root>/
  3. 逐份 JSON 用 ShotLabel.model_validate 校验，bad files 报到 stderr
"""

from __future__ import annotations
import argparse, json, os, shlex, subprocess, sys
from pathlib import Path
from typing import Any

import yaml

try:
    from src.runpod.schemas import ShotLabel
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.runpod.schemas import ShotLabel  # noqa


def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _known_hosts_path() -> str:
    """项目专用 known_hosts，避免污染用户 ~/.ssh/known_hosts。"""
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
    """SSH 选项：accept-new TOFU + 项目独立 known_hosts，可抵御 MITM。"""
    return ["-i", os.path.expanduser(pod["ssh_key"]),
            "-p", str(pod["port"]),
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={_known_hosts_path()}",
            "-o", "ConnectTimeout=10"]


def _rsync_from_pod(pod: dict[str, Any], pod_workspace: str,
                    local_root: str, dry_run: bool) -> int:
    os.makedirs(local_root, exist_ok=True)
    ssh_cmd = "ssh " + " ".join(shlex.quote(x) for x in _ssh_opts(pod))
    src  = f"{pod['user']}@{pod['host']}:{pod_workspace}/output/"
    cmd  = ["rsync", "-avz", "--progress", "-e", ssh_cmd,
            src, local_root + "/"]
    print("  $", " ".join(shlex.quote(c) for c in cmd))
    if dry_run:
        return 0
    r = subprocess.run(cmd, check=False)
    return r.returncode


def _validate_all(local_root: str) -> tuple[int, int]:
    ok = bad = 0
    for f in sorted(Path(local_root).rglob("*.json")):
        try:
            ShotLabel.model_validate(json.loads(f.read_text(encoding="utf-8")))
            ok += 1
        except Exception as e:
            print(f"  [BAD] {f}: {e}", file=sys.stderr)
            bad += 1
    return ok, bad


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage 5: 从 Runpod Pod 拉回标注 JSON")
    ap.add_argument("--config",      required=True, help="configs/runpod.yaml 路径")
    ap.add_argument("--dry-run",     action="store_true", help="只打印 rsync 命令，不执行")
    ap.add_argument("--no-validate", action="store_true", help="跳过 schema 校验")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    pod   = cfg["pod"]
    paths = cfg["paths"]

    pod_workspace = paths["pod_workspace"]
    local_labels  = os.path.expanduser(paths["local_labels_root"])

    print(f"  从 Pod 拉回: {pod['user']}@{pod['host']}:{pod_workspace}/output/")
    print(f"  本地落地到 : {local_labels}")
    rc = _rsync_from_pod(pod, pod_workspace, local_labels, args.dry_run)
    if rc != 0:
        print(f"[ERR] rsync 失败 rc={rc}", file=sys.stderr)
        return rc
    print("  ✅ rsync 完成。")

    if args.dry_run or args.no_validate:
        return 0

    print("\n  schema 校验中...")
    ok, bad = _validate_all(local_labels)
    print(f"  ✅ {ok} 份通过  ❌ {bad} 份失败")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
