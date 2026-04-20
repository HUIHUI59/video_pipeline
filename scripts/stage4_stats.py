#!/usr/bin/env python3
"""scripts/stage4_stats.py — Stage 4 manifest 统计

扫 Stage 4 的 jsonl manifest 目录，按 shot_category 分桶：
  - single     单人特写（脸占帧 ≥ 15%）
  - dominant   2-3 人，主角脸显著更大
  - multi      2-3 人均衡 or ≥4 人
  - wide       有人但看不清脸 or 远景
  - landscape  无人

用法：
  python scripts/stage4_stats.py /mnt/movies/Films/forCloudKorOutput/manifest
  python scripts/stage4_stats.py --config configs/runpod.122b.yaml
  python scripts/stage4_stats.py --config configs/runpod.122b.yaml --per-movie
  python scripts/stage4_stats.py --config configs/runpod.122b.yaml --quality-only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _manifest_from_config(cfg_path: str) -> str:
    import yaml
    cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    p = (cfg.get("paths") or {}).get("local_manifest_dir")
    if not p:
        raise ValueError(f"{cfg_path} 里缺少 paths.local_manifest_dir")
    return os.path.expanduser(p)


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("manifest_dir", nargs="?",
                    help="manifest 目录，含 <movie>.jsonl")
    ap.add_argument("--config",
                    help="从 configs/runpod*.yaml 读 paths.local_manifest_dir")
    ap.add_argument("--quality-only", action="store_true",
                    help="只统计 quality_ok=True 的镜头（过滤太黑/太亮/模糊/低对比度）")
    ap.add_argument("--per-movie", action="store_true",
                    help="按电影分解 Top 30 single 计数")
    ap.add_argument("--top", type=int, default=30,
                    help="--per-movie 展示多少部（默认 30）")
    args = ap.parse_args()

    if args.manifest_dir:
        manifest_dir = args.manifest_dir
    elif args.config:
        manifest_dir = _manifest_from_config(args.config)
    else:
        ap.error("需要 manifest_dir 位置参数或 --config 选项")

    root = Path(manifest_dir).expanduser()
    if not root.is_dir():
        print(f"[ERR] 目录不存在：{root}", file=sys.stderr)
        return 1

    jsonl_files = sorted(root.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[ERR] {root} 里找不到任何 .jsonl", file=sys.stderr)
        return 1

    totals: Counter = Counter()
    by_movie: dict[str, Counter] = defaultdict(Counter)
    total_rows = 0
    bad_quality = 0
    parse_errors = 0
    duration_by_cat: dict[str, float] = defaultdict(float)

    for f in jsonl_files:
        movie = f.stem
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception as e:
            print(f"[warn] 读 {f.name} 失败：{e}", file=sys.stderr)
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            total_rows += 1
            cat = row.get("shot_category", "unknown")
            q_ok = row.get("quality_ok")
            if q_ok is False:
                bad_quality += 1
            if args.quality_only and q_ok is False:
                continue
            totals[cat] += 1
            by_movie[movie][cat] += 1
            dur = row.get("duration_sec") or 0
            try:
                duration_by_cat[cat] += float(dur)
            except (TypeError, ValueError):
                pass

    kept = sum(totals.values())
    print(f"════ {root} ════")
    print(f"  文件数       : {len(jsonl_files)} 部电影的 manifest")
    print(f"  原始镜头总数 : {total_rows}")
    print(f"  画质不合格   : {bad_quality}  "
          f"({100.0 * bad_quality / (total_rows or 1):.1f}%)")
    if parse_errors:
        print(f"  JSON 解析错误: {parse_errors}")
    if args.quality_only:
        print(f"  (本次 --quality-only，只统计 quality_ok=True 的 {kept} 条)")
    print()

    print("════ shot_category 分布 ════")
    category_order = ["single", "dominant", "multi", "wide", "landscape"]
    for cat in category_order + sorted(k for k in totals if k not in category_order):
        n = totals.get(cat, 0)
        if n == 0 and cat not in category_order:
            continue
        pct = 100.0 * n / (kept or 1)
        hours = duration_by_cat.get(cat, 0.0) / 3600.0
        bar = "█" * int(pct / 2)
        print(f"  {cat:<12} {n:>7}  {pct:>5.1f}%  {hours:>6.1f} h  {bar}")

    print()
    single_n = totals.get("single", 0)
    print(f"════ 单人特写 (single) 合计 ════")
    print(f"  {single_n} 个 shot  ({100.0 * single_n / (kept or 1):.1f}% of kept)")
    print(f"  累计时长 {duration_by_cat.get('single', 0) / 3600:.1f} 小时 / "
          f"{duration_by_cat.get('single', 0) / 60:.1f} 分钟")

    if args.per_movie:
        print()
        print(f"════ Top {args.top} 部电影按 single 数量排序 ════")
        ranked = sorted(by_movie.items(),
                        key=lambda kv: kv[1].get("single", 0),
                        reverse=True)
        print(f"  {'single':>6}  {'dom':>5}  {'multi':>5}  "
              f"{'wide':>5}  {'land':>5}  {'total':>5}   movie")
        for movie, counts in ranked[:args.top]:
            s = counts.get("single", 0)
            d = counts.get("dominant", 0)
            m = counts.get("multi", 0)
            w = counts.get("wide", 0)
            l = counts.get("landscape", 0)
            tot = sum(counts.values())
            if tot == 0:
                continue
            print(f"  {s:>6}  {d:>5}  {m:>5}  {w:>5}  {l:>5}  {tot:>5}   {movie}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
