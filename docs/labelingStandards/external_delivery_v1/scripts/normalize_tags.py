#!/usr/bin/env python3
"""
normalize_tags.py — Post-process VLM inference JSON to normalize action tags.

Applies motion_synonyms.yaml rules in order:
  1. verb_forms: walks → walking, sat → sitting
  2. synonyms: strolling → walking, grabbing → grasping
  3. camera_terms_forbidden: remove any camera terms from action fields
  4. intensity_remap: move intensity words to action_quality.intensity
  5. tone_remap: move tone words to action_quality.tone

Usage:
  python scripts/normalize_tags.py \\
    --input output/movies/MovieName/shots/ \\
    --taxonomy docs/motion_taxonomy.yaml \\
    --synonyms docs/motion_synonyms.yaml \\
    --output output/movies/MovieName/shots/  # in-place OK

  # Dry-run (no file modification, report only)
  python scripts/normalize_tags.py \\
    --input output/movies/MovieName/shots/ \\
    --taxonomy docs/motion_taxonomy.yaml \\
    --synonyms docs/motion_synonyms.yaml \\
    --dry-run \\
    --output normalization_report.json
"""

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml required. pip install pyyaml", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger("normalize")


class TagNormalizer:
    """Apply synonym/verb-form/camera-term normalization rules."""

    def __init__(self, synonyms_path: Path, taxonomy_path: Path | None = None):
        with open(synonyms_path, encoding="utf-8") as f:
            rules = yaml.safe_load(f)

        self.verb_forms: dict[str, str] = {
            k.lower().strip(): v.lower().strip()
            for k, v in rules.get("verb_forms", {}).items()
        }
        self.synonyms: dict[str, str] = {
            k.lower().strip(): v.lower().strip()
            for k, v in rules.get("synonyms", {}).items()
        }
        self.camera_terms: set[str] = {
            t.lower().strip() for t in rules.get("camera_terms_forbidden", [])
        }
        self.intensity_remap: dict[str, dict] = {
            k.lower().strip(): v
            for k, v in rules.get("intensity_remap", {}).items()
        }
        self.tone_remap: dict[str, dict] = {
            k.lower().strip(): v
            for k, v in rules.get("tone_remap", {}).items()
        }

        # Load taxonomy leaves for validation
        self.taxonomy_leaves: set[str] = set()
        if taxonomy_path and taxonomy_path.exists():
            with open(taxonomy_path, encoding="utf-8") as f:
                tax = yaml.safe_load(f)
            for cat_data in tax.get("action_primary", {}).values():
                if isinstance(cat_data, dict) and "leaves" in cat_data:
                    self.taxonomy_leaves.update(
                        leaf.lower().strip() for leaf in cat_data["leaves"]
                    )

        self.stats = Counter()

    def normalize_action_primary(self, value: str | None) -> str | None:
        """Normalize an action_primary value."""
        if not value:
            return value

        original = value
        v = value.lower().strip()

        # 1. Verb forms
        if v in self.verb_forms:
            v = self.verb_forms[v]
            self.stats["verb_form_fixed"] += 1

        # 2. Synonyms
        if v in self.synonyms:
            v = self.synonyms[v]
            self.stats["synonym_replaced"] += 1

        # 3. Camera terms → remove (return None)
        if v in self.camera_terms or any(ct in v for ct in self.camera_terms):
            self.stats["camera_term_removed"] += 1
            return None

        # 4. Intensity remap → remove from action, note for axis
        if v in self.intensity_remap:
            self.stats["intensity_remapped"] += 1
            return None

        # 5. Tone remap → remove from action, note for axis
        if v in self.tone_remap:
            self.stats["tone_remapped"] += 1
            return None

        # 6. Validate against taxonomy
        if v not in self.taxonomy_leaves and not v.startswith("other/"):
            # Check if any leaf is a substring
            matched = [leaf for leaf in self.taxonomy_leaves if leaf in v]
            if matched:
                # Use the longest matching leaf
                best = max(matched, key=len)
                self.stats["fuzzy_matched_to_leaf"] += 1
                return best
            else:
                # Wrap as other/
                self.stats["wrapped_as_other"] += 1
                clean = v.replace(" ", "_")[:32]
                return f"other/{clean}"

        if v != original.lower().strip():
            self.stats["total_changed"] += 1

        return v

    # ── 自由文本里剥 camera 术语 ──────────────────────────────────
    # 规范 § 5.3 禁止 camera 术语出现在 action/body/kinematics 字段，同样不应
    # 污染 captions、shot_context summary 等描述性文字。这里做保守的整词替换：
    # 只在 word boundary 出现时删除，避免误删"closeness"这种非 camera 词。
    def _strip_camera_terms_text(self, text):
        """Remove camera terms from free-text. Returns (new_text, changed)."""
        if not text or not isinstance(text, str):
            return text, False
        import re
        changed = False
        out = text
        for term in self.camera_terms:
            if not term:
                continue
            pattern = re.compile(
                r"(?i)(?<![A-Za-z0-9])" + re.escape(term) + r"(?![A-Za-z0-9])"
            )
            new_out = pattern.sub("", out)
            if new_out != out:
                changed = True
                out = new_out
        if changed:
            # 清理多余空格/重复逗号/首尾标点
            out = re.sub(r"\s{2,}", " ", out)
            out = re.sub(r"\s*,\s*,", ",", out)
            out = re.sub(r"^[,\s]+|[,\s]+$", "", out)
            self.stats["camera_term_stripped_in_text"] += 1
        return out, changed

    def normalize_shot(self, shot: dict) -> tuple[dict, int]:
        """Normalize a single shot JSON. Returns (normalized_shot, changes_count)."""
        changes = 0

        def _strip_dict_text(d, keys):
            """对 d 里列出的字符串 keys 跑 _strip_camera_terms_text。返回 changes 数。"""
            if not isinstance(d, dict):
                return 0
            c = 0
            for k in keys:
                v = d.get(k)
                new_v, changed = self._strip_camera_terms_text(v)
                if changed:
                    d[k] = new_v
                    c += 1
            return c

        # 1) persons[].body_analysis.action_primary（原有行为，保持）
        for person in shot.get("persons", []) or []:
            body = person.get("body_analysis") if isinstance(person, dict) else None
            if isinstance(body, dict):
                ap = body.get("action_primary")
                if ap:
                    new_ap = self.normalize_action_primary(ap)
                    if new_ap != ap:
                        body["action_primary"] = new_ap
                        changes += 1

        # 2) camera_terms_forbidden 扩散到全部自由文本字段（delivery_v1 § 5.3）
        #    verb_forms / synonyms / intensity_remap / tone_remap 保持只作用
        #    于 action_primary —— 它们对自然语言描述是有损改写，易改变语义。
        for person in shot.get("persons", []) or []:
            if not isinstance(person, dict):
                continue
            fa = person.get("face_analysis")
            if isinstance(fa, dict):
                changes += _strip_dict_text(fa, ["expression_caption"])
                changes += _strip_dict_text(
                    fa.get("alternative_captions"),
                    ["direct", "literary", "direction", "situational"],
                )
            ba = person.get("body_analysis")
            if isinstance(ba, dict):
                changes += _strip_dict_text(
                    ba, ["motion_caption", "gesture_detail"])
                changes += _strip_dict_text(
                    ba.get("alternative_captions"),
                    ["direct", "literary", "direction", "situational"],
                )
                changes += _strip_dict_text(
                    ba.get("upper_body_detail"),
                    ["head", "neck", "shoulders", "arms", "hands", "torso"],
                )

        sc = shot.get("shot_context")
        if isinstance(sc, dict):
            changes += _strip_dict_text(
                sc, ["shot_emotion_summary", "shot_motion_summary"])
            changes += _strip_dict_text(
                sc.get("scene_context"),
                ["visible_setting", "narrative_situation"],
            )

        return shot, changes


def process_directory(
    normalizer: TagNormalizer,
    input_dir: Path,
    output_dir: Path | None,
    dry_run: bool,
) -> dict:
    """Process all JSON files in directory."""
    json_files = sorted(input_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("validation_") and not f.name.startswith("movie_meta")]

    total = len(json_files)
    modified = 0

    log.info(f"Processing {total} JSON files from {input_dir}")

    for i, fpath in enumerate(json_files, 1):
        try:
            with open(fpath, encoding="utf-8") as f:
                shot = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            log.warning(f"Skip {fpath.name}: {e}")
            continue

        normalized, changes = normalizer.normalize_shot(shot)

        if changes > 0:
            modified += 1
            if not dry_run:
                out_path = (output_dir or input_dir) / fpath.name
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(normalized, f, indent=2, ensure_ascii=False)

        if i % 500 == 0:
            log.info(f"  {'[DRY-RUN] ' if dry_run else ''}Processed {i}/{total}")

    report = {
        "report_version": "1.0",
        "report_date": datetime.now().isoformat(),
        "dry_run": dry_run,
        "total_files": total,
        "files_modified": modified,
        "normalization_stats": dict(normalizer.stats),
        "taxonomy_leaves_loaded": len(normalizer.taxonomy_leaves),
    }

    log.info(f"{'DRY-RUN ' if dry_run else ''}Complete: {modified}/{total} files modified")
    log.info(f"Stats: {dict(normalizer.stats)}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Normalize action tags in VLM inference output")
    parser.add_argument("--input", required=True, type=Path, help="Input directory with per-shot JSON")
    parser.add_argument("--taxonomy", required=True, type=Path, help="motion_taxonomy.yaml path")
    parser.add_argument("--synonyms", required=True, type=Path, help="motion_synonyms.yaml path")
    parser.add_argument("--output", type=Path, help="Output dir (default: in-place) or report path for dry-run")
    parser.add_argument("--dry-run", action="store_true", help="Report only, do not modify files")
    args = parser.parse_args()

    normalizer = TagNormalizer(args.synonyms, args.taxonomy)
    log.info(f"Loaded {len(normalizer.verb_forms)} verb forms, {len(normalizer.synonyms)} synonyms, "
             f"{len(normalizer.camera_terms)} camera terms, {len(normalizer.taxonomy_leaves)} taxonomy leaves")

    if args.dry_run:
        report = process_directory(normalizer, args.input, None, dry_run=True)
        out = args.output or Path("normalization_report.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info(f"Report: {out}")
    else:
        output_dir = args.output or args.input
        report = process_directory(normalizer, args.input, output_dir, dry_run=False)
        report_path = output_dir / "normalization_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        log.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
