"""Forge CLI - thin command-line front door for the forge engine."""

import argparse
import os
import sys

from forge.core import run_job
from forge import FORGE_VERSION


def main():
    p = argparse.ArgumentParser(
        prog="forge",
        description=f"Forge {FORGE_VERSION} - video analysis & sample extraction")
    p.add_argument("input", help="input video file")
    p.add_argument("--scenes", action="store_true", help="scene detection + clip splitting")
    p.add_argument("--detect", action="store_true", help="object/person detection")
    p.add_argument("--transcribe", action="store_true", help="speech-to-text SRT")
    p.add_argument("--frames", action="store_true", help="contact sheet + thumbnails")
    p.add_argument("--all", action="store_true", help="enable every feature")
    p.add_argument("--output", default=None, help="output root (default: project output\\)")

    g = p.add_argument_group("feature tunables")
    g.add_argument("--threshold", type=float, default=27.0, help="scene sensitivity (default 27)")
    g.add_argument("--min-scene", type=float, default=1.0, help="min scene length sec (default 1)")
    g.add_argument("--labels", default=None, help="comma list: person,car,truck (default all)")
    g.add_argument("--sample-step", type=int, default=15, help="detect every Nth frame (default 15)")
    g.add_argument("--language", default=None, help="force language code, e.g. en, fr, ja")
    g.add_argument("--words-per-line", type=int, default=8, help="SRT words per line (default 8)")
    g.add_argument("--max-lines", type=int, default=2, help="SRT max lines per cue (default 2)")
    g.add_argument("--sheet", default="4x4", help="contact sheet grid, e.g. 4x4")

    args = p.parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input not found: {args.input}")
        return 1

    features = {
        "scenes": args.all or args.scenes,
        "detect": args.all or args.detect,
        "transcribe": args.all or args.transcribe,
        "frames": args.all or args.frames,
    }
    if not any(features.values()):
        print("Nothing to do - add --scenes --detect --transcribe --frames (or --all)")
        return 1

    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
    else:
        labels = None

    try:
        cols, rows = args.sheet.lower().split("x")
        sheet_cols, sheet_rows = int(cols), int(rows)
    except (ValueError, AttributeError):
        sheet_cols, sheet_rows = 4, 4

    try:
        summary = run_job(
            args.input, output_root=args.output, features=features,
            scene_threshold=args.threshold, scene_min_len=args.min_scene,
            detect_labels=labels, detect_sample_step=args.sample_step,
            language=args.language, words_per_line=args.words_per_line,
            max_lines=args.max_lines, sheet_cols=sheet_cols, sheet_rows=sheet_rows,
        )
    except Exception as e:
        print(f"FAILED: {e}")
        return 1

    print("\n" + "=" * 60)
    print(f"FORGE {FORGE_VERSION} - SUMMARY  ({summary['elapsed']})")
    print("=" * 60)
    print(f"Video   : {summary['video']}")
    print(f"Duration: {summary['duration']:.1f}s")
    print(f"Output  : {summary['output_dir']}")
    for feat, res in summary["stages"].items():
        line = f"  {feat:12s}: "
        if "cancelled" in res:
            line += "CANCELLED"
        elif "error" in res:
            line += f"ERROR - {res['error']}"
        elif feat == "scenes":
            line += f"{res['scenes']} scenes -> {res['clips']} clips"
        elif feat == "detect":
            line += f"{res['hits']} hits ({', '.join(f'{k}:{v}' for k, v in list(res['labels'].items())[:5])})"
        elif feat == "transcribe":
            line += f"{res['segments']} segments [{res['language']}] -> {os.path.basename(res['srt'])}"
        elif feat == "frames":
            line += f"contact sheet + {res['thumbs']} thumbnails"
        if "time" in res:
            line += f"  ({res['time']}s)"
        print(line)
    if summary["errors"]:
        print(f"\nErrors: {len(summary['errors'])}")
        for stage, err in summary["errors"]:
            print(f"  [{stage}] {err[:200]}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())