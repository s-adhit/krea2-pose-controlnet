"""Compare a stratified authoritative reconstruction against stored control PNGs."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pose_controlnet.control_reconstruction import compare_control, summarize_reconstruction
from pose_controlnet.dataset_index import DatasetIndex
from pose_controlnet.pose_targets import load_sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=16)
    parser.add_argument("--min-foreground-iou", type=float, default=0.995)
    parser.add_argument("--max-mae", type=float, default=0.25)
    args = parser.parse_args()
    if args.per_source < 1:
        parser.error("--per-source must be positive")
    _, records = load_sidecar(args.sidecar); index = DatasetIndex.discover(args.dataset_root)
    selected = defaultdict(list)
    for record in records:
        if len(selected[record["source"]]) < args.per_source:
            selected[record["source"]].append(record)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows, panels = [], []
    for source in sorted(selected):
        for record in selected[source]:
            metrics, expected, reconstructed, difference = compare_control(record, index.control_by_stem[record["stem"]])
            metrics["source"] = source; rows.append(metrics)
            panels.extend([expected, reconstructed, difference])
    _contact_sheet(panels, args.output_dir / "contact_sheet.png")
    summary = summarize_reconstruction(rows, min_foreground_iou=args.min_foreground_iou, max_mae=args.max_mae)
    (args.output_dir / "mismatches.json").write_text(json.dumps({"samples": rows, **summary}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise SystemExit(1)


def _contact_sheet(images: list[Image.Image], output: Path) -> None:
    thumb = (192, 192); columns = 6; rows = max(1, (len(images) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * thumb[0], rows * thumb[1]), "black")
    for index, image in enumerate(images):
        copy = image.copy(); copy.thumbnail(thumb)
        x, y = (index % columns) * thumb[0], (index // columns) * thumb[1]
        sheet.paste(copy, (x + (thumb[0] - copy.width) // 2, y + (thumb[1] - copy.height) // 2))
    sheet.save(output)


if __name__ == "__main__":
    main()
