"""Convert reviewed BAMBI matched RGB flights to a YOLO detection dataset.

BAMBI stores one MOT annotation file per flight. This converter matches MOT
rows to the extracted RGB frames, keeps only tracks explicitly accepted for
RGB, maps all supported red-deer labels to class 0, and writes one normalized
YOLO label file per image. Images without an accepted deer receive an empty
label file and therefore act as background examples.

Split complete flights rather than frames. Example:

    python dataloaders/bambi_matched/convert.py \
        --train-flight data/bambi/flight_104 \
        --out data/bambi_yolo

When another flight is available for validation:

    python dataloaders/bambi_matched/convert.py \
        --train-flight data/bambi/flight_104 \
        --val-flight data/bambi/flight_132 \
        --out data/bambi_yolo --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


SUPPORTED_CLASSES = {"red deer", "red deer (male)"}


@dataclass(frozen=True)
class MotBox:
    track_id: int
    left: float
    top: float
    width: float
    height: float


@dataclass
class ConvertedImage:
    flight_id: str
    frame: int
    split: str
    source: Path
    output: Path
    width: int
    height: int
    boxes: list[MotBox]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BAMBI matched RGB flights from MOT to YOLO format."
    )
    parser.add_argument(
        "--train-flight",
        action="append",
        required=True,
        type=Path,
        help="Extracted BAMBI matched flight directory. Repeat for multiple training flights.",
    )
    parser.add_argument(
        "--val-flight",
        action="append",
        default=[],
        type=Path,
        help="Complete flight reserved for validation. Repeat for multiple validation flights.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output YOLO dataset directory.")
    parser.add_argument(
        "--preview-count",
        type=int,
        default=12,
        help="Maximum deterministic preview images to generate per non-empty split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory. Never modifies source flights.",
    )
    return parser.parse_args()


def prepare_output(out: Path, overwrite: bool) -> None:
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {out}. Use --overwrite to replace it.")
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "preview" / split).mkdir(parents=True, exist_ok=True)


def load_manifest(flight_dir: Path) -> tuple[str, dict]:
    manifest_path = flight_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing BAMBI manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    flight_id = str(manifest.get("flight_id", ""))
    if not flight_id:
        raise ValueError(f"Manifest has no flight_id: {manifest_path}")
    if not manifest.get("labelled"):
        raise ValueError(f"Flight {flight_id} is not marked as labelled")
    if "rgb" not in manifest.get("modalities", []):
        raise ValueError(f"Flight {flight_id} has no RGB modality")
    return flight_id, manifest


def accepted_rgb_tracks(labels_json: Path) -> tuple[set[int], Counter]:
    if not labels_json.is_file():
        raise FileNotFoundError(f"Missing BAMBI track metadata: {labels_json}")
    data = json.loads(labels_json.read_text(encoding="utf-8"))
    accepted: set[int] = set()
    reasons: Counter = Counter()
    for raw_id, track in data.get("tracks", {}).items():
        track_id = int(track.get("track_id", raw_id))
        class_name = str(track.get("class_name", "")).strip().casefold()
        if track.get("status") != "accepted":
            reasons["track_not_accepted"] += 1
        elif "rgb" not in track.get("accepted_modalities", []):
            reasons["track_not_accepted_for_rgb"] += 1
        elif class_name not in SUPPORTED_CLASSES:
            reasons["unsupported_class"] += 1
        else:
            accepted.add(track_id)
            reasons["accepted_rgb_deer_tracks"] += 1
    if not accepted:
        raise ValueError(f"No accepted RGB red-deer tracks found in {labels_json}")
    return accepted, reasons


def parse_mot(
    mot_path: Path, accepted_tracks: set[int], available_frames: set[int]
) -> tuple[dict[int, list[MotBox]], Counter]:
    if not mot_path.is_file():
        raise FileNotFoundError(f"Missing BAMBI RGB MOT labels: {mot_path}")
    boxes: dict[int, list[MotBox]] = defaultdict(list)
    stats: Counter = Counter()
    with mot_path.open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row:
                continue
            if len(row) < 6:
                raise ValueError(f"{mot_path}:{line_number} has fewer than six columns")
            stats["mot_rows_total"] += 1
            try:
                frame = int(float(row[0]))
                track_id = int(float(row[1]))
                left, top, width, height = map(float, row[2:6])
            except ValueError as error:
                raise ValueError(f"Invalid MOT values at {mot_path}:{line_number}: {row}") from error
            if track_id not in accepted_tracks:
                stats["mot_rows_rejected_track"] += 1
                continue
            if frame not in available_frames:
                stats["mot_rows_without_extracted_frame"] += 1
                continue
            if width <= 0 or height <= 0:
                raise ValueError(f"Non-positive box at {mot_path}:{line_number}: {row}")
            boxes[frame].append(MotBox(track_id, left, top, width, height))
            stats["mot_rows_converted"] += 1
    return boxes, stats


def validate_box(box: MotBox, image_width: int, image_height: int, context: str) -> None:
    right = box.left + box.width
    bottom = box.top + box.height
    if box.left < 0 or box.top < 0 or right > image_width or bottom > image_height:
        raise ValueError(
            f"Box outside image in {context}: "
            f"left={box.left}, top={box.top}, width={box.width}, height={box.height}, "
            f"image={image_width}x{image_height}"
        )


def yolo_line(box: MotBox, image_width: int, image_height: int) -> str:
    centre_x = (box.left + box.width / 2) / image_width
    centre_y = (box.top + box.height / 2) / image_height
    width = box.width / image_width
    height = box.height / image_height
    return f"0 {centre_x:.6f} {centre_y:.6f} {width:.6f} {height:.6f}"


def convert_flight(flight_dir: Path, split: str, out: Path) -> tuple[list[ConvertedImage], Counter]:
    flight_dir = flight_dir.resolve()
    flight_id, manifest = load_manifest(flight_dir)
    rgb_dir = flight_dir / "rgb"
    labels_dir = flight_dir / "labels"
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"Missing RGB directory: {rgb_dir}")

    image_paths = sorted(rgb_dir.glob("*.jpg"))
    if not image_paths:
        raise ValueError(f"No RGB JPG images found in {rgb_dir}")
    try:
        available_frames = {int(path.stem) for path in image_paths}
    except ValueError as error:
        raise ValueError(f"RGB image names must be numeric frame IDs in {rgb_dir}") from error
    if len(available_frames) != len(image_paths):
        raise ValueError(f"Duplicate numeric frame IDs found in {rgb_dir}")

    labels_json = labels_dir / f"{flight_id}_labels.json"
    mot_path = labels_dir / f"{flight_id}_rgb_mot.txt"
    accepted_tracks, track_stats = accepted_rgb_tracks(labels_json)
    boxes_by_frame, mot_stats = parse_mot(mot_path, accepted_tracks, available_frames)

    converted: list[ConvertedImage] = []
    stats = Counter(track_stats)
    stats.update(mot_stats)
    declared_frames = manifest.get("frames", {}).get("rgb")
    if declared_frames is not None and int(declared_frames) != len(image_paths):
        raise ValueError(
            f"Flight {flight_id} manifest declares {declared_frames} RGB frames, "
            f"but {len(image_paths)} were found"
        )

    for source in image_paths:
        frame = int(source.stem)
        name = f"flight{flight_id}_{source.stem}"
        output_image = out / "images" / split / f"{name}.jpg"
        output_label = out / "labels" / split / f"{name}.txt"
        boxes = sorted(boxes_by_frame.get(frame, []), key=lambda box: box.track_id)
        with Image.open(source) as image:
            width, height = image.size
        for box in boxes:
            validate_box(box, width, height, f"flight {flight_id}, frame {frame}")
        shutil.copy2(source, output_image)
        lines = [yolo_line(box, width, height) for box in boxes]
        output_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        converted.append(
            ConvertedImage(flight_id, frame, split, source, output_image, width, height, boxes)
        )
        stats["images"] += 1
        stats["positive_images" if boxes else "negative_images"] += 1
        stats["boxes"] += len(boxes)
    return converted, stats


def evenly_spaced(records: list[ConvertedImage], count: int) -> list[ConvertedImage]:
    if count <= 0 or not records:
        return []
    if count >= len(records):
        return records
    if count == 1:
        return [records[len(records) // 2]]
    return [records[round(index * (len(records) - 1) / (count - 1))] for index in range(count)]


def make_previews(records: list[ConvertedImage], out: Path, maximum: int) -> int:
    if not records or maximum <= 0:
        return 0
    positives = [record for record in records if record.boxes]
    negatives = [record for record in records if not record.boxes]
    if positives and negatives:
        positive_count = min(len(positives), max(1, round(maximum * 2 / 3)))
        negative_count = min(len(negatives), maximum - positive_count)
    else:
        positive_count = min(len(positives), maximum)
        negative_count = min(len(negatives), maximum - positive_count)
    selected = evenly_spaced(positives, positive_count) + evenly_spaced(negatives, negative_count)
    selected.sort(key=lambda record: (record.flight_id, record.frame))

    for record in selected:
        with Image.open(record.source).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for box in record.boxes:
                draw.rectangle(
                    [box.left, box.top, box.left + box.width, box.top + box.height],
                    outline="lime",
                    width=3,
                )
                draw.text((box.left, max(0, box.top - 12)), f"deer t{box.track_id}", fill="lime")
            if not record.boxes:
                draw.rectangle([0, 0, 235, 24], fill="black")
                draw.text((5, 5), "NO LABELLED RGB DEER", fill="yellow")
            destination = out / "preview" / record.split / record.output.name
            image.save(destination, quality=90)
    return len(selected)


def write_manifest(records: list[ConvertedImage], out: Path) -> None:
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "flight_id",
                "frame",
                "split",
                "source_image",
                "output_image",
                "width",
                "height",
                "n_boxes",
                "track_ids",
            ]
        )
        for record in sorted(records, key=lambda item: (item.split, item.flight_id, item.frame)):
            writer.writerow(
                [
                    record.flight_id,
                    record.frame,
                    record.split,
                    str(record.source),
                    str(record.output),
                    record.width,
                    record.height,
                    len(record.boxes),
                    ";".join(str(box.track_id) for box in record.boxes),
                ]
            )


def main() -> None:
    args = parse_args()
    if args.preview_count < 0:
        raise ValueError("--preview-count must be non-negative")

    train_dirs = [path.resolve() for path in args.train_flight]
    val_dirs = [path.resolve() for path in args.val_flight]
    overlap = set(train_dirs) & set(val_dirs)
    if overlap:
        raise ValueError(f"Flights cannot be in both train and val: {sorted(map(str, overlap))}")

    flight_ids: dict[str, str] = {}
    for split, directories in (("train", train_dirs), ("val", val_dirs)):
        for directory in directories:
            flight_id, _ = load_manifest(directory)
            if flight_id in flight_ids:
                raise ValueError(
                    f"Flight ID {flight_id} appears more than once: {flight_ids[flight_id]} and {directory}"
                )
            flight_ids[flight_id] = f"{split}:{directory}"

    out = args.out.resolve()
    prepare_output(out, args.overwrite)
    all_records: list[ConvertedImage] = []
    summary: dict[str, object] = {
        "class_names": {"0": "deer"},
        "selection_policy": "all extracted RGB frames; accepted RGB red-deer tracks only",
        "splits": {},
    }

    for split, directories in (("train", train_dirs), ("val", val_dirs)):
        split_records: list[ConvertedImage] = []
        split_stats: Counter = Counter()
        per_flight: dict[str, dict[str, int]] = {}
        for directory in directories:
            records, stats = convert_flight(directory, split, out)
            split_records.extend(records)
            split_stats.update(stats)
            flight_id = records[0].flight_id
            per_flight[flight_id] = dict(sorted(stats.items()))
        previews = make_previews(split_records, out, args.preview_count)
        split_stats["previews"] = previews
        summary["splits"][split] = {
            "flights": per_flight,
            "totals": dict(sorted(split_stats.items())),
        }
        all_records.extend(split_records)

    write_manifest(all_records, out)
    (out / "data.yaml").write_text(
        "train: images/train\nval: images/val\nnc: 1\nnames: ['deer']\n", encoding="utf-8"
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Converted dataset: {out}")


if __name__ == "__main__":
    main()
