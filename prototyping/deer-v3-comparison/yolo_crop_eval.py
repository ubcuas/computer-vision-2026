"""Compare YOLO26s full-frame inference with overlapping crops on Deer v3."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataloaders.deer_shared.common import digest
from dataloaders.deer_shared.metrics import detection_metrics, iou

THRESHOLDS = (0.31, 0.25, 0.20, 0.15, 0.10, 0.075, 0.05, 0.025, 0.01, 0.001)


def views(frame):
    """Four overlapping crops covering the frame. Each spans 60% per axis."""
    height, width = frame.shape[:2]
    crop_w, crop_h = (3 * width + 4) // 5, (3 * height + 4) // 5
    for y in dict.fromkeys((0, height - crop_h)):
        for x in dict.fromkeys((0, width - crop_w)):
            yield frame[y:y + crop_h, x:x + crop_w], x, y


def merge_boxes(rows, overlap=0.55):
    kept = []
    for row in sorted(rows, key=lambda item: -item['score']):
        if all(iou(row['bbox'], other['bbox']) < overlap for other in kept):
            kept.append(row)
    return kept


def predict(model, frame, image_id, device, x=0, y=0):
    result = model.predict(frame, imgsz=1280, device=device,
                           conf=0.001, max_det=300, verbose=False)[0]
    height, width = frame.shape[:2]
    rows = []
    for x1, y1, x2, y2, score, label in result.boxes.data.cpu().tolist():
        x1, x2 = [max(0., min(width, v)) for v in (x1, x2)]
        y1, y2 = [max(0., min(height, v)) for v in (y1, y2)]
        if label == 0 and x2 > x1 and y2 > y1:
            rows.append(dict(image_id=image_id, category_id=1, score=score,
                             bbox=[x1 + x, y1 + y, x2 - x1, y2 - y1]))
    return rows


def measure(gt, predictions, threshold):
    result = detection_metrics(gt, predictions, threshold)
    boxes = [r['predicted'] for r in result.pop('per_image')]
    result['boxes_per_frame'] = statistics.mean(boxes)
    result['p95_boxes'] = sorted(boxes)[(95 * len(boxes) + 99) // 100 - 1]
    return result


def main():
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--sources', nargs='+', default=['bambi', 'mississippi_decoy'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    model = YOLO(str(args.weights))
    if list(model.names.values()) != ['deer']:
        raise ValueError('Expected the trained one-class deer checkpoint')
    (args.output / 'config.json').write_text(json.dumps(dict(
        weights_sha256=digest(args.weights), device=args.device, precision='fp32',
        torch=torch.__version__, ultralytics=ultralytics.__version__,
        image_size=1280, crop_fraction=0.6, merge_iou=0.55,
        sources=args.sources, thresholds=THRESHOLDS), indent=2))

    curves = []
    for split in ('val', 'test'):
        gt = json.loads((args.data / 'annotations' / f'{split}.coco.json').read_text())
        groups = defaultdict(list)
        for im in gt['images']:
            source = Path(im['file_name']).name.split('__', 1)[0]
            if source in args.sources:
                groups[source].append(im)
        if set(groups) != set(args.sources):
            raise ValueError(f'Missing sources in {split}')
        predictions = dict(full=[], crops=[], combined=[])
        timings = defaultdict(list)
        frames = [im for group in groups.values() for im in group]
        for index, im in enumerate(frames):
            frame = cv2.imread(str(args.data / 'images' / im['file_name']))
            if frame is None or frame.shape[:2] != (im['height'], im['width']):
                raise ValueError(f'Cannot decode expected dimensions of {im["file_name"]}')
            k = im['id']
            if index == 0:  # Warm the predictor before timing.
                predict(model, frame, k, args.device)
                crop, x, y = next(views(frame))
                predict(model, crop, k, args.device, x, y)
            start = time.perf_counter()
            full = predict(model, frame, k, args.device)
            mid = time.perf_counter()
            crops = [p for crop, x, y in views(frame)
                     for p in predict(model, crop, k, args.device, x, y)]
            merged_crops = merge_boxes(crops)
            cropped = time.perf_counter()
            combined = merge_boxes(full + crops)
            end = time.perf_counter()
            for mode, rows, elapsed in (
                ('full', full, mid - start),
                ('crops', merged_crops, cropped - mid),
                ('combined', combined, (cropped - start) + (end - cropped)),
            ):
                predictions[mode].extend(rows)
                timings[mode].append(elapsed * 1000)
            if (index + 1) % 10 == 0 or index + 1 == len(frames):
                print(f'{split}: {index + 1}/{len(frames)} frames', flush=True)

        for mode, rows in predictions.items():
            (args.output / f'{split}_{mode}_predictions.json').write_text(json.dumps(rows))
            for source, images in {**groups, 'all': frames}.items():
                ids = {im['id'] for im in images}
                subset = dict(images=images, annotations=[
                    a for a in gt['annotations'] if a['image_id'] in ids])
                selected = [p for p in rows if p['image_id'] in ids]
                for threshold in THRESHOLDS:
                    curves.append(dict(split=split, source=source, mode=mode,
                                       **measure(subset, selected, threshold)))
        (args.output / f'{split}_timings.json').write_text(json.dumps({
            mode: dict(median_ms=statistics.median(times),
                       p95_ms=sorted(times)[(95 * len(times) + 99) // 100 - 1])
            for mode, times in timings.items()}, indent=2))
        with (args.output / 'sweep.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)


if __name__ == '__main__':
    main()
