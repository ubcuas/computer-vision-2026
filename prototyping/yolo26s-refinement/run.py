"""Bounded YOLO26s refinement. Preparation is CPU-only; GPU work is explicit."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'prototyping/deer-v3-comparison')]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + '.partial')
    pending.write_text(json.dumps(value, indent=2))
    pending.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def capture(cmd):
    return subprocess.check_output(cmd, text=True, cwd=REPO, timeout=20).strip()


def source(image):
    return Path(image['file_name']).name.split('__', 1)[0]


def inventory(data):
    """Require the canonical split files/annotations, without hashing every photo."""
    frozen = REPO / 'dataloaders/deer_v3'
    result = {}
    expected_hashes = read(frozen / 'checksums.json')
    if (data / 'splits/train_decoy_4x.txt').read_text() != (frozen / 'splits/train_decoy_4x.txt').read_text():
        raise ValueError('Not the frozen 4x-decoy training list')
    sets = {}
    for split, expected in [('train', 3193), ('val', 358), ('test', 238)]:
        rels = (data / f'splits/{split}.txt').read_text().splitlines()
        if rels != (frozen / f'splits/{split}.txt').read_text().splitlines():
            raise ValueError(f'{split}: not the frozen v3 split')
        ann_path = f'annotations/{split}.coco.json'
        if sha(data / ann_path) != expected_hashes[ann_path]:
            raise ValueError(f'{split}: COCO annotation content changed')
        gt = read(data / ann_path)
        if len(rels) != expected or len(gt['images']) != expected:
            raise ValueError(f'Unexpected {split} image count')
        names = {str(Path('images') / im['file_name']) for im in gt['images']}
        if names != set(rels):
            raise ValueError(f'{split}: COCO images disagree with split list')
        if {c['id']: c['name'] for c in gt['categories']} != {1: 'deer'}:
            raise ValueError('Expected COCO deer category 1')
        for rel in rels:
            image = data / rel
            label = data / 'labels' / image.parent.name / image.with_suffix('.txt').name
            if not image.is_file() or not label.is_file():
                raise FileNotFoundError(image if not image.is_file() else label)
            label_rel = label.relative_to(data).as_posix()
            if sha(label) != expected_hashes[label_rel]:
                raise ValueError(f'YOLO label content changed: {label_rel}')
        sets[split] = set(rels)
        result[split] = dict(images=expected, boxes=len(gt['annotations']),
                             by_source=dict(Counter(map(source, gt['images']))),
                             split_sha256=sha(data / f'splits/{split}.txt'),
                             coco_sha256=sha(data / f'annotations/{split}.coco.json'))
    if any(sets[a] & sets[b] for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]):
        raise ValueError('Split overlap')
    result['weighted_train_sha256'] = sha(data / 'splits/train_decoy_4x.txt')
    return result


def tiles(data, run, cfg):
    """Supplement training only with crops; never mine validation/test misses."""
    from PIL import Image, ImageDraw
    gt = read(data / 'annotations/train.coco.json')
    boxes = {}
    for ann in gt['annotations']:
        boxes.setdefault(ann['image_id'], []).append(ann['bbox'])
    images = sorted(gt['images'], key=lambda im: im['file_name'])
    rng = random.Random(cfg['seed'])
    rng.shuffle(images)
    counts, rows = Counter(), []
    previews = []
    for im in images:
        src = source(im)
        if counts[src] >= cfg['tiles_per_source']:
            continue
        w, h = im['width'], im['height']
        candidates = [b for b in boxes.get(im['id'], [])
                      if b[2] * b[3] / (w * h) <= cfg['tile_max_box_area_fraction']]
        if not candidates:
            continue
        x, y, bw, bh = rng.choice(candidates)
        cw, ch = round(w * cfg['tile_fraction']), round(h * cfg['tile_fraction'])
        left = max(0, min(w - cw, round(x + bw / 2 - cw / 2)))
        top = max(0, min(h - ch, round(y + bh / 2 - ch / 2)))
        clipped, invalid = [], False
        for bx, by, ww, hh in boxes[im['id']]:
            x1, y1 = max(left, bx), max(top, by)
            x2, y2 = min(left + cw, bx + ww), min(top + ch, by + hh)
            if x2 > x1 and y2 > y1:
                # Skip the entire crop if it produces an ambiguous deer sliver.
                if min(x2 - x1, y2 - y1) < 2 or (x2-x1)*(y2-y1)/(ww*hh) < .5:
                    invalid = True
                    break
                clipped.append([x1-left, y1-top, x2-x1, y2-y1])
        if invalid or not clipped:
            continue
        with Image.open(data / 'images' / im['file_name']) as original:
            if original.size != (w, h):
                raise ValueError(f'Dimensions disagree: {im["file_name"]}')
            tile = original.convert('RGB').crop((left, top, left+cw, top+ch))
        name = Path(im['file_name']).stem + '__tile.jpg'
        dest = run / 'prepared/images/train' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        tile.save(dest, quality=95)
        label = run / 'prepared/labels/train' / Path(name).with_suffix('.txt')
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(''.join(f'0 {(xx+ww/2)/cw:.9f} {(yy+hh/2)/ch:.9f} {ww/cw:.9f} {hh/ch:.9f}\n'
                                 for xx, yy, ww, hh in clipped))
        counts[src] += 1
        rows.append(dict(image=str(dest), original=im['file_name'], source=src,
                         crop_xywh=[left, top, cw, ch], boxes=clipped))
        if sum(p[0] == src for p in previews) < 4:
            draw = ImageDraw.Draw(tile)
            for xx, yy, ww, hh in clipped:
                draw.rectangle((xx, yy, xx+ww, yy+hh), outline='lime', width=3)
            tile.thumbnail((320, 200))
            previews.append((src, tile.copy()))
    sheet = Image.new('RGB', (1280, max(1, (len(previews)+3)//4)*230), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (src, tile) in enumerate(previews):
        x, y = (i % 4)*320, (i//4)*230
        sheet.paste(tile, (x, y+25))
        draw.text((x+4, y+4), src, fill='black')
    sheet.save(run / 'prepared/tile_review.jpg')
    write(run / 'prepared/tiles.json', rows)
    return [r['image'] for r in rows], dict(counts)


def prepare(args):
    if args.run.exists():
        raise FileExistsError('Use a fresh run directory; preparation never overwrites a run.')
    cfg = read(args.config)
    inv = inventory(args.data)
    if not args.weights.is_file():
        raise FileNotFoundError('Supply the saved YOLO26s deer best.pt; no automatic downloads.')
    args.run.mkdir(parents=True)
    write(args.run / 'recipe.json', cfg)
    subprocess.run([sys.executable, str(REPO / 'dataloaders/deer_v3/prepare.py'),
                    '--root', str(args.data)], check=True)
    tile_paths, tile_counts = tiles(args.data, args.run, cfg)
    base = (args.data / 'runtime/train_decoy_4x.txt').read_text().splitlines()
    (args.run / 'prepared/train.txt').write_text('\n'.join(base + tile_paths) + '\n')
    # Shared loader owns the original paths/weights; only train gains derived tiles.
    import yaml
    dataset = yaml.safe_load((args.data / 'runtime/data.yaml').read_text())
    dataset['train'] = str(args.run / 'prepared/train.txt')
    (args.run / 'prepared/data.yaml').write_text(yaml.safe_dump(dataset))
    snapshot = args.run / 'code_snapshot'
    snapshot.mkdir()
    for file in HERE.iterdir():
        if file.suffix in ('.py', '.ipynb', '.json', '.md'):
            (snapshot / file.name).write_bytes(file.read_bytes())
    prepared_hashes = {str(p.relative_to(args.run)): sha(p) for p in (args.run / 'prepared').rglob('*') if p.is_file()}
    write(args.run / 'manifest.json', dict(prepared_hashes=prepared_hashes, recipe_sha256=sha(args.run / 'recipe.json'), data=str(args.data), weights=str(args.weights),
          weights_sha256=sha(args.weights), inventory=inv, tiles=tile_counts,
          base_weighted_train=len(base), effective_train=len(base)+len(tile_paths),
          revision=capture(['git', 'rev-parse', 'HEAD']),
          working_tree=capture(['git', 'status', '--short']),
          runner_sha256=sha(Path(__file__)), prepared_at=time.time(),
          note='Fine-tune existing deer model, not a COCO restart. Historical test has been inspected.'))
    print(json.dumps(dict(tiles=tile_counts, effective_train=len(base)+len(tile_paths),
                          review=str(args.run / 'prepared/tile_review.jpg')), indent=2))


def gpu_idle():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU fallback for training.')
    processes = capture(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'])
    active = []
    for row in processes.splitlines():
        parts = [v.strip() for v in row.split(',')]
        # NVIDIA lists this server's remote desktop renderer as a compute app.
        if len(parts) == 3 and parts[1] == '/usr/libexec/gnome-remote-desktop-daemon':
            try:
                if int(parts[2].split()[0]) <= 256:
                    continue
            except ValueError:
                pass
        active.append(row)
    if active:
        raise RuntimeError('GPU has compute processes; inspect before training:\n' + '\n'.join(active))


def verify_run(args):
    manifest = read(args.run / 'manifest.json')
    if inventory(args.data) != manifest['inventory'] or str(args.data) != manifest['data']:
        raise ValueError('Dataset/split metadata changed since preparation')
    if sha(args.run / 'recipe.json') != manifest['recipe_sha256']:
        raise ValueError('Recipe changed after preparation; use a separate run')
    for rel, checksum in manifest['prepared_hashes'].items():
        if sha(args.run / rel) != checksum:
            raise ValueError(f'Prepared training input changed: {rel}')
    return manifest


def train(args):
    import fcntl
    from ultralytics import YOLO
    manifest = verify_run(args)
    cfg = read(args.run / 'recipe.json')
    with (args.run / 'gpu.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.run / 'training_complete.json').exists():
            raise RuntimeError('Training already complete; evaluate the saved weights.')
        gpu_idle()
        smoke = args.mode == 'smoke'
        last = args.run / 'fit/weights/last.pt'
        if not smoke and not (args.run / 'smoke_complete.json').exists():
            raise RuntimeError('Run the GPU smoke test first.')
        if last.exists() and not args.resume and not smoke:
            raise RuntimeError('Saved training exists; use --resume, never restart implicitly.')
        if args.resume and (not last.exists() or smoke):
            raise RuntimeError('--resume needs an existing fit/weights/last.pt and train mode')
        weights = last if args.resume else Path(manifest['weights'])
        if not args.resume and sha(weights) != manifest['weights_sha256']:
            raise ValueError('Initial checkpoint changed')
        model = YOLO(str(weights))
        if list(model.names.values()) != ['deer']:
            raise ValueError('Expected the saved one-class deer detector')
        (args.run / 'pip-freeze.txt').write_text(capture([sys.executable, '-m', 'pip', 'freeze']))
        write(args.run / 'hardware.json', dict(gpu=capture(['nvidia-smi']), timestamp=time.time()))
        if args.resume:
            model.train(resume=True)
        else:
            options = {k: v for k, v in cfg.items() if not k.startswith(('tile_', 'tiles_', 'review_'))}
            options.update(data=str(args.run / 'prepared/data.yaml'),
                           project=str(args.run), name='smoke' if smoke else 'fit', exist_ok=False)
            if smoke:
                options.update(epochs=1, fraction=.02, val=False, plots=False)
            model.train(**options)
        folder = args.run / ('smoke' if smoke else 'fit')
        best = folder / 'weights/best.pt'
        if not best.exists() or not (folder / 'weights/last.pt').exists():
            raise RuntimeError('Training returned without expected saved weights')
        write(args.run / ('smoke_complete.json' if smoke else 'training_complete.json'),
              dict(best=str(best), sha256=sha(best), finished_at=time.time(), resumed=args.resume,
                   selection='Ultralytics native validation fitness; review with shared evaluator'))


def review_report(data, output, calibration=None):
    import numpy as np
    from dataloaders.deer_shared.metrics import detection_metrics
    gt = read(data / f'annotations/{output.name}.coco.json')
    pred = read(output / 'predictions.json')
    ids = {}
    for im in gt['images']:
        ids.setdefault(source(im), set()).add(im['id'])
    def measure(t):
        m = detection_metrics(gt, pred, t)
        slices = {}
        for src, subset_ids in ids.items():
            subset = dict(gt, images=[im for im in gt['images'] if im['id'] in subset_ids],
                          annotations=[a for a in gt['annotations'] if a['image_id'] in subset_ids])
            result = detection_metrics(subset, [p for p in pred if p['image_id'] in subset_ids], t)
            slices[src] = {k: v for k, v in result.items() if k != 'per_image'}
        m.pop('per_image')
        m['by_source'] = slices
        m['macro_source_recall'] = float(np.mean([s['recall'] for s in slices.values() if s['tp']+s['fn']]))
        m['fp_per_image'] = m['fp']/m['images']
        m['candidates_per_image'] = (m['tp']+m['fp'])/m['images']
        return m
    if calibration:
        result = measure(calibration['threshold'])
    else:
        cfg = read(output.parent / 'recipe.json')
        curve = [measure(float(t)) for t in np.unique(np.r_[.001, .005, np.arange(.01, 1, .01)])]
        feasible = [m for m in curve if m['fp_per_image'] <= cfg['review_max_fp_per_image']]
        if not feasible:
            raise RuntimeError('No validation threshold satisfies the candidate-review budget')
        reached = [m for m in feasible if m['macro_source_recall'] >= cfg['review_recall_target']]
        result = min(reached, key=lambda m: (m['fp_per_image'], -m['threshold'])) if reached else max(
            feasible, key=lambda m: (m['macro_source_recall'], -m['fp_per_image']))
        calibration = dict(threshold=result['threshold'], selected_on='validation only',
                           target=cfg['review_recall_target'], max_fp_per_image=cfg['review_max_fp_per_image'],
                           target_met=bool(reached), objective='macro source recall subject to FP/image budget',
                           curve=curve)
        write(output.parent / 'review_calibration.json', calibration)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([m['fp_per_image'] for m in curve], [m['macro_source_recall'] for m in curve], '.-')
        ax.axvline(cfg['review_max_fp_per_image'], color='grey', linestyle='--')
        ax.set(xlabel='False candidates / image', ylabel='Mean recall across sources',
               title='Validation: human-review workload vs deer recall')
        fig.tight_layout(); fig.savefig(output / 'review_tradeoff.png', dpi=150); plt.close(fig)
    write(output / 'review_metrics.json', result)
    return result


def evaluate(args):
    verify_run(args)
    from workflow import evaluate_split
    if not (args.run / 'training_complete.json').exists():
        raise RuntimeError('Finish training first')
    if args.test and not (args.run / 'review_calibration.json').exists():
        raise RuntimeError('Evaluate validation first; test never selects the threshold')
    split = 'test' if args.test else 'val'
    dest = args.run / split
    best = args.run / 'fit/weights/best.pt'
    if sha(best) != read(args.run / 'training_complete.json')['sha256']:
        raise ValueError('Saved best weights changed after training')
    if dest.exists() and not ((dest / 'predictions.json').exists() and (dest / 'profile.json').exists()):
        raise FileExistsError(f'Partial inference preserved at {dest}; inspect before retrying')
    env = dict(os.environ, PYTHONPATH=str(REPO), MPLBACKEND='Agg')
    if not dest.exists():
        gpu_idle()
        subprocess.run([sys.executable, str(REPO / 'prototyping/deer-v3-comparison/train_and_profile.py'),
                    'profile', 'yolo26s', '--data', str(args.data), '--run', str(args.run),
                    '--weights', str(args.run / 'fit/weights/best.pt'), '--split', split,
                    '--size', '1280', '--precision', 'fp32', '--output', str(dest)], check=True, env=env)
    evaluate_split(args.data, args.run, split)
    cal = read(args.run / 'review_calibration.json') if args.test else None
    result = review_report(args.data, dest, cal)
    print(json.dumps({k: v for k, v in result.items() if k != 'by_source'}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'smoke', 'train', 'evaluate'])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--weights', type=Path)
    p.add_argument('--config', type=Path, default=HERE / 'recipe.json')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--test', action='store_true', help='Explicit historical-test regression check after validation')
    args = p.parse_args()
    args.data = args.data.expanduser().resolve(); args.run = args.run.expanduser().resolve()
    if args.mode == 'prepare' and args.weights is None:
        p.error('prepare requires --weights pointing to saved YOLO26s deer best.pt')
    if args.weights:
        args.weights = args.weights.expanduser().resolve()
    if args.mode == 'prepare': prepare(args)
    elif args.mode in ('smoke', 'train'): train(args)
    else: evaluate(args)


if __name__ == '__main__':
    main()
