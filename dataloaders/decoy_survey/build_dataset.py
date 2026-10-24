"""Build a YOLO detection dataset from the decoy-survey PDF + point CSV.

CSV rows are (survey id, image, class, x, y). "SlideN.JPG" means PDF page N
(NOT the number printed on the slide), and x/y are pixel coords on the slide
rendered at 300 DPI (4000x2250). Each slide has one photo on a white page, so
we crop the photo out, shift the points into crop coordinates, and turn every
point into a box (fitted to the animal by colour, falling back to fixed size).

Usage:
  python build_dataset.py --pdf X.pdf --csv Y.csv --out dataset [--classes deer]
         [--negatives] [--val 0.2] [--render-dir DIR] [--fit sam|color|fixed]
Needs pdftoppm (poppler) on PATH, plus the packages in requirements.txt.
SAM boxes are cached in sam_boxes_cache.json next to this script, so a rebuild
only runs SAM for points it has not seen before.
"""
import argparse, csv, os, random, re, shutil, subprocess, collections
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

DPI = 300


def render(pdf, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    if not any(f.endswith('.jpg') for f in os.listdir(out_dir)):
        subprocess.run(['pdftoppm', '-r', str(DPI), '-jpeg', '-jpegopt', 'quality=95',
                        pdf, os.path.join(out_dir, 'p')], check=True)
    files = sorted(f for f in os.listdir(out_dir) if f.endswith('.jpg'))
    return {int(re.findall(r'(\d+)\.jpg$', f)[0]): os.path.join(out_dir, f) for f in files}


def photo_bbox(arr):
    """Bounding box of the photo = union of large non-white blobs.
    The slide-number text glyphs are small blobs, so they are dropped."""
    nonwhite = (arr < 235).any(axis=2)
    lab, n = ndimage.label(nonwhite)
    sizes = ndimage.sum(nonwhite, lab, range(1, n + 1))
    big = np.isin(lab, np.where(sizes >= 8000)[0] + 1)
    ys, xs = np.where(big)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def fit_box(arr, x, y, fixed, lo, hi):
    """Grow a box from 'non-grass' pixels (R >= G: tan, brown, white) connected
    to the point. Falls back to a fixed square if the blob looks wrong."""
    H, W = arr.shape[:2]
    x0, y0 = max(0, int(x - hi)), max(0, int(y - hi))
    x1, y1 = min(W, int(x + hi)), min(H, int(y + hi))
    win = arr[y0:y1, x0:x1].astype(int)
    R, G = win[..., 0], win[..., 1]
    mask = (R >= G - 3) & (win.sum(2) > 150)
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=3)
    lab, _ = ndimage.label(mask)
    px, py = int(x - x0), int(y - y0)
    k = 12
    patch = lab[max(0, py - k):py + k, max(0, px - k):px + k]
    ids = patch[patch > 0]
    if ids.size:
        cid = collections.Counter(ids.tolist()).most_common(1)[0][0]
        ys, xs = np.where(lab == cid)
        bx0, by0, bx1, by1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        w, h = bx1 - bx0, by1 - by0
        touches = bx0 == 0 or by0 == 0 or bx1 == lab.shape[1] or by1 == lab.shape[0]
        # touching the window edge on a clipped photo border is fine; elsewhere it means leakage
        edge_ok = (bx0 == 0 and x0 == 0) or (by0 == 0 and y0 == 0) or \
                  (bx1 == lab.shape[1] and x1 == W) or (by1 == lab.shape[0] and y1 == H)
        if w >= lo and h >= lo and (not touches or edge_ok) and len(xs) > 0.15 * w * h:
            pad = 0.08
            return (x0 + bx0 - pad * w, y0 + by0 - pad * h,
                    x0 + bx1 + pad * w, y0 + by1 + pad * h), True
    return (x - fixed / 2, y - fixed / 2, x + fixed / 2, y + fixed / 2), False


class SamBoxer:
    """Point-prompted Segment Anything: mask from the click -> box.
    Runs on a window around the point (SAM resizes input to 1024, so small
    windows keep small deer at decent resolution). Results cached to JSON."""

    def __init__(self, weights, cache_path, win=300):
        import json
        self.json, self.win, self.cache_path = json, win, cache_path
        self.cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        self.weights, self.model = weights, None

    def __call__(self, crop, x, y, key, lo, hi):
        if key not in self.cache:
            if self.model is None:
                from ultralytics import SAM
                self.model = SAM(self.weights)
            H, W = crop.shape[:2]
            R = self.win
            x0, y0 = max(0, int(x - R)), max(0, int(y - R))
            x1, y1 = min(W, int(x + R)), min(H, int(y + R))
            res = self.model(np.ascontiguousarray(crop[y0:y1, x0:x1, ::-1]),
                             points=[[x - x0, y - y0]], labels=[1], verbose=False)[0]
            b = res.boxes.xyxy.cpu().numpy()
            self.cache[key] = [float(b[0][0] + x0), float(b[0][1] + y0),
                               float(b[0][2] + x0), float(b[0][3] + y0)] if len(b) else None
            self.json.dump(self.cache, open(self.cache_path, 'w'))
            print('  sam', key, flush=True)
        b = self.cache[key]
        if b is None:
            return None
        w, h = b[2] - b[0], b[3] - b[1]
        # reject masks that grabbed a grass patch / whole window, or a speck
        if not (lo <= max(w, h) <= hi) or not (b[0] <= x <= b[2] and b[1] <= y <= b[3]):
            return None
        return tuple(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--render-dir', default=None, help='cache for 300 DPI page renders')
    ap.add_argument('--classes', default='deer', help="comma list, or 'all'")
    ap.add_argument('--negatives', action='store_true',
                    help='also include slides with none of the classes (empty label files)')
    ap.add_argument('--val', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fixed', type=float, default=120, help='fallback box side, px at 300 DPI')
    ap.add_argument('--fit', choices=['sam', 'color', 'fixed'], default='sam')
    ap.add_argument('--sam-weights', default='mobile_sam.pt')
    ap.add_argument('--sam-cache', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        'sam_boxes_cache.json'))
    a = ap.parse_args()
    sam = SamBoxer(a.sam_weights, a.sam_cache) if a.fit == 'sam' else None
    os.makedirs(a.out, exist_ok=True)

    rows = list(csv.DictReader(open(a.csv, newline='')))
    allc = sorted({r['class'] for r in rows})
    classes = allc if a.classes == 'all' else a.classes.split(',')
    cid = {c: i for i, c in enumerate(classes)}
    pts = collections.defaultdict(list)
    for r in rows:
        page = int(re.findall(r'\d+', r['image'])[0])
        pts[page].append((r['class'], float(r['x']), float(r['y'])))

    pages = render(a.pdf, a.render_dir or os.path.join(a.out, '_render'))
    keep = sorted(pages) if a.negatives else \
        [p for p in sorted(pages) if any(c in cid for c, _, _ in pts.get(p, []))]
    order = keep[:]
    random.Random(a.seed).shuffle(order)
    nval = round(len(order) * a.val)
    split = {p: ('val' if i < nval else 'train') for i, p in enumerate(order)}

    for d in ('images', 'labels', 'preview'):
        shutil.rmtree(os.path.join(a.out, d), ignore_errors=True)
    for s in ('train', 'val'):
        os.makedirs(os.path.join(a.out, 'images', s))
        os.makedirs(os.path.join(a.out, 'labels', s))
    os.makedirs(os.path.join(a.out, 'preview'))

    stats = collections.Counter()
    manifest = []
    for p in keep:
        arr = np.asarray(Image.open(pages[p]).convert('RGB'))
        cx0, cy0, cx1, cy1 = photo_bbox(arr)
        crop = arr[cy0:cy1, cx0:cx1]
        W, H = cx1 - cx0, cy1 - cy0
        name = f'slide{p:03d}'
        lines, boxes = [], []
        for c, x, y in pts.get(p, []):
            if c not in cid:
                continue
            x, y = x - cx0, y - cy0
            if a.fit == 'sam':
                b = sam(crop, x, y, f'{p}:{x + cx0:.2f}:{y + cy0:.2f}', lo=20, hi=350)
                fitted = b is not None
                (bx0, by0, bx1, by1) = b if fitted else (x - a.fixed / 2, y - a.fixed / 2,
                                                         x + a.fixed / 2, y + a.fixed / 2)
            elif a.fit == 'color':
                (bx0, by0, bx1, by1), fitted = fit_box(crop, x, y, a.fixed, lo=25, hi=260)
            else:
                (bx0, by0, bx1, by1), fitted = (x - a.fixed / 2, y - a.fixed / 2,
                                                x + a.fixed / 2, y + a.fixed / 2), False
            bx0, by0, bx1, by1 = max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)
            stats['fitted' if fitted else 'fallback'] += 1
            stats[c] += 1
            lines.append(f'{cid[c]} {(bx0 + bx1) / 2 / W:.6f} {(by0 + by1) / 2 / H:.6f} '
                         f'{(bx1 - bx0) / W:.6f} {(by1 - by0) / H:.6f}')
            boxes.append((c, bx0, by0, bx1, by1, fitted, x, y))
        s = split[p]
        img = Image.fromarray(crop)
        img.save(os.path.join(a.out, 'images', s, name + '.jpg'), quality=95)
        with open(os.path.join(a.out, 'labels', s, name + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        manifest.append((name, p, s, W, H, len(lines)))
        if boxes:
            d = ImageDraw.Draw(img)
            for c, bx0, by0, bx1, by1, fitted, x, y in boxes:
                d.rectangle([bx0, by0, bx1, by1], outline='yellow' if fitted else 'red', width=4)
                d.ellipse([x - 6, y - 6, x + 6, y + 6], fill='red')
                d.text((bx0, max(0, by0 - 40)), c, fill='yellow', font_size=36)
            img.save(os.path.join(a.out, 'preview', name + '.jpg'), quality=80)

    with open(os.path.join(a.out, 'data.yaml'), 'w', newline='\n') as f:
        # No 'path:' key, so Ultralytics resolves images/ relative to this file on any machine.
        f.write(f"train: images/train\nval: images/val\nnc: {len(classes)}\nnames: {classes}\n")
    with open(os.path.join(a.out, 'manifest.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file', 'pdf_page', 'split', 'width', 'height', 'n_boxes'])
        w.writerows(manifest)
    print('images:', len(keep), dict(collections.Counter(split.values())), dict(stats))


if __name__ == '__main__':
    main()
