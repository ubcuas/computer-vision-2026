"""Convert the AWIR-Deer folder (YOLO .txt next to each .JPG, deer = class 1)
into the same dataset layout as decoy_survey, with deer remapped to class 0.

Only the RGB images in the top folder are used; the Thermal_archive_excluded_from_RGB
subfolder is skipped (different camera, our data is RGB).

Usage:
  python convert.py --src AWIR-Deer --out ../../prototyping/yolo-fine-tune/dataset_awir [--val 0.2]
"""
import argparse, glob, os, random, shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--val', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    # set(): Windows globbing is case-insensitive, so both patterns match the same files there
    imgs = sorted(set(glob.glob(os.path.join(a.src, '*.JPG')) + glob.glob(os.path.join(a.src, '*.jpg'))))
    order = imgs[:]
    random.Random(a.seed).shuffle(order)
    nval = round(len(order) * a.val)
    split = {p: ('val' if i < nval else 'train') for i, p in enumerate(order)}

    for d in ('images', 'labels'):
        shutil.rmtree(os.path.join(a.out, d), ignore_errors=True)
        for s in ('train', 'val'):
            os.makedirs(os.path.join(a.out, d, s))

    nbox = 0
    for p in imgs:
        name = os.path.splitext(os.path.basename(p))[0]
        lines = []
        for line in open(os.path.splitext(p)[0] + '.txt').read().split('\n'):
            if line.strip():
                cls, *xywh = line.split()
                assert cls == '1' and len(xywh) == 4, (p, line)
                lines.append(' '.join(['0'] + xywh))
        nbox += len(lines)
        s = split[p]
        shutil.copyfile(p, os.path.join(a.out, 'images', s, name + '.jpg'))
        with open(os.path.join(a.out, 'labels', s, name + '.txt'), 'w', newline='\n') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))

    with open(os.path.join(a.out, 'data.yaml'), 'w', newline='\n') as f:
        # No 'path:' key, so Ultralytics resolves images/ relative to this file on any machine.
        f.write("train: images/train\nval: images/val\nnc: 1\nnames: ['deer']\n")
    print('images:', len(imgs), 'val:', nval, 'boxes:', nbox)


if __name__ == '__main__':
    main()
