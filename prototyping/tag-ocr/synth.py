"""Synthetic ear-tag crops for training and stress-testing a tag-ID recognizer.

The appearance is modelled on CONOPS v1.0 Figure 2: a white laminated paper
card, portrait orientation, punched hole near the top, and two black capital
alphanumeric characters in a Times-style serif. Proportions were measured from
that figure (card width ~2.6x and height ~3.9x the character height). Anything
not stated in the CONOPS (font face, exact card size, how tags hang) is an
assumption, so training mixes several fonts and the evaluation keeps unseen
fonts as a separate slice (see RULE-003 in docs/competition/ambiguities.md).

Each sample imitates what a tag detector would hand the recognizer: a loose
crop around one tag at a controlled character height in pixels, after
viewpoint warp, blur, noise, glare and JPEG compression.

    python synth.py --out ../../artifacts/tag-ocr/preview --n 64 --sheet
"""
import argparse
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Fonts are looked up by file name in these directories. Missing fonts are skipped.
FONT_DIRS = [
    os.environ.get("TAG_FONT_DIR", ""),
    "C:/Windows/Fonts",
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/msttcorefonts",
    "/Library/Fonts",
    "/System/Library/Fonts/Supplemental",
]
# Times-like serif first: it is what Figure 2 shows.
TRAIN_FONTS = [
    "times.ttf", "timesbd.ttf", "Times New Roman.ttf", "LiberationSerif-Regular.ttf",
    "LiberationSerif-Bold.ttf", "cambria.ttc", "BOOKOS.TTF", "constan.ttf",
    "DejaVuSerif.ttf", "arial.ttf", "arialbd.ttf", "calibri.ttf", "LiberationSans-Regular.ttf",
    "DejaVuSans.ttf", "consola.ttf", "bahnschrift.ttf",
]
# Never used in training: unseen serifs and sans faces.
HELDOUT_FONTS = ["georgia.ttf", "GARA.TTF", "verdana.ttf", "Candara.ttf",
                 "Georgia.ttf", "LiberationMono-Regular.ttf", "FreeSerif.ttf", "FreeSans.ttf"]
# The Figure 2 look. Liberation Serif is metric-compatible with Times New Roman (for Linux boxes).
TIMES_FONTS = ["times.ttf", "Times New Roman.ttf", "LiberationSerif-Regular.ttf"]
FONT_SETS = {"train": TRAIN_FONTS, "heldout": HELDOUT_FONTS, "times": TIMES_FONTS}

IN_H, IN_W = 96, 64   # recognizer input: portrait, matching the tag's card shape

RENDER_CAP_PX = 64          # character height when drawing the high-resolution card
CARD_W, CARD_H = 2.6, 3.9   # card size in character heights (from Figure 2)
TEXT_CY = 0.57              # text centre, fraction of card height from the top


def find_fonts(names):
    """Resolve font file names (or a FONT_SETS key) to paths; first Times match only for 'times'."""
    key = names if isinstance(names, str) else None
    names = FONT_SETS[key] if key else names
    found = []
    for n in names:
        for d in FONT_DIRS:
            if d and (Path(d) / n).is_file() and str(Path(d) / n) not in found:
                found.append(str(Path(d) / n))
                break
    if key == "times":
        found = found[:1]
    if not found:
        raise FileNotFoundError(f"none of {names} found in {FONT_DIRS}; set TAG_FONT_DIR "
                                "(on Ubuntu: apt install fonts-liberation fonts-dejavu fonts-freefont-ttf)")
    return found


def to_input(gray):
    """Letterbox a grayscale crop of any size to IN_H x IN_W uint8 (recognizer input)."""
    h, w = gray.shape[:2]
    s = min(IN_H / h, IN_W / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    im = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR if s > 1 else cv2.INTER_AREA)
    out = np.full((IN_H, IN_W), int(np.median(gray)), np.uint8)
    y, x = (IN_H - nh) // 2, (IN_W - nw) // 2
    out[y:y + nh, x:x + nw] = im
    return out


def _font_for_cap(path, cap_px):
    """Pick a point size whose capital 'H' is cap_px tall."""
    f = ImageFont.truetype(path, 100)
    l, t, r, b = f.getbbox("H")
    return ImageFont.truetype(path, max(8, round(100 * cap_px / (b - t))))


def render_card(code, font_path, rng):
    """Frontal high-resolution tag on a transparent canvas. Returns RGBA uint8."""
    S = RENDER_CAP_PX
    w, h = int(CARD_W * S * rng.uniform(0.9, 1.15)), int(CARD_H * S * rng.uniform(0.9, 1.1))
    pad = int(0.08 * S)                       # clear laminate border around the paper
    canvas = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    lam = tuple(int(v) for v in rng.uniform(215, 245, 3)) + (int(rng.uniform(90, 170)),)
    d.rounded_rectangle([0, 0, w + 2 * pad - 1, h + 2 * pad - 1], radius=pad, fill=lam)
    paper = int(rng.uniform(225, 255))
    tint = rng.uniform(-8, 8, 3)
    d.rectangle([pad, pad, pad + w, pad + h], fill=tuple(int(np.clip(paper + t, 0, 255)) for t in tint) + (255,))
    hr = 0.13 * S
    hx, hy = pad + w / 2 + rng.uniform(-0.05, 0.05) * w, pad + 0.1 * h
    d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(0, 0, 0, 0))

    font = _font_for_cap(font_path, S * rng.uniform(0.95, 1.1))
    ink = int(rng.uniform(5, 60))
    tw = d.textlength(code, font=font)
    l, t, r, b = font.getbbox("H")
    x = pad + (w - tw) / 2 + rng.uniform(-0.04, 0.04) * w
    y = pad + TEXT_CY * h - (t + b) / 2 + rng.uniform(-0.04, 0.04) * h
    d.text((x, y), code, font=font, fill=(ink, ink, ink, 255))
    return np.array(canvas)


def background(hw, rng, bg_images):
    h, w = hw
    if bg_images and rng.random() < 0.7:
        img = bg_images[rng.integers(len(bg_images))]
        if img.shape[0] > h and img.shape[1] > w:
            y, x = rng.integers(img.shape[0] - h), rng.integers(img.shape[1] - w)
            return img[y:y + h, x:x + w].astype(np.float32)
    # Procedural: decoy browns, dry/green grass, sky and shadow.
    palette = [(120, 85, 55), (150, 110, 70), (90, 65, 45), (170, 160, 110), (95, 120, 60),
               (60, 85, 40), (180, 195, 210), (70, 70, 75), (200, 180, 140)]
    base = np.array(palette[rng.integers(len(palette))], np.float32) * rng.uniform(0.7, 1.2)
    low = cv2.resize(rng.normal(0, 25, (4, 4, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    hi = rng.normal(0, rng.uniform(2, 18), (h, w, 3)).astype(np.float32)
    return np.clip(base + low + hi, 0, 255)


def view_homography(w, h, yaw, pitch, roll, f_scale=1.6):
    """Homography for rotating a w x h planar card in 3D and re-projecting it."""
    f = f_scale * max(w, h)
    y, p, r = map(math.radians, (yaw, pitch, roll))
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    R = Rz @ Rx @ Ry
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    pts3 = np.c_[src - [w / 2, h / 2], np.zeros(4)] @ R.T + [0, 0, f]
    dst = (pts3[:, :2] * f / pts3[:, 2:3]).astype(np.float32)
    dst -= dst.min(0)
    return cv2.getPerspectiveTransform(src, dst), dst


def make_sample(code, char_px, font_path, rng, bg_images=(), hard=1.0):
    """One recognizer input crop (uint8 grayscale) with characters ~char_px tall.

    `hard` scales the strength of viewpoint and image degradations (0 = clean frontal).
    """
    card = render_card(code, font_path, rng)
    ch, cw = card.shape[:2]
    yaw = rng.uniform(-55, 55) * hard
    pitch = rng.uniform(-40, 40) * hard
    roll = rng.normal(0, 12) * hard
    H, dst = view_homography(cw, ch, yaw, pitch, roll)
    ow, oh = int(np.ceil(dst[:, 0].max())) + 1, int(np.ceil(dst[:, 1].max())) + 1
    warped = cv2.warpPerspective(card, H, (ow, oh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))

    # Loose crop, as from an imperfect tag detector.
    m = rng.uniform(0.05, 0.35, 4)
    L, T, R, B = (int(m[0] * ow), int(m[1] * oh), int(m[2] * ow), int(m[3] * oh))
    cH, cW = oh + T + B, ow + L + R
    bg = background((cH, cW), rng, bg_images)
    a = warped[..., 3:4].astype(np.float32) / 255
    region = bg[T:T + oh, L:L + ow]
    bg[T:T + oh, L:L + ow] = region * (1 - a) + warped[..., :3].astype(np.float32) * a

    # Glare on the laminate.
    if rng.random() < 0.35 * hard:
        gx, gy = rng.uniform(L, L + ow), rng.uniform(T, T + oh)
        sx, sy = rng.uniform(0.1, 0.5) * ow, rng.uniform(0.05, 0.3) * oh
        yy, xx = np.mgrid[0:cH, 0:cW]
        g = np.exp(-(((xx - gx) / sx) ** 2 + ((yy - gy) / sy) ** 2))
        bg += (g * rng.uniform(40, 160))[..., None]
    # Illumination gradient / partial shadow.
    if rng.random() < 0.4 * hard:
        k = rng.uniform(0.45, 1.0)
        ramp = np.linspace(1, k, cW if rng.random() < 0.5 else cH, dtype=np.float32)
        bg *= ramp[None, :, None] if ramp.size == cW else ramp[:, None, None]
    bg = np.clip(bg * rng.uniform(0.6, 1.25) + rng.uniform(-30, 20), 0, 255).astype(np.uint8)

    # Downscale so the frontal character height is char_px.
    scale = char_px / RENDER_CAP_PX
    small = cv2.resize(bg, (max(4, round(cW * scale)), max(4, round(cH * scale))), interpolation=cv2.INTER_AREA)
    small = small.astype(np.float32)
    sig = rng.uniform(0, 1.0) * hard
    if sig > 0.15:
        small = cv2.GaussianBlur(small, (0, 0), sig)
    if rng.random() < 0.3 * hard:
        L_ = int(rng.integers(2, 5))
        kern = np.zeros((L_, L_), np.float32)
        kern[L_ // 2] = 1 / L_
        M = cv2.getRotationMatrix2D(((L_ - 1) / 2, (L_ - 1) / 2), rng.uniform(0, 180), 1)
        small = cv2.filter2D(small, -1, cv2.warpAffine(kern, M, (L_, L_)))
    small += rng.normal(0, rng.uniform(0, 8) * hard, small.shape)
    small = np.clip(small, 0, 255).astype(np.uint8)
    q = int(rng.uniform(35, 95)) if hard > 0 else 95
    small = cv2.imdecode(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def random_code(rng):
    return CHARSET[rng.integers(36)] + CHARSET[rng.integers(36)]


def load_bg_images(bg_dir, limit=200):
    if not bg_dir:
        return []
    paths = [p for p in Path(bg_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")][:limit]
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is not None:
            out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--px", type=float, nargs=2, default=[5, 24], help="character height range in pixels")
    ap.add_argument("--fonts", choices=sorted(FONT_SETS), default="train")
    ap.add_argument("--bg-dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sheet", action="store_true", help="also write an upscaled contact sheet")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    fonts = find_fonts(a.fonts)
    bgs = load_bg_images(a.bg_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i in range(a.n):
        code, px = random_code(rng), rng.uniform(*a.px)
        img = make_sample(code, px, fonts[rng.integers(len(fonts))], rng, bgs)
        cv2.imwrite(str(out / f"{i:05d}_{code}_{px:.1f}px.png"), img)
        tiles.append((img, f"{code} {px:.0f}px"))
    if a.sheet:
        cell = 128
        cols = 8
        rows = math.ceil(len(tiles) / cols)
        sheet = np.full((rows * (cell + 16), cols * cell), 255, np.uint8)
        for k, (img, lab) in enumerate(tiles):
            r, c = divmod(k, cols)
            s = cell / max(img.shape)
            im = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_NEAREST)
            sheet[r * (cell + 16):r * (cell + 16) + im.shape[0], c * cell:c * cell + im.shape[1]] = im
            cv2.putText(sheet, lab, (c * cell + 2, r * (cell + 16) + cell + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
        cv2.imwrite(str(out / "_sheet.png"), sheet)
    print(f"wrote {a.n} samples to {out} using {len(fonts)} fonts")


if __name__ == "__main__":
    main()
