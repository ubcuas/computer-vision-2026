"""Exact tag-ID accuracy versus character height in pixels.

This answers ENG-001 on synthetic data: below how many pixels does the ID
become unreadable? Combine the result with geometry.py to get the maximum
camera range for a given camera.

    python evaluate.py --model ../../artifacts/tag-ocr/run1/tagnet.pt

Slices (all fresh seeds, never seen in training):
  times    - Figure 2 look (Times / Liberation Serif), the expected case
  heldout  - fonts never used in training: robustness if organizers change font
  easy     - Times at half viewpoint/degradation strength (closer to a good oblique pass)
Writes eval.csv and eval.md next to the model.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import synth
from data import generate
from recognizer import batch_tensor, load

PX_GRID = [4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24]
SLICES = {"times": ("times", 1.0), "heldout": ("heldout", 1.0), "easy": ("times", 0.5)}


@torch.no_grad()
def score(model, X, Y, dev):
    probs1, probs2 = [], []
    for i in range(0, len(X), 2048):
        l1, l2 = model(batch_tensor(torch.as_tensor(X[i:i + 2048], device=dev), dev))
        probs1.append(l1.softmax(1).cpu())
        probs2.append(l2.softmax(1).cpu())
    p1, p2 = torch.cat(probs1), torch.cat(probs2)
    c1, i1 = p1.max(1)
    c2, i2 = p2.max(1)
    Y = torch.as_tensor(Y)
    ok = (i1 == Y[:, 0]) & (i2 == Y[:, 1])
    return ok.numpy(), (c1 * c2).numpy(), i1.numpy(), i2.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=1000, help="samples per slice per pixel height")
    ap.add_argument("--conf", type=float, default=0.9, help="confidence threshold for the selective-accuracy columns")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    model = load(a.model, dev)
    out = Path(a.model).parent
    cache = out.parent / "cache"

    rows = ["slice,char_px,n,exact_acc,coverage_at_conf,acc_at_conf"]
    table = {}
    confusions = Counter()
    for name, (fontset, hard) in SLICES.items():
        for k, px in enumerate(PX_GRID):
            seed = 900000 + 1000 * list(SLICES).index(name) + k
            X, Y, _ = generate(a.n, seed, px, px, fontset=fontset, hard=hard, workers=a.workers,
                               cache=cache / f"eval_{name}_px{px}_n{a.n}.npz")
            ok, conf, i1, i2 = score(model, X, Y, dev)
            keep = conf >= a.conf
            acc_sel = ok[keep].mean() if keep.any() else float("nan")
            rows.append(f"{name},{px},{a.n},{ok.mean():.4f},{keep.mean():.4f},{acc_sel:.4f}")
            table[(name, px)] = (ok.mean(), keep.mean(), acc_sel)
            if name == "times":
                for t, p in ((Y[:, 0], i1), (Y[:, 1], i2)):
                    for tt, pp in zip(t, p):
                        if tt != pp:
                            confusions[(synth.CHARSET[tt], synth.CHARSET[pp])] += 1
            print(rows[-1], flush=True)

    (out / "eval.csv").write_text("\n".join(rows) + "\n")
    md = [f"Exact two-character ID accuracy (n={a.n} per cell). "
          f"`sel` = accuracy on readings with confidence >= {a.conf} (`cov` = share kept).", "",
          "| char px | " + " | ".join(f"{s} acc | {s} cov / sel" for s in SLICES) + " |",
          "|---:|" + "---:|---:|" * len(SLICES)]
    for px in PX_GRID:
        md.append(f"| {px} | " + " | ".join(
            f"{table[(s, px)][0]:.1%} | {table[(s, px)][1]:.0%} / "
            + ("-" if np.isnan(table[(s, px)][2]) else f"{table[(s, px)][2]:.1%}") for s in SLICES) + " |")
    md += ["", "Most frequent character confusions on the `times` slice (true -> predicted, all heights):", ""]
    md += [f"- {t} -> {p}: {c}" for (t, p), c in confusions.most_common(12)]
    (out / "eval.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
