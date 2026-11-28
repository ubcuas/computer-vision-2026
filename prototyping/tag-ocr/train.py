"""Generate synthetic tag crops and train TagNet.

    python train.py --out ../../artifacts/tag-ocr/run1

Uses CUDA when available. Rendering is CPU-only and cached under
<out>/../cache, so later runs with the same sizes and seeds skip it. Writes
<out>/tagnet.pt (best validation exact-ID accuracy), config.json and
train_log.csv. Train and validation use the training fonts with different
seeds; evaluate.py uses fresh seeds, a fixed pixel-height grid and held-out fonts.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import synth
from data import generate
from recognizer import TagNet, batch_tensor


def augment(x):
    """Cheap tensor-side augmentation on top of the rendered variation."""
    flip = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < 0.5
    x = torch.where(flip, -x, x)   # polarity: stops the model keying on absolute brightness
    return x + 0.05 * torch.randn_like(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=200000)
    ap.add_argument("--n-val", type=int, default=5000)
    ap.add_argument("--px", type=float, nargs=2, default=[4, 28], help="training character-height range (px)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, help="CPU processes for rendering (default: cores - 2)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = out.parent / "cache"
    torch.manual_seed(a.seed)
    dev = torch.device(a.device)

    t0 = time.time()
    px = f"{a.px[0]:g}-{a.px[1]:g}"
    Xtr, Ytr, _ = generate(a.n_train, a.seed, *a.px, workers=a.workers,
                           cache=cache / f"train_n{a.n_train}_s{a.seed}_px{px}.npz")
    Xva, Yva, _ = generate(a.n_val, a.seed + 1000, *a.px, workers=a.workers,
                           cache=cache / f"val_n{a.n_val}_s{a.seed + 1000}_px{px}.npz")
    print(f"data ready: {len(Xtr)} train / {len(Xva)} val in {time.time() - t0:.0f}s; device {dev}", flush=True)

    # The whole set fits in GPU memory as uint8 (200k x 96 x 64 = 1.2 GB), so skip DataLoader overhead.
    Xtr_t, Ytr_t = torch.as_tensor(Xtr, device=dev), torch.as_tensor(Ytr, device=dev)
    Xva_t, Yva_t = torch.as_tensor(Xva, device=dev), torch.as_tensor(Yva, device=dev)

    model = TagNet().to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    nb = len(Xtr) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * nb, pct_start=0.1)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)
    log = ["epoch,train_loss,val_exact,val_char,seconds"]
    best = -1.0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=dev)
        tl = 0.0
        for b in range(nb):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            x = augment(batch_tensor(Xtr_t[idx], dev))
            y = Ytr_t[idx]
            with torch.autocast(dev.type, enabled=amp):
                l1, l2 = model(x)
                loss = (F.cross_entropy(l1, y[:, 0], label_smoothing=0.05)
                        + F.cross_entropy(l2, y[:, 1], label_smoothing=0.05))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
        model.eval()
        with torch.no_grad():
            outs = [model(batch_tensor(Xva_t[i:i + 2048], dev)) for i in range(0, len(Xva_t), 2048)]
            p1 = torch.cat([o[0] for o in outs]).argmax(1)
            p2 = torch.cat([o[1] for o in outs]).argmax(1)
        ok1, ok2 = p1 == Yva_t[:, 0], p2 == Yva_t[:, 1]
        exact = (ok1 & ok2).float().mean().item()
        char = torch.cat([ok1, ok2]).float().mean().item()
        line = f"{ep + 1},{tl / nb:.4f},{exact:.4f},{char:.4f},{time.time() - t0:.0f}"
        log.append(line)
        print(line, flush=True)
        if exact > best:
            best = exact
            torch.save(model.state_dict(), out / "tagnet.pt")
    (out / "train_log.csv").write_text("\n".join(log) + "\n")
    (out / "config.json").write_text(json.dumps({
        **vars(a), "params": nparam, "best_val_exact": best,
        "train_fonts": synth.find_fonts("train"), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if amp else None}, indent=2))
    print(f"done: best val exact {best:.4f}, {nparam} params, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
