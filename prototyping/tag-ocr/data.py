"""Parallel generation and caching of synthetic recognizer inputs.

Deliberately imports no torch: on Windows every worker process re-imports this
module, and each worker is pinned to one OpenCV thread so N workers do not
spawn N x cores threads (that oversubscription made the first smoke run ~30x
slower than the single-process timing).
"""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import synth


def _init_worker():
    cv2.setNumThreads(1)


def _gen_chunk(args):
    seed, n, px_lo, px_hi, fontset, hard = args
    rng = np.random.default_rng(seed)
    fonts = synth.find_fonts(fontset)
    X = np.empty((n, synth.IN_H, synth.IN_W), np.uint8)
    Y = np.empty((n, 2), np.int64)
    P = np.empty(n, np.float32)
    for i in range(n):
        code = synth.random_code(rng)
        px = rng.uniform(px_lo, px_hi) if px_hi > px_lo else px_lo
        X[i] = synth.to_input(synth.make_sample(code, px, fonts[rng.integers(len(fonts))], rng, hard=hard))
        Y[i] = [synth.CHARSET.index(code[0]), synth.CHARSET.index(code[1])]
        P[i] = px
    return X, Y, P


def generate(n, seed, px_lo, px_hi, fontset="train", hard=1.0, workers=None, chunk=500, cache=None):
    """Returns (X uint8 [n,H,W], Y int64 [n,2], P float32 [n] char px). Cached to `cache` (.npz) if given."""
    if cache and Path(cache).is_file():
        z = np.load(cache)
        if len(z["X"]) == n:
            return z["X"], z["Y"], z["P"]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    jobs = [(seed * 100003 + k, min(chunk, n - k * chunk), px_lo, px_hi, fontset, hard)
            for k in range((n + chunk - 1) // chunk)]
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        parts = list(ex.map(_gen_chunk, jobs))
    X, Y, P = (np.concatenate(p) for p in zip(*parts))
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, X=X, Y=Y, P=P)
    return X, Y, P
