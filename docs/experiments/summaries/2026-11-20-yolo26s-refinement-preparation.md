# YOLO26s small-deer fine-tune

**Status:** running on greglinux; final accuracy results pending. GPU smoke passed and saved checkpoints. Full training was observed advancing without CUDA errors. This is not evidence of improved accuracy yet.

**Goal:** improve small-deer recall without flooding the human reviewer with false boxes. Motivation: [the earlier model comparison and crop pilot](2026-11-10-deer-v3-modal-comparison.md).

**Data:** unchanged Deer v3 split: 3,193 train / 358 validation / 238 test, one deer class, 4x training-decoy exposure. No LVIS additions. Added 467 training-only crops containing 1,777 boxes; effective training list is 4,077 entries. Related crops never enter validation/test. Original split, COCO and YOLO label hashes were checked; clipped crop labels passed CPU checks.

**Recipe:** fine-tune our saved Modal YOLO26s deer weights at 1280, AdamW LR 0.0001, seed 42, up to 25 epochs, patience 7, batch 1, workers 0, FP32. Native best-checkpoint selection uses validation fitness. Initial weights SHA256: `a6374c47b5d1b97c8e4f8f16f586463a626de7a8a5eb1dbcdfd51bb0b3d1f672`.

**Code/runtime:** launch revision `1687177`, branch `experiment/yolo26s-refinement`; RTX 3070 Laptop 8 GB, PyTorch 2.8.0+cu128, Ultralytics 8.4.155. The run manifest stores code/data identity; saved environment/config files record the full setup. Yash's checkout and training outputs were not changed.

**Outputs on greglinux:** `/home/mrd/yolo26s_refinement/run1`; console log `/home/mrd/yolo26s_refinement/pipeline.log`; checkout `/home/mrd/yolo26s_refinement/code`. Detached service: `deer-yolo26s-refinement-v1.service`. Smoke → train → validation/calibration → historical-test regression; stops on failure. Weights and data stay outside Git.

**Evaluation:** preserve native loss/metric graphs, shared AP/P/R/F1/count reports, source slices, inference memory/latency and a human-review tradeoff plot. Select thresholds on validation only: existing F1 threshold plus a proposed 95% mean-source-recall target under two false boxes/image. Report when that target cannot be met. Deer v3 test has already informed development; it is a regression check, not a fresh unbiased field test. No Jetson performance claim.

[Notebook](../../../prototyping/yolo26s-refinement/YOLO26s_Refinement.ipynb) · [Run instructions](../../../prototyping/yolo26s-refinement/README.md) · [Exact recipe](../../../prototyping/yolo26s-refinement/recipe.json). Leave the notebook's training toggle off while the service runs. Final completion requires saved weights, validation/test reports and `pipeline_complete.json`; update this record with measured results after completion.
