# YOLO26s small-deer fine-tune

**Status (checked 2026-11-21):** failed during epoch 1 at approximately 2,370/4,077 steps with `torch.AcceleratorError: CUDA error: an illegal instruction was encountered`. No full-training epoch checkpoint or results CSV was saved. Only smoke-test weights exist; there are no new accuracy results. Logs and outputs are preserved; no automatic restart was attempted. The GPU still responds to nvidia-smi, but the cause is not established. A full-run checkpoint resume is unavailable; recovery would require an explicitly acknowledged fresh attempt from the original saved deer weights.

**Goal:** improve small-deer recall without flooding the human reviewer with false boxes. Motivation: [the earlier model comparison and crop pilot](2026-11-10-deer-v3-modal-comparison.md).

**Data:** unchanged Deer v3 split: 3,193 train / 358 validation / 238 test, one deer class, 4x training-decoy exposure. No LVIS additions. Added 467 training-only crops containing 1,777 boxes; effective training list is 4,077 entries. Related crops never enter validation/test. Original split, COCO and YOLO label hashes were checked; clipped crop labels passed CPU checks.

**Recipe:** fine-tune our saved Modal YOLO26s deer weights at 1280, AdamW LR 0.0001, seed 42, up to 25 epochs, patience 7, batch 1, workers 0, FP32. Native best-checkpoint selection uses validation fitness. Initial weights SHA256: `a6374c47b5d1b97c8e4f8f16f586463a626de7a8a5eb1dbcdfd51bb0b3d1f672`.

**Code/runtime:** launch revision `1687177`, branch `experiment/yolo26s-refinement`; RTX 3070 Laptop 8 GB, PyTorch 2.8.0+cu128, Ultralytics 8.4.155. The run manifest stores code/data identity; saved environment/config files record the full setup. Yash's checkout and training outputs were not changed.

**Outputs on greglinux:** `/home/mrd/yolo26s_refinement/run1`; console log `/home/mrd/yolo26s_refinement/pipeline.log`; checkout `/home/mrd/yolo26s_refinement/code`. Detached service: `deer-yolo26s-refinement-v1.service`. Smoke → train → validation/calibration → historical-test regression; stops on failure. Weights and data stay outside Git.

**Evaluation:** preserve native loss/metric graphs, shared AP/P/R/F1/count reports, source slices, inference memory/latency and a human-review tradeoff plot. Select thresholds on validation only: existing F1 threshold plus a proposed 95% mean-source-recall target under two false boxes/image. Report when that target cannot be met. Deer v3 test has already informed development; it is a regression check, not a fresh unbiased field test. No Jetson performance claim.

[Notebook](../../../prototyping/yolo26s-refinement/YOLO26s_Refinement.ipynb) · [Run instructions](../../../prototyping/yolo26s-refinement/README.md) · [Exact recipe](../../../prototyping/yolo26s-refinement/recipe.json). Leave the notebook's training toggle off while the service runs. Final completion requires saved weights, validation/test reports and `pipeline_complete.json`; update this record with measured results after completion.

Recovery authorized 2026-11-21: one fresh attempt from the original saved deer weights (no full-run checkpoint exists). Preserve the first failure and smoke outputs under `run1/interruption_2026-11-21/`. Retry the smoke test before training. Use `CUDA_LAUNCH_BLOCKING=1` for synchronous error reporting; this is diagnostic, not a proven CUDA fix. Same recipe, data and prepared crops. Queue up to 12 hours while the aircraft ROS object-detection process uses the GPU; never stop another team's process. Recovery service: `deer-yolo26s-refinement-recovery-1`; log: `/home/mrd/yolo26s_refinement/recovery-pipeline.log`; queue state: `/home/mrd/yolo26s_refinement/run1.queue.json`. Stops on another failure, with no repeat loop.
