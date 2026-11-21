# YOLO26s: next run

I'm keeping YOLO26s and trying to improve its small-deer recall without making the human review queue unusable. This is prepared code, not a new result.

## What changes

- Start from our saved Deer v3 YOLO26s `best.pt`. This is a new fine-tune, not a restart of the Modal optimizer.
- Keep the same 3,193 / 358 / 238 original images. Training decoys still get 4x exposure.
- Add up to 100 training crops per source, centred on small deer. Each crop covers 60% of the original width/height. All intersecting deer get clipped labels; crops leaving ambiguous deer slivers are discarded. Validation/test never supply crops or training examples.
- Fine-tune for at most 25 epochs at 1280, AdamW LR 0.0001, early-stopping patience 7. Mild brightness, scale and rotation augmentation; no mosaic or vertical flips in this trial. These settings are a starting hypothesis, not an established optimum.
- Start batch 1, workers 0, FP32 on Michael's laptop. This is a conservative memory/precision choice after the previous GPU failures, not a fix for a faulty GPU. It may be slower than AMP. Stop on CUDA errors and inspect them; no endless automatic restarts.

The crop index and a contact sheet are saved before training so we can inspect the derived labels. Extra crops change the sampling distribution: this isn't another identical-recipe comparison with the first Modal run.

## Server setup

Use a separate environment and run folder so we don't overwrite Yash's work. Python 3.11 is suitable. Use the existing working CUDA PyTorch installation if available; otherwise install the appropriate CUDA build from pytorch.org for the server driver first.

```bash
python -m venv .venv-yolo-refine
source .venv-yolo-refine/bin/activate
pip install ultralytics==8.4.155 numpy==1.26.4 pillow pyyaml matplotlib pandas pycocotools ipykernel nbformat
python -m ipykernel install --user --name deer-refine --display-name 'Deer YOLO refinement'
```

Open `YOLO26s_Refinement.ipynb` with that kernel. Set the repo, existing extracted Deer v3 data, saved deer weights and new output directory in the first code cell. No download is built into this runner. The original checkpoint is currently local at `artifacts/modal_comparison/yolo26s/weights/best.pt` (SHA256 `a6374c47b5d1b97c8e4f8f16f586463a626de7a8a5eb1dbcdfd51bb0b3d1f672`). Transfer it separately if it isn't on the server.

CLI equivalent, from the repo root:

```bash
python prototyping/yolo26s-refinement/run.py prepare --data "$DEER_DATA" --run "$DEER_RUN" --weights "$DEER_WEIGHTS"
python prototyping/yolo26s-refinement/run.py smoke --data "$DEER_DATA" --run "$DEER_RUN"
python prototyping/yolo26s-refinement/run.py train --data "$DEER_DATA" --run "$DEER_RUN"
python prototyping/yolo26s-refinement/run.py evaluate --data "$DEER_DATA" --run "$DEER_RUN"
```

Only `prepare` is CPU-only. Smoke runs one short epoch in its own folder and saves weights; full training always starts from the original deer checkpoint. If full training fails, preserve its log and use `train ... --resume` to load `fit/weights/last.pt` with its saved optimizer/schedule. No automatic restart from scratch. Run the long command in tmux or a detached service, not a terminal that will close with SSH. The runner refuses another active compute process, but the team still needs to coordinate GPU access.

## What we get

`fit/` has best/last weights, epoch checkpoints, native loss/metric graphs, CSV and arguments. `manifest.json` records split/COCO hashes, input weights, tile inventory, code revision and dirty state; `pip-freeze.txt` and `hardware.json` record the runtime. Shared validation reports add AP50/AP50–95, P/R/F1, count error, source slices, confidence curves and batch-one FP32 memory/latency on the actual GPU. `review_tradeoff.png` shows recall versus false boxes per image.

We select two validation thresholds: the existing F1 threshold for continuity, and a human-review threshold targeting mean source recall of 95% under a budget of two false boxes/image. The latter is a configurable workload assumption. If 95% isn't possible, the report says so and gives the best feasible recall. A small number of decoy validation boxes means that slice remains uncertain. Native best-checkpoint selection still uses Ultralytics validation fitness, not the new review metric.

After choosing the recipe on validation, `evaluate ... --test` runs the saved checkpoint with the locked validation thresholds. Deer v3 test has already influenced our thinking, so this is a historical regression check, not a fresh unbiased flight test. Collect a new held-out camera session before claiming field performance.

The earlier [crop pilot](../deer-v3-comparison/yolo_crop_eval.py) remains available for optional inference experiments. It made four crop passes in addition to the full-frame pass; don't assume that workload will fit the Jetson. Training crops don't require five-pass inference.

## New flight setup

Optical zoom can help distant deer occupy more pixels, but it narrows coverage. We need labelled oblique/distant views and real camera validation, not just stronger synthetic rotation. A gimbal does not guarantee a top-down view. Tag/anomaly models remain separate workstreams; this notebook trains deer detection only.

`launch_server.sh PYTHON DATA RUN WEIGHTS` runs preparation, smoke, training, validation and historical-test regression sequentially. Use it in a detached service and save its console log outside the fresh run directory. It stops on a failure; rerunning explicitly resumes saved training rather than discarding it. Reports and `pipeline_complete.json` mark the end, not merely the final epoch.
