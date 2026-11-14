# Michael's handoff: Gurnek's GFL deer prototype

**October 31 restart:** see [BASELINE.md](BASELINE.md) and `notebooks/GFL_Anant_Baseline_Server.ipynb` for the new comparison. The paused historical experiment below remains separate.

**Status: paused on October 25, 2026.** Gurnek stopped the Colab GPU session and deleted its automatic monitor. Nothing will auto-start. Two learning-rate trials finished; the third was interrupted. The planned full source fit, adaptation and final comparison did not finish. The exact-Anant-data baseline is prepared but has never trained. Partial checkpoints remain useful; they are not final competition models.

This branch adds GFL model code under `prototyping/gfl-deer/` and reusable preparation/evaluation under `dataloaders/deer_shared/`. It is based on Anant's `deer-yolo-dataset` branch. It does not merge Yash's `ai-setup` documentation branch. `current_run_snapshot/` preserves the original Colab package; `deer_pipeline/` is the reorganized active code. The snapshot is provenance, not the server entry point.

## What we attempted

Reconstruct [Moreni et al.'s aerial-deer methodology](https://doi.org/10.1371/journal.pone.0284449): COCO-pretrained GFL, one deer class, fixed resize, horizontal/vertical flips, SGD, 500-iteration warmup, 10 epochs with LR reductions at 6/8. We inferred the closest standard backbone, ResNeXt-101-32x4d with deformable convolutions; the authors' exact config and chosen LR were unavailable. See [RESEARCH.md](RESEARCH.md) for evidence and departures. This is a reconstructed method, not exact reproduction.

| Experiment | Training data | Validation / test | State |
| --- | --- | --- | --- |
| Wildlife source stage | 328 positive Sherbrooke summer images | All 512 supplied summer validation images | LR 0.0025 and 0.00025 completed 3 epochs each; 0.000025 interrupted |
| Planned adaptation | 184 decoy + 42 RGB AWIR images | 40 decoy + 6 AWIR validation; 39 decoy diagnostic test | Not reached |
| Exact-Anant baseline | His 211 decoy training images | His 53 validation images; no independent test | Prepared, not trained |

The original experiment screened three LRs, planned a fresh 10-epoch source fit, then 10 adaptation epochs. Resolution was nominal 2650x1768, batch 1, FP16, seed 42. It is expensive: about 2.8 seconds/source training image on T4, plus two validation passes per epoch and post-trial prediction. Duplicate validation is an identified optimization opportunity; it has not been changed in the executed experiment.

## Saved data and checkpoints — do not download source data again

**Gurnek's private Drive:** [Gurnek_GFL_Research](https://drive.google.com/drive/folders/1jzRxAyvJ0U3jTVfa67Zd-IkjWKWRMfWY). Michael may need Gurnek to grant access or copy the folder to approved team storage; this branch does not change sharing.

| Drive path under `MyDrive/Gurnek_GFL_Research/` | Contents |
| --- | --- |
| `workspace/` | Cached resized images, prepared annotations/manifests, original code and provenance. Copy this entire directory to the server. Full-resolution Sherbrooke originals were streamed and discarded. |
| `runs/lr_0.0025/x101_2650/` | First completed 3-epoch trial: epoch checkpoints, best weights, config, calibration, logs, predictions and examples |
| `runs/lr_0.00025/x101_2650/` | Second completed 3-epoch trial, same artifact layout |
| `runs/lr_2.5e-05/x101_2650/` | Interrupted third trial; inspect files to determine last fully saved epoch |
| `runs/lr_trials.json` | Completed trial-selection records; not a final model comparison |
| `runs/driver.log` | Experiment-level progress/errors |
| `gpu_smoke/`, `hook_diagnostic/`, `runs/smoke*` | Engineering checks, not usable evidence of a final trained model |

Copy **both `workspace/` and `runs/`**, preserving their relationship; keep immutable copies of the originals. Use actual `epoch_N.pth` files when transferring: `latest.pth` is a symlink and some download tools preserve only its link text. `best_f1.pth` is for inference; an epoch checkpoint includes optimizer state for resume. First trial's `epoch_3.pth` was verified present in Drive at **426,353,223 bytes**. Full tensor/hash verification must be done after transfer.

[Original Colab notebook](https://colab.research.google.com/drive/1EX586MOBp2doRofrx2FqHUbKo_Lfwa6C) retains outputs. Ignore its stale Connecting display: Drive proved training continued while browser control was disconnected. The runtime has since been explicitly terminated.

## Decisions needed before spending GPU time

Agree with the team on image/annotation version, grouped train/validation/test split, exact camera capture mode and model input size. Team chat says 1080p; exact dimensions/capture pipeline remain unverified. Camera dimensions alone do not match deer pixel size, altitude, blur or field of view. Preserve aspect ratio; do not stretch every source image to 1920x1080.

Current decoy annotations differ from Anant's: Gurnek used 263 images/145 manually assisted agent-reviewed boxes, excluded uncertain page 76, and grouped related views into 184/40/39. Anant used 264 images/146 SAM-or-fixed boxes with a 211/53 image-random split. **33 of Gurnek's 39 test images are in Anant's training set.** Do not use these existing scores to declare a winner. Neither annotation set is asserted to be human-approved ground truth; provenance/license and final benchmark approval remain team work.

The exact-data baseline intentionally uses **Anant's existing files**, including his uncertain example, negatives and split. The adapter does not rerun SAM, regenerate crops or resplit. It converts normalized YOLO class 0 to COCO category 1 using actual image dimensions, recording SHA-256 hashes of every JPEG and label. Its test checks all 146 box conversions. Input pixels and annotations match; augmentation, architecture and training budget do not.

## GPU server setup

Linux x86_64 NVIDIA server. The stack was verified on Tesla T4: Python 3.10, PyTorch 1.13.1+cu117, MMCV-full 1.7.1, MMDetection 2.28.2. Check the server GPU/driver supports this older CUDA/PyTorch stack; a much newer GPU may require a separate compatibility update. The setup script is provided but has **not been executed on Michael's server**. No server credentials are stored here.

```bash
git clone --branch experiment/gfl-shared-decoy-baseline https://github.com/ubcuas/computer-vision-2026.git
cd computer-vision-2026
export REPO_ROOT="$PWD"
# Copy the Drive workspace to this chosen location before the next steps.
export DEER_WORKSPACE="/absolute/server/path/gfl_workspace"
export GFL_ENV="$DEER_WORKSPACE/.venv-gfl"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/prototyping/gfl-deer${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi
bash prototyping/gfl-deer/scripts/setup_server.sh
export PY="$GFL_ENV/bin/python"
# Rewrite old /content/deer_research/... annotation paths to this workspace.
"$PY" -m deer_pipeline.relocate
"$PY" -m unittest discover -s prototyping/gfl-deer/tests -v
```

The first setup downloads dependencies and the official model initialization as needed; it does not download datasets. Reuse the environment and cached data afterward. Run training inside tmux or another server job manager and save logs/checkpoints on persistent storage.

## Option A: exact-Anant-data baseline after team agreement

The branch contains Anant's committed dataset from `23c257eb884b05cf21043d5bbaef627c38e280cf`. Verify it has not been edited (`git diff 23c257e -- prototyping/yolo-fine-tune/dataset` should be empty). Then:

```bash
"$PY" prototyping/gfl-deer/run.py prepare-matched --workspace "$DEER_WORKSPACE"
# Inspect data/anant_exact/inventory.json and annotations/{train,val}.json.
# Expected: 264 images, 146 boxes total; 211 train / 53 val.
"$PY" prototyping/gfl-deer/run.py train-matched \
  --workspace "$DEER_WORKSPACE" \
  --workdir "$DEER_WORKSPACE/runs/decoy_matched_anant_v1"
```

Defaults: official COCO initialization (never wildlife-trained weights), GFL X101/DCN, long edge 1280 with aspect preserved, batch 1, 10 epochs, LR 0.00025, seed 42. Anant's YOLO recipe uses image size 1280, up to 100 epochs and patience 30. This is a **matched-data validation baseline**, not matched compute or a final untouched-test comparison. If team choices change, update/version the config before running; these defaults do not settle that agreement.

To resume an interrupted matched baseline, use the same command and settings plus `--resume /absolute/path/to/epoch_N.pth`. Do not pass a `best` checkpoint as `--checkpoint` and call it a full optimizer resume. The baseline refuses to retrain a completed epoch-10 run without an explicit resume argument.

## Option B: continue a particular original wildlife-data stage

Preserve the saved resolution, LR, dataset and backbone when resuming. Changing them makes a new experiment. After copying runs to `$DEER_WORKSPACE/runs`, select an actual saved epoch file and inspect it:

```bash
export CHECKPOINT="$DEER_WORKSPACE/runs/lr_2.5e-05/x101_2650/epoch_1.pth"  # Example only: check what actually exists!
"$PY" -c 'import os,torch; p=os.environ["CHECKPOINT"]; d=torch.load(p,map_location="cpu"); print(d["meta"]); assert "optimizer" in d'
"$PY" -m deer_pipeline train \
  --dataset "$DEER_WORKSPACE/data/sherbrooke/prepared" \
  --workdir "$DEER_WORKSPACE/runs/lr_2.5e-05/x101_2650" \
  --model x101 --lr 0.000025 --epochs 3 --long-edge 2650 \
  --resume "$CHECKPOINT"
```

Do not rerun completed trials. The reorganized experiment driver has resume support, but original run metadata contains `/content` paths; use explicit stage commands after relocating data and inspecting configs. Do not blindly launch the full driver or the old notebook's Run all on the server. A changed lower-resolution plan should be recorded separately rather than described as continuation of the high-resolution paper reconstruction.

## Use a trained model without retraining

```bash
"$PY" -m deer_pipeline predict \
  --config "$DEER_WORKSPACE/runs/decoy_matched_anant_v1/config.py" \
  --checkpoint "$DEER_WORKSPACE/runs/decoy_matched_anant_v1/best_f1.pth" \
  --annotations "$DEER_WORKSPACE/data/anant_exact/annotations/val.json" \
  --output "$DEER_WORKSPACE/results/matched_val_predictions.json"
"$PY" -m deer_pipeline calibrate \
  --annotations "$DEER_WORKSPACE/data/anant_exact/annotations/val.json" \
  --predictions "$DEER_WORKSPACE/results/matched_val_predictions.json" \
  --output "$DEER_WORKSPACE/results/matched_calibration.json"
"$PY" -m deer_pipeline evaluate \
  --annotations "$DEER_WORKSPACE/data/anant_exact/annotations/val.json" \
  --predictions "$DEER_WORKSPACE/results/matched_val_predictions.json" \
  --calibration "$DEER_WORKSPACE/results/matched_calibration.json" \
  --output "$DEER_WORKSPACE/results/matched_val_metrics.json"
```

These paths exist only after that baseline has trained. Predictions contain image ID, deer confidence and `[x,y,width,height]` in input-image pixels. Evaluation reports precision/recall/F1 at IoU 0.5, AP50/AP50-95 and count metrics. Thresholds come from validation only. Applying calibration and scoring on that same validation set is model-selection evidence, not final generalization evidence.

Keep data, weights and generated outputs outside Git. Preserve configs, seed, package versions, code commit, dataset hashes, splits, checkpoint hashes, metrics and hardware/latency with each run. The current branch includes [artifact locations](artifacts.json), [research evidence](RESEARCH.md), and an [experiment record](../../docs/experiments/summaries/2026-10-25-gfl-prototypes.md).
