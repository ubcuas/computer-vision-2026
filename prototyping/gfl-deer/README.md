# Gurnek: research-derived GFL and exact-data decoy baseline

**Paused by Gurnek, 2026-10-25.** Colab training abandoned in favour of team GPU compute after agreeing shared data, splits and camera resolution. Five-minute monitor deleted; do not auto-launch either experiment. Existing checkpoints/cached data remain in Drive.

Two separate experiments; weights and datasets remain in external Drive storage. See `artifacts.json` for locations and `RESEARCH.md` for methodology and limitations. The live wildlife-data Colab run is not modified by this repository reorganization. `current_run_snapshot/` preserves the original executed package for provenance; the active package uses shared loaders under `dataloaders/deer_shared/`.

## Current wildlife-data experiment

COCO initialization -> 328 positive Sherbrooke summer training images (512 original validation images) -> adaptation on 184 decoy + 42 RGB AWIR train images, with 40 decoy + 6 AWIR validation images. Decoy diagnostic test: 39 images. Source recipe: 3 learning rates screened for 3 epochs, fresh 10-epoch source fit, 10-epoch adaptation. Nominal long edge 2650, batch 1, FP16, seed 42. X101/DCN is an inferred reconstruction of Moreni's backbone. Exact reproduction is not claimed. The source code snapshot and output configs identify the actual run, independently of subsequent repo changes.

Prepared images and annotations are cached at MyDrive/Gurnek_GFL_Research/workspace. All run checkpoints and logs are under MyDrive/Gurnek_GFL_Research/runs. `complete.json` marks successful completion of source fit, adaptation and comparison. Smoke and hook diagnostic checkpoints are not final models. Existing decoy test images overlap Anant's training data; these results cannot establish a winner against his current checkpoint.

## Exact-data baseline against Anant

Use the committed Anant dataset at commit `23c257eb884b05cf21043d5bbaef627c38e280cf`: 264 JPEGs, 146 point-derived boxes; 211 train (70 positive), 53 validation (18 positive). The adapter reads his existing JPEGs and YOLO labels verbatim and preserves every split, negative and box. It does not regenerate SAM boxes, resize images on disk, remove uncertain examples, or apply our own grouping. YOLO class 0 becomes COCO category 1, with normalized coordinates converted using actual dimensions. `inventory.json` records image and label hashes.

From repository root:

```bash
python prototyping/gfl-deer/run.py prepare-matched --workspace /path/to/cached/workspace
DEER_WORKSPACE=/path/to/cached/workspace bash prototyping/gfl-deer/scripts/setup_colab.sh
/content/gfl-env/bin/python prototyping/gfl-deer/run.py train-matched \
  --workspace /content/deer_research \
  --workdir /content/drive/MyDrive/Gurnek_GFL_Research/decoy_matched_anant_v1
```

Run after the current experiment finishes successfully. This baseline starts from official COCO GFL weights, never the wildlife-trained checkpoint. Defaults: X101/DCN, long edge 1280, batch 1, 10 epochs, LR 0.00025, seed 42, paper-style flips/warmup. These training choices differ from Anant's YOLO 100-epoch/patience-30 recipe; this is matched data, not matched compute or an isolated architecture experiment. Resume an interrupted baseline using the same command plus `--resume <run>/latest.pth`. Do not start it concurrently with the current T4 run.

Validation selects best_f1.pth and its confidence threshold. Use that checkpoint for later prediction; do not train again to use the detector. The 53 validation images are not an untouched test set. Final comparison policy is pending next week's team discussion. A new common test set is still needed for final claims.

Shared metrics: `dataloaders/deer_shared/metrics.py`. Existing `deer_pipeline` CLI remains available with PYTHONPATH containing the repo root and this prototype, and DEER_WORKSPACE pointing to the cached workspace. Legacy notebooks are preserved for provenance; their standalone paths refer to the original workspace. The repo `run.py` is the entry point for matched training.
