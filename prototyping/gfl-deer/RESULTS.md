# Decoy baseline results — 2026-10-31

Completed 10 epochs on Michael's RTX 3070 Laptop GPU. Dataset: exact Anant Run #1 pixels, boxes and 211 train / 53 validation images (112 / 34 boxes). COCO initialization; X101/DCN GFL; batch 1, FP16, initial LR 0.00025, milestones 6/8, seed 42, aspect-preserving 1280 resize with minimum-size padding.

| Validation metric | GFL best-F1 epoch 4 | Anant YOLO11s Run #1 (teammate reported) |
|---|---:|---:|
| Precision | 0.9143 | 0.66 |
| Recall | 0.9412 | 0.75 |
| F1 | 0.9275 | Not independently supplied |
| AP50 | 0.9501 | 0.73 |
| AP50–95 | 0.5991 | 0.37 |

GFL confidence selected on validation: 0.34; matching IoU 0.5; TP32 / FP3 / FN2. Count MAE 0.09434, count bias +0.01887 deer/image; exact count accuracy 0.90566. Mean loading + inference + postprocessing 0.12907 seconds/image (52 timed images, one warmup excluded; output rendering excluded).

These are validation results, not independent test performance. The split includes correlated same-site views. P/R conventions differ from Ultralytics; rerun both models through a common evaluator before ranking them. Best-AP epoch 9 was 0.6767 AP50–95 in the training evaluator; the exported best-F1 checkpoint and table instead use epoch 4. Do not mix these checkpoints' scores.

## Persistent artifacts

Server: `/home/mrd/gfl_workspace/runs/gfl_decoy_run1_v1/`.
Executed notebook: `/home/mrd/gfl_workspace/GFL_Anant_Baseline_Server.executed.ipynb`.
Dataset cache + inventory: `/home/mrd/gfl_workspace/data/anant_exact/`.
Local downloaded bundle/reports: repository ignored `artifacts/gfl_decoy_run1_v1/`.

- `best_f1.pth`: SHA256 `f2ba67d31f298e639047e80eb5c15d6dba3f33d6d47e1fc3bc55cd5fa20a8575`.
- `best_bbox_mAP_epoch_9.pth`: SHA256 `f859b10a037d34eccfc8dff606a20d467bd0c9ecea8d31c0afbee3833865421f`.
- `epoch_10.pth`: SHA256 `28d4f2cdc6846dd35454b3f6349722108cc155d0c82315d086fc7b1230268bd9` (resumable optimizer state).

Verified portable ZIP: `gfl_model_bundle.zip` (~207 MiB). Contains trained inference weights, config, calibration, metrics and graphs. Keep epoch checkpoints separately for resuming. Training code revision 36f543a; documentation was subsequently expanded.

## Longer comparison

Separate run `gfl_decoy_long_v1`, same architecture/data/preprocessing/seed, fresh official COCO initialization. Up to 100 epochs, LR 0.00025 with drops 60/80, validation-AP early stopping patience 30 (strict improvement). This investigates training budget; it does not equalize YOLO's optimizer or augmentation. Default notebook remains 10 epochs; select RUN_NAME, EPOCHS=100 and PATIENCE=30 for the longer run. All outputs stay separate; do not overwrite the short run. Launched 2026-10-31 at about 15:02 Vancouver time, code revision `09ae669`, persistent launch marker `gfl_decoy_long_v1/launch.json`; real epoch-1 GPU progress verified. All 15 pipeline tests completed (14 passed, one unavailable-source test skipped), and the registered stopping hook/config passed server checks. Results pending.

Annotation verification: Gurnek confirmed the supplied annotations are correct; prior automated visual error suspicions are withdrawn. All 146 YOLO-to-COCO boxes and 264 source image/label hashes were verified. Evaluate against supplied labels unchanged; see ANNOTATION_AUDIT.md.

Data-quality follow-up: Gurnek identifies oversized bounding boxes; better reviewed box annotations are needed. Their contribution to Anant's overlapping predictions is plausible but unconfirmed. See ANNOTATION_AUDIT.md.

Long-run interruption: epoch 13 completed and saved; training failed during epoch 14 on 2026-10-31 with CUDA CUBLAS_STATUS_INTERNAL_ERROR in deform_conv_backward_parameters (FP16). Epoch13, best-F1 and best-AP epoch11 checkpoints remain on persistent server storage. No restart was launched. At the previous pace, the remaining full 100-epoch budget was approximately 2h15m plus reporting; recovery time is unknown and early stopping can finish sooner.
