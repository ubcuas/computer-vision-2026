# Deer v3 model comparison

- Owner: Gurnek; status: inconclusive for deployment, completed held-out evaluation.
- Data: [final Deer v3 release](../../datasets/deer-v3.md), frozen ZIP SHA256 `37dcff344fde1b29998ebcc9466772e0fda6e04bc94ddeefb09fc6c1d6eb0372`.
- Split: 3,193 train / 358 validation / 238 test images. Fourfold decoy training exposure gives 3,610 training entries. Same split and labels across models; model-specific resizing and augmentation differ.
- Code: dirty working tree based on `1a9e41cd1cb909c603f6b91c97f384ff5bd69a88`; implementation archived in the [comparison prototype](../../../prototyping/deer-v3-comparison/README.md). Seed 42.
- Purpose: compare deer detection on a held-out split using shared COCO evaluation and validation-only confidence calibration.
- Nominal input size 1280. YOLO11s and YOLO26s: 100-epoch limit; GFL X101/DCN: 100 limit, validation AP patience 30, stopped epoch 55, best epoch 25. Official COCO initialization. YOLO11 resumed from the laptop checkpoint; its total timing cannot be inferred from the Modal continuation time.

| Model | Precision | Recall | F1 | AP50 | AP50–95 |
| --- | --- | --- | --- | --- | --- |
| yolo11s | 80.3% | 43.8% | 56.7% | 60.7% | 29.5% |
| yolo26s | 71.2% | 66.3% | 68.6% | 70.3% | 33.6% |
| gfl_long | 71.8% | 67.3% | 69.5% | 69.5% | 31.4% |
| dfine_l (epoch 20) | 42.4% | 74.6% | 54.1% | 65.6% | 33.1% |

Precision/recall/F1 use IoU 0.5 and each model's validation-selected threshold. All test rows use the same 238 images (575 deer annotations). YOLO26s leads AP; GFL is slightly ahead on F1/count error. These differences do not establish a statistically reliable winner. This is one seed per model with different training recipes, not an architecture-only controlled trial. Held-out sources include inferred capture groups; this is not competition field validation. Jetson latency and memory still need testing on the actual device.

Artifacts: Modal workspace `software-9441`, environment `main`. In each volume, the model directory contains the training/evaluation outputs, including validation/test metrics and predictions, calibration, plots or inference profiles where completed. Local copies live in ignored `artifacts/modal_comparison/`.

| Model | Modal volume | Checkpoint path | SHA-256 |
| --- | --- | --- | --- |
| YOLO11s | `uas-deer-v3-yolo11s` | `/yolo11s/fit/weights/best.pt` | `0731bb141afd130f9ef4b005e66ff877368e3c5de5874549a9fdb099e83a61c7` |
| YOLO26s | `uas-deer-v3-yolo26s` | `/yolo26s/fit/weights/best.pt` | `a6374c47b5d1b97c8e4f8f16f586463a626de7a8a5eb1dbcdfd51bb0b3d1f672` |
| GFL | `uas-deer-v3-gfl_long` | `/gfl_long/best_bbox_mAP_epoch_25.pth` | `2b75bf451d131b3356a81fcd9c272f8370cdd7c7d07f3727ccd4cce4bccf920d` |
| D-FINE-L | `uas-deer-v3-dfine_l` | `/dfine_l/fit/checkpoint0019.pth` | `25a32c886463ca4efcf90830f66627142829cd3777691d9893230283d52dbeb5` |

Training and shared profiling used an A100-40GB (YOLO11s started on a laptop RTX 3070 before resuming on Modal). The Modal image definitions pin the model environments and the D-FINE/RT-DETR revisions. The exploratory 960/640 YOLO26s, short GFL, RTMDet and RT-DETR candidates have no completed comparable result.
GFL's first reporting attempt failed because shared metrics ran in Modal's Python 3.11 wrapper, while NumPy existed only in its legacy Python 3.10 venv. Added wrapper NumPy/pycocotools/Matplotlib/Pillow with an import smoke check; completed remaining evaluation without retraining. Applied the same fix to the modern-model image.

D-FINE-L resumed from epoch 1 on Modal app `ap-cwesEKfdX7UsLfP0sJUc1C`, with its optimizer, scaler, EMA and scheduler restored. Its separate checkpoint evaluations are documented below.

## YOLO26s and GFL error audit

Joined saved per-image scores to the frozen COCO annotations by image ID and verified ground-truth counts on all 358 validation and 238 test frames. Reviewed eight selected test images with annotation overlays. Prediction coordinates were not used for this audit, so a false positive could be a background detection or a poorly localized deer.

- Dry grass is a priority. On 56 BAMBI test frames from one held-out site, YOLO26s matched 24/73 deer and GFL 20/73. Both missed every deer on frames `227_2223` and `227_5871`. The annotated boxes are visible on the contact sheet, and the deer blend into the dry grass. The test site is one capture group.
- Rucervus differs sharply by held-out group. On validation, YOLO26s matched 11/313 and GFL 56/313. On test, they matched 268/393 and 285/393. On the test frame `UAVSurveyDeer_mp4-0032`, neither matched any of eight labels. The image is upside down and oblique over marsh. Review source transforms and keep a separate in-domain flight evaluation.
- The models are complementary on some images. On `young_deer__28967_1.jpg`, YOLO26s matched all three labels and GFL matched none. On Rucervus frame `UAVSurveyDeer_mp4-0060`, GFL matched five of six and YOLO26s two. Both missed `mississippi_decoy__slide103.jpg`.
- All 99 empty training frames and all 20 empty test frames come from Mississippi decoys. Current test results give limited evidence about false alarms on empty grassland. Add reviewed empty grass frames and independent dry-field flights with deer or decoys. Preserve capture groups.
- GFL's 1280 setting fits images into 1280 × 854. A square source can shrink to 854 × 854 while YOLO26s can use 1280 × 1280. Check effective scale as a controlled variable. GFL stopped at epoch 55 before either scheduled LR drop (60 and 80). If rerunning GFL, test a schedule that decays before patience expires, separately from resize.
- Before retraining, compare the existing YOLO26s weights at full frame versus overlapping tiles, and test operating thresholds on validation. Measure recall, false alarms per frame, and device latency. Use a fresh flight holdout for final evaluation after using this test set to guide development.

Review sheet (green ground-truth boxes): ignored `artifacts/modal_comparison/test_error_examples.jpg`.

## BAMBI prediction audit

Matched the 304 saved YOLO26s boxes on the 56 BAMBI test frames against 73 labels at IoU 0.50, reproducing 24 matches and eight unmatched predictions at confidence 0.31. Of the 49 misses, 14 had a valid box below 0.31, five had a nearby box with insufficient overlap, and 30 had no nearby saved box (best IoU below 0.10 or no output). Lowering confidence to 0.20 gives 28 matches and 12 unmatched predictions. At 0.001 the saved outputs give 38 matches and 266 unmatched predictions. These are retrospective test diagnostics; threshold selection remains on validation.

By flight prefix, matched labels were 226: 7/21, 227: 0/15, 228: 14/17, 229: 2/12, 230: 1/8. All are from one held-out site. Multiple frames can contain the same animal. On `227_2223` neither label overlaps a saved box, and `227_5871` has no saved boxes. On `226_6291`, a 0.623-score prediction appears on a deer but reaches only 0.485 IoU. On `228_11211`, the two predictions both match (IoU 0.753 and 0.829). Dry vegetation is visible in successful and failed examples, so background alone does not explain the disparity. Review full-resolution labels and compare full-frame inference with crops on a separate dry-field validation flight. Overlay: ignored `artifacts/modal_comparison/bambi_yolo26s_predictions.jpg`.

## D-FINE-L checkpoint evaluation

Stopped training after 52 completed epochs. Selected the saved epoch-20 periodic checkpoint using native validation history. It scored 58.58% AP50–95, within 0.02 percentage points of the historical peak at epoch 19. The exact epoch-19 weights were unverified after the resume. The evaluated checkpoint was `fit/checkpoint0019.pth` (zero-based name), SHA256 `25a32c886463ca4efcf90830f66627142829cd3777691d9893230283d52dbeb5`.

The evaluation-only Modal app `ap-xOghbDkzNf1sj4zbn58gJl` used the shared evaluator and calibrated confidence 0.45 on the 358 validation images. Results on the same 238 held-out images: AP50 65.55%, AP50–95 33.11%, precision 42.39%, recall 74.61%, F1 54.06%, 429 TP / 583 FP / 146 FN, mean absolute count error 2.11. YOLO26s scored AP50 70.28%, AP50–95 33.61%, F1 68.65%, 381 TP / 154 FP / 194 FN, and mean absolute count error 0.87.

D-FINE-L matched 26/73 BAMBI labels with 50 unmatched predictions versus 24/73 and eight for YOLO26s. Flight 227 was 2/15 with 27 unmatched predictions versus 0/15 with zero. On decoys, D-FINE-L matched 10/14 with one unmatched prediction versus 11/14 with two. These results favor keeping YOLO26s as the Jetson baseline and targeting dry-field failures in the next experiment. D-FINE-L reserved 864 MiB GPU memory during FP16 test inference on an A100. Jetson throughput remains unmeasured.

Artifacts are under `/dfine_l/epoch20_eval/` in Modal volume `uas-deer-v3-dfine_l`, with local copies in ignored `artifacts/modal_comparison/dfine_l/`.

### D-FINE-L epoch 50 follow-up

Evaluated saved `fit/checkpoint0049.pth` (epoch 50, SHA256 `061526c3d678b39e4346eebda08218c4b67ef96bd78a7e658d98b3d564021796`) using the same shared evaluator and splits as epoch 20. Validation AP50/AP50–95 were 76.44%/54.65%. The validation-selected confidence threshold was 0.42.

On 238 test images, epoch 50 scored 51.75% AP50, 26.81% AP50–95, 45.07% precision, 58.78% recall, and 51.02% F1 (338 TP, 412 FP, 237 FN). Mean absolute count error was 1.37. BAMBI was 20/73 matched with 58 unmatched predictions, including 0/15 matched with 36 unmatched predictions on flight 227. Decoys were 10/14 matched with two unmatched predictions. Epoch 20 remains the stronger of the two D-FINE-L checkpoints by test AP and F1. YOLO26s remains the onboard baseline.

Evaluation-only app `ap-Hgqs8xadEZtxBwybeiJqOo` completed. Artifacts are in the D-FINE-L Modal volume under `/dfine_l/epoch50_eval/` and copied to ignored `artifacts/modal_comparison/dfine_l_epoch50/`.

### YOLO26s audit for human review

Swept saved FP16 predictions from both splits. On 238 test frames, cutoff 0.31 matched 381/575 labels with 154 unmatched boxes. At 0.20 it was 411/575 with 248 unmatched, at 0.10 it was 443/575 with 428 unmatched, and at 0.001 it was 509/575 with 7,437 unmatched. The last setting averaged 33.4 boxes per frame with 129 at the 95th percentile. A looser review diagnostic checks whether any candidate covers the center of each labelled box. At 0.001 this reached 535/575, leaving 40 labels without a center-covering candidate. Center coverage does not enforce one box per animal or measure distinct animals over time.

At 0.001, 35/73 BAMBI labels and 3/14 Mississippi decoy labels still lacked IoU 0.50 matches. BAMBI flight 227 was 3/15 with 60 unmatched predictions, and 4/15 by center coverage. In slide 179 a confident near-match and duplicate explain the two unmatched boxes at the baseline cutoff. The coyote in that same frame appears only as a 0.014 candidate. Checked predictions against original coyote key points scaled to packaged image dimensions. No saved boxes covered the eight validation coyote points. One of five test points was covered, and none at confidence 0.10 or above. The two coyotes named in the older GFL audit (slides 077 and 206) are in v3 training and are not part of this held-out check. The small sample and point-based check limit conclusions on coyote confusion.

Rucervus accounts for 135/154 unmatched boxes at 0.31. Frame 0097 shows misses, a near-match and boxes on or around marsh vegetation and water. Unmatched boxes can also include unlabelled deer, so review original imagery before mining any frame as a negative. Prioritize crops as an inference diagnostic, then diverse dry-field deer and decoy footage, verified background examples, and consistent box extents. Use fresh whole-flight validation to set a candidate budget and a separate flight to measure operator-level count accuracy and time.

Ignored artifacts under `artifacts/modal_comparison/yolo26s/` include original prediction JSON, per-frame threshold sweeps, unmatched box CSVs, and the annotated test contact sheet.

### YOLO26s overlapping crop pilot

Ran the saved baseline weights locally on Apple M1 Pro (PyTorch 2.15.0 MPS, Ultralytics 8.4.155, FP32) with Deer v3 BAMBI and Mississippi decoy images (75 validation frames, 84 test frames). Used a nominal size of 1280 in all modes. Compared one full-frame pass, four corner crops each covering 60% of frame width and height, and the union of full-frame and crop predictions. Remapped boxes into original coordinates and merged overlapping boxes at IoU 0.55 in the crop and combined modes. All modes used the shared IoU 0.50 matcher.

At confidence 0.31, four crops recovered 46/87 test labels versus 35/87 for the full frame, with 26 versus 10 unmatched boxes. BAMBI improved from 24/73 to 34/73 and Mississippi from 11/14 to 12/14. On flight 227, crops at 0.10 recovered 3/15 with five unmatched boxes versus 1/15 and zero for the full frame. At the same cutoff, adding the full-frame detections to the crops recovered no further test labels and added five unmatched boxes on the combined test subset.

The same cutoff does not equalize review workload. For example, full-frame 0.05 and combined 0.20 had similar unmatched counts on validation (27 and 28) and matched 60 and 62 of 93 labels. On the test subset these settings matched 43 and 47 of 87, with 25 and 34 unmatched boxes. These sweeps are exploratory because prior analysis used the test set. Use a fresh flight to select settings and assess unique deer per pass and operator time.

Median test inference on the Mac was 72 ms full frame, 286 ms crops, and 362 ms combined, including model input preparation, inference and box handling after image decode. Jetson and live camera throughput are unmeasured. The four-crop mode is a candidate for review or a slower second pass.

Reproduce with `prototyping/deer-v3-comparison/yolo_crop_eval.py --data <Deer-v3-directory> --weights <best.pt> --output <new-directory> --device mps`. Outputs in ignored `artifacts/modal_comparison/yolo26s/crop_eval/` include checkpoint SHA-256, predictions, timings and the confidence sweep. A separate `training_review_queue.csv` samples 96 train-only frames for label review, including 48 BAMBI frames across flights 249–252 (all four from the same brown terrain site), 24 decoy positives and 24 decoy background frames. The queue is for review and does not change the training split.

## Follow-up actions

1. **Collect and label dry-field footage.** Use several separate flights or sites with deer/decoys, varied altitude, angle and lighting, and verified empty passes. Split by whole flight (prefer a separate site for test). Do not repurpose the current test groups for training.
2. **Run one short YOLO26s fine-tune from the saved checkpoint.** Choose settings on new validation flights. Compare against the unchanged baseline on a fresh holdout, checking distinct deer found per pass, candidate workload and review time. Retain the original v3 numbers as a historical baseline.
3. **Measure on the actual Jetson.** Start with full-frame 1280 FP16. Record latency, frame drops and memory on the live video pipeline. Check crops as a second pass only if the added detections justify the time and extra boxes.

These are proposed next experiments; no dry-field retraining or Jetson validation has been completed.
