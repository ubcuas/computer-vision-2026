# Restart: GFL versus Anant, 2026-10-31

Entry point: `notebooks/GFL_Anant_Baseline_Server.ipynb`. Start Jupyter from repository root on Michael's NVIDIA server. Edit WORKSPACE before running. Setup, data verification, training and evaluation are separate cells; nothing auto-starts. See README.md for the pinned CUDA environment and server compatibility limits.

Default `DATA_VARIANT='decoy'` matches the progress document's reported Run #1: exactly Anant's 264 images/146 boxes, 211 train and 53 validation, including negatives. GFL initializes from COCO. It uses no Rucervus, Sherbrooke, AWIR or previous GFL weights. Image/label SHA-256 inventories are saved outside Git.

Latest Anant branch `dbe1c2b` changes Run #2 to decoy + AWIR RGB. `DATA_VARIANT='combined'` matches its 253 train / 64 validation images and 242 total boxes. This option uses the existing prepared AWIR files from his branch without rebuilding or resplitting them. Both decoy and AWIR files are already tracked in Anant's history; future large datasets should stay in external storage. This branch adds no Rucervus images or model weights.

The source files/splits are matched, not the optimizer, augmentation or training budget. GFL defaults: X101/DCN, 10 epochs, LR 0.00025, batch 1, long edge 1280 (aspect preserved), seed 42. Anant Run #1: YOLO11s, up to 100 epochs, patience 30; latest Run #2: up to 300 epochs, patience 50. His IOU=0.45 is NMS in validation/prediction, not the AP matching threshold. No independent test set exists in either recipe.

The pasted progress document reports YOLO Run #1 P=.66, R=.75, AP50=.73, AP50-95=.37. These are teammate-reported values, not independently reproduced results; the averaging/threshold details and trained checkpoint have not been verified. Do not equate our calibrated F1 metrics directly to Ultralytics P/R without aligning evaluation. The notebook reports common COCO AP and IoU-.5 detection/count metrics as validation diagnostics.

For SSH, use a server tmux session and port-forward Jupyter to your laptop. Outputs stay under WORKSPACE/runs/RUN_NAME, with config, package freeze, hardware, optimizer checkpoints, best_f1.pth, calibration and predictions. Use a new RUN_NAME for a fresh experiment. To resume use an actual epoch_N.pth file, retaining the same variant/config. Never rerun training merely to perform prediction.

Training has not been run for this restart; GPU execution will happen after Gurnek connects to Michael's server. Original stopped Colab artifacts remain separate in Drive; this notebook does not download or resume them.
