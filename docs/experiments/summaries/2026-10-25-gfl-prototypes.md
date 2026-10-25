# GFL prototypes: wildlife-data run and Anant-matched baseline

Owner: Gurnek. Status: wildlife-data run running; matched-data run prepared, not started. See `prototyping/gfl-deer/README.md` and artifacts.json for commands, data counts, storage and limitations.

Objective: reconstruct an accessible aerial-deer GFL methodology and compare with YOLO under identical decoy input data. Current run observed in Colab at first LR trial, epoch 1 batch 140/328, on Tesla T4. Source download complete and prepared workspace cached in Drive. No final model metrics claimed. PyTorch 1.13.1+cu117, MMCV 1.7.1, MMDetection 2.28.2; original executed source preserved in current_run_snapshot. Runtime configs, package freeze and MMDetection source revision accompany checkpoints.

Matched baseline preserves Anant's commit 23c257e images, 146 boxes and 211/53 split, including negatives. Conversion produces COCO boxes from the existing normalized labels and records byte hashes. Start only after successful current-run completion; initialize from COCO. Full comparison contract is intentionally deferred to the team. Different training recipes remain confounders, and validation performance is not independent test performance.

Recovery check: Drive independently confirmed first LR trial epoch_3.pth (426,353,223 bytes), saved validation predictions at 20:04 UTC, and second LR trial lr_0.00025 started at 20:05 UTC on 2026-10-25. Browser remained Connecting; this was not evidence of backend termination. No duplicate training or source download was launched. Resume-aware driver added in commit 2ec32de is local and has not been deployed to the active runtime.
