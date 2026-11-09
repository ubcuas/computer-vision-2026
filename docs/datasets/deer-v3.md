# Deer dataset v3 — team handoff

**Use this version for the next GFL/YOLO comparison. Upload the whole folder to restricted team Drive.**

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| young_deer | 1570 | 48 | 60 |
| bambi | 831 | 44 | 56 |
| mississippi_decoy | 139 | 31 | 28 |
| awir | 34 | 6 | 13 |
| sherbrooke | 328 | 147 | 0 |
| **Total** | 2902 | 276 | 157 |

Why: more Young Deer diversity without sampling 29k augmented files blindly; grouped augmentation/scene families, BAMBI locations and AWIR dates; summer-oriented selection and fewer repeated video frames. Source-key-confirmed decoy negatives teach rejection of coyotes/birds. Saved human corrections take precedence.

Young Deer evaluation represents 16 inferred families in validation and 20 in test (three representatives each), not 108 independent flights. BAMBI test is one held-out location; AWIR and decoys are same-site diagnostics. Decoys were used in older experiments. Grouping and similarity checks reduce leakage but cannot prove independence. Image proportions follow available groups; they are intentionally not a random 70/15/15 split.

Checks: every selected source/COCO/YOLO box compared; dimensions, classes, bounds, file integrity, group separation and exact duplicates checked. Additional Young Deer cross-partition SIFT retrieval found matches, removed lower-priority groups and finished with no detected matches. The retrieval is approximate. All selected evaluation images were skimmed using contact sheets and target zooms; not every training image or box was individually certified.

Run `python prepare.py --verify` after downloading. Use runtime/data.yaml for YOLO; runtime/mmdet_datasets.py provides equivalent GFL/MMDetection 2.x datasets. Both give TRAIN decoy images 4x exposure, retain negative images, and leave validation/test unchanged. No model trained yet. Start from official COCO weights, agree preprocessing, tune on validation, then report test metrics per source. Freeze changes under a new version.

Detailed rationale and source attribution: README.md. Integrity report: audit/verification.json. Image gallery: review/index.html. Original source files and earlier versions remain unchanged.
