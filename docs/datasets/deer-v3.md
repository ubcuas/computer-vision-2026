# Deer dataset v3 — team handoff

**Final archive used for the Modal comparison: 3,789 images, including the reviewed Rucervus selection.** This supersedes the earlier 3,335-image v3 draft. Request the archive from the CV team; [artifact metadata](../../dataloaders/deer_v3/artifact.json) identifies it by SHA-256.

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| young_deer | 1570 | 48 | 60 |
| bambi | 831 | 44 | 56 |
| mississippi_decoy | 139 | 31 | 28 |
| awir | 34 | 6 | 13 |
| sherbrooke | 328 | 147 | 0 |
| Rucervus reviewed aerial | 291 | 82 | 81 |
| **Total** | 3193 | 358 | 238 |

Why: more Young Deer diversity without sampling 29k augmented files blindly; grouped augmentation/scene families, BAMBI locations and AWIR dates; summer-oriented selection and fewer repeated video frames. Source-key-confirmed decoy negatives teach rejection of coyotes/birds. Saved human corrections take precedence.

Young Deer evaluation represents 16 inferred families in validation and 20 in test (three representatives each), not 108 independent flights. BAMBI test is one held-out location; AWIR and decoys are same-site diagnostics. Decoys were used in older experiments. Grouping and similarity checks reduce leakage but cannot prove independence. Image proportions follow available groups; they are intentionally not a random 70/15/15 split.

For the original five sources, every selected source/COCO/YOLO box compared; dimensions, classes, bounds, file integrity, group separation and exact duplicates checked. Additional Young Deer cross-partition SIFT retrieval found matches, removed lower-priority groups and finished with no detected matches. The retrieval is approximate. All selected evaluation images were skimmed using contact sheets and target zooms; not every training image or box was individually certified.

Run `python prepare.py --verify` after downloading. Use runtime/data.yaml for YOLO; runtime/mmdet_datasets.py provides equivalent GFL/MMDetection 2.x datasets. Both give TRAIN decoy images 4x exposure, retain negative images, and leave validation/test unchanged. The [Modal comparison](../experiments/summaries/2026-11-10-deer-v3-modal-comparison.md) used these splits. Freeze future data or label changes under a new version.

Gurnek accepted 454 of 1,029 aerial-review candidates from [Rucervus](https://data.mendeley.com/datasets/53nvjhh5pg/1), retaining the supplied boxes. We grouped related filename sequences and geometric scene matches; no verified flight or site IDs were available. Measure this source separately: its frames and background differ across splits. Keep the archive internal until redistribution permissions are confirmed. The archive README has the full source attribution; its checksum index covers images, labels and split lists.
