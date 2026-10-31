# Decoy annotation audit — 2026-10-31

## Verified conversion and provenance

Compared the actual server COCO train/val annotations used for training with the committed original YOLO labels. All 264 images and label files match the server dataset inventory SHA256 hashes. All 146 boxes have exactly zero pixel-coordinate difference from the direct YOLO normalized-center/size to COCO top-left/size conversion. Empty labels and train/validation assignments are preserved. The dataset directory also matches the locally fetched origin/deer-yolo-dataset branch byte-for-byte (git diff).

The suspicious source labels therefore predate the GFL format adapter; they are shared with any YOLO run using this dataset revision. This does not independently verify which files were on a historical teammate training machine.

## Visual review shortlist

All 146 labelled regions were screened on contextual contact sheets, followed by full-image inspection of the shortlist. This is not an exhaustive audit of missing objects in 176 negative images. Box numbers below are 1-based lines in the original YOLO label file. Species judgments on small/occluded objects require human review.

| Image | Split | Label line | Concern / review action |
|---|---|---|---|
| slide076 | val | 1 | Strongest suspected misplaced label: 120x120 box covers grass/path without an obvious deer. Original CSV calls the point deer; check image/point alignment. |
| slide248 | val | 1 | Extremely small box on an obscured object at the right edge. CSV calls it deer; identity/visible extent cannot be reliably judged from this crop. Do not automatically delete or relabel as bird. |
| slide051 | train | 2, 3 | Fitted boxes appear to capture too little of one deer and excessive background/too little of another. Review and redraw visible deer extents. |
| slide111 | train | 5, 6 | Similar poor-fitting boxes in the same scene configuration as slide051: narrow partial box and background-heavy box cutting off visible animal. |
| slide028 | train | 1 | Tight body fragment on upper deer, with visible head/neck outside the box; review box extent. |
| slide156 | train | 1 | Upper deer box tightly covers body while head/neck extend outside; review extent. |

### Correction to preliminary visual suspicions

Read the original local decoy_survey_key_points.csv. It explicitly identifies the white animal in slide206 as coyote (no deer points on that slide); slide077 has a coyote and a deer; slide046 has a coyote plus two deer. Excluding coyotes is intentional for a deer-only dataset. These are NOT supported missing-deer annotation claims. Our corresponding predictions may be genuine coyote/deer classification errors. Earlier suggestions of missing deer on slide206/077 were premature and are withdrawn.

The shortlist contains probable fitting issues plus ambiguous review cases, not six confirmed incorrect class labels. No source labels or the running experiment were changed.

## Upstream builder risks to investigate

The shared dataloaders/decoy_survey/build_dataset.py maps CSV image numbers directly to PDF page numbers, crops photos, and uses point-prompted SAM or a 120px fallback square. Pages with no deer points become negatives. An incorrect page/point mapping, incomplete class key, incorrect prompt, or failed fitted box can therefore create these symptoms. None has yet been confirmed as the root cause. Audit original CSV/PDF identifiers before rebuilding; do not rerun SAM as a substitute for verifying correspondence.

## Evidence

Local ignored artifacts/gfl_annotation_audit/verification.json and slide*_original_labels.jpg contain the numerical check and overlays drawn directly from YOLO source labels. This audit did not modify datasets, checkpoints or the running job.

Before a final comparison, agree on corrections as a team, version a reviewed shared dataset, and evaluate both architectures under one evaluator on a separate flight/site if available. Recalibrate confidence using reviewed validation data. Training-label corrections require new labelled runs; validation-only corrections can be evaluated with existing checkpoints, while preserving original results and documenting selection bias.
