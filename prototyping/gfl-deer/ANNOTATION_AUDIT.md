# Decoy annotation audit — 2026-10-31

## Verified conversion and provenance

Compared the actual server COCO train/val annotations used for training with the committed original YOLO labels. All 264 images and label files match the server dataset inventory SHA256 hashes. All 146 boxes have exactly zero pixel-coordinate difference from the direct YOLO normalized-center/size to COCO top-left/size conversion. Empty labels and train/validation assignments are preserved. The dataset directory also matches the locally fetched origin/deer-yolo-dataset branch byte-for-byte (git diff).

The suspicious source labels therefore predate the GFL format adapter; they are shared with any YOLO run using this dataset revision. This does not independently verify which files were on a historical teammate training machine.

## Visual review candidates (all validation)

- slide076: original YOLO box covers grass/path with no obvious deer. Box center ~756,302 pixels; fixed-size ~120x120. A fallback box is consistent with the source builder, but precise upstream cause remains unverified.
- slide248: original tiny box covers a distant bird-like/ambiguous object; needs human review before declaring its species or deleting it.
- slide206: original label file is empty although a deer-shaped decoy is visible at bottom right. Our detector prediction is counted as a false positive under the existing annotations. Verify target identity against source survey key.
- slide077: one labelled tan deer and a second white animal-shaped decoy; review intended species/exclusion policy rather than automatically labelling it deer.

These observations do not establish the error rate in the training partition. That requires a separate visual audit. Do not silently correct labels or change the ongoing experiment. Both GFL runs retain the original dataset for reproducibility; interpret their validation scores as provisional pending annotation review.

## Upstream builder risks to investigate

The shared dataloaders/decoy_survey/build_dataset.py maps CSV image numbers directly to PDF page numbers, crops photos, and uses point-prompted SAM or a 120px fallback square. Pages with no deer points become negatives. An incorrect page/point mapping, incomplete class key, incorrect prompt, or failed fitted box can therefore create these symptoms. None has yet been confirmed as the root cause. Audit original CSV/PDF identifiers before rebuilding; do not rerun SAM as a substitute for verifying correspondence.

## Evidence

Local ignored artifacts/gfl_annotation_audit/verification.json and slide*_original_labels.jpg contain the numerical check and overlays drawn directly from YOLO source labels. This audit did not modify datasets, checkpoints or the running job.

Before a final comparison, agree on corrections as a team, version a reviewed shared dataset, and evaluate both architectures under one evaluator on a separate flight/site if available. Recalibrate confidence using reviewed validation data. Training-label corrections require new labelled runs; validation-only corrections can be evaluated with existing checkpoints, while preserving original results and documenting selection bias.
