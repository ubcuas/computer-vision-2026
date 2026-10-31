# Annotation conversion verification — 2026-10-31

Gurnek confirmed the supplied annotations are correct. Earlier automated visual suspicions are withdrawn; do not treat the generated review shortlist as established annotation errors. Keep the dataset unchanged and count errors against its supplied ground truth.

Verified against actual server COCO train/validation files: all 264 image/label source SHA256 hashes match the committed YOLO dataset inventory; all 146 converted boxes match direct normalized YOLO-to-pixel COCO conversion with zero pixel difference. Splits and empty labels are preserved. The dataset directory matches the locally fetched origin/deer-yolo-dataset branch.

The original survey key also identifies the white animals in slide206/slide077 as coyotes; their exclusion is intentional for deer-only labels. The model's deer predictions on those coyotes are classification errors, not missing deer annotations.

Local ignored artifacts/gfl_annotation_audit contains conversion verification and visual overlays. These overlays were exploratory and are not an annotation correction list. Neither datasets nor running training were changed.
