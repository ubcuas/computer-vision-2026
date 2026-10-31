# Annotation conversion verification — 2026-10-31

Gurnek confirmed the target annotations, then clarified that bounding boxes are too large. Treat oversized/loose box extents as a dataset-quality issue requiring better reviewed annotations. Earlier automated species/missing-target suspicions remain withdrawn. Keep the current experiment dataset unchanged for reproducibility.

Verified against actual server COCO train/validation files: all 264 image/label source SHA256 hashes match the committed YOLO dataset inventory; all 146 converted boxes match direct normalized YOLO-to-pixel COCO conversion with zero pixel difference. Splits and empty labels are preserved. The dataset directory matches the locally fetched origin/deer-yolo-dataset branch.

The original survey key also identifies the white animals in slide206/slide077 as coyotes; their exclusion is intentional for deer-only labels. The model's deer predictions on those coyotes are classification errors, not missing deer annotations.

Local ignored artifacts/gfl_annotation_audit contains conversion verification and visual overlays. These overlays were exploratory and are not an annotation correction list. Neither datasets nor running training were changed.

## Bounding-box quality issue identified by Gurnek

Some supplied boxes are too large and contain excessive background. We need better data: review box extents against the visible deer, use consistent rules for occlusion and truncated animals, and version corrected annotations jointly before a new comparison. Oversized boxes can reduce localization precision and may increase overlap between nearby detections. This is a plausible contributor to Anant's reported overlapping boxes, not a demonstrated root cause; duplicate detections, NMS settings and model predictions also need inspection. Do not attribute all overlapping predictions to labels without comparing the actual examples.

This affects both architectures using the shared labels. Correct species/point identity does not guarantee a good fitted bounding box. Record original-data results separately from future corrected-data runs.
