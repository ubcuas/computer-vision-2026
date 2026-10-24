# Decoy survey → YOLO dataset

Converts the decoy survey PDF (one drone photo per slide) and its point-label CSV into a YOLO dataset.

```bash
python build_dataset.py --pdf Master1_no_key_0s_randomized.pdf --csv decoy_survey_key_points.csv \
    --out ../../prototyping/yolo-fine-tune/dataset --negatives          # --classes all for every class
```

Needs `pdftoppm` (poppler) on PATH.

- `SlideN.JPG` in the CSV means page N of the PDF, not the number printed on the slide.
- `x, y` are pixel coordinates on the slide rendered at 300 DPI.
- The labels are points. Each one becomes a box using MobileSAM, falling back to a 120 px box if SAM fails.
  Results are cached in `sam_boxes_cache.json`, so rebuilding the deer set needs no SAM runs.
