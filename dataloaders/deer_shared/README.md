# Shared deer preparation and evaluation

`anant.py` converts the existing committed YOLO dataset to COCO without altering pixels, annotations or splits. `prepare.py` and `finalize.py` preserve Gurnek's original wildlife/decoy preparation methodology. `common.py` supplies validation/export and an external DEER_WORKSPACE root. `metrics.py` provides common IoU-0.5 detection/count metrics, COCO AP and validation-only threshold calibration.

The exact-data baseline uses Anant's manifest, not `metadata/decoy_shared_v1.json`. That earlier grouped manifest is retained as provenance for the current wildlife-data experiment and is not a team-approved replacement split. Annotation metadata is agent reviewed, not human reviewed. Dataset redistribution/license approval and a final team evaluation contract remain unresolved. No raw data or weights are added here.
