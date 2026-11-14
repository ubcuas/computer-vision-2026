# 0002: Start flight trials with full-frame YOLO26s

- Status: proposed for team review
- Owners: Gurnek / CV team
- Related work: [Deer v3 comparison and follow-up](../experiments/summaries/2026-11-10-deer-v3-modal-comparison.md)

## Decision

Use the saved YOLO26s model at input size 1280 as the first Jetson flight-test baseline. Keep GFL as a comparison point and test overlapping crops as a separate second pass. Give the operator a way to add deer the detector misses.

## Rationale and limits

YOLO26s had the highest AP on the common held-out test. GFL had slightly better F1 and count error. D-FINE-L found more labeled deer but added substantially more unmatched boxes. The crop pilot recovered additional deer at the cost of more unmatched boxes and about four times the Mac inference latency. These results support the trial order; they do not establish a deployment winner or measure Jetson performance.

## Review trigger

Revisit after a live Jetson profile and a fresh, whole-flight dry-field holdout. Compare unique deer found, review workload, runtime and retained frames before choosing a deployment mode.
