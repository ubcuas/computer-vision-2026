# 0001: Freeze Gurnek detector dataset v3

- Status: accepted for Gurnek’s requested experiments; team-wide adoption pending review
- Date: 2026-11-08
- Owner: Gurnek Sandhu

## Context and decision

Earlier source-count caps were practical guesses. Expanded review found extensive Young Deer augmentation and repeated BAMBI locations across flights. Freeze the source-aware v3 split and coordinate-consistent COCO/YOLO exports. Young Deer inferred families, BAMBI locations, AWIR capture dates and heuristic decoy view groups determine partitions. Keep key-confirmed decoy negatives and saved corrections; use 4x TRAIN-only decoy exposure consistently in both architectures. See ../datasets/deer-v3.md and ../../dataloaders/deer_v3/README.md.

## Alternatives and consequences

Random file splitting risks evaluating transformed copies of training content. Author flight splits alone do not separate known BAMBI sites. Larger raw file counts can overstate diversity. The resulting held-out image counts are smaller and not a literal 70/15/15 ratio. Source-specific results are mandatory; same-site decoys/AWIR, prior decoy exposure and inferred Young Deer families limit external validity.

## Review trigger

New original-scene metadata, verified labelling errors, evidence of cross-partition relatives, or newly collected competition-like imagery requires a new dataset version. Never silently overwrite this split after comparing test results. Team source-license/storage review remains separate from this experiment-specific decision.
