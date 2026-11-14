# Deer dataset final_v1

Status: Gurnek-approved local split for the next detector experiments, 2026-11-08; not a team-wide policy or certified competition benchmark. Maintainer: Gurnek. Loader: ../../dataloaders/deer_final_v1/. Exact counts and split checksums: loader summary.json. Files stored externally in `UAS/Deer Dataset Review`; shared hosting/transfer and source license verification remain outstanding. No training run uses this version yet.

## Intentional choices

We are fine-tuning pretrained single-class deer detectors, not training from scratch. Prioritize aerial RGB/open-field scenes for a summer Medicine Hat deployment (brown and green grass); this operating-scene expectation is a team assumption, not a new rule interpretation. This dataset only addresses per-frame detection, not temporal counting, clustering or tag recognition.

The raw combined candidate pool has 35,604 images. The frozen training subset caps Young Deer at 1,000 and BAMBI at 20 time-spaced frames per training flight, preventing frame count alone from dictating source influence. Other candidates remain available outside these lists. BAMBI whole flights remain in one partition; existing test flights reserved. Twelve source training flights augment its original one-flight validation. Young Deer and AWIR are training-only because original scene/augmentation independence cannot be reliably established. Sherbrooke retains provided summer partitions.

Decoy eligible view groups use approximately 70/15/15 train/validation/test, deterministic seed 20261109. Group sizes mean image proportions need not equal these percentages. Original SIFT groups are heuristic. Training decoys receive 4x sampling weight, not copied files; validation/test are never oversampled into training. This is a starting hypothesis for adaptation, not evidence that 4x is optimal. Retrain YOLO and GFL from official COCO checkpoints. Previously trained decoy weights must not initialize these runs.

## Annotation learnings

Mississippi decoys and Mississippi AWIR are different datasets. Decoy imagery matches competition targets well but some original boxes contain too much ground/shadow or omit visible animal parts. User reviewed the imagery; 26 saved image reviews are verifiable on disk, and saved corrections override originals. Unsaved positive labels retain original extents; do not claim every label was edited. Unsaved empty decoy labels are excluded until checked. Wolf/coyote decoys are not deer. Tight visible-animal boxes exclude cast shadows; no mechanical uniform box shrinking.

BAMBI uses matched RGB labels only: unmatched images are not safe negatives. Young Deer contributes fawns on mostly green vegetation, including rotated/padded derivatives; map only fawn class 1, exclude person class 2. Sherbrooke has crowded/shadowed scenes; cached training positives do not represent a complete negative distribution. AWIR has repeated pen views. Thermal data, unlabelled Swamp Deer and mixed-view augmented Rucervus/Deer Detection collections are excluded.

YOLO class 0 and COCO category 1 both mean deer. Coordinates were compared against source annotations for all 60,604 pool boxes; conversion passed with negligible rounding. JPEG images are unchanged; copied via independent APFS clones locally. COCO and YOLO exports use the same underlying boxes. Split and preprocessing are separate choices: resolution/augmentation must be documented for every model.

## How to interpret evaluation

Report BAMBI and decoy test metrics separately, plus per-source validation slices. Decoy test groups are held out from NEW runs, but this same site/data was already used in prior training, model selection and manual exploration. They are not a pristine external test or proof of competition generalization. Flights may share sites/animals. Young Deer source-family provenance is unresolved. Exact-byte cross-partition duplicates were checked; perceptual/cross-source duplicate audits and verified brown/green/snow scene stratification remain incomplete. Calling this version final freezes its files, not its scientific limitations.

Choose thresholds and checkpoints using validation only. Test once after selection; preserve P/R/F1 at IoU .5, AP50, AP50–95, count errors, latency and per-source failure examples. Any future label/split edits require a new dataset version. Keep eventual independently collected competition-like decoys for a stronger external test.

## Source provenance

Local files were already downloaded; exact retrieval dates/licenses are not fully verified. Sources: BAMBI https://github.com/bambi-eco/Dataset (matched RGB release recorded as Zenodo 19034999); Young Deer https://zenodo.org/records/18206908; Sherbrooke https://doi.org/10.5683/SP3/TELIYD; AWIR https://scholarsjunction.msstate.edu/gri-publications/2/; decoys local original PDF `Master1_no_key_0s_randomized.pdf` and point CSV, via Anant's repository dataset. Verify licenses before redistribution. Point CSV was not originally bounding-box annotation; repository boxes and later human corrections are derived annotations.
