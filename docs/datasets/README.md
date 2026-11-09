# Dataset Register

Document a dataset here before it becomes a shared dependency. Dataset files normally live in approved external storage rather than Git.

## Required information

For each dataset, record:

- canonical name and version;
- source URL and retrieval date;
- owner or maintainer;
- license and redistribution restrictions;
- storage location and integrity information;
- image, annotation, and class formats;
- collection platform, viewpoint, altitude, resolution, and environment when known;
- class counts and important distributions;
- train/validation/test split definition;
- known quality problems, biases, duplicates, and leakage risks;
- preprocessing or conversion steps; and
- experiments that use the dataset.

## Candidate datasets

These candidates came from initial team notes and still require source and license verification.

| Dataset | Potential value | Verification needed |
| --- | --- | --- |
| BAMBI | Multimodal nadir UAV wildlife imagery | Canonical URL, license, deer count, annotations, and relevance to decoys |
| Mississippi State deer-decoy dataset | Plastic deer-decoy imagery; initial notes mark it as previously used | Canonical source, license, capture geometry, annotation format, and exact counts; the notes claim 264 total images but list 112 positive and 141 negative images, which sum to 253 |
| BorealisV2 | Birds-eye imagery that may contain deer | Canonical source, license, number of annotated deer, and class suitability |

Do not call a dataset approved until its provenance and license are verified and the team has defined a stable split. Preserve the final competition-like test set from training and model selection.

The statement that a dataset was previously used is not yet a reproducible experiment record. Before relying on the Mississippi State result, recover or rerun the associated code, split, configuration, metrics, and conclusions.

## Gurnek shared deer split

See [deer final_v1](deer-final-v1.md) and [portable loader](../../dataloaders/deer_final_v1/README.md) for the locally frozen mixed-source split, decoy corrections, 4x training weighting, provenance and evaluation limitations. Source licensing and shared artifact hosting remain unresolved; this is not a blanket redistribution approval.

### Local v2 selection (supersedes v1 training cap)

Following Gurnek's review, a self-contained `UAS/Deer Training Dataset v2` package now uses 2,500 Young Deer training images instead of 1,000; all validation/test assignments remain unchanged. Counts: 4,172 train, 863 validation, 745 test (5,780 total). The 2,500 choice is a practical starting budget, not an experimentally proven optimum or fully reviewed cohort. Package README, checksums, preparation script, source provenance, identical COCO/YOLO split exports and saved decoy corrections are included. No Drive URL is established yet; Gurnek will upload it. The existing deer_final_v1 loader remains a v1 snapshot, not the v2 loader. Use the v2 package's prepare.py after transfer. Do not silently mix versions in a comparison.

## Current Gurnek dataset: v3

[Deer v3 data card](deer-v3.md) supersedes the v1/v2 selections for the next experiments. [Loader and frozen inventory](../../dataloaders/deer_v3/README.md). It includes grouped Young Deer and capture-date AWIR evaluation, site-grouped BAMBI, and source-key-confirmed decoy negatives. Images stay external; the Drive URL is pending Gurnek’s upload. Licensing/quality limitations remain in the package README; do not equate the local frozen split with team-wide approval or pristine competition data.
