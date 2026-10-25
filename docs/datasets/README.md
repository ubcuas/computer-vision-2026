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
