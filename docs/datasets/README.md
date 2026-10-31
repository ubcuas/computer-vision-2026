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
| BAMBI matched RGB | Multimodal nadir UAV wildlife imagery. The reviewed `matched` release provides paired RGB/thermal frames and MOT labels for red deer. Flight 104 was inspected locally: 160 extracted RGB frames, 14 RGB-accepted tracks, 107 positive frames, 53 negative frames, and 206 RGB boxes after matching annotations to extracted frames. | Dataset files are CC BY 4.0; repository tools are MIT. Canonical repository: <https://github.com/bambi-eco/Dataset>. Matched Part 1: <https://doi.org/10.5281/zenodo.21291168>. The repository includes a general `mot_to_yolo.py`, but matched Flight 104 also requires joining its 10-column RGB MOT file with RGB acceptance/class metadata and extracted-frame names. Real deer, Austrian environments, and correlated video frames limit direct relevance to plastic decoys. Split by complete flight and visually review converted labels before use. |
| BAMBI Detection Dataset v1.0.0 | Ready-made YOLOv12 detection dataset with 19,252 thermal frames and a video/location-grouped train/validation/test split. | <https://doi.org/10.5281/zenodo.15773102>. The release combines red deer, roe deer, and wild boar into one `animal` class and is thermal-only, so it is not interchangeable with the RGB, deer-only competition experiment. The Zenodo record does not currently display an explicit license value; verify usage terms before adoption. |
| Mississippi State deer-decoy dataset | Plastic deer-decoy imagery; initial notes mark it as previously used | Canonical source, license, capture geometry, annotation format, and exact counts; the notes claim 264 total images but list 112 positive and 141 negative images, which sum to 253 |
| BorealisV2 | Birds-eye imagery that may contain deer | Canonical source, license, number of annotated deer, and class suitability |

Do not call a dataset approved until its provenance and license are verified and the team has defined a stable split. Preserve the final competition-like test set from training and model selection.

The statement that a dataset was previously used is not yet a reproducible experiment record. Before relying on the Mississippi State result, recover or rerun the associated code, split, configuration, metrics, and conclusions.
