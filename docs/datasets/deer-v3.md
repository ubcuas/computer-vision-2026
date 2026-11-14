# Deer dataset preparation

[Deer Data Preparation on Google Docs](https://docs.google.com/document/d/1f6ofkU0H8atJxT8Gj9dldD2-_1Vpz2T3f2wGQT5W2uM/edit) · [Deer Training Dataset v3.zip](https://drive.google.com/file/d/12UtXf_Jfgh9FkDu1e-sWtKCx2sEkzX_s/view?usp=sharing)

I put Deer v3 together so GFL and YOLO could use the same images, labels and splits. This dataset covers deer detection. For the competition we expect mostly top-down views of decoys in an open field in Medicine Hat, with either dry yellow grass or greener cover.

## What I noticed in each source

- **Mississippi Decoy Deer.** Our closest match and only decoy source. Some boxes included too much grass or shadow, so I corrected the boxes by hand. Coyote and bird decoys are background. I checked the original point key before keeping frames with no deer.
- **Young Deer.** Good aerial views and generally tight boxes, mostly over green vegetation. Many images are rotated, blurred or color-adjusted versions of related views.
- **BAMBI.** Useful aerial footage over open brown ground, with repeated views from video. Snow and dense woodland are less representative of our field.
- **AWIR.** A small RGB set with generally reasonable boxes and repeated views of deer pens.
- **Sherbrooke.** Summer aerial images with varied deer sizes, crowded scenes and labeled backgrounds. Some frames have heavy tree cover or shadows.
- **Rucervus / Swamp Deer.** Overhead views in tan vegetation, dry grass and greener field edges. I used the separate COCO export for labels. I reviewed 1,029 candidates and kept 454. The review filtered out front-on photographs, collages and heavily transformed examples.

## How we selected and split the data

I started with 35,604 candidate images and added the Rucervus review later. For the initial sources, we reviewed four random samples of 20 labeled images per source, then expanded the review to cover repeated scenes, capture groups and evaluation images. We removed obvious snow frames, some degraded or unverifiable images, and narrow fragments from the decoy PDF. Original files and review decisions were preserved.

We kept related views together. Young Deer used inferred scene families, BAMBI locations, AWIR capture dates, and the decoys inferred groups of related views. For Rucervus, related filename sequences stayed together and geometric image matching linked additional scenes. Exact duplicate checks and approximate image retrieval helped remove Young Deer matches across splits.

## Final dataset

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| Young Deer | 1,570 | 48 | 60 |
| BAMBI | 831 | 44 | 56 |
| Mississippi decoys | 139 | 31 | 28 |
| AWIR RGB | 34 | 6 | 13 |
| Sherbrooke summer | 328 | 147 | 0 |
| Rucervus reviewed aerial | 291 | 82 | 81 |
| **Total** | 3,193 | 358 | 238 |

The final archive has **3,789 images**. YOLO and COCO exports represent the same labels. Training decoys get four times the exposure using the training list. Validation and test keep their original counts. This release supersedes the 3,335-image v3 draft.

## Limits and reproducibility

Competition-like yellow grass remains underrepresented, and the decoys come from one field that was used in earlier experiments. BAMBI test covers one held-out location. AWIR was split across capture dates at the same site. Young Deer validation and test cover 16 and 20 inferred families respectively, and Rucervus lacks verified flight or site IDs. Related frames are not independent. For the original five sources we checked source-to-export geometry, dimensions, classes, bounds and split separation. Rucervus review retained the supplied boxes. The archive README lists sources and attribution. Access belongs with the CV team while redistribution permissions are still being confirmed.

After extracting, run `python prepare.py --verify`. YOLO reads `runtime/data.yaml`. GFL uses the MMDetection helper at `runtime/mmdet_datasets.py`. The metadata gives the archive SHA-256 and the loader records the frozen split. Freeze future label or split changes under a new version. [Artifact metadata](../../dataloaders/deer_v3/artifact.json) · [Loader and frozen inventory](../../dataloaders/deer_v3/README.md) · [Modal experiment record](../experiments/summaries/2026-11-10-deer-v3-modal-comparison.md)
