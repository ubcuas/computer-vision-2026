# Shared deer v3 loader and frozen inventory

Use this version, superseding deer_final_v1 and the local v2 package. Obtain `Deer Training Dataset v3` from Gurnek/team Drive (URL not yet supplied). Images/labels remain external.

```sh
python dataloaders/deer_v3/prepare.py --root '/path/to/Deer Training Dataset v3' --verify
```

This writes local runtime/data.yaml for YOLO and runtime/mmdet_datasets.py for MMDetection 2.x. Call training_dataset(pipeline) and evaluation_dataset(split,pipeline) in GFL configs. Both routes repeat all TRAIN decoys four times, preserve negative images, and never mix held-out images into training. The lists under splits/ are relative, frozen references; hashes and source image hashes are committed. `--verify` checks the package's images, labels, annotations, splits and metadata against checksums.json. It does not authenticate an independently modified checksum file; compare its hash with the repository CHECKSUMS.sha256 after transfer.

No hosted downloader can be configured until a Drive URL is provided. Download/extract once, then reuse the local cache. See ../../docs/datasets/deer-v3.md for counts, data choices and limitations. No training has been run with this dataset version.
