# Shared deer dataset final_v1

Dataset images and annotations stay external. Obtain `Deer Dataset Review` from Gurnek; no hosted download location has been established. Do not upload images or weights to ordinary Git. Preserve `images/`, `labels/`, `annotations/deer.coco.json`.

Run from repository root:

```sh
python dataloaders/deer_final_v1/prepare.py --root '/path/to/Deer Dataset Review' --output /path/to/run-data
```

The generated data.yaml uses 4x decoy training sampling, validation/test once each. For GFL use generated train/val/test.coco.json and the external images directory; implement matching sampling weights: 4 for `mississippi_decoy__` filenames, 1 otherwise. Do not use an unweighted GFL run as an architecture-only comparison to weighted YOLO. The optional train.txt is unweighted. There is no image resizing in this loader; agree and record identical evaluation preprocessing separately.

See ../../docs/datasets/deer-final-v1.md for scope, provenance and limitations. Split lists are relative and versioned; file hashes are in summary.json. The local preparation command verifies split list hashes, image/label existence and image disjointness. It does not certify annotation quality or enforce an image-content checksum on every file.
