# BAMBI matched RGB to YOLO

Converts complete flights from the reviewed BAMBI `matched` release into a one-class YOLO detection dataset.

The converter uses RGB images only. It keeps tracks whose metadata says `status: accepted`, whose accepted modalities include `rgb`, and whose class is `red deer` or `red deer (male)`. Both classes become YOLO class `0` (`deer`). Images without an accepted RGB deer receive an empty label file and act as negative examples.

## Why this converter exists

BAMBI already publishes two related YOLO resources, but neither is a drop-in replacement for this RGB red-deer experiment:

- The BAMBI repository includes `mot_to_yolo.py`, a useful general conversion utility for its 13-column base MOT annotations. The reviewed `matched` Flight 104 files use a different 10-column layout with class fields set to `-1`, store RGB acceptance and deer class information in `104_labels.json`, name extracted frames independently, and need empty label files for negative images. This converter joins those sources and produces the exact one-class layout used here.
- The separate [BAMBI Detection Dataset](https://doi.org/10.5281/zenodo.15773102) is already distributed in YOLOv12 format, but it contains thermal imagery and combines red deer, roe deer, and wild boar into one `animal` class. It is a different dataset and task from the reviewed RGB red-deer subset used by this experiment.

Prefer the upstream tools when their data and label semantics match the intended experiment. Do not convert or duplicate data merely to reproduce an existing suitable release.

## Split policy

Assign complete flights to one split. Never divide frames from the same flight between training and validation because neighbouring frames contain the same animals and nearly identical scenery.

With only Flight 104 available, use all of it for training and retain the existing decoy-survey validation set as the experiment's proxy validation set. When another suitable BAMBI flight is available, reserve the complete additional flight for BAMBI-specific validation.

## Convert Flight 104

Run from the repository root:

```powershell
python dataloaders/bambi_matched/convert.py `
  --train-flight data/bambi/flight_104 `
  --out data/bambi_yolo
```

The output is under the ignored root `data/` directory and should not be committed.

```text
data/bambi_yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── preview/
│   ├── train/
│   └── val/
├── data.yaml
├── manifest.csv
└── summary.json
```

Review every generated file under `preview/` before training. Green rectangles must surround RGB deer, and images marked `NO LABELLED RGB DEER` should not contain an obvious unlabelled deer.

To rebuild an existing output after intentionally changing inputs or conversion logic, add `--overwrite`. This deletes only the specified output directory and never modifies the downloaded BAMBI flight.

## Add a held-out BAMBI flight later

```powershell
python dataloaders/bambi_matched/convert.py `
  --train-flight data/bambi/flight_104 `
  --val-flight data/bambi/flight_132 `
  --out data/bambi_yolo `
  --overwrite
```

The committed experiment configuration `prototyping/yolo-fine-tune/data_bambi_decoy.yaml` trains on the existing decoy training set plus converted BAMBI training flights. It deliberately selects checkpoints using the unchanged decoy validation set so the comparison with the decoy-only baseline is interpretable.
