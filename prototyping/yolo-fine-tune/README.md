# YOLO fine-tune: deer detector

- `dataset/`: YOLO dataset of the decoy survey slides, 1 class (`deer`).
  264 images (88 with deer, 176 without), 146 boxes, split 211 train / 53 val.
  Built by `dataloaders/decoy_survey/build_dataset.py`.
- `dataset_awir/`: AWIR-Deer RGB drone photos, 1 class (`deer`).
  53 images (all with deer), 96 boxes, split 42 train / 11 val.
  Built by `dataloaders/awir_deer/convert.py`.
- `data_bambi_decoy.yaml`: experiment configuration that adds converted BAMBI matched RGB training images while retaining the existing decoy validation split.
- `data/bambi_yolo/` at the repository root: generated, ignored BAMBI YOLO dataset built by `dataloaders/bambi_matched/convert.py`.
- `train.ipynb`: trains YOLO11, plots curves, validates on the decoy proxy set, and shows predictions. Its first code cell owns the selected data configuration and run name.

## Training on a GPU server

```bash
git clone https://github.com/ubcuas/computer-vision-2026.git
cd computer-vision-2026/prototyping/yolo-fine-tune

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

nvidia-smi                        # check the GPU is visible
tmux                              # optional: training keeps running if SSH drops (reattach: tmux attach)
yolo detect train data=data_bambi_decoy.yaml model=yolo11s.pt imgsz=1280 epochs=300 patience=50 seed=0 project=runs name=yolo11s_bambi104_decoy_v1
```

Weights end up in `runs/yolo11s_bambi104_decoy_v1/weights/` (`best.pt` = best validation epoch, `last.pt` = final epoch), with `results.csv` and plots beside them. Use a new `name=` for each run so earlier runs are kept. The earlier decoy-plus-AWIR configuration remains available as `data_combined.yaml`; change both `data=` and the run name when intentionally running that experiment.

Or use the notebook instead of the `yolo` command:

```bash
jupyter lab --no-browser --port 8888
# on your laptop: ssh -L 8888:localhost:8888 user@server, then open the printed localhost:8888 link
```

If `import cv2` fails with `libGL.so.1`, run `sudo apt install -y libgl1`.

The notebook's validation and prediction cells are set up to fit an 8 GB GPU (one image per
`predict` call, `workers=0` for `val`). Training uses AutoBatch, so it adapts to the GPU on its own.
