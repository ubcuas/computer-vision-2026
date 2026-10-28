# YOLO fine-tune: deer detector

- `dataset/`: YOLO dataset of the decoy survey slides, 1 class (`deer`).
  264 images (88 with deer, 176 without), 146 boxes, split 211 train / 53 val.
  Built by `dataloaders/decoy_survey/build_dataset.py`.
- `train.ipynb`: trains YOLO11, plots curves, validates, and shows predictions.

## Training on a GPU server

```bash
git clone https://github.com/ubcuas/computer-vision-2026.git
cd computer-vision-2026/prototyping/yolo-fine-tune

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

nvidia-smi                        # check the GPU is visible
tmux                              # optional: training keeps running if SSH drops (reattach: tmux attach)
yolo detect train data=dataset/data.yaml model=yolo11s.pt imgsz=1280 epochs=100 patience=30 project=runs name=deer
```

Weights end up in `runs/deer/weights/` (`best.pt` = best validation epoch, `last.pt` = final epoch),
with `results.csv` and plots in `runs/deer/`.

Or use the notebook instead of the `yolo` command:

```bash
jupyter lab --no-browser --port 8888
# on your laptop: ssh -L 8888:localhost:8888 user@server, then open the printed localhost:8888 link
```

If `import cv2` fails with `libGL.so.1`, run `sudo apt install -y libgl1`.

The notebook's validation and prediction cells are set up to fit an 8 GB GPU (one image per
`predict` call, `workers=0` for `val`). Training uses AutoBatch, so it adapts to the GPU on its own.
