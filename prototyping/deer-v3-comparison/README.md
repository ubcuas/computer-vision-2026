# Deer detector comparison

The [experiment record](../../docs/experiments/summaries/2026-11-10-deer-v3-modal-comparison.md) has the measured YOLO11s, YOLO26s, GFL and D-FINE-L results, checkpoint locations and crop audit. [Deer v3](../../docs/datasets/deer-v3.md) identifies the exact training and evaluation data. The notebook preserves the original experiment plan, including candidates we stopped or never finished; use the experiment record for conclusions.

- `modal_yolo11s.py`: A100 entry points and container dependencies. `main` is the historical YOLO11s resume (requires its seed volume); `launch_parallel` uses the staged dataset and per-model output volumes; `evaluate_dfine` evaluates a saved checkpoint.
- `Deer_v3_Comparison.ipynb`: original workstation notebook. GPU execution is opt-in via `DEER_COMPARE_RUN=1`. See [recovery notes](RECOVERY.md) for the original host setup.
- `yolo_crop_eval.py`: inference-only full-frame vs overlapping crops using existing YOLO26s weights.

## Run with the saved artifacts

Request the final ZIP or Drive file ID from the CV team. For the workstation workflow, put the archive at `$DEER_COMPARE_WORKSPACE/input/deer-v3.drive.zip` or set `DEER_V3_DRIVE_ID` and provide `gdown` in `$DEER_COMPARE_WORKSPACE/.download-env`. For Modal, the existing `uas-deer-v3-dataset` volume holds `input/deer-v3.drive.zip` and `input/weights/` (COCO starting weights; see `workflow.py` for filenames). To stage it on another workspace, provision those paths before launching. Use a separate output workspace or volumes for new experiments so the completed run stays intact.

The historical entry point accepts explicit model names, for example:

```sh
modal run --detach --env main modal_yolo11s.py::launch_parallel --models yolo26s
```

Run it from this directory only when you intend to use a billable GPU. Existing run markers may cause the original workspace to reuse or skip stages. For reviewing results or running the crop pilot, use the saved checkpoint instead of launching training:

```sh
python yolo_crop_eval.py --data /path/to/extracted-v3 --weights /path/to/best.pt \
  --output /path/to/new-crop-report --device mps
```

The crop pilot needs PyTorch, Ultralytics, OpenCV and NumPy. Use `--device cuda:0` on a CUDA host. GitHub Actions checks the notebook, frozen inventory, and existing CPU pipeline tests without the dataset, weights, or cloud credentials.
