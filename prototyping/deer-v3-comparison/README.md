# Deer detector comparison

We evaluated YOLO11s, YOLO26s, GFL and D-FINE-L on Modal. We abandoned the remaining runs because of compute cost.

- [Final notebook](Deer_v3_Comparison.ipynb) shows the validation and test results directly.
- [Experiment record](../../docs/experiments/summaries/2026-11-10-deer-v3-modal-comparison.md) lists checkpoint paths, hashes and evaluation artifacts.
- [Deer v3 data card](../../docs/datasets/deer-v3.md) identifies the exact training and evaluation splits.

`modal_yolo11s.py` preserves the original Modal training and evaluation code. `yolo_crop_eval.py` preserves the separate local inference audit using the saved YOLO26s checkpoint. The historical workstation notebook and its runner are paired at commit `de41fbf`.
