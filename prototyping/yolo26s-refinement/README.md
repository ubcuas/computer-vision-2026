# YOLO26s refinement

[Notebook](YOLO26s_Refinement.ipynb) · [Experiment record](../../docs/experiments/summaries/2026-11-20-yolo26s-refinement-preparation.md) · [Recipe](recipe.json)

The first attempt failed before saving an epoch. Recovery is queued as `deer-yolo26s-refinement-recovery-1.service` on greglinux. Open the notebook there to view progress; leave `START_TRAINING=False` so it doesn't start another job.

- Code: `/home/mrd/yolo26s_refinement/code`
- Data: `/home/mrd/deer-data/Deer Training Dataset v3`
- Outputs: `/home/mrd/yolo26s_refinement/run1`
- Log: `/home/mrd/yolo26s_refinement/recovery-pipeline.log`
- Python: `/home/mrd/code/computer-vision-2026-v4/.venv/bin/python`

## Running it again

Use the existing working CUDA environment (Ultralytics 8.4.155), extracted v3 data, saved deer checkpoint and a **new** output directory. No downloads are built in. The notebook explains the training changes and evaluation outputs; the experiment record has the input checkpoint hash.

```bash
bash prototyping/yolo26s-refinement/launch_server.sh "$PYTHON" "$DEER_DATA" "$DEER_RUN" "$DEER_WEIGHTS"
```

Run inside tmux or a detached service. The script prepares crops, smoke-tests, trains, validates, then evaluates the historical test with validation-selected thresholds. It stops on failure. Rerunning explicitly restores `fit/weights/last.pt` if training was interrupted; it doesn't discard progress.

For individual stages, use `run.py prepare --data DATA --run RUN --weights WEIGHTS`, then `smoke`, `train`, and `evaluate` with the same data/run paths. `train --resume` restores the saved optimizer/schedule. `evaluate --test` requires validation calibration first.

Keep the run directory: it contains best/last and epoch weights, configuration, provenance, graphs, prediction JSON, metrics and memory/latency profiles. `pipeline_complete.json` marks completion. We can edit notebook commentary later without retraining. Do not commit datasets or outputs to Git.

Set `DEER_WAIT_FOR_GPU_SECONDS=43200` to queue for up to 12 hours before starting. The queue checks compute processes once a minute and never stops them. Recovery state is recorded next to the run as `run1.queue.json`.
