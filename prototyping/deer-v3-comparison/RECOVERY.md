# Leaving the comparison running

The detached user service on greglinux is called `deer-v3-unattended.service`.
Follow it with `journalctl --user -fu deer-v3-unattended.service`. The workspace
is `/home/mrd/deer_v3_compare`. Under `runs/comparison_v1`, the queue writes
`unattended.json`, each model writes `recovery/history.json` and attempt logs,
and the queue writes `unattended_complete.json` once all models have finished or
been skipped. The executed notebook refreshes after each model.

The queue runs one model at a time. A recoverable CUDA crash or a 45-minute
silence starts a fresh process from the latest loadable optimizer checkpoint.
It can retry the original initialization if no checkpoint has been written.
A model is skipped after three attempts at the same stage without progress,
ten failures across its stages, or a deterministic failure such as CUDA out of memory. For a skipped
model the history records the reason. Completed stages and model files remain
available. Retrying evaluation does not retrain the model.

Training keeps the original dataset, size and precision. Training retries use
synchronous CUDA launches for clearer failure logs. Failed attempts can lose
work since the last saved epoch. For frameworks that can reset their best score
on resume, the queue keeps pre-resume best checkpoints and compares candidates
on validation before running the held-out test.

The service survives an SSH disconnect and can restart after a supervisor crash.
If greglinux reboots, start the queue again once the user session is available.
With the service stopped, rerun the same launcher command on greglinux.

```sh
systemd-run --user --unit=deer-v3-unattended \
  --property=Restart=on-failure --property=RestartSec=120 \
  --property=StartLimitIntervalSec=1800 --property=StartLimitBurst=3 \
  /home/mrd/gfl_workspace/.venv-gfl/bin/python -u \
  /home/mrd/deer_v3_compare/code/prototyping/deer-v3-comparison/unattended.py \
  /home/mrd/deer_v3_compare
```
