#!/usr/bin/env bash
# Run inside tmux or a detached service. No automatic retries after a GPU failure.
set -euo pipefail
PYTHON="${1:?Python executable}"
DATA="${2:?Extracted Deer v3 directory}"
RUN="${3:?New run directory}"
WEIGHTS="${4:?Original YOLO26s deer checkpoint}"
SCRIPT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run.py"
export MPLBACKEND=Agg
if [[ "${DEER_WAIT_FOR_GPU_SECONDS:-0}" -gt 0 ]]; then
    "$PYTHON" -u "$(dirname -- "$SCRIPT")/wait_for_gpu.py" --seconds "$DEER_WAIT_FOR_GPU_SECONDS" --status "${RUN}.queue.json"
fi
if [[ ! -f "$RUN/manifest.json" ]]; then
    "$PYTHON" -u "$SCRIPT" prepare --data "$DATA" --run "$RUN" --weights "$WEIGHTS"
fi
if [[ ! -f "$RUN/smoke_complete.json" ]]; then
    "$PYTHON" -u "$SCRIPT" smoke --data "$DATA" --run "$RUN"
fi
if [[ ! -f "$RUN/training_complete.json" ]]; then
    if [[ -f "$RUN/fit/weights/last.pt" ]]; then
        "$PYTHON" -u "$SCRIPT" train --data "$DATA" --run "$RUN" --resume
    else
        "$PYTHON" -u "$SCRIPT" train --data "$DATA" --run "$RUN"
    fi
fi
"$PYTHON" -u "$SCRIPT" evaluate --data "$DATA" --run "$RUN"
# Historical test regression; thresholds are already locked by validation.
"$PYTHON" -u "$SCRIPT" evaluate --data "$DATA" --run "$RUN" --test
"$PYTHON" -c 'import json,sys,time,pathlib; p=pathlib.Path(sys.argv[1]); (p/"pipeline_complete.json").write_text(json.dumps({"finished_at":time.time(),"test_is_historical_regression":True},indent=2))' "$RUN"
