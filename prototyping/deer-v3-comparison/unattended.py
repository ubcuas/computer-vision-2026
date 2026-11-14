"""Run the comparison serially with bounded, checkpoint-aware crash recovery."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workflow import HERE, MODELS, MODEL_SIZES, MODERN_WEIGHTS, candidate_stage, prepare, require_idle_gpu
from dataloaders.deer_shared.common import read_json, write_json

STAGES = ('train', 'val', 'test', 'profile')
EPOCHS = dict(zip(MODELS, (100, 100, 100, 100, 10, 100, 60, 60, 60, 60)))
TRANSIENT = re.compile(
    r'CUDA error: (?!out of memory)|CUBLAS_STATUS_(?:INTERNAL|EXECUTION)_FAILED|'
    r'CUDNN_STATUS_(?:INTERNAL|EXECUTION)_ERROR|device-side assert|unspecified launch failure|'
    r'CUDA-capable device is busy|GPU has active compute processes', re.I)
NO_OUTPUT_SECONDS = 45 * 60


def paths(workspace):
    gfl = Path.home() / 'gfl_workspace'
    return gfl / '.venv-gfl/bin/python', Path.home() / 'code/.venv/bin/python', gfl


def checkpoint_info(workspace, run, model):
    gfl, yolo, _ = paths(workspace)
    python = workspace / '.edge-env/bin/python' if model in MODERN_WEIGHTS else (
        gfl if model.startswith('gfl') else yolo)
    result = subprocess.check_output([str(python), str(HERE / 'checkpoint_recovery.py'),
                                      model, str(run), str(EPOCHS[model])],
                                     text=True, timeout=180)
    return json.loads(result.splitlines()[-1])


def backup_checkpoint(run, checkpoint):
    checkpoint = Path(checkpoint)
    backup = run / 'recovery' / ('resume_checkpoint' + checkpoint.suffix)
    if checkpoint != backup:
        pending = backup.with_suffix('.pending')
        shutil.copy2(checkpoint, pending)
        pending.replace(backup)
    return backup


def snapshot_best(run):
    # Some frameworks reset their best-score tracking on resume. Keep the
    # previous candidate for validation-only selection after training.
    for path in [*run.glob('best_*.pth'), *run.glob('fit/best*.pth'), *run.glob('fit/weights/best.pt')]:
        name = f'{path.stem}-{path.stat().st_mtime_ns}{path.suffix}'
        archive = run / 'recovery/best' / name
        if not archive.is_file():
            archive.parent.mkdir(exist_ok=True)
            pending = archive.with_suffix('.pending')
            shutil.copy2(path, pending)
            pending.replace(archive)


def tail(path, size=32768):
    if not path.is_file():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode(errors='replace')


def wait_for_gpu():
    for _ in range(60):
        try:
            require_idle_gpu()
            return
        except (subprocess.SubprocessError, RuntimeError) as exc:
            print(f'Waiting for GPU: {exc}', flush=True)
            time.sleep(30)
    raise RuntimeError('GPU unavailable for 30 minutes')


def wait_for_existing(workspace):
    recovery = workspace / 'notebook_launch/yolo11s_recovery'
    pid_file = recovery / 'pid'
    if not pid_file.is_file() or (recovery / 'exit_code').is_file():
        return
    pid = int(pid_file.read_text())
    process = Path(f'/proc/{pid}/cmdline')
    print(f'Waiting for existing YOLO11s runner PID {pid}', flush=True)
    while not (recovery / 'exit_code').is_file():
        try:
            if b'run_notebook.py' not in process.read_bytes():
                return
        except FileNotFoundError:
            return
        log = workspace / 'runs/comparison_v1/yolo11s/resume-train.log'
        if log.is_file() and time.time() - log.stat().st_mtime > NO_OUTPUT_SECONDS:
            stop_stalled_existing_training(workspace, pid)
        time.sleep(20)


def stop_stalled_existing_training(workspace, runner):
    # The old notebook kernel has its own process group. Only stop it if
    # its verified descendant is still training this exact YOLO11s run.
    processes = {}
    for line in subprocess.check_output(['ps', '-eo', 'pid=,ppid=,pgid=,args='], text=True).splitlines():
        pid, parent, group, command = line.strip().split(maxsplit=3)
        processes[int(pid)] = (int(parent), int(group), command)
    expected = str(workspace / 'runs/comparison_v1/yolo11s')
    for pid, (parent, group, command) in processes.items():
        if 'train_and_profile.py train yolo11s ' not in command or f'--run {expected} ' not in command:
            continue
        kernel = processes.get(parent)
        if not kernel or 'ipykernel_launcher' not in kernel[2] or group != parent:
            continue
        if processes.get(kernel[0], (None,))[0] == runner:
            print(f'Existing YOLO11s stalled. Stopping its isolated kernel group {group}.', flush=True)
            terminate_group(group)
            return


def terminate_group(group):
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(15)
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_attempt(workspace, model, stage, checkpoint, log, retry):
    cmd = [sys.executable, '-u', __file__, str(workspace), '--stage', stage, '--model', model]
    if checkpoint:
        cmd.extend(('--checkpoint', str(checkpoint)))
    env = dict(os.environ, MPLBACKEND='Agg', DEER_COMPARE_ATTEMPT_LOG=str(log),
               DEER_COMPARE_CUDA_RETRY=str(int(retry)))
    with log.open('a') as stream:
        process = subprocess.Popen(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while process.poll() is None:
            if time.time() - log.stat().st_mtime > NO_OUTPUT_SECONDS:
                terminate_group(process.pid)
                process.wait()
                return process.returncode, '45 minutes without trainer output'
            time.sleep(5)
    return process.returncode, None


def stage_done(run, model, stage):
    if stage == 'train':
        marker = run / 'training_complete.json'
        return marker.is_file() and Path(read_json(marker)['best']).is_file()
    if stage == 'val':
        names = ('val/metrics.json', 'val/profile.json', 'val/threshold_curves.png', 'calibration.json')
    elif stage == 'test':
        names = ('test/metrics.json', 'test/profile.json', 'test/count_errors.png', 'test/error_gallery.png')
    else:
        size = MODEL_SIZES[model]
        names = [f'profile_{size}_fp32/profile.json', f'profile_{size}_fp16/profile.json']
        if model.startswith('yolo26'):
            names += [f'profile_{s}_fp16/sample_metrics.json' for s in (640, 960, 1280)]
    return all((run / name).is_file() for name in names)


def supervise_stage(workspace, run, model, stage, history):
    records = history.setdefault(stage, [])
    if records and records[-1]['state'] == 'running':
        records[-1].update(state='interrupted', recoverable=True)
    while not stage_done(run, model, stage):
        failed = [item for item in records if item['state'] != 'complete']
        if failed and not failed[-1].get('recoverable'):
            return failed[-1]['reason']
        if sum(item['state'] != 'complete' for attempts in history.values() for item in attempts) >= 10:
            return 'reached ten failed attempts for this model'
        wait_for_gpu()
        try:
            info = checkpoint_info(workspace, run, model) if stage == 'train' else {'epoch': 0}
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            return f'checkpoint inspection failed: {exc}'
        epoch = info['epoch']
        stalled = 0
        for item in reversed(failed):
            if epoch > item['epoch_at_start']:
                break
            stalled += 1
        if stalled >= 3:
            return 'three consecutive attempts without a new saved epoch' if stage == 'train' else (
                'three failed attempts at this evaluation stage')
        checkpoint = info.get('checkpoint')
        if stage == 'train' and info['files'] and checkpoint is None:
            return 'checkpoint files exist but none passed the resume check'
        if checkpoint:
            checkpoint = backup_checkpoint(run, checkpoint)
            snapshot_best(run)
        number = len(records) + 1
        log = run / 'recovery' / f'{stage}-{number:02}.log'
        record = dict(state='running', epoch_at_start=epoch, checkpoint=str(checkpoint) if checkpoint else None,
                      log=str(log), started=time.time())
        records.append(record)
        write_json(run / 'recovery/history.json', history)
        if failed:
            time.sleep(min(60 * stalled + 30, 180))
        print(f'{model} {stage} attempt {number}, saved epochs {epoch}: {log}', flush=True)
        retry_cuda = bool(failed) or (stage == 'train' and bool(
            TRANSIENT.search(tail(run / 'train.log') + tail(run / 'resume-train.log'))))
        code, timeout = run_attempt(workspace, model, stage, checkpoint, log, retry_cuda)
        complete = stage_done(run, model, stage)
        output = tail(log)
        retry = bool(timeout or TRANSIENT.search(output))
        reason = timeout or ('CUDA or GPU availability error' if retry else (
            f'exit {code} with missing stage artifacts' if not complete else None))
        record.update(state='complete' if complete else 'failed', recoverable=retry,
                      reason=reason, exit_code=code, finished=time.time())
        write_json(run / 'recovery/history.json', history)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('workspace', type=Path)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--stage', choices=STAGES)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = workspace / 'runs/comparison_v1'
    if args.stage:
        candidate_stage(args.stage, args.model, workspace, workspace / 'data', output,
                        *paths(workspace), unattended=True, checkpoint=args.checkpoint)
        return
    with (workspace / 'unattended.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_file = output / 'unattended.json'
        status = read_json(status_file) if status_file.is_file() else {}
        if not status:
            status['yolo11s'] = dict(state='waiting for existing run')
            write_json(status_file, status)
        wait_for_existing(workspace)
        prepare(workspace)
        for model in MODELS:
            if status.get(model, {}).get('state') in ('complete', 'skipped'):
                continue
            run = output / model
            (run / 'recovery').mkdir(parents=True, exist_ok=True)
            history_file = run / 'recovery/history.json'
            history = read_json(history_file) if history_file.is_file() else {}
            for stage in STAGES:
                status[model] = dict(state='running', stage=stage)
                write_json(status_file, status)
                reason = supervise_stage(workspace, run, model, stage, history)
                if reason:
                    status[model] = dict(state='skipped', stage=stage, reason=reason,
                                         history=str(history_file))
                    print(f'{model} skipped during {stage}: {reason}', flush=True)
                    break
            else:
                status[model] = dict(state='complete')
            write_json(status_file, status)
            # Render from the artifacts only. GPU work is owned by the queue.
            try:
                subprocess.run([sys.executable, str(HERE / 'run_notebook.py')],
                               env=dict(os.environ, DEER_COMPARE_WORKSPACE=str(workspace),
                                        DEER_COMPARE_RUN='0', DEER_COMPARE_REPORT='1'),
                               timeout=600, check=True)
            except subprocess.SubprocessError as exc:
                print(f'Notebook refresh failed: {exc}. Stage artifacts are saved.', flush=True)
        write_json(output / 'unattended_complete.json', status)


if __name__ == '__main__':
    main()
