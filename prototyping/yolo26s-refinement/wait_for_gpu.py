"""Wait for other compute jobs to leave the GPU; never stop their processes."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def busy_rows(output):
    active = []
    for line in output.splitlines():
        parts = [s.strip() for s in line.split(',')]
        if len(parts) == 3 and parts[1] == '/usr/libexec/gnome-remote-desktop-daemon':
            try:
                if int(parts[2].split()[0]) <= 256:
                    continue
            except ValueError:
                pass
        if line.strip():
            active.append(line)
    return active


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=int, required=True)
    p.add_argument('--status', type=Path, required=True)
    args = p.parse_args()
    deadline = time.monotonic() + args.seconds
    while True:
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                          '--format=csv,noheader'], text=True, timeout=15)
        active = busy_rows(output)
        state = dict(state='waiting_for_other_gpu_processes' if active else 'gpu_available',
                     observed_at=time.time(), active=active)
        args.status.write_text(json.dumps(state, indent=2))
        if not active:
            print('GPU available; continuing the pipeline.', flush=True)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('GPU still in use after the allowed queue time; no jobs stopped.')
        print('Waiting for GPU: ' + '; '.join(active), flush=True)
        time.sleep(min(60, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
