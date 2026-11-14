"""Find an intact, resumable epoch in the training framework's own environment."""
import argparse
import inspect
import json
import pickle
from pathlib import Path
import sys
import zipfile

import torch


def candidates(run, model):
    fit = run / 'fit'
    backup = run / 'recovery' / 'resume_checkpoint'
    if model.startswith('yolo'):
        weights = fit / 'weights'
        return [weights / 'last.pt', *weights.glob('epoch*.pt'), backup.with_suffix('.pt'), run / 'before_resume/last.pt']
    if model.startswith('gfl') or model == 'rtmdet_tiny':
        return [*run.glob('epoch_*.pth'), backup.with_suffix('.pth')]
    return [fit / 'last.pth', *fit.glob('checkpoint*.pth'), backup.with_suffix('.pth')]


def epoch_of(path, model, limit):
    # These are local checkpoints produced by our own trainers.
    options = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    ckpt = torch.load(path, map_location='cpu', **options)
    if model.startswith('yolo'):
        epoch = ckpt['epoch'] + 1
        ready = ckpt.get('optimizer') and ckpt.get('ema') is not None and ckpt.get('train_args')
        if epoch >= limit:  # Ultralytics rejects resume after the last scheduled epoch.
            return None
    elif model.startswith('gfl') or model == 'rtmdet_tiny':
        epoch = ckpt['meta']['epoch']
        ready = ckpt.get('optimizer') and ckpt.get('state_dict')
    else:
        epoch = ckpt['last_epoch'] + 1
        ready = ckpt.get('optimizer') and ckpt.get('model') and ckpt.get('lr_scheduler')
    return epoch if ready and 0 < epoch <= limit else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('run', type=Path)
    parser.add_argument('limit', type=int)
    args = parser.parse_args()
    files = [p for p in candidates(args.run, args.model) if p.is_file()]
    found = []
    vendor_log = args.run / 'fit/log.txt'
    vendor_epochs = set()
    if args.model.startswith('rtdetr') or args.model == 'dfine_l':
        if vendor_log.is_file():
            for line in vendor_log.read_text().splitlines():
                try:
                    vendor_epochs.add(json.loads(line)['epoch'] + 1)
                except (ValueError, KeyError):
                    continue
    for path in files:
        try:
            epoch = epoch_of(path, args.model, args.limit)
        except (OSError, ValueError, RuntimeError, KeyError, EOFError, TypeError, zipfile.BadZipFile,
                pickle.UnpicklingError) as exc:
            print(f'Cannot load {path}: {exc}', file=sys.stderr)
            continue
        if epoch == args.limit and (args.model.startswith('rtdetr') or args.model == 'dfine_l') and epoch not in vendor_epochs:
            continue
        if epoch:
            found.append((epoch, path.stat().st_mtime_ns, str(path)))
            if path.name in ('last.pt', 'last.pth'):
                break
    selected = max(found, default=None)
    print(json.dumps({'files': len(files), 'epoch': selected[0] if selected else 0,
                      'checkpoint': selected[2] if selected else None}))


if __name__ == '__main__':
    main()
