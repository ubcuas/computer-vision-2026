"""Notebook helpers for staging, serialized GPU jobs, and shared evaluation."""
from collections import Counter, defaultdict
import csv
import fcntl
from pathlib import Path, PurePosixPath
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile

from dataloaders.deer_shared.common import digest, read_json, write_json

SPLITS = {'train': 3193, 'val': 358, 'test': 238}
MODERN_WEIGHTS = {'rtmdet_tiny': 'rtmdet-tiny.pth', 'rtdetrv2_r18': 'rtdetrv2-r18.pth',
                  'dfine_l': 'dfine-l.pth', 'rtdetrv2_r50': 'rtdetrv2-r50.pth'}
YOLO26_RESOLUTION_RUNS = ('yolo26s', 'yolo26s_960', 'yolo26s_640')
MODELS = ('yolo11s', *YOLO26_RESOLUTION_RUNS, 'gfl_short', 'gfl_long', *MODERN_WEIGHTS)
MODEL_SIZES = dict.fromkeys(MODELS, 1280)
MODEL_SIZES.update(yolo26s_960=960, yolo26s_640=640)
GFL_WEIGHT = 'gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco_20200630_102002-14a2bf25.pth'
HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]


def prepare(workspace, *, downloader=None, drive_id=None):
    """Download once and extract once. Finder metadata is omitted."""
    data = workspace / 'data'
    archive = workspace / 'input/deer-v3.drive.zip'
    if not data.exists():
        if not archive.exists():
            drive_id = drive_id or os.environ.get('DEER_V3_DRIVE_ID')
            if not drive_id:
                raise FileNotFoundError(f'Place the team ZIP at {archive} or set DEER_V3_DRIVE_ID.')
            archive.parent.mkdir(parents=True, exist_ok=True)
            downloader = downloader or workspace / '.download-env/bin/gdown'
            pending = archive.with_suffix('.partial')
            subprocess.run([str(downloader), '--continue', drive_id, '-O', str(pending)], check=True)
            pending.replace(archive)
        with tempfile.TemporaryDirectory(dir=workspace) as tmp, zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                path = PurePosixPath(member.filename)
                if not path.parts or path.parts[0] == '__MACOSX' or path.name == '.DS_Store':
                    continue
                if path.is_absolute() or '..' in path.parts or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f'Unexpected archive member {member.filename}')
                z.extract(member, tmp)
            extracted = Path(tmp) / 'Deer Training Dataset v3'
            if not (extracted / 'prepare.py').is_file():
                raise ValueError('The ZIP is missing the Deer Training Dataset v3 folder')
            extracted.replace(data)
    subprocess.run([sys.executable, str(data / 'prepare.py')], check=True)
    counts = {s: len((data / f'splits/{s}.txt').read_text().splitlines()) for s in SPLITS}
    if counts != SPLITS:
        raise ValueError(f'Expected {SPLITS}. Got {counts}.')
    print(f'Data ready at {data} with {counts}.', flush=True)
    return data


def require_idle_gpu():
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
        text=True, timeout=15).splitlines()
    # GNOME's remote desktop registers a CUDA context for display encoding.
    active = [row for row in rows if row.strip() and 'gnome-remote-desktop-daemon' not in row]
    if active:
        raise RuntimeError(f'GPU has active compute processes: {active}')


def candidate_stage(stage, model, workspace, data, output, gfl_python, yolo_python, gfl_workspace, *, unattended=False, checkpoint=None):
    run = output / model
    run.mkdir(parents=True, exist_ok=True)
    env = (gfl_python, yolo_python, gfl_workspace)
    with (workspace / 'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        complete = run / 'training_complete.json'
        if stage == 'train':
            if complete.exists():
                print(f'Reusing checkpoint in {run}')
                return
            resume = os.environ.get('DEER_COMPARE_RESUME') == model
            if (run / 'launch.json').exists() and not (resume or unattended):
                raise RuntimeError(f'Previous launch in {run}. Inspect its log and checkpoints.')
            if model.startswith('gfl'):
                filename = GFL_WEIGHT
            elif model in YOLO26_RESOLUTION_RUNS:
                filename = 'yolo26s.pt'
            else:
                filename = MODERN_WEIGHTS.get(model, f'{model}.pt')
            weight = workspace / 'input/weights' / filename
            if not weight.is_file():
                raise RuntimeError(f'Missing starting weights at {weight}')
            if unattended:
                if checkpoint is None and not (run / 'smoke/training_complete.json').exists():
                    invoke('smoke', model, data, run / 'smoke', weight, *env)
                write_json(run / 'launch.json', {'pid': os.getpid(), 'model': model})
                flags = ('--resume', str(checkpoint)) if checkpoint else ()
                invoke('train', model, data, run, weight, *env, *flags)
                return
            if resume:
                checkpoint = run / 'fit/weights/last.pt'
                marker = run / 'resume_launch.json'
                if not model.startswith('yolo') or not checkpoint.is_file() or marker.exists():
                    raise RuntimeError(f'Cannot resume {model}. Inspect {checkpoint} and {marker}.')
                # Keep a copy of the pre-recovery checkpoint as Ultralytics overwrites last.pt.
                saved = run / 'before_resume'
                saved.mkdir(exist_ok=False)
                shutil.copy2(checkpoint, saved / checkpoint.name)
                write_json(marker, {'pid': os.getpid(), 'checkpoint': str(checkpoint),
                                    'cuda_launch_blocking': True})
                invoke('train', model, data, run, weight, *env, '--resume', str(checkpoint))
                return
            if not (run / 'smoke/training_complete.json').exists():
                invoke('smoke', model, data, run / 'smoke', weight, *env)
            write_json(run / 'launch.json', {'pid': os.getpid(), 'model': model})
            invoke('train', model, data, run, weight, *env)
            return

        if unattended and stage == 'val':
            select_recovered_best(model, data, run, complete, env)
        best = Path(read_json(complete)['best'])
        if stage in ('val', 'test'):
            if stage == 'test' and not (run / 'calibration.json').exists():
                raise RuntimeError('Finish validation and freeze the threshold before test.')
            folder = run / stage
            if not (folder / 'profile.json').exists():
                size = MODEL_SIZES.get(model, 1280)
                invoke('profile', model, data, run, best, *env, '--split', stage, '--size', str(size),
                       '--output', str(folder))
            evaluate_split(data, run, stage)
        elif stage == 'profile':
            main_size = MODEL_SIZES.get(model, 1280)
            sizes = (640, 960, 1280) if model in YOLO26_RESOLUTION_RUNS else (1280,)
            for size, precision in [(main_size, 'fp32'), *((size, 'fp16') for size in sizes)]:
                folder = run / f'profile_{size}_{precision}'
                if not (folder / 'profile.json').exists():
                    invoke('profile', model, data, run, best, *env, '--split', 'val', '--limit', '48',
                           '--size', str(size), '--precision', precision, '--output', str(folder))
                if model in YOLO26_RESOLUTION_RUNS and precision == 'fp16':
                    evaluate_profile_sample(data, run, folder)
        else:
            raise ValueError(stage)


def invoke(mode, model, data, run, weights, gfl_python, yolo_python, gfl_workspace, *extra):
    require_idle_gpu()
    run.mkdir(parents=True, exist_ok=True)
    python = gfl_python if model.startswith('gfl') else yolo_python
    workspace = data.parent
    if model in MODERN_WEIGHTS:
        python = workspace / '.edge-env/bin/python'
        if not (workspace / 'edge_setup_complete').is_file():
            raise RuntimeError(f'Edge environment setup incomplete. See {workspace / "edge-setup.log"}.')
    env = dict(os.environ, DEER_WORKSPACE=str(gfl_workspace), MPLBACKEND='Agg',
               DEER_COMPARE_WORKSPACE=str(workspace),
               YOLO_CONFIG_DIR=str(run / 'ultralytics'),
               PYTHONPATH=os.pathsep.join(map(str, [CODE, CODE / 'prototyping/gfl-deer'])))
    log = run / (mode + ('-' + '-'.join(extra[1::2]).split('/')[-1] if extra else '') + '.log')
    if mode == 'train' and '--resume' in extra:
        if env.get('DEER_COMPARE_BLOCK_ON_RESUME', '1') == '1':
            env['CUDA_LAUNCH_BLOCKING'] = '1'
        log = run / 'resume-train.log'
    if os.environ.get('DEER_COMPARE_ATTEMPT_LOG'):
        log = Path(os.environ['DEER_COMPARE_ATTEMPT_LOG'])
    elif mode == 'profile':
        log = Path(extra[extra.index('--output') + 1]) / 'profile.log'
        log.parent.mkdir(parents=True, exist_ok=True)
    if mode in ('train', 'smoke') and os.environ.get('DEER_COMPARE_CUDA_RETRY') == '1':
        env['CUDA_LAUNCH_BLOCKING'] = '1'
    print(f'{model} {mode}. Log at {log}', flush=True)
    with log.open('a') as stream:
        command = [str(python), '-u', str(HERE / 'train_and_profile.py'), mode, model,
                   '--data', str(data), '--run', str(run), '--weights', str(weights), *extra]
        if os.environ.get('DEER_COMPARE_STREAM_LOG') != '1':
            subprocess.run(command, cwd=run, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
            return
        with subprocess.Popen(command, cwd=run, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors='replace') as child:
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                if line.strip() and not line.lstrip().startswith(('\x1b[K', '\x1b[?25')):
                    print(line, end='', flush=True)
            if child.wait():
                raise subprocess.CalledProcessError(child.returncode, command)


def select_recovered_best(model, data, run, complete, env):
    from dataloaders.deer_shared.metrics import ap_metrics

    saved = sorted((run / 'recovery/best').glob('*.pt*'))
    manifest = read_json(complete)
    if not saved or manifest.get('recovery_validation'):
        return
    candidates = list(dict.fromkeys([Path(manifest['best']), *saved,
                                    *sorted(run.glob('best_*.pth'))]))
    gt = read_json(data / 'annotations/val.coco.json')
    gt.setdefault('info', {})
    results = []
    for index, path in enumerate(candidates):
        folder = run / 'recovery/selection' / str(index)
        if not (folder / 'profile.json').is_file():
            invoke('profile', model, data, run, path, *env, '--split', 'val',
                   '--size', str(MODEL_SIZES[model]), '--output', str(folder))
        ap = ap_metrics(gt, read_json(folder / 'predictions.json'))['ap50_95']
        results.append((ap, path, folder))
    _, best, folder = max(results, key=lambda row: row[0])
    val = run / 'val'
    val.mkdir(exist_ok=True)
    for name in ('predictions.json', 'profile.json'):
        write_json(val / name, read_json(folder / name))
    manifest.update(best=str(best), sha256=digest(best), recovery_validation=[
        {'checkpoint': str(path), 'ap50_95': ap} for ap, path, _ in results])
    write_json(complete, manifest)


def evaluate_profile_sample(data, run, folder):
    from dataloaders.deer_shared.metrics import ap_metrics, detection_metrics

    truth = read_json(data / 'annotations/val.coco.json')
    truth.setdefault('info', {})
    measured = read_json(folder / 'profile.json')['per_image_ms']
    truth['images'] = [image for image in truth['images'] if image['file_name'] in measured]
    ids = {image['id'] for image in truth['images']}
    truth['annotations'] = [box for box in truth['annotations'] if box['image_id'] in ids]
    predictions = read_json(folder / 'predictions.json')
    threshold = read_json(run / 'calibration.json')['threshold']
    scores = detection_metrics(truth, predictions, threshold)
    scores.update(ap_metrics(truth, predictions))
    write_json(folder / 'sample_metrics.json', scores)


def evaluate_split(data, run, split):
    from dataloaders.deer_shared.metrics import ap_metrics, calibrate, detection_metrics
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    folder = run / split
    gt = read_json(data / f'annotations/{split}.coco.json')
    gt.setdefault('info', {})
    pred = read_json(folder / 'predictions.json')
    if split == 'val':
        cal = calibrate(gt, pred)
        write_json(run / 'calibration.json', cal)
        curve = cal['curve']
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([m['recall'] for m in curve], [m['precision'] for m in curve])
        axes[0].set(xlabel='Recall', ylabel='Precision', title='Validation PR at IoU 0.5')
        for key in ('precision', 'recall', 'f1'):
            axes[1].plot([m['threshold'] for m in curve], [m[key] for m in curve], label=key)
        axes[1].set(xlabel='Confidence threshold', title='Validation threshold sweep')
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(folder / 'threshold_curves.png', dpi=150)
        plt.close(fig)
    threshold = read_json(run / 'calibration.json')['threshold']
    result = detection_metrics(gt, pred, threshold)
    result.update(ap_metrics(gt, pred))
    source_ids = defaultdict(set)
    for image in gt['images']:
        source_ids[Path(image['file_name']).name.split('__', 1)[0]].add(image['id'])
    result['by_source'] = {}
    for source, ids in sorted(source_ids.items()):
        subset = dict(gt, images=[i for i in gt['images'] if i['id'] in ids],
                      annotations=[a for a in gt['annotations'] if a['image_id'] in ids])
        pp = [p for p in pred if p['image_id'] in ids]
        metrics = detection_metrics(subset, pp, threshold)
        metrics.update(ap_metrics(subset, pp))
        metrics.pop('per_image')
        result['by_source'][source] = metrics
    negatives = [row for row in result['per_image'] if row['truth'] == 0]
    result['negative_images'] = len(negatives)
    result['negative_images_with_false_positives'] = sum(row['fp'] > 0 for row in negatives)
    write_json(folder / 'metrics.json', result)
    if split == 'test':
        render_examples(data, folder, gt, pred, result, Image, ImageDraw, plt)


def render_examples(data, folder, gt, pred, result, Image, ImageDraw, plt):
    images = {i['id']: i for i in gt['images']}
    truth, predictions = defaultdict(list), defaultdict(list)
    for box in gt['annotations']:
        truth[box['image_id']].append(box)
    for box in pred:
        if box['score'] >= result['threshold']:
            predictions[box['image_id']].append(box)
    rows = result['per_image']
    ordered = sorted(rows, key=lambda row: -(row['fp'] + row['fn']))
    decoys = [row for row in ordered if Path(images[row['image_id']]['file_name']).name.startswith('mississippi_decoy__')]
    other = [row for row in ordered if row not in decoys]
    gallery = decoys[:4] + other[:4]
    worst = {row['image_id']: row for row in gallery + ordered[:16]}
    examples = folder / 'examples'
    examples.mkdir(exist_ok=True)
    for row in worst.values():
        key = row['image_id']
        file = data / 'images' / images[key]['file_name']
        with Image.open(file) as source:
            image = source.convert('RGB')
        draw = ImageDraw.Draw(image)
        for boxes, color in ((truth[key], 'lime'), (predictions[key], 'red')):
            for box in boxes:
                x, y, w, h = box['bbox']
                draw.rectangle((x, y, x+w, y+h), outline=color, width=3)
                if 'score' in box:
                    draw.text((x, y), f"{box['score']:.2f}", fill=color)
        image.thumbnail((1280, 1280))
        image.save(examples / file.name)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for ax in axes.flat:
        ax.axis('off')
    for ax, row in zip(axes.flat, gallery):
        file = Path(images[row['image_id']]['file_name']).name
        with Image.open(examples / file) as image:
            ax.imshow(image)
        ax.set_title(f"{file.split('__')[0]} · {row['fn']} missed, {row['fp']} extra", fontsize=9)
    fig.suptitle('Test examples · green labels, red predictions')
    fig.tight_layout()
    fig.savefig(folder / 'error_gallery.png', dpi=145)
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.scatter([r['truth'] for r in rows], [r['predicted'] for r in rows], alpha=.35)
    end = max(max(r['truth'], r['predicted']) for r in rows)
    ax.plot([0, end], [0, end], '--', color='gray')
    ax.set(xlabel='Labelled deer', ylabel='Predicted deer', title='Held-out counts per image')
    fig.savefig(folder / 'count_errors.png', dpi=150)
    plt.close(fig)


def summarize(output):
    import matplotlib.pyplot as plt

    table = []
    fields = ['ap50', 'ap50_95', 'precision', 'recall', 'f1', 'count_mae',
              'count_bias', 'exact_count_accuracy', 'negative_images_with_false_positives']
    profiles = ['mean_ms', 'p95_ms', 'peak_allocated_mib', 'peak_reserved_mib',
                'sampled_process_peak_mib', 'peak_host_rss_mib', 'parameters', 'checkpoint_bytes',
                'requested_size']
    for model in MODELS:
        run = output / model
        for split in ('val', 'test'):
            folder = run / split
            if not (folder / 'metrics.json').exists():
                continue
            metrics, profile = [read_json(folder / f'{key}.json') for key in ('metrics', 'profile')]
            decoys = metrics['by_source'].get('mississippi_decoy', {})
            table.append(dict(model=model, split=split, threshold=metrics['threshold'],
                              **{key: metrics[key] for key in fields},
                              **{key: profile[key] for key in profiles},
                              **{f'decoy_{key}': decoys.get(key) for key in
                                 ('images', 'ap50_95', 'recall', 'count_mae', 'tp', 'fp', 'fn')}))
    if not table:
        return []
    with (output / 'comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    test = [r for r in table if r['split'] == 'test']
    if test:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, key, title in zip(axes, ('ap50_95', 'p95_ms', 'peak_reserved_mib'),
                                  ('Test AP50–95', 'Batch 1 p95 latency (ms)', 'Peak CUDA reserved (MiB)')):
            ax.bar([r['model'] for r in test], [r[key] for r in test])
            ax.set_title(title)
            ax.tick_params(axis='x', rotation=30)
        fig.tight_layout()
        fig.savefig(output / 'comparison.png', dpi=160)
        plt.close(fig)
    training_curves(output, plt)
    deployment_profiles(output, table)
    return table


def resolution_rows(output):
    rows = []
    for model in YOLO26_RESOLUTION_RUNS:
        for size in (640, 960, 1280):
            folder = output / model / f'profile_{size}_fp16'
            if not (folder / 'sample_metrics.json').is_file():
                continue
            metrics = read_json(folder / 'sample_metrics.json')
            profile = read_json(folder / 'profile.json')
            rows.append(dict(training_size=MODEL_SIZES[model], inference_size=size,
                             **{k: metrics[k] for k in ('ap50_95', 'recall', 'count_mae')},
                             **{k: profile[k] for k in ('images', 'p95_ms', 'peak_reserved_mib')}))
    if rows:
        with (output / 'yolo26_resolution.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def deployment_profiles(output, table):
    """Keep the deployment measurements together for comparing input settings."""
    columns = ('model', 'precision', 'requested_size', 'images', 'mean_ms', 'p95_ms',
               'peak_allocated_mib', 'peak_reserved_mib', 'sampled_process_peak_mib',
               'peak_host_rss_mib', 'model_load_to_first_detection_ms')
    rows = []
    for model in {r['model']: None for r in table}:
        for setting in ('val', 'profile_1280_fp32', 'profile_960_fp32', 'profile_640_fp32',
                        'profile_640_fp16', 'profile_960_fp16', 'profile_1280_fp16'):
            path = output / model / setting / 'profile.json'
            if path.exists():
                rows.append(dict(model=model, **read_json(path)))
    with (output / 'deployment_profiles.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def training_curves(output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model in MODELS:
        series = candidate_series(output / model)
        ap = validation_ap_history(series)
        losses = series.get('train/loss', [])
        if not losses:
            totals = defaultdict(float)
            for key, values in series.items():
                if key.startswith('train/'):
                    for epoch, value in values:
                        totals[epoch] += value
            losses = sorted(totals.items())
        for axis, values in zip(axes, (ap, losses)):
            if values:
                axis.plot(*zip(*values), label=model)
    for axis, title in zip(axes, ('Native validation AP50–95', 'Training loss (model-specific scale)')):
        axis.set(xlabel='Epoch', title=title)
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=150)
    plt.close(fig)


def candidate_series(run):
    series = defaultdict(list)
    yolo_log = run / 'fit/results.csv'
    if yolo_log.exists():
        with yolo_log.open() as stream:
            for row in csv.DictReader(stream):
                row = {key.strip(): value for key, value in row.items()}
                for key, value in row.items():
                    if key.startswith(('train/', 'val/', 'metrics/')):
                        series[key].append((int(row['epoch']), float(value)))
    elif run.name in ('rtdetrv2_r18', 'rtdetrv2_r50', 'dfine_l'):
        log = run / 'fit/log.txt'
        if log.exists():
            for line in log.read_text().splitlines():
                row = json.loads(line)
                epoch = row['epoch'] + 1
                for key, value in row.items():
                    if key.startswith('train_loss'):
                        series[key.replace('train_', 'train/', 1)].append((epoch, value))
                ap = row.get('test_coco_eval_bbox')
                if ap:
                    series['mAP50–95'].append((epoch, ap[0]))
                    series['mAP50'].append((epoch, ap[1]))
    elif run.name == 'rtmdet_tiny':
        losses = defaultdict(list)
        for log in sorted(run.glob('*/vis_data/scalars.json')):
            for line in log.read_text().splitlines():
                row = json.loads(line)
                if 'epoch' in row:
                    for key, value in row.items():
                        if key.startswith('loss'):
                            losses[key, row['epoch']].append(value)
                for key in ('coco/bbox_mAP', 'coco/bbox_mAP_50'):
                    if key in row:
                        series[key].append((row['step'], row[key]))
        for (key, epoch), values in sorted(losses.items()):
            series['train/' + key].append((epoch, sum(values) / len(values)))
    else:
        losses = defaultdict(list)
        for path in sorted(run.glob('*.log.json')):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get('mode') == 'train':
                    for key, value in row.items():
                        if key.startswith('loss'):
                            losses[key, row['epoch']].append(value)
                elif row.get('mode') == 'val':
                    for key in ('bbox_mAP', 'bbox_mAP_50'):
                        if key in row:
                            series[key].append((row['epoch'], row[key]))
        for (key, epoch), values in sorted(losses.items()):
            series['train/' + key].append((epoch, sum(values) / len(values)))
        history = run / 'epoch_precision_recall.jsonl'
        if history.exists():
            for line in history.read_text().splitlines():
                row = json.loads(line)
                for key in ('precision', 'recall'):
                    series[key].append((row['epoch'], row[key]))
    return series


def validation_ap_history(series):
    return next((series[key] for key in ('metrics/mAP50-95(B)', 'coco/bbox_mAP',
                                        'bbox_mAP', 'mAP50–95') if key in series), [])


def plot_candidate_training(run):
    import matplotlib.pyplot as plt

    series = candidate_series(run)
    if not series:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    for key, values in series.items():
        if key.startswith(('train/', 'val/')):
            axis = axes[0]
        elif 'mAP' in key:
            axis = axes[1]
        elif 'precision' in key or 'recall' in key:
            axis = axes[2]
        else:
            continue
        axis.plot(*zip(*values), label=key.replace('metrics/', '').replace('(B)', ''))
    if run.name in MODERN_WEIGHTS:
        axes[2].text(.5, .5, 'See validation PR curve', ha='center', transform=axes[2].transAxes)
    for axis, title in zip(axes, ('Loss components', 'Validation AP', 'Validation precision and recall')):
        axis.set(xlabel='Epoch', title=title)
        if axis.lines:
            axis.legend(fontsize=7)
    ap = validation_ap_history(series)
    if ap:
        epoch, score = max(ap, key=lambda item: item[1])
        axes[1].axvline(epoch, color='gray', linestyle='--', alpha=.7)
        print(f'Peak logged validation AP50–95: {score:.3f} at epoch {epoch} of {max(e for e, _ in ap)}.')
    fig.tight_layout()
    fig.savefig(run / 'training_curves.png', dpi=150)
    plt.close(fig)
