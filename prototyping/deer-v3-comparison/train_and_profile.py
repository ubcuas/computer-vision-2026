"""Train and profile one candidate in its own CUDA process."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from dataloaders.deer_shared.common import digest, read_json, write_json
from edge_models import edge_predictor, train_detr, train_rtmdet
from workflow import MODEL_SIZES, MODERN_WEIGHTS

SEED = 42
SIZE = 1280
RECIPES = {
    'yolo11s': (100, 30),
    'yolo26s': (100, 30),
    'yolo26s_960': (100, 30),
    'yolo26s_640': (100, 30),
    'gfl_short': (10, 0),
    'gfl_long': (100, 30),
    'rtmdet_tiny': (60, 0),
    'rtdetrv2_r18': (60, 0),
    'dfine_l': (60, 0),
    'rtdetrv2_r50': (60, 0),
}


def gfl_config(data, run, weights, epochs, patience, smoke=False):
    from mmcv import Config
    from deer_pipeline.train import config, register_hooks

    register_hooks()
    cfg = config(data, run, epochs=epochs, long_edge=SIZE,
                 checkpoint=str(weights), smoke=smoke, patience=patience)
    base = dict(cfg.data.train, ann_file=str(data / 'annotations/train.coco.json'),
                img_prefix=str(data / 'images'))
    decoys = dict(base, ann_file=str(data / 'runtime/train_decoys.coco.json'))
    cfg.data.train = dict(type='ConcatDataset', datasets=[
        base, dict(type='RepeatDataset', times=3, dataset=decoys)])
    for split in ('val', 'test'):
        annotations = data / f'annotations/{split}.coco.json'
        coco = read_json(annotations)
        if 'info' not in coco:
            # pycocotools.loadRes requires info even when the dataset omits it.
            coco['info'] = {}
            annotations = run / f'{split}.coco.json'
            write_json(annotations, coco)
        cfg.data[split].update(ann_file=str(annotations), img_prefix=str(data / 'images'))
    # AP selects the checkpoint. Calibrate F1 after training to avoid two val passes.
    cfg.custom_hooks = [hook for hook in cfg.custom_hooks if hook['type'] != 'ValidationF1Hook']
    cfg.log_config['interval'] = 50
    return Config(cfg._cfg_dict.to_dict())


def train_gfl(data, run, weights, epochs, patience, smoke, resume=None):
    from mmcv.runner import HOOKS, Hook
    from mmdet.apis import set_random_seed, train_detector
    from mmdet.core import EvalHook
    from mmdet.datasets import build_dataset
    from mmdet.models import build_detector

    cfg = gfl_config(data, run, weights, epochs, patience, smoke)
    if resume:
        cfg.resume_from = str(resume)
        cfg.load_from = None
    if not smoke:
        from dataloaders.deer_shared.metrics import detection_metrics
        from deer_pipeline.train import results_to_coco

        truth = read_json(data / 'annotations/val.coco.json')

        @HOOKS.register_module(force=True)
        class EpochPrecisionRecallHook(Hook):
            def after_train_epoch(self, runner):
                evaluation = next(h for h in runner.hooks if isinstance(h, EvalHook))
                results = evaluation.latest_results
                if results is None:
                    return
                predictions = results_to_coco(results, truth)
                metrics = detection_metrics(truth, predictions, threshold=.25)
                row = {key: metrics[key] for key in ('precision', 'recall', 'f1')}
                with (run / 'epoch_precision_recall.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(row, epoch=runner.epoch + 1)) + '\n')

        cfg.custom_hooks.append(dict(type='EpochPrecisionRecallHook', priority=75))
    cfg.dump(str(run / 'config.py'))
    set_random_seed(SEED, deterministic=True)
    model = build_detector(cfg.model)
    model.init_weights()
    model.CLASSES = ('deer',)
    train = build_dataset(cfg.data.train)
    if len(train) != 3610:
        raise RuntimeError(f'GFL expected 3610 training exposures, got {len(train)}')
    train_detector(model, [train], cfg, distributed=False, validate=not smoke,
                   meta=dict(CLASSES=('deer',), config=cfg.pretty_text, seed=SEED))
    if smoke:
        state = torch.load(run / 'iter_2.pth', map_location='cpu')
        assert 'optimizer' in state and all(
            torch.isfinite(value).all() for value in state['state_dict'].values()
            if value.is_floating_point())
        return run / 'iter_2.pth'
    candidates = sorted(run.glob('best_bbox_mAP_epoch_*.pth'))
    if not candidates or (len(candidates) != 1 and not resume):
        raise RuntimeError(f'Expected a single best AP checkpoint, got {candidates}')
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def train_yolo(data, run, weights, epochs, patience, smoke, size, resume=None):
    from ultralytics import YOLO, settings

    settings.update({'sync': False})
    model = YOLO(str(resume or weights))
    expected = Counter((data / 'runtime/train_decoy_4x.txt').read_text().splitlines())

    def check_training_loader(trainer):
        if not smoke and Counter(trainer.train_loader.dataset.im_files) != expected:
            raise RuntimeError('YOLO loader changed the frozen training image exposure counts')

    if os.environ.get('DEER_COMPARE_EPOCH_LOG') == '1':
        def log_epoch(trainer):
            if trainer.epoch == trainer.start_epoch:
                print(f'Resumed at epoch {trainer.start_epoch + 1} of {trainer.epochs}', flush=True)
            print(f'Epoch {trainer.epoch + 1}/{trainer.epochs}: checkpoint saved', flush=True)

        model.add_callback('on_model_save', log_epoch)
    model.add_callback('on_train_start', check_training_loader)
    if resume:
        model.train(resume=True, data=str(data / 'runtime/data.yaml'), save_dir=str(run / 'fit'))
        return run / 'fit/weights/best.pt'
    yaml = run / 'smoke.yaml' if smoke else data / 'runtime/data.yaml'
    if smoke:
        for split in ('train', 'val'):
            paths = (data / f'runtime/{split}.txt').read_text().splitlines()
            (run / f'{split}.txt').write_text('\n'.join(paths[::max(1, len(paths)//8)][:8]) + '\n')
        yaml.write_text(f'train: {run}/train.txt\nval: {run}/val.txt\nnames: [deer]\n')
    model.train(data=str(yaml), epochs=1 if smoke else epochs, patience=patience,
                imgsz=size, batch=4, workers=2, seed=SEED, device=0, amp=True,
                optimizer='AdamW', lr0=.001, lrf=.01, plots=True, save_period=5,
                project=str(run), name='fit', exist_ok=True, cache=False)
    return run / 'fit/weights/best.pt'


def nvidia_sample():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
             '--format=csv,noheader,nounits'], text=True, timeout=3)
        for line in output.splitlines():
            pid, mib = line.split(',')
            if int(pid) == os.getpid():
                return int(mib)
    except (ValueError, OSError, subprocess.SubprocessError):
        pass
    return None


def profile(args):
    """Profile synchronized, batch-one path-to-detections inference."""
    gt = read_json(args.data / f'annotations/{args.split}.coco.json')
    if args.limit:
        gt['images'] = gt['images'][::max(1, len(gt['images']) // args.limit)][:args.limit]
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    cpu_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    half = args.precision == 'fp16'
    if args.model in MODERN_WEIGHTS:
        predict, parameters = edge_predictor(args, half)
    elif args.model.startswith('gfl'):
        from mmcv.runner import wrap_fp16_model
        from mmdet.apis import inference_detector, init_detector
        from deer_pipeline.train import register_padding, results_to_coco

        register_padding()
        model = init_detector(str(args.run / 'config.py'), str(args.weights), device='cuda:0')
        if half:
            wrap_fp16_model(model)
        if args.size != SIZE:
            model.cfg.data.test.pipeline[1].img_scale = (args.size, round(args.size * 1768 / 2650))

        def predict(path, image):
            result = inference_detector(model, str(path))
            return results_to_coco([result], {'images': [image]})

        parameters = sum(p.numel() for p in model.parameters())
    else:
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
        parameters = sum(p.numel() for p in model.model.parameters())

        def predict(path, image):
            result = model.predict(str(path), imgsz=args.size, device=0, half=half,
                                   conf=.001, max_det=300, verbose=False)[0]
            rows = []
            for x1, y1, x2, y2, confidence, category in result.boxes.data.cpu().tolist():
                x1, x2 = [max(0., min(x, image['width'])) for x in (x1, x2)]
                y1, y2 = [max(0., min(y, image['height'])) for y in (y1, y2)]
                if x2 > x1 and y2 > y1 and category == 0:
                    rows.append(dict(image_id=image['id'], category_id=1,
                                     bbox=[x1, y1, x2 - x1, y2 - y1], score=confidence))
            return rows

    def infer(image):
        return predict(args.data / 'images' / image['file_name'], image)

    def time_image(image):
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = infer(image)
        torch.cuda.synchronize()
        return result, (time.perf_counter() - start) * 1000

    cold, _ = time_image(gt['images'][0])
    first_result_ms = (time.perf_counter() - start) * 1000
    for image in gt['images'][1:6]:
        infer(image)
    torch.cuda.synchronize()
    warm_allocated = torch.cuda.memory_allocated()
    warm_reserved = torch.cuda.memory_reserved()
    startup_peak_allocated = torch.cuda.max_memory_allocated()
    startup_peak_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    stop = threading.Event()
    samples = []

    def sample_process():
        while not stop.is_set():
            mib = nvidia_sample()
            if mib is not None:
                samples.append(mib)
            stop.wait(1)

    monitor = threading.Thread(target=sample_process, daemon=True)
    monitor.start()
    predictions, times = [], []
    try:
        for index, image in enumerate(gt['images'], 1):
            if index % 25 == 0:
                print(f'Profiling image {index}/{len(gt["images"])}', flush=True)
            result, elapsed = time_image(image)
            predictions.extend(result)
            times.append(elapsed)
    finally:
        stop.set()
        monitor.join()
    mib = 2**20
    result = dict(gpu=torch.cuda.get_device_name(), precision=args.precision, batch=1,
                  requested_size=args.size, images=len(times), parameters=parameters,
                  checkpoint_bytes=args.weights.stat().st_size,
                  model_load_to_first_detection_ms=first_result_ms,
                  warm_allocated_mib=warm_allocated/mib, warm_reserved_mib=warm_reserved/mib,
                  startup_peak_allocated_mib=startup_peak_allocated/mib,
                  startup_peak_reserved_mib=startup_peak_reserved/mib,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/mib,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/mib,
                  sampled_process_peak_mib=max(samples) if samples else None,
                  process_samples=len(samples), peak_host_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                  pre_model_host_peak_rss_mib=cpu_start/1024,
                  mean_ms=float(np.mean(times)), median_ms=float(np.median(times)),
                  p95_ms=float(np.percentile(times, 95)), p99_ms=float(np.percentile(times, 99)),
                  max_ms=float(max(times)), serial_images_per_second=1000/float(np.mean(times)),
                  scope='file decode, preprocess, host/device transfer, model and postprocess; no drawing',
                  per_image_ms=dict(zip((i['file_name'] for i in gt['images']), times)))
    write_json(args.output / 'predictions.json', predictions)
    write_json(args.output / 'profile.json', result)
    print({k: v for k, v in result.items() if k != 'per_image_ms'}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'smoke', 'profile'])
    parser.add_argument('model', choices=RECIPES)
    for arg in ('data', 'run', 'weights'):
        parser.add_argument(f'--{arg}', type=Path, required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
    parser.add_argument('--size', type=int, default=SIZE)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.resume and args.mode != 'train':
        parser.error('--resume requires train mode')
    if args.mode == 'profile':
        profile(args)
        return
    args.run.mkdir(parents=True, exist_ok=True)
    if (args.run / 'training_complete.json').exists():
        raise RuntimeError(f'Completed training exists at {args.run}')
    epochs, patience = RECIPES[args.model]
    (args.run / 'pip-freeze.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True))
    smoke = args.mode == 'smoke'
    start = time.time()
    if args.model == 'rtmdet_tiny':
        best = train_rtmdet(args.data, args.run, args.weights, 1 if smoke else epochs, smoke, args.resume)
    elif args.model in MODERN_WEIGHTS:
        best = train_detr(args.model, args.data, args.run, args.weights, 1 if smoke else epochs, smoke, args.resume)
    elif args.model.startswith('gfl'):
        best = train_gfl(args.data, args.run, args.weights, epochs, patience, smoke, args.resume)
    else:
        size = MODEL_SIZES.get(args.model, SIZE)
        best = train_yolo(args.data, args.run, args.weights, epochs, patience, smoke, size, args.resume)
    write_json(args.run / 'training_complete.json', dict(
        best=str(best), sha256=digest(best), seconds=time.time()-start,
        pretrained=str(args.weights), pretrained_sha256=digest(args.weights),
        epochs_limit=epochs, patience=patience, seed=SEED, size=MODEL_SIZES.get(args.model, SIZE),
        resumed_from=str(args.resume) if args.resume else None))


if __name__ == '__main__':
    main()
