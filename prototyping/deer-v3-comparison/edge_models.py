"""Adapters for RTMDet and the DETR candidates."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from dataloaders.deer_shared.common import read_json, write_json
from workflow import MODEL_SIZES

RTDETR_REVISION = 'df2a96c8c3a098567a1149bf0248da6efe6d495d'
DFINE_REVISION = '956d1709314c2c6a4df6f34de232054578a7449f'
DETR_CONFIGS = {
    'rtdetrv2_r18': 'configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
    'rtdetrv2_r50': 'configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml',
    'dfine_l': 'configs/dfine/dfine_hgnetv2_l_coco.yml',
}


def rtmdet_config(data, run, weights, epochs, smoke):
    import mmdet
    from mmengine.config import Config

    config = Path(mmdet.__file__).parent / '.mim/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py'
    cfg = Config.fromfile(str(config))
    size = MODEL_SIZES['rtmdet_tiny']
    cfg.work_dir = str(run)
    cfg.load_from = str(weights)
    cfg.model.backbone.init_cfg = None
    cfg.model.bbox_head.num_classes = 1
    # The feature pyramid combines maps at strides up to 32.
    cfg.model.data_preprocessor.pad_size_divisor = 32
    cfg.model.test_cfg.score_thr = .001
    cfg.model.test_cfg.max_per_img = 300
    cfg.randomness = dict(seed=42)
    cfg.train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=1)
    cfg.param_scheduler = [
        dict(type='LinearLR', start_factor=.01, by_epoch=False, begin=0, end=250),
        dict(type='CosineAnnealingLR', eta_min=1e-6, T_max=epochs, by_epoch=True,
             begin=0, end=epochs, convert_to_iter_based=True),
    ]
    cfg.optim_wrapper.type = 'AmpOptimWrapper'
    cfg.optim_wrapper.optimizer.lr = .0002
    cfg.auto_scale_lr = dict(enable=False, base_batch_size=2)
    # Use a simple fixed-size fine-tune pipeline with no large mosaic cache.
    train_pipe = [
        dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
        dict(type='Resize', scale=(size, size), keep_ratio=True),
        dict(type='RandomFlip', prob=.5), dict(type='PackDetInputs'),
    ]
    for split in ('train', 'val'):
        ann = data / f'annotations/{split}.coco.json'
        if smoke:
            gt = read_json(ann)
            images = gt['images'][::max(1, len(gt['images'])//8)][:8]
            ids = {i['id'] for i in images}
            gt['images'] = images
            gt['annotations'] = [a for a in gt['annotations'] if a['image_id'] in ids]
            ann = run / f'{split}_smoke.json'
            write_json(ann, gt)
        loader = cfg[f'{split}_dataloader']
        loader.update(batch_size=2 if split == 'train' else 1, num_workers=2)
        pipeline = train_pipe if split == 'train' else deepcopy(loader.dataset.pipeline)
        if split == 'val':
            for transform in pipeline:
                if transform['type'] == 'Resize':
                    transform['scale'] = (size, size)
                elif transform['type'] == 'Pad':
                    transform['size'] = (size, size)
        ds = dict(type='CocoDataset', ann_file=str(ann), data_prefix=dict(img=str(data / 'images')),
                  metainfo=dict(classes=('deer',)), test_mode=split != 'train', pipeline=pipeline,
                  filter_cfg=dict(filter_empty_gt=False, min_size=32))
        if split == 'train' and not smoke:
            decoys = dict(ds, ann_file=str(data / 'runtime/train_decoys.coco.json'))
            loader.dataset = dict(type='ConcatDataset', datasets=[
                ds, dict(type='RepeatDataset', times=3, dataset=decoys)])
        else:
            loader.dataset = ds
        if split == 'val':
            cfg.val_evaluator.ann_file = str(ann)
    cfg.test_dataloader = cfg.val_dataloader
    cfg.test_evaluator = cfg.val_evaluator
    cfg.default_hooks.checkpoint.update(interval=1, max_keep_ckpts=2, save_best='coco/bbox_mAP')
    # The official EMA hook is retained. Its pipeline-switch hook expects 300 epochs.
    cfg.custom_hooks = [h for h in cfg.custom_hooks if h['type'] != 'PipelineSwitchHook']
    return cfg


def train_rtmdet(data, run, weights, epochs, smoke, resume=None):
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules

    register_all_modules()
    cfg = rtmdet_config(data, run, weights, epochs, smoke)
    if resume:
        cfg.resume = True
        cfg.load_from = str(resume)
    cfg.dump(str(run / 'config.py'))
    runner = Runner.from_cfg(cfg)
    assert len(runner.train_dataloader.dataset) == (8 if smoke else 3610)
    runner.train()
    best = list(run.glob('best_coco_bbox_mAP_epoch_*.pth'))
    if smoke and not best:
        return run / 'epoch_1.pth'
    if not best or (len(best) != 1 and not resume):
        raise RuntimeError(f'Expected one RTMDet best checkpoint, found {best}')
    return max(best, key=lambda path: path.stat().st_mtime_ns)


def detr_source(model):
    dfine = model == 'dfine_l'
    name, expected = ('D-FINE', DFINE_REVISION) if dfine else ('RT-DETR', RTDETR_REVISION)
    source = Path(os.environ['DEER_COMPARE_WORKSPACE']) / 'vendor' / name
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if revision != expected:
        raise RuntimeError(f'Unexpected {name} source revision {revision}')
    if not dfine:
        source = source / 'rtdetrv2_pytorch'
    sys.path.insert(0, str(source))
    return source


def detr_annotations(data, run, split, smoke):
    gt = read_json(data / f'annotations/{split}.coco.json')
    images = {i['file_name']: i for i in gt['images']}
    annotations = {}
    for annotation in gt['annotations']:
        annotations.setdefault(annotation['image_id'], []).append(annotation)
    paths = (data / f'splits/{"train_decoy_4x" if split == "train" else "val"}.txt').read_text().splitlines()
    if smoke:
        paths = paths[::max(1, len(paths)//8)][:8]
    packed = dict(gt, images=[], annotations=[], categories=[dict(id=0, name='deer')])
    packed.setdefault('info', {})
    for name in paths:
        image = images[name.removeprefix('images/')]
        image_id = len(packed['images']) + 1
        packed['images'].append(dict(image, id=image_id))
        for box in annotations.get(image['id'], []):
            packed['annotations'].append(dict(box, id=len(packed['annotations']) + 1,
                                               category_id=0, image_id=image_id))
    assert len(packed['images']) == (8 if smoke else (3610 if split == 'train' else 358))
    path = run / f'{split}.coco.json'
    write_json(path, packed)
    return path


def detr_config(model, source, data, run, epochs, smoke):
    import yaml
    from src.core.yaml_utils import load_config

    size = MODEL_SIZES[model]
    dfine = model == 'dfine_l'
    config = load_config(str(source / DETR_CONFIGS[model]), {})
    config.pop('__include__', None)
    config.update(num_classes=1, remap_mscoco_category=False, output_dir=str(run / 'fit'),
                  checkpoint_freq=5, eval_spatial_size=[size, size])
    config['epochs' if dfine else 'epoches'] = epochs
    config['HGNetv2' if dfine else 'PResNet']['pretrained'] = False
    config['optimizer']['lr'] = .0001
    config['lr_scheduler']['milestones'] = [40, 55]
    for split in ('train', 'val'):
        loader = config[f'{split}_dataloader']
        batch_size = 2 if split == 'train' and model == 'rtdetrv2_r18' else 1
        loader.update(total_batch_size=batch_size,
                      num_workers=2, drop_last=False)
        loader['dataset'].update(img_folder=str(data / 'images'),
                                 ann_file=str(detr_annotations(data, run, split, smoke)))
        for operation in loader['dataset']['transforms']['ops']:
            if operation['type'] == 'Resize':
                operation['size'] = [size, size]
        if split == 'train':
            if dfine:
                # Keep the first-stage checkpoint policy throughout this fine-tune.
                loader['collate_fn'].update(base_size=size, base_size_repeat=None, stop_epoch=epochs)
            else:
                loader['collate_fn']['scales'] = None
    path = run / 'config.yml'
    path.write_text(yaml.safe_dump(config))
    return path


def train_detr(model, data, run, weights, epochs, smoke, resume=None):
    source = detr_source(model)
    path = detr_config(model, source, data, run, epochs, smoke)
    entry = 'train.py' if model == 'dfine_l' else 'tools/train.py'
    subprocess.run([sys.executable, str(source / entry), '-c', str(path),
                    '-r' if resume else '-t', str(resume or weights), '--use-amp', '--seed', '42'], cwd=source, check=True)
    best = run / 'fit' / ('best_stg1.pth' if model == 'dfine_l' else 'best.pth')
    return best if best.exists() or not smoke else run / 'fit/last.pth'


def edge_predictor(args, half):
    if args.model == 'rtmdet_tiny':
        from mmdet.apis import inference_detector, init_detector

        model = init_detector(str(args.run / 'config.py'), str(args.weights), device='cuda:0')

        def predict(path, image):
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=half):
                result = inference_detector(model, str(path)).pred_instances
            return prediction_rows(image, result.labels, result.bboxes, result.scores)
    else:
        from PIL import Image
        from torchvision.transforms import Compose, Resize, ToTensor

        detr_source(args.model)
        from src.core import YAMLConfig

        config = YAMLConfig(str(args.run / 'config.yml'))
        checkpoint = torch.load(args.weights, map_location='cpu')
        state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
        model = config.model
        model.load_state_dict(state)
        if args.size != MODEL_SIZES[args.model]:
            # Recompute position encodings and anchors for a different profile size.
            model.encoder.eval_spatial_size = None
            model.decoder.eval_spatial_size = None
        model = model.deploy().cuda().eval()
        postprocess = config.postprocessor.deploy().cuda()
        transform = Compose([Resize((args.size, args.size)), ToTensor()])

        def predict(path, image):
            with Image.open(path) as source:
                frame = transform(source.convert('RGB'))[None].cuda()
            target_size = torch.tensor([[image['width'], image['height']]], device='cuda')
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=half):
                labels, boxes, scores = postprocess(model(frame), target_size)
            return prediction_rows(image, labels[0], boxes[0], scores[0])

    return predict, sum(p.numel() for p in model.parameters())


def prediction_rows(image, labels, boxes, scores):
    rows = []
    for label, (x1, y1, x2, y2), score in zip(labels.tolist(), boxes.tolist(), scores.tolist()):
        if label != 0 or score < .001:
            continue
        x1, x2 = [min(image['width'], max(0, x)) for x in (x1, x2)]
        y1, y2 = [min(image['height'], max(0, y)) for y in (y1, y2)]
        if x2 > x1 and y2 > y1:
            rows.append(dict(image_id=image['id'], category_id=1, score=score,
                             bbox=[x1, y1, x2-x1, y2-y1]))
    return rows
