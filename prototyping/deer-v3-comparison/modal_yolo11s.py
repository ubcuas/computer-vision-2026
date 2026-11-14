"""Resume the laptop's YOLO11s comparison run in a detached Modal A100 job."""
import os
from pathlib import Path
import shutil
import sys

import modal

HERE = Path(__file__).resolve().parent
CODE = Path('/code')
WORKSPACE = Path('/dataset')
OUTPUT = Path('/outputs')
RUN = OUTPUT / 'yolo11s'
SEED = OUTPUT / 'seed'
SOURCE_HASH = 'd83817b4bf6823ab945f5274214679b386b89adff8ae1dd80c1676f9bd79d0b4'
BEST_HASH = '03eeafcc9def06aeb4cbe4e5ca2690a0f154b34c8394cb45f80538b33acba638'

app = modal.App('uas-deer-v3-yolo11s')
dataset_volume = modal.Volume.from_name('uas-deer-v3-dataset')
output_volume = modal.Volume.from_name('uas-deer-v3-yolo11s')


def with_code(image):
    if not modal.is_local():
        return image
    repo = HERE.parents[1]
    image = image.add_local_dir(HERE, CODE / 'prototyping/deer-v3-comparison',
                                ignore=['__pycache__', '*.pyc'])
    for name in ('__init__.py', 'common.py', 'metrics.py'):
        image = image.add_local_file(repo / 'dataloaders/deer_shared' / name,
                                     str(CODE / 'dataloaders/deer_shared' / name))
    return image


cpu_image = with_code(modal.Image.debian_slim(python_version='3.11').pip_install('gdown==5.2.0'))
gpu_image = with_code(
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('libgl1', 'libglib2.0-0')
    .pip_install('torch==2.8.0', 'torchvision==0.23.0', 'numpy==1.26.4',
                 'opencv-python==4.11.0.86', 'ultralytics==8.4.155',
                 'matplotlib==3.10.9', 'pycocotools==2.0.10',
                 'PyYAML==6.0.2', 'pandas==2.2.3')
)


def configure():
    sys.path[:0] = [str(CODE), str(CODE / 'prototyping/deer-v3-comparison')]
    os.environ.update(MPLBACKEND='Agg', DEER_COMPARE_STREAM_LOG='1',
                      DEER_COMPARE_EPOCH_LOG='1')


@app.function(image=cpu_image, volumes={WORKSPACE: dataset_volume}, timeout=2700, max_containers=1,
              cpu=2, memory=4096)
def stage_dataset(drive_id: str = ""):
    configure()
    from workflow import prepare

    data = prepare(WORKSPACE, downloader=shutil.which('gdown'), drive_id=drive_id)
    dataset_volume.commit()
    return str(data)


def restore_seed():
    from dataloaders.deer_shared.common import digest, write_json

    marker = RUN / 'modal_import.json'
    if marker.is_file():
        return
    for name, expected in [('last.pt', SOURCE_HASH), ('best.pt', BEST_HASH)]:
        if digest(SEED / 'fit/weights' / name) != expected:
            raise RuntimeError(f'Imported {name} checksum mismatch')
    shutil.copytree(SEED / 'fit', RUN / 'fit', dirs_exist_ok=True)
    write_json(marker, {'source': 'greglinux', 'source_epoch': 73, 'last_sha256': SOURCE_HASH,
                        'best_sha256': BEST_HASH})
    output_volume.commit()


def find_checkpoint():
    from checkpoint_recovery import candidates, epoch_of

    choices = [*candidates(RUN, 'yolo11s'), SEED / 'fit/weights/last.pt']
    for path in sorted((p for p in choices if p.is_file()),
                       key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            epoch = epoch_of(path, 'yolo11s', 100)
        except (OSError, EOFError, RuntimeError, ValueError, KeyError) as exc:
            print(f'Cannot load checkpoint {path}: {exc}', flush=True)
            continue
        if epoch and epoch >= 73:
            print(f'Using {path} with {epoch} completed epochs', flush=True)
            return path
    raise RuntimeError('No resumable checkpoint from epoch 73 or later')


@app.function(image=gpu_image, gpu='A100-40GB', cpu=4, memory=16384, timeout=3 * 60 * 60,
              retries=1, single_use_containers=True, max_containers=1,
              volumes={WORKSPACE: dataset_volume, OUTPUT: output_volume})
def resume_and_evaluate():
    configure()
    from unattended import stage_done
    from workflow import candidate_stage, invoke, plot_candidate_training, summarize

    data = WORKSPACE / 'data'
    if not (data / 'runtime/data.yaml').is_file():
        raise RuntimeError('Stage the frozen dataset first')
    restore_seed()
    if not stage_done(RUN, 'yolo11s', 'train'):
        checkpoint = find_checkpoint()
        invoke('train', 'yolo11s', data, RUN, SEED / 'yolo11s.pt',
               sys.executable, sys.executable, WORKSPACE, '--resume', str(checkpoint))
        output_volume.commit()
    for stage in ('val', 'test', 'profile'):
        if not stage_done(RUN, 'yolo11s', stage):
            candidate_stage(stage, 'yolo11s', WORKSPACE, data, OUTPUT,
                            sys.executable, sys.executable, WORKSPACE)
            output_volume.commit()
    plot_candidate_training(RUN)
    summarize(OUTPUT)
    output_volume.commit()


@app.local_entrypoint()
def main():
    stage_dataset.remote(os.environ.get('DEER_V3_DRIVE_ID', ''))
    call = resume_and_evaluate.spawn()
    print(f'A100 function call: {call.object_id}', flush=True)
    call.get()


# Each parallel job stages a local copy of the shared archive to avoid training
# directly from the Volume. Only its own output Volume is writable.
SCRATCH = Path('/tmp/deer-workspace')
ENV_PYTHON = Path('/opt/deer-env/bin/python')
PARALLEL_MODELS = (
    'yolo26s', 'yolo26s_960', 'yolo26s_640', 'gfl_short', 'gfl_long',
    'rtmdet_tiny', 'rtdetrv2_r18', 'dfine_l', 'rtdetrv2_r50',
)
VENDOR = Path('/opt/deer-vendor')
DATASET_READ_ONLY = dataset_volume.with_mount_options(read_only=True)

legacy_base = (
    modal.Image.debian_slim(python_version='3.11')
    .apt_install('git', 'libgl1', 'libglib2.0-0')
    .pip_install('uv==0.8.22')
    .run_commands(
        'uv python install 3.10',
        'uv venv --python 3.10 --seed /opt/deer-env',
        '/opt/deer-env/bin/python -m pip install setuptools==69.5.1 wheel==0.45.1 numpy==1.26.4',
    )
)
legacy_dependencies = (
    'Pillow==11.3.0 PyYAML==6.0.2 opencv-python==4.11.0.86 '
    'scipy==1.10.1 pycocotools==2.0.10 matplotlib==3.7.5 '
    'terminaltables==3.1.10 yapf==0.40.1 addict==2.4.0'
)
gfl_image = legacy_base.run_commands(
    '/opt/deer-env/bin/python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 '
    '--extra-index-url https://download.pytorch.org/whl/cu117',
    '/opt/deer-env/bin/python -m pip install --only-binary mmcv-full mmcv-full==1.7.1 '
    '-f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html',
    f'/opt/deer-env/bin/python -m pip install {legacy_dependencies}',
    'git clone --depth 1 --branch v2.28.2 https://github.com/open-mmlab/mmdetection.git '
    '/opt/deer-vendor/mmdetection',
    '/opt/deer-env/bin/python -m pip install --no-deps --no-build-isolation '
    '-e /opt/deer-vendor/mmdetection',
    '/opt/deer-env/bin/python -c "import mmdet; from mmcv.ops import deform_conv2d, nms"',
)
# Shared evaluation runs in the Modal wrapper, not the legacy training venv.
gfl_image = gfl_image.pip_install(
    'numpy==1.26.4', 'pycocotools==2.0.10', 'matplotlib==3.10.9', 'Pillow==11.3.0',
).run_commands(
    'python -c "import numpy; from pycocotools.cocoeval import COCOeval; '
    'import matplotlib.pyplot; from PIL import Image"',
)
if modal.is_local():
    gfl_image = gfl_image.add_local_dir(
        HERE.parents[1] / 'prototyping/gfl-deer/deer_pipeline',
        '/code/prototyping/gfl-deer/deer_pipeline', ignore=['__pycache__', '*.pyc'],
    )
gfl_image = with_code(gfl_image)
edge_image = with_code(legacy_base.run_commands(
    '/opt/deer-env/bin/python -m pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 '
    '--extra-index-url https://download.pytorch.org/whl/cu121',
    '/opt/deer-env/bin/python -m pip install '
    'https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv-2.1.0-cp310-cp310-manylinux1_x86_64.whl',
    '/opt/deer-env/bin/python -m pip install mmdet==3.3.0 mmengine==0.10.7 '
    'numpy==1.26.4 opencv-python==4.11.0.86 scipy==1.15.3 Pillow==11.3.0 '
    'pycocotools==2.0.11 matplotlib==3.10.9 PyYAML==6.0.2 faster-coco-eval==1.6.6 '
    'tensorboard==2.21.0 calflops==0.3.2 transformers==4.46.3 loguru==0.7.3 '
    'onnx==1.17.0 onnxruntime-gpu==1.19.2 yapf==0.40.1',
    'git clone https://github.com/lyuwenyu/RT-DETR.git /opt/deer-vendor/RT-DETR '
    '&& git -C /opt/deer-vendor/RT-DETR checkout df2a96c8c3a098567a1149bf0248da6efe6d495d',
    'git clone https://github.com/Peterande/D-FINE.git /opt/deer-vendor/D-FINE '
    '&& git -C /opt/deer-vendor/D-FINE checkout 956d1709314c2c6a4df6f34de232054578a7449f',
    '/opt/deer-env/bin/python -c "import mmdet, mmengine; from mmcv.ops import nms"',
).pip_install(
    'numpy==1.26.4', 'pycocotools==2.0.10', 'matplotlib==3.10.9', 'Pillow==11.3.0',
).run_commands(
    'python -c "import numpy; from pycocotools.cocoeval import COCOeval; '
    'import matplotlib.pyplot; from PIL import Image"',
))


def parallel_run(model, python):
    import json
    import subprocess
    import time

    configure()
    from dataloaders.deer_shared.common import write_json
    from unattended import EPOCHS, stage_done
    from workflow import MODERN_WEIGHTS, candidate_stage, plot_candidate_training, prepare, resolution_rows, summarize

    volume = modal.Volume.from_name(f'uas-deer-v3-{model}')
    run = OUTPUT / model
    run.mkdir(exist_ok=True)
    if (run / 'modal_complete.json').is_file():
        print(f'{model} already finished', flush=True)
        return
    status = {'model': model, 'started_at': time.time(), 'gpu': 'A100-40GB'}
    stage = 'dataset'
    try:
        local_input = SCRATCH / 'input'
        local_input.mkdir(parents=True, exist_ok=True)
        for name in ('deer-v3.drive.zip', 'weights'):
            (local_input / name).symlink_to(WORKSPACE / 'input' / name)
        if VENDOR.exists():
            (SCRATCH / 'vendor').symlink_to(VENDOR, target_is_directory=True)
        if model in MODERN_WEIGHTS:
            (SCRATCH / '.edge-env').symlink_to(ENV_PYTHON.parents[1], target_is_directory=True)
            (SCRATCH / 'edge_setup_complete').touch()
        data = prepare(SCRATCH)
        seed_fit = SEED / 'fit'
        if model == 'yolo26s_960' and seed_fit.is_dir() and not (run / 'fit').exists():
            pending = run / '.importing-fit'
            shutil.copytree(seed_fit, pending, dirs_exist_ok=True)
            pending.replace(run / 'fit')
            volume.commit()

        for stage in ('train', 'val', 'test', 'profile'):
            if stage_done(run, model, stage):
                continue
            write_json(run / 'modal_status.json', dict(status, stage=stage, state='running'))
            checkpoint = None
            if stage == 'train':
                probe = subprocess.check_output(
                    [str(python), str(CODE / 'prototyping/deer-v3-comparison/checkpoint_recovery.py'),
                     model, str(run), str(EPOCHS[model])], text=True,
                )
                checkpoint_info = json.loads(probe.splitlines()[-1])
                if checkpoint_info['files'] and not checkpoint_info['checkpoint']:
                    raise RuntimeError(f'{model} has checkpoints but none passed the resume check')
                checkpoint = checkpoint_info['checkpoint']
                if checkpoint:
                    print(f"{model} resuming from {checkpoint_info['epoch']} completed epochs", flush=True)
            candidate_stage(stage, model, SCRATCH, data, OUTPUT, python, python, SCRATCH,
                            unattended=True, checkpoint=checkpoint)
            if not stage_done(run, model, stage):
                raise RuntimeError(f'{model} {stage} did not produce the expected artifacts')
            volume.commit()

        stage = 'report'
        plot_candidate_training(run)
        summarize(OUTPUT)
        resolution_rows(OUTPUT)
        done = dict(status, state='complete', finished_at=time.time())
        write_json(run / 'modal_complete.json', done)
        write_json(run / 'modal_status.json', done)
    except Exception as exc:
        write_json(run / 'modal_status.json', dict(status, stage=stage, state='failed', error=str(exc)))
        raise
    finally:
        volume.commit()


parallel_options = dict(gpu='A100-40GB', cpu=4, memory=16384,
                        timeout=86400, max_containers=1, retries=0, single_use_containers=True)


@app.function(image=gpu_image, **parallel_options)
def train_parallel_yolo(model: str):
    parallel_run(model, Path(sys.executable))


@app.function(image=gfl_image, **parallel_options)
def train_parallel_gfl(model: str):
    os.environ['DEER_COMPARE_BLOCK_ON_RESUME'] = '0'
    parallel_run(model, ENV_PYTHON)


@app.function(image=edge_image, **parallel_options)
def train_parallel_edge(model: str):
    parallel_run(model, ENV_PYTHON)


@app.function(image=modal.Image.debian_slim(python_version='3.11'), timeout=86400)
def dispatch_parallel(models: list[str]):
    functions = {}
    for model in models:
        if model.startswith('yolo'):
            worker = train_parallel_yolo
        elif model.startswith('gfl'):
            worker = train_parallel_gfl
        else:
            worker = train_parallel_edge
        volume = modal.Volume.from_name(f'uas-deer-v3-{model}', create_if_missing=True)
        call = worker.with_options(volumes={WORKSPACE: DATASET_READ_ONLY, OUTPUT: volume}).spawn(model)
        functions[model] = call
        print(f'{model} submitted {call.object_id}', flush=True)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=len(functions)) as pool:
        pending = {pool.submit(call.get): model for model, call in functions.items()}
        for task in as_completed(pending):
            model = pending[task]
            try:
                task.result()
            except Exception as exc:
                print(f'{model} failed: {exc}', flush=True)
            else:
                print(f'{model} completed', flush=True)


@app.local_entrypoint()
def launch_parallel(models: str = ''):
    selected = [model.strip() for model in models.split(',')] if models else []
    if not selected:
        raise ValueError('Pass --models explicitly; each selected model uses a billable GPU.')
    if len(set(selected)) != len(selected) or set(selected).difference(PARALLEL_MODELS):
        raise ValueError('Select distinct model names from PARALLEL_MODELS')
    dispatch_parallel.spawn(selected).get()


@app.function(image=edge_image, gpu='A100-40GB', cpu=4, memory=16384, timeout=1800,
              max_containers=1, retries=0, single_use_containers=True)
def evaluate_dfine_checkpoint(epoch: int):
    configure()
    from dataloaders.deer_shared.common import digest, write_json
    from unattended import stage_done
    from workflow import evaluate_split, invoke, prepare

    model = 'dfine_l'
    original_run = OUTPUT / model
    weights = original_run / f'fit/checkpoint{epoch - 1:04d}.pth'
    run = original_run / f'epoch{epoch}_eval'
    volume = modal.Volume.from_name('uas-deer-v3-dfine_l')
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if all(stage_done(run, model, split) for split in ('val', 'test')):
        return

    run.mkdir(exist_ok=True)
    shutil.copy2(original_run / 'config.yml', run / 'config.yml')
    (SCRATCH / 'input').mkdir(parents=True, exist_ok=True)
    (SCRATCH / 'input/deer-v3.drive.zip').symlink_to(WORKSPACE / 'input/deer-v3.drive.zip')
    (SCRATCH / 'vendor').symlink_to(VENDOR, target_is_directory=True)
    (SCRATCH / '.edge-env').symlink_to(ENV_PYTHON.parents[1], target_is_directory=True)
    (SCRATCH / 'edge_setup_complete').touch()
    data = prepare(SCRATCH)
    write_json(run / 'checkpoint.json', {'epoch': epoch, 'path': str(weights), 'sha256': digest(weights)})
    for split in ('val', 'test'):
        folder = run / split
        if stage_done(run, model, split):
            continue
        if not (folder / 'profile.json').is_file():
            invoke('profile', model, data, run, weights, ENV_PYTHON, ENV_PYTHON, SCRATCH,
                   '--split', split, '--size', '1280', '--output', str(folder))
        evaluate_split(data, run, split)
        if not stage_done(run, model, split):
            raise RuntimeError(f'D-FINE-L {split} evaluation artifacts are incomplete')
        volume.commit()
    write_json(run / 'evaluation_complete.json', {'state': 'complete', 'epoch': epoch})
    volume.commit()


@app.local_entrypoint()
def evaluate_dfine(epoch: int = 20):
    if epoch < 1:
        raise ValueError("Epoch must be at least 1")
    volume = modal.Volume.from_name('uas-deer-v3-dfine_l')
    evaluate_dfine_checkpoint.with_options(
        volumes={WORKSPACE: DATASET_READ_ONLY, OUTPUT: volume}
    ).remote(epoch)
