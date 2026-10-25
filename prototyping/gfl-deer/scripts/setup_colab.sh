#!/usr/bin/env bash
set -euo pipefail
RESEARCH_ROOT="${DEER_WORKSPACE:?Set DEER_WORKSPACE to the cached external workspace}"
python3 -m pip install -q uv==0.8.22
python3 -m uv python install 3.10
if [ ! -x /content/gfl-env/bin/python ]; then
  python3 -m uv venv /content/gfl-env --python 3.10 --seed
fi
PY=/content/gfl-env/bin/python
"$PY" -m pip install -q setuptools==69.5.1 wheel==0.45.1 numpy==1.23.5
"$PY" -m pip install -q torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
"$PY" -m pip install -q mmcv-full==1.7.1 --only-binary mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
"$PY" -m pip install -q Pillow==11.3.0 PyMuPDF==1.26.5 requests==2.32.5 scipy==1.10.1 pycocotools==2.0.10 matplotlib==3.7.5 six==1.17.0 terminaltables==3.1.10 yapf==0.40.1 addict==2.4.0 packaging==24.2 pyyaml==6.0.2 opencv-python==4.11.0.86
mkdir -p "$RESEARCH_ROOT/vendor"
if [ ! -d "$RESEARCH_ROOT/vendor/mmdetection/.git" ]; then
  git clone --depth 1 --branch v2.28.2 https://github.com/open-mmlab/mmdetection.git "$RESEARCH_ROOT/vendor/mmdetection"
fi
"$PY" -m pip install -q --no-deps --no-build-isolation -e "$RESEARCH_ROOT/vendor/mmdetection"
"$PY" -c 'import torch,mmcv,mmdet; from mmcv.ops import nms,deform_conv2d; assert torch.cuda.is_available(); print(torch.__version__,mmcv.__version__,mmdet.__version__,torch.cuda.get_device_name())'
