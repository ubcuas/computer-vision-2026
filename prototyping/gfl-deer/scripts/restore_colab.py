"""Restore cached prepared data/code from mounted Drive; never trains or downloads."""
import shutil
from pathlib import Path

saved=Path('/content/drive/MyDrive/Gurnek_GFL_Research/workspace')
root=Path('/content/deer_research')
if not (saved/'data/sherbrooke/prepared/annotations/val.json').exists():
    raise RuntimeError('Persistent source cache is not complete yet; let the initial preparation finish.')
shutil.copytree(saved,root,dirs_exist_ok=True,ignore=shutil.ignore_patterns('vendor','__pycache__'))
print('Restored prepared workspace:',root)
print('Run setup_colab.sh to recreate runtime dependencies, then deer_pipeline.relocate.')
print('Saved models remain in /content/drive/MyDrive/Gurnek_GFL_Research/runs')
print('Use predict/evaluate commands to reuse checkpoints. Training is not started by this script.')
