import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def cell(kind,s):
 d=dict(cell_type=kind,metadata={},source=s.splitlines(True))
 if kind=='code':d.update(execution_count=None,outputs=[])
 return d
cells=[cell('markdown','# Reuse saved GFL detector and data\nDefault: restore and inspect only. No dataset downloads or training. Run after the initial preparation has saved the workspace.\n'),cell('code', '''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import shutil, subprocess, json, os
BASE=Path('/content/drive/MyDrive/Gurnek_GFL_Research')
SAVED=BASE/'workspace'
assert (SAVED/'data/sherbrooke/prepared/annotations/val.json').exists(), 'Initial data cache is not complete yet'
ROOT=Path('/content/deer_research')
shutil.copytree(SAVED, ROOT, dirs_exist_ok=True, ignore=shutil.ignore_patterns('vendor','__pycache__'))
os.chdir(ROOT)
print('Using cached data:',SAVED)
'''),cell('code', '''# Runtime dependencies are temporary; model weights and data stay on Drive.
subprocess.run(['bash','scripts/setup_colab.sh'],check=True)
PY='/content/gfl-env/bin/python'
subprocess.run([PY,'-m','deer_pipeline.relocate'],check=True)
RUNS=BASE/'runs'
completed=RUNS/'complete.json'
if completed.exists():
    MODELS=json.loads(completed.read_text());print(MODELS)
else:
    MODELS={}
    print('Full experiment has not completed. Saved stage checkpoints:')
    print('\n'.join(str(p) for p in RUNS.rglob('*.pth')))
'''),cell('markdown','## Optional prediction with an existing trained checkpoint\nThis cell performs inference only. It never retrains. Set `RUN_PREDICTIONS=True` when wanted.\n'),cell('code', '''RUN_PREDICTIONS=False
if RUN_PREDICTIONS:
    assert 'adapted' in MODELS, 'Choose a completed trained checkpoint first'
    weights=Path(MODELS['adapted'])
    subprocess.run([PY,'-m','deer_pipeline','predict',
        '--config',str(weights.parent/'config.py'),
        '--checkpoint',str(weights),
        '--annotations',str(ROOT/'data/decoys/prepared/annotations/test.json'),
        '--output',str(BASE/'reuse_predictions/predictions.json')],check=True)
''')]
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(accelerator='GPU',colab={'name':'GFL_Reuse_Saved_Model.ipynb'},kernelspec={'name':'python3','display_name':'Python 3'}),cells=cells)
(root/'notebooks/GFL_Reuse_Saved_Model.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
