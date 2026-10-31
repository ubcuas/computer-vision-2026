import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def cell(kind,text):
 d={'cell_type':kind,'metadata':{},'source':text.splitlines(True)}
 if kind=='code':d.update(execution_count=None,outputs=[])
 return d
cells=[cell('markdown','''# GFL against Anant’s baseline
Run on Michael’s Linux NVIDIA server. Default matches the **reported Run #1**: 211 decoy train / 53 decoy validation images. No Rucervus, Sherbrooke or AWIR. Run #2 in the latest repository uses decoy + AWIR; choose that explicitly below if comparing it. Same source images/boxes/splits; different architecture, augmentation and training budget. Validation is not an independent test. No training runs until the training cell is executed.
'''),cell('code','''from pathlib import Path
import os,sys,subprocess,json
# Start Jupyter from the repository root.
REPO=Path.cwd().resolve()
assert (REPO/'prototyping/gfl-deer/run.py').exists(), 'Start Jupyter from repository root'
WORKSPACE=Path('/absolute/server/path/gfl_workspace')  # EDIT
assert str(WORKSPACE) != '/absolute/server/path/gfl_workspace', 'Set your persistent workspace'
RUN_NAME='gfl_decoy_run1_v1'
DATA_VARIANT='decoy'  # decoy = reported Run #1; combined = latest decoy+AWIR recipe
os.environ['DEER_WORKSPACE']=str(WORKSPACE)
os.environ['PYTHONPATH']=str(REPO)+os.pathsep+str(REPO/'prototyping/gfl-deer')
subprocess.run(['nvidia-smi'],check=True)
'''),cell('code','''# Install once per server environment; downloads software, not datasets.
subprocess.run(['bash','prototyping/gfl-deer/scripts/setup_server.sh'],check=True)
PY=str(WORKSPACE/'.venv-gfl/bin/python')
RUN=WORKSPACE/'runs'/RUN_NAME
'''),cell('code','''# Preserve committed images/labels and splits; format conversion only.
if DATA_VARIANT=='combined':
    import tarfile,io
    destination=WORKSPACE/'anant_source';destination.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','fetch','origin','deer-yolo-dataset'],check=True)
    raw=subprocess.check_output(['git','archive','dbe1c2b','prototyping/yolo-fine-tune/dataset_awir'])
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            assert (destination/member.name).resolve().is_relative_to(destination.resolve())
        archive.extractall(destination)
    os.environ['ANANT_AWIR_ROOT']=str(destination/'prototyping/yolo-fine-tune/dataset_awir')
args=[PY,'prototyping/gfl-deer/run.py','prepare-matched','--workspace',str(WORKSPACE),'--variant',DATA_VARIANT]
subprocess.run(args,check=True)
DATA=WORKSPACE/'data'/('anant_exact' if DATA_VARIANT=='decoy' else 'anant_combined')
for split in ['train','val']:
 d=json.loads((DATA/'annotations'/f'{split}.json').read_text())
 print(split,len(d['images']),'images',len(d['annotations']),'boxes')
print('Byte hashes:',DATA/'inventory.json')
'''),cell('code','''# Explicit training launch. A fresh run refuses existing checkpoint directories.
RESUME=None  # For continuation, set an actual saved epoch_N.pth path.
cmd=[PY,'prototyping/gfl-deer/run.py','train-matched','--workspace',str(WORKSPACE),
     '--workdir',str(RUN),'--variant',DATA_VARIANT]
if RESUME:cmd+=['--resume',str(RESUME)]
RUN.mkdir(parents=True,exist_ok=True)
with (RUN/'launcher.log').open('a') as log:
 result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
result.check_returncode()
'''),cell('code','''# Validation only, using F1-selected checkpoint. No retraining.
PRED=RUN/'val_predictions.json';CAL=RUN/'validation_calibration.json'
def run(*args):subprocess.run([PY,'-m','deer_pipeline',*map(str,args)],check=True)
run('predict','--config',RUN/'config.py','--checkpoint',RUN/'best_f1.pth','--annotations',DATA/'annotations/val.json','--output',PRED)
run('calibrate','--annotations',DATA/'annotations/val.json','--predictions',PRED,'--output',CAL)
run('evaluate','--annotations',DATA/'annotations/val.json','--predictions',PRED,'--calibration',CAL,'--output',RUN/'validation_metrics.json')
print('Saved weights/config/logs/metrics:',RUN)
''')]
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':cells}
(root/'notebooks/GFL_Anant_Baseline_Server.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
