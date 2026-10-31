"""Detached server entry point for the separate longer decoy experiment.
Run from repo root with the pinned GFL Python; DEER_WORKSPACE must be set.
An exclusive persistent launch marker prevents duplicate jobs. Inspect failures
before manually clearing that marker; this entry point never restarts training.
"""
import json,os,subprocess,shutil
from pathlib import Path
import nbformat
from nbclient import NotebookClient
repo=Path.cwd().resolve()
workspace=Path(os.environ['DEER_WORKSPACE']).resolve()
run=workspace/'runs/gfl_decoy_long_v1'
assert (workspace/'notebook_complete.json').exists(), 'Finish short-run notebook first'
assert (workspace/'runs/gfl_decoy_run1_v1/gfl_model_bundle.zip').exists()
assert not any(run.glob('*.pth')), 'Existing weights; inspect and resume explicitly'
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not active, f'GPU compute already active: {active}'
run.mkdir(parents=True,exist_ok=True)
with (run/'launch.json').open('x') as f:
 json.dump(dict(pid=os.getpid(),code_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),epochs=100,patience=30,initialization='official COCO',dataset='anant_exact'),f,indent=2)
shutil.copy2(workspace/'data/anant_exact/inventory.json',run/'dataset_inventory.json')
nb=nbformat.read(repo/'prototyping/gfl-deer/notebooks/GFL_Anant_Baseline_Server.ipynb',as_version=4)
for c in nb.cells:
 if c.cell_type!='code':continue
 c.source=c.source.replace("Path('/absolute/server/path/gfl_workspace')",f'Path({str(workspace)!r})')
 c.source=c.source.replace("RUN_NAME='gfl_decoy_run1_v1'","RUN_NAME='gfl_decoy_long_v1'")
 c.source=c.source.replace('EPOCHS=10;PATIENCE=0','EPOCHS=100;PATIENCE=30')
 c.source=c.source.replace("subprocess.run(['bash','prototyping/gfl-deer/scripts/setup_server.sh'],check=True)","print('Reusing verified GFL environment and cached dataset')")
output=workspace/'GFL_Decoy_Long.executed.ipynb'
class SavingClient(NotebookClient):
 def process_message(self,*args,**kwargs):
  result=super().process_message(*args,**kwargs)
  nbformat.write(self.nb,output)
  return result
try:
 SavingClient(nb,timeout=None,kernel_name='gfl-server',resources={'metadata':{'path':str(repo)}}).execute()
 (run/'complete.json').write_text(json.dumps(dict(notebook=str(output),run=str(run))))
except BaseException as e:
 (run/'failed.json').write_text(json.dumps(dict(error=str(e)[-4000:])))
 raise
finally:nbformat.write(nb,output)
