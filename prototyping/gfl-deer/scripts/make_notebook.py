import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def md(s):return dict(cell_type='markdown',metadata={},source=s.splitlines(True))
def code(s):return dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True))
cells=[md('''# Gurnek: GFL aerial deer -> decoys
Reconstructed Moreni et al. methodology, not exact replication. Choose **Runtime -> Change runtime type -> T4 GPU**. No paid compute. Upload the bundle ZIP (or every numbered ZIP shard). Source annotations were agent-reviewed, not human-reviewed. Decoy test is a same-site diagnostic.
'''),code('''from google.colab import files
from pathlib import Path
import zipfile, os, subprocess, sys
uploaded = files.upload()
for name in sorted(uploaded, key=lambda n: (n == 'gfl_code_update.zip', n)):
    if name.endswith('.zip'):
        with zipfile.ZipFile(name) as archive:
            for member in archive.infolist():
                target=(Path('/content') / member.filename).resolve()
                assert target.is_relative_to(Path('/content')), 'Unsafe archive member'
            archive.extractall('/content')
ROOT=Path('/content/deer_research')
assert ROOT.exists(), 'Upload the research bundle, not a raw dataset archive'
os.chdir(ROOT)
'''),md('''## Persist outputs
Drive mounting asks for Google authorization. Save outputs to your own Drive; this does not alter sharing. If you decline mounting, explicitly use `/content/gfl_runs` and download all results before the runtime disconnects.
'''),code('''from google.colab import drive
drive.mount('/content/drive')
RUNS=Path('/content/drive/MyDrive/Gurnek_GFL_Research/runs')
RUNS.mkdir(parents=True,exist_ok=True)
'''),code('''r=subprocess.run(['bash','scripts/setup_colab.sh'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
Path('setup.log').write_text(r.stdout)
print(r.stdout[-10000:])
r.check_returncode()
PY='/content/gfl-env/bin/python'
'''),code('''subprocess.run([PY,'-m','deer_pipeline.relocate'],check=True)
subprocess.run([PY,'-m','unittest','discover','-s','tests','-v'],check=True)
'''),md('''## Complete/resume the Sherbrooke download
A complete bundle may already contain this data. Otherwise the downloader resumes verified receipts, selects 328 positive summer train images and all 512 summer validation images, checks publisher MD5, resizes to the paper's nominal dimensions and discards full-resolution bytes. Several tens of GB traverse the network; only resized images are retained.
'''),code('''r=subprocess.run([PY,'-u','-m','deer_pipeline','sherbrooke','--workers','4'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
Path('download.log').write_text(r.stdout)
print(r.stdout[-8000:])
r.check_returncode()
'''),md('''## Train and evaluate
The complete driver performs GPU backward/checkpoint/resume smoke checks, three 3-epoch LR trials, a fresh 10-epoch source training run, then 10 adaptation epochs. It records OOM/resolution/backbone changes. Stop on unexplained failures; logs remain under RUNS. Run duration is GPU/data dependent.
'''),code('''r=subprocess.run([PY,'-u','-m','deer_pipeline','run','--workdir',str(RUNS),'--model','x101'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(RUNS/'driver.log').write_text(r.stdout)
print(r.stdout[-10000:])
r.check_returncode()
'''),code('''import json
for stage in ['pre_adaptation','adapted']:
    p=RUNS/'comparison'/stage/'metrics.json'
    if p.exists():
        m=json.loads(p.read_text())
        print(stage,{k:v for k,v in m.items() if k!='per_image'})
print('Checkpoints and run metadata:',RUNS)
'''),md('''## Compare teammates
Use exactly `data/decoys/prepared/annotations/test.json` and the same source images for all checkpoints. Calibrate each model on `val.json` only. Convert predictions to COCO JSON `{image_id, category_id:1, bbox:[x,y,w,h], score}`. Report hardware, resolution and latency. Do not claim a winner from literature scores or an incomplete training run. See `README.md` and `RESEARCH.md`.
''')]
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab={'name':'GFL_Deer_Colab.ipynb','gpuType':'T4'},accelerator='GPU',kernelspec={'name':'python3','display_name':'Python 3'},language_info={'name':'python'}),cells=cells)
(ROOT/'notebooks/GFL_Deer_Colab.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
