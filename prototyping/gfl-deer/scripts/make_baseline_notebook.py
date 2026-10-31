import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def cell(kind,text):
 d={'cell_type':kind,'metadata':{},'source':text.splitlines(True)}
 if kind=='code':d.update(execution_count=None,outputs=[])
 return d
cells=[cell('markdown','''# GFL against Anant’s baseline
Run on Michael’s Linux NVIDIA server. Default matches the **reported Run #1**: 211 decoy train / 53 decoy validation images. No Rucervus, Sherbrooke or AWIR. Run #2 in the latest repository uses decoy + AWIR; choose that explicitly below if comparing it. Same source images/boxes/splits; different architecture, augmentation and training budget. Validation is not an independent test. No training runs until the training cell is executed.
'''),cell('markdown','''## Completed short-run results (2026-10-31)
Ten epochs completed on Michael's RTX 3070 Laptop GPU. The best-F1 checkpoint was epoch 4, selected strictly on validation F1. On 53 validation images / 34 deer boxes: precision **0.9143**, recall **0.9412**, F1 **0.9275**, AP50 **0.9501**, AP50-95 **0.5991**, at confidence **0.34** and matching IoU 0.5. There were 32 TP, 3 FP and 2 FN; count MAE 0.0943 deer/image and exact-count accuracy 90.57%. Mean single-image load + inference + postprocessing latency was 0.1291 s (one warmup image excluded; rendering excluded).

Anant's teammate-reported Run #1: precision 0.66, recall 0.75, AP50 0.73, AP50-95 0.37. Our selected checkpoint scores higher on these reported validation metrics; evaluator conventions differ and neither result establishes independent test performance. The best-AP GFL checkpoint is separately retained at epoch 9 (training evaluator AP50-95 0.6767); the metrics above and portable ZIP consistently use **best_f1.pth**, not the epoch-9 checkpoint.

Artifacts on Michael's server: `/home/mrd/gfl_workspace/runs/gfl_decoy_run1_v1/`. The verified `gfl_model_bundle.zip` includes inference weights, config, calibration and reports; epoch checkpoints preserve optimizer state separately. See `RESULTS.md` for checkpoint hashes and comparison limitations.

## Shared data-quality limitation
Gurnek identified oversized/loose bounding boxes in the shared decoy data. Better reviewed box extents are needed for both detectors. Excessive background may impair localization and could contribute to Anant's reported overlapping boxes, but this explanation remains unverified; prediction examples and NMS settings also need inspection. Target/species identity and box extent are separate annotation questions. Current runs preserve original labels; any corrected dataset should be jointly versioned and evaluated separately.

## Why two GFL training runs?
These are **two training recipes for the same GFL architecture**, producing separate trained checkpoints. Anant's YOLO11s is the other detector architecture being compared. The short run asks whether the reconstructed paper recipe transfers to our decoys; the longer run asks whether giving GFL a training budget closer to Anant's improves validation performance. Ten epochs is a paper-based starting point, not evidence that our dataset has converged.

| Setting | GFL short baseline (current) | GFL longer comparison | Anant YOLO11s Run #1 (reported) |
|---|---|---|---|
| Architecture | GFL ResNeXt-101 32x4d + deformable convolutions | Same GFL architecture | YOLO11s |
| Initialization | Official COCO pretrained weights | Fresh official COCO weights | COCO pretrained YOLO11s |
| Data | 211 decoy train / 53 validation | Identical images, boxes and split | Same decoy split |
| Training budget | 10 epochs | Up to 100 epochs; validation AP patience 30 | Up to 100 epochs; patience 30 |
| Learning-rate schedule | Initial LR 0.00025; drops at 6 and 8 | Planned LR 0.00025; drops at 60 and 80 | YOLO training configuration |
| Input preprocessing | Aspect-preserving 1280 long edge; minimum-size padding | Same preprocessing | Reported image size 1280; YOLO letterboxing |
| Status | Running; results pending | Ready for verified separate launch | Teammate-reported results |

The longer recipe uses scaled learning-rate milestones (60/80 for 100 epochs) and stops after 30 epochs without a strict improvement in validation COCO AP50-95. This is an adaptation of the paper recipe. It uses a **new run directory** (`gfl_decoy_long_v1`) and starts independently from COCO, rather than continuing the short run after its learning rate has already decayed. Retain `gfl_decoy_run1_v1` and all its outputs. Do not run both GPU jobs concurrently. The executable cells default to the **10-epoch baseline**; select the longer run using RUN_NAME, EPOCHS and PATIENCE in the configuration cell.

Both runs save best-F1/AP weights and resumable optimizer checkpoints. After each finishes, record the actual epochs, selected checkpoint, validation precision/recall/F1, AP50, AP50-95, count errors, latency, configuration and artifact hashes. Add the measured comparison here once available; do not replace pending results with estimates.

Matching data and maximum epochs makes the comparison more useful, but does not equalize augmentation, optimizer, architecture or compute. Compare AP using the same evaluation implementation where possible; our validation-calibrated P/R/F1 may differ from Ultralytics' reporting conventions. This validation set is used for checkpoint/threshold selection and is **not an independent test set**, so no team winner is established by these numbers.
'''),cell('code','''from pathlib import Path
import os,sys,subprocess,json
# Start Jupyter from the repository root.
REPO=Path.cwd().resolve()
assert (REPO/'prototyping/gfl-deer/run.py').exists(), 'Start Jupyter from repository root'
WORKSPACE=Path('/absolute/server/path/gfl_workspace')  # EDIT
assert str(WORKSPACE) != '/absolute/server/path/gfl_workspace', 'Set your persistent workspace'
RUN_NAME='gfl_decoy_run1_v1'
EPOCHS=10;PATIENCE=0  # Longer run: new RUN_NAME='gfl_decoy_long_v1', EPOCHS=100, PATIENCE=30
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
     '--workdir',str(RUN),'--variant',DATA_VARIANT,'--epochs',str(EPOCHS),'--patience',str(PATIENCE)]
if RESUME:cmd+=['--resume',str(RESUME)]
RUN.mkdir(parents=True,exist_ok=True)
with (RUN/'launcher.log').open('a') as log:
 process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 for line in process.stdout:
  print(line,end='');log.write(line);log.flush()
 assert process.wait()==0, 'Training failed; inspect launcher.log and resume saved checkpoint'
'''),cell('code','''# Validation only, using F1-selected checkpoint. No retraining.
PRED=RUN/'val_predictions.json';CAL=RUN/'validation_calibration.json'
def run(*args):subprocess.run([PY,'-m','deer_pipeline',*map(str,args)],check=True)
run('predict','--config',RUN/'config.py','--checkpoint',RUN/'best_f1.pth','--annotations',DATA/'annotations/val.json','--output',PRED)
run('calibrate','--annotations',DATA/'annotations/val.json','--predictions',PRED,'--output',CAL)
run('evaluate','--annotations',DATA/'annotations/val.json','--predictions',PRED,'--calibration',CAL,'--output',RUN/'validation_metrics.json')
print('Saved weights/config/logs/metrics:',RUN)
''')]
cells += [cell('markdown','''## Saved metrics, curves and prediction examples
Run these cells after training, or when reopening an existing run. They do not retrain. P/R/F1 use the validation-selected confidence at IoU 0.5; AP averages over confidence thresholds. This differs from Ultralytics' reporting conventions. No validation-loss graph is claimed: our loop logs training losses and validation detection metrics.
'''),cell('code','''subprocess.run([PY,'prototyping/gfl-deer/scripts/report.py','--run',str(RUN),
                '--annotations',str(DATA/'annotations/val.json')],check=True)
from IPython.display import display, Image as DisplayImage
print(json.loads((RUN/'validation_metrics.json').read_text()).keys())
print((RUN/'report/metrics.csv').read_text())
display(DisplayImage(filename=str(RUN/'report/training_and_validation.png')))
display(DisplayImage(filename=str(RUN/'report/prediction_examples.png')))
print('Checkpoint inventory:',(RUN/'report/checkpoint_inventory.json').read_text())
'''),cell('markdown','''## Reuse and resume
`best_f1.pth` and `best_bbox_mAP_epoch_*.pth` hold trained weights. `epoch_N.pth` also preserves optimizer/training state; the last two epochs are retained. `latest.pth` may be a symlink, so transfer actual checkpoint files. Keep the complete run directory on persistent storage, together with the dataset inventory. To use the model later, rerun configuration/data cells and prediction/evaluation cells only; skip training. To continue interrupted training, set RESUME to an actual epoch checkpoint and retain the same data/config. Training refuses existing checkpoints without explicit resume. Never delete the run folder to perform inference.
''')]
cells += [cell('markdown','''## Take the trained model off the server
This creates a verified ZIP with trained inference weights, architecture config, confidence calibration, validation results and graphs. No dataset images are bundled. Keep original epoch checkpoints separately if you want to continue training.
'''),cell('code','''subprocess.run([PY,'prototyping/gfl-deer/scripts/export_model.py','--run',str(RUN)],check=True)
BUNDLE=RUN/'gfl_model_bundle.zip'
print('Copy this file off the server:',BUNDLE)
from IPython.display import FileLink
import shutil
# Jupyter serves files under its root; expose a copy in the ignored artifacts directory.
DOWNLOAD=REPO/'artifacts'/RUN_NAME
DOWNLOAD.mkdir(parents=True,exist_ok=True)
shutil.copy2(BUNDLE,DOWNLOAD/BUNDLE.name)
display(FileLink(str((DOWNLOAD/BUNDLE.name).relative_to(REPO))))
print('Alternatively, on your laptop: scp USER@SERVER:'+str(BUNDLE)+' .')
''')]
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'}},'cells':cells}
(root/'notebooks/GFL_Anant_Baseline_Server.ipynb').write_text(json.dumps(nb,indent=2)+'\n')
