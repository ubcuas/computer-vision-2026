"""End-to-end bounded experiment, preserving selection/test separation and run lineage."""
import subprocess
import sys
from pathlib import Path
from .common import ROOT,read_json,write_json
from .metrics import calibrate,detection_metrics,evaluate
from .train import predict

def train_command(dataset,folder,model,lr,epochs,edge,checkpoint=None,smoke=False,resume=None):
    cmd=[sys.executable,'-m','deer_pipeline','train','--dataset',str(dataset),'--workdir',str(folder),'--model',model,
         '--lr',str(lr),'--epochs',str(epochs),'--long-edge',str(edge)]
    if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
    if smoke:cmd+=['--smoke']
    if resume:cmd+=['--resume',str(resume)]
    return cmd

def checkpoint(folder,prefer_f1=False):
    folder=Path(folder)
    if prefer_f1 and (folder/'best_f1.pth').exists():return folder/'best_f1.pth'
    candidates=list(folder.glob('best_bbox_mAP*.pth'))
    if candidates:return sorted(candidates)[-1]
    if (folder/'latest.pth').exists():return folder/'latest.pth'
    raise FileNotFoundError('No completed checkpoint in '+str(folder))

def train_retry(dataset,folder,model,lr,epochs,edge,initial=None,smoke=False):
    """Only documented memory/kernel failures trigger fallback; other errors stop."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    choices=[(model,e) for e in [2650,2048,1600] if e<=edge]
    if model=='x101':choices += [('r50',1600)]
    kernel_failed=False
    for selected,e in choices:
        if kernel_failed and selected=='x101':continue
        if initial and selected!=model:raise RuntimeError('Adaptation backbone fallback requires repeating pretraining with R50; checkpoint cannot be silently reused')
        attempt=folder/f'{selected}_{e}';attempt.mkdir(exist_ok=True)
        log=attempt/'console.log'
        with open(log,'w') as out:
            proc=subprocess.run(train_command(dataset,attempt,selected,lr,epochs,e,initial,smoke),stdout=out,stderr=subprocess.STDOUT,cwd=ROOT)
        if proc.returncode==0:
            write_json(folder/'selected.json',dict(model=selected,long_edge=e,attempt=str(attempt),departure=selected!=model or e!=2650))
            return attempt,selected,e
        txt=log.read_text().lower()
        memory='out of memory' in txt
        kernel=any(k in txt for k in ['not implemented for','no kernel image','deform_conv_forward','invalid device function'])
        if not (memory or (kernel and selected=='x101')):raise RuntimeError('Training failed; see '+str(log))
        if kernel:kernel_failed=True
    raise RuntimeError('All documented fallback attempts failed; see '+str(folder))

def run(workdir,model='x101'):
    workdir=Path(workdir).resolve();workdir.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data/sherbrooke/prepared';adapt=ROOT/'data/adaptation'
    for d in [source,adapt]:
        for s in ['train','val']:
            if not read_json(d/'annotations'/f'{s}.json')['images']:raise ValueError('Empty training/validation data')
    smoke,model,edge=train_retry(source,workdir/'smoke',model,.00025,10,2650,smoke=True)
    # The second smoke resumes optimiser state from the two-iteration checkpoint.
    with open(workdir/'smoke_resume.log','w') as out:
        cmd=train_command(source,workdir/'smoke_resume',model,.00025,10,edge,smoke=True,resume=smoke/'iter_1.pth')
        subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
    trials=[]
    for lr in [.0025,.00025,.000025]:
        f,selected,e=train_retry(source,workdir/f'lr_{lr}',model,lr,3,edge)
        pred=f/'val_predictions.json';predict(f/'config.py',checkpoint(f),source/'annotations/val.json',pred)
        gt=read_json(source/'annotations/val.json');ps=read_json(pred);cal=calibrate(gt,ps)
        metric=detection_metrics(gt,ps,cal['threshold']);trials.append(dict(lr=lr,f1=metric['f1'],folder=str(f),model=selected,long_edge=e))
        write_json(workdir/'lr_trials.json',trials)
    best=max(trials,key=lambda t:t['f1']);model=best['model'];edge=best['long_edge']
    # Restart from COCO for a full 10 epochs; screening runs are never reported as the final fit.
    pre,model,edge=train_retry(source,workdir/'pretrain',model,best['lr'],10,edge)
    pretrained=checkpoint(pre)
    post,model,edge=train_retry(adapt,workdir/'adapt',model,best['lr']/10,10,edge,pretrained)
    adapted=checkpoint(post,prefer_f1=True)
    # One common decoy-only validation set calibrates both comparison models.
    decoy=ROOT/'data/decoys/prepared'
    for label,folder,weights in [('pre_adaptation',pre,pretrained),('adapted',post,adapted)]:
        dest=workdir/'comparison'/label;dest.mkdir(parents=True,exist_ok=True)
        pv=dest/'val_predictions.json';pt=dest/'test_predictions.json'
        predict(folder/'config.py',weights,decoy/'annotations/val.json',pv)
        cal=calibrate(read_json(decoy/'annotations/val.json'),read_json(pv));write_json(dest/'calibration.json',cal)
        predict(folder/'config.py',weights,decoy/'annotations/test.json',pt)
        evaluate(decoy/'annotations/test.json',pt,dest/'metrics.json',calibration=dest/'calibration.json')
    write_json(workdir/'complete.json',dict(pretrained=str(pretrained),adapted=str(adapted),
        limitation='Reconstructed method; same-site decoy diagnostic; no claim of matching paper or winning team comparison'))
