"""Repo entry point. Run from any directory; data and weights stay external."""
import os,sys,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path[:0]=[str(REPO),str(HERE)]
p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare-matched','train-matched']);p.add_argument('--workspace',required=True);p.add_argument('--workdir');p.add_argument('--resume');p.add_argument('--variant',choices=['decoy','combined'],default='decoy');p.add_argument('--epochs',type=int,default=10);p.add_argument('--patience',type=int,default=0);a=p.parse_args()
if a.epochs<3 or a.patience<0:p.error('epochs must be >=3 and patience >=0')
os.environ['DEER_WORKSPACE']=str(Path(a.workspace).resolve())
from dataloaders.deer_shared.anant import build
workspace=Path(a.workspace).resolve();dataset=workspace/'data/anant_exact'
build(REPO,dataset)
if a.variant=='combined':
    from dataloaders.deer_shared.combined import combine
    dataset=combine(REPO,workspace,dataset)

if a.action=='train-matched':
    from deer_pipeline.train import run_train
    dest=Path(a.workdir).resolve() if a.workdir else workspace/'runs/decoy_matched_v1'
    if any(dest.glob('*.pth')) and not a.resume:raise RuntimeError('Existing checkpoints: use --resume or choose a new workdir')
    run_train(dataset,dest,model='x101',lr=.00025,epochs=a.epochs,long_edge=1280,resume=a.resume,patience=a.patience)
