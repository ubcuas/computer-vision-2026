"""Repo entry point. Run from any directory; data and weights stay external."""
import os,sys,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path[:0]=[str(REPO),str(HERE)]
p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare-matched','train-matched']);p.add_argument('--workspace',required=True);p.add_argument('--workdir');p.add_argument('--resume');a=p.parse_args()
os.environ['DEER_WORKSPACE']=str(Path(a.workspace).resolve())
from dataloaders.deer_shared.anant import build
workspace=Path(a.workspace).resolve();dataset=workspace/'data/anant_exact'
build(REPO,dataset)
if a.action=='train-matched':
    from deer_pipeline.train import run_train
    dest=Path(a.workdir).resolve() if a.workdir else workspace/'runs/decoy_matched_v1'
    if (dest/'epoch_10.pth').exists() and not a.resume:raise RuntimeError('Completed baseline exists; use saved checkpoint for inference')
    run_train(dataset,dest,model='x101',lr=.00025,epochs=10,long_edge=1280,resume=a.resume)
