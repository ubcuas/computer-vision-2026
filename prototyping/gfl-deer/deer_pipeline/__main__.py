import argparse
from pathlib import Path
from .common import ROOT,read_json,write_json,validate

def main():
    p=argparse.ArgumentParser(description='UBC UAS reconstructed GFL deer research pipeline')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('sherbrooke');s.add_argument('--output',default=str(ROOT/'data/sherbrooke'));s.add_argument('--workers',type=int,default=4);s.add_argument('--limit',type=int);s.add_argument('--validation',choices=['full','positive'],default='full')
    s=sub.add_parser('prepare-local');s.add_argument('--uas',default=str(ROOT.parent))
    s=sub.add_parser('finalize')
    s=sub.add_parser('validate');s.add_argument('manifest')
    s=sub.add_parser('train');s.add_argument('--dataset',required=True);s.add_argument('--workdir',required=True);s.add_argument('--model',choices=['x101','r50'],default='x101');s.add_argument('--lr',type=float,default=.00025);s.add_argument('--epochs',type=int,default=10);s.add_argument('--long-edge',type=int,default=2650);s.add_argument('--checkpoint');s.add_argument('--resume');s.add_argument('--smoke',action='store_true')
    s=sub.add_parser('predict');s.add_argument('--config',required=True);s.add_argument('--checkpoint',required=True);s.add_argument('--annotations',required=True);s.add_argument('--output',required=True);s.add_argument('--device',default='cuda:0')
    s=sub.add_parser('evaluate');s.add_argument('--annotations',required=True);s.add_argument('--predictions',required=True);s.add_argument('--output',required=True);s.add_argument('--calibration');s.add_argument('--threshold',type=float,default=.5)
    s=sub.add_parser('calibrate');s.add_argument('--annotations',required=True);s.add_argument('--predictions',required=True);s.add_argument('--output',required=True)
    s=sub.add_parser('run');s.add_argument('--workdir',required=True);s.add_argument('--model',choices=['x101','r50'],default='x101')
    a=p.parse_args();d=vars(a).copy();d.pop('command')
    if a.command=='sherbrooke':
        from .prepare import sherbrooke
        sherbrooke(**d)
    elif a.command=='prepare-local':
        from .prepare import awir,decoys
        awir(Path(a.uas)/'AWIR-Deer',ROOT/'data/awir')
        decoys(Path(a.uas)/'Deer Decoy Dataset/Master1_no_key_0s_randomized.pdf',Path(a.uas)/'Deer Decoy Dataset/decoy_survey_key_points.csv',ROOT/'data/decoys')
    elif a.command=='finalize':
        from .finalize import finalize_awir,finalize_decoys,adaptation
        finalize_awir();finalize_decoys();adaptation()
    elif a.command=='validate':validate(read_json(a.manifest),check_files=True);print('Manifest validation passed')
    elif a.command=='train':
        from .train import run_train
        run_train(**d)
    elif a.command=='predict':
        from .train import predict
        d['config_path']=d.pop('config');predict(**d)
    elif a.command=='evaluate':
        from .metrics import evaluate
        r=evaluate(a.annotations,a.predictions,a.output,a.threshold,a.calibration);print({k:v for k,v in r.items() if k!='per_image'})
    elif a.command=='calibrate':
        if Path(a.annotations).stem!='val':raise ValueError('Calibration requires a file named val.json')
        from .metrics import calibrate
        write_json(a.output,calibrate(read_json(a.annotations),read_json(a.predictions)))
    elif a.command=='run':
        from .experiment import run
        run(a.workdir,a.model)

if __name__=='__main__':main()
