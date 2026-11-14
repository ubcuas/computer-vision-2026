"""Use the GFL environment from repo root, with --workspace set to the saved runs. No training."""
import json,hashlib,argparse
from pathlib import Path
from dataloaders.deer_shared.metrics import calibrate,evaluate
parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True);args=parser.parse_args()
root=Path(args.workspace);out=root/'comparison';gt=root/'data/anant_exact/annotations/val.json';truth=json.loads(gt.read_text());rows={}
for name,pred in [('yolo',out/'yolo_predictions.json'),('gfl_short',root/'runs/gfl_decoy_run1_v1/val_predictions.json'),('gfl_short_best_ap',out/'gfl_short_best_ap_predictions.json'),('gfl_long',root/'runs/gfl_decoy_long_v1/val_predictions.json')]:
 cal=calibrate(truth,json.loads(pred.read_text()));cp=out/(name+'_calibration.json');cp.write_text(json.dumps(cal,indent=2))
 m=evaluate(gt,pred,out/(name+'_metrics.json'),calibration=cp)
 rows[name]={k:v for k,v in m.items() if k not in ['per_image','ground_truth','predictions','calibration']}
 print(name,json.dumps(rows[name]))
(out/'summary.json').write_text(json.dumps(dict(models=rows,ground_truth_sha256=hashlib.sha256(gt.read_bytes()).hexdigest(),images=53,deer=34,partition='validation only; no independent test',nms_iou=.6,score_floor=.001,max_det=300),indent=2))
