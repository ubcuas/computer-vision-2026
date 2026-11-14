"""Export saved YOLO detections for the shared GFL evaluator; no training.
Run with Anant's Ultralytics environment. Ground truth is the audited COCO val.
"""
import argparse,json,time,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--annotations',required=True);p.add_argument('--output',required=True);a=p.parse_args()
import torch,ultralytics
from ultralytics import YOLO
model=YOLO(a.checkpoint);gt=json.loads(Path(a.annotations).read_text());pred=[];times=[]
for n,im in enumerate(gt['images']):
 torch.cuda.synchronize();start=time.perf_counter()
 result=model.predict(im['file_name'],imgsz=1280,conf=.001,iou=.6,max_det=300,device=0,verbose=False)[0]
 torch.cuda.synchronize();elapsed=time.perf_counter()-start
 if n:times.append(elapsed)
 for box,score,cls in zip(result.boxes.xyxy.tolist(),result.boxes.conf.tolist(),result.boxes.cls.tolist()):
  assert cls==0
  x,y,X,Y=box;x=max(0,min(x,im['width']));y=max(0,min(y,im['height']));X=max(0,min(X,im['width']));Y=max(0,min(Y,im['height']))
  if X>x and Y>y:pred.append(dict(image_id=im['id'],category_id=1,bbox=[x,y,X-x,Y-y],score=score))
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(pred))
meta=dict(checkpoint=a.checkpoint,sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),torch=torch.__version__,ultralytics=ultralytics.__version__,parameters=sum(t.numel() for t in model.model.parameters()),imgsz=1280,conf=.001,nms_iou=.6,max_det=300,mean_seconds=sum(times)/len(times),warmup_images=1,images=len(gt['images']),includes='loading + inference + postprocessing; excludes output rendering',gpu=torch.cuda.get_device_name(),checkpoint_train_metrics=model.ckpt.get('train_metrics'))
Path(str(out)+'.meta.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
