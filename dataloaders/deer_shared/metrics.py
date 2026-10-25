"""Framework-independent image-level detection/count metrics; never tune on test."""
import contextlib
import io
import math
from collections import defaultdict
import numpy as np
from .common import read_json, write_json

def iou(a,b):
    x,y,w,h=a;X,Y,W,H=b
    inter=max(0,min(x+w,X+W)-max(x,X))*max(0,min(y+h,Y+H)-max(y,Y))
    return inter/(w*h+W*H-inter) if w*h+W*H>inter else 0.

def validate_predictions(gt,predictions):
    ids={i['id']:i for i in gt['images']}
    for p in predictions:
        if p['image_id'] not in ids:raise ValueError('Prediction for unknown image ID')
        if p.get('category_id',1)!=1:raise ValueError('Only deer category_id=1 supported')
        if not math.isfinite(p['score']) or not 0<=p['score']<=1:raise ValueError('Invalid confidence')
        x,y,w,h=p['bbox'];im=ids[p['image_id']]
        if not all(math.isfinite(z) for z in [x,y,w,h]) or min(x,y)<0 or min(w,h)<=0 or x+w>im['width']+.01 or y+h>im['height']+.01:raise ValueError('Prediction bbox outside image')

def detection_metrics(gt,predictions,threshold=.5):
    validate_predictions(gt,predictions)
    truth=defaultdict(list);pred=defaultdict(list)
    for a in gt['annotations']:truth[a['image_id']].append(a['bbox'])
    for p in predictions:
        if p['score']>=threshold:pred[p['image_id']].append(p)
    tp=fp=fn=0;errors=[];rows=[]
    for im in gt['images']:
        boxes=truth[im['id']];used=set();t=0
        for p in sorted(pred[im['id']],key=lambda a:-a['score']):
            candidates=[(iou(p['bbox'],b),j) for j,b in enumerate(boxes) if j not in used]
            overlap,j=max(candidates,default=(0,-1))
            if overlap>=.5:used.add(j);t+=1
        count=len(pred[im['id']]);tp+=t;fp+=count-t;fn+=len(boxes)-t
        errors.append(count-len(boxes));rows.append(dict(image_id=im['id'],source_id=im.get('source_id'),truth=len(boxes),predicted=count,tp=t,fp=count-t,fn=len(boxes)-t))
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    return dict(threshold=threshold,iou_threshold=.5,tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,
      f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
      count_mae=float(np.mean(np.abs(errors))) if errors else 0.,count_bias=float(np.mean(errors)) if errors else 0.,
      exact_count_accuracy=float(np.mean(np.array(errors)==0)) if errors else 0.,images=len(errors),per_image=rows)

def ap_metrics(gt,predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not gt['annotations']:return {'ap50':None,'ap50_95':None}
    with contextlib.redirect_stdout(io.StringIO()):
        c=COCO();c.dataset=gt;c.createIndex()
        if predictions:d=c.loadRes(predictions)
        else:
            d=COCO();d.dataset=dict(images=gt['images'],annotations=[],categories=gt['categories']);d.createIndex()
        e=COCOeval(c,d,'bbox');e.evaluate();e.accumulate();e.summarize()
    return dict(ap50=float(e.stats[1]),ap50_95=float(e.stats[0]))

def calibrate(gt,predictions):
    curve=[detection_metrics(gt,predictions,float(t)) for t in np.linspace(.01,.99,99)]
    best=max(curve,key=lambda m:(m['f1'],-m['count_mae'],m['threshold']))
    return dict(threshold=best['threshold'],selected_on='validation only',selection_metric='F1@IoU0.5, then count MAE, then higher threshold',
                curve=[{k:v for k,v in m.items() if k!='per_image'} for m in curve])

def evaluate(gtpath,predpath,output,threshold=.5,calibration=None):
    gt=read_json(gtpath);pred=read_json(predpath)
    if not gt['images']:raise ValueError('Cannot evaluate an empty partition')
    if calibration:threshold=read_json(calibration)['threshold']
    result=detection_metrics(gt,pred,threshold);result.update(ap_metrics(gt,pred))
    result.update(ground_truth=str(gtpath),predictions=str(predpath),calibration=str(calibration) if calibration else None)
    write_json(output,result);return result
