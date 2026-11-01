"""Plot recorded learning histories; no training. --artifacts points to comparison files."""
import argparse,json,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--artifacts',required=True);a=p.parse_args();root=Path(a.artifacts)
def history(name):
 rows=[]
 for line in (root/name).read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 return rows
short=history('gfl_short_history.jsonl');long=history('gfl_long_history.jsonl');yolo=list(csv.DictReader((root/'yolo_results.csv').open()))
fig,axes=plt.subplots(2,2,figsize=(13,9))
for label,rows in [('GFL short',short),('GFL long',long)]:
 epochs=sorted({r['epoch'] for r in rows if 'loss'in r})
 losses=[sum(r['loss'] for r in rows if r.get('epoch')==e and 'loss'in r)/sum(1 for r in rows if r.get('epoch')==e and 'loss'in r) for e in epochs]
 axes[0,0].plot(epochs,losses,label=label)
 vals=[r for r in rows if 'bbox_mAP'in r];axes[0,1].plot([r['epoch'] for r in vals],[r['bbox_mAP'] for r in vals],label=label)
axes[0,0].set(title='GFL training loss (mean logged minibatches)',xlabel='Epoch',ylabel='Loss')
axes[0,1].plot([int(r['epoch']) for r in yolo],[float(r['metrics/mAP50-95(B)']) for r in yolo],label='YOLO11s')
axes[0,1].set(title='Native validation AP50–95 by epoch',xlabel='Epoch',ylabel='AP50–95',ylim=(0,1))
for ax,key,title in [(axes[1,0],'box_loss','YOLO box loss'),(axes[1,1],'cls_loss','YOLO classification loss')]:
 for split in ['train','val']:
  start=20 if key=='cls_loss' else 5
  rs=[r for r in yolo if int(r['epoch'])>=start]
  ax.plot([int(r['epoch']) for r in rs],[float(r[split+'/'+key]) for r in rs],label=split)
 ax.set(title=title+f' (epochs ≥{start})',xlabel='Epoch',ylabel='Loss')
for ax in axes.flat:ax.legend();ax.grid(alpha=.2)
fig.suptitle('Learning curves: losses are comparable within a recipe, not across architectures')
fig.tight_layout();fig.savefig(root/'learning_curve_comparison.png',dpi=170)
