"""Apply recorded visual reviews and freeze reproducible same-site diagnostic splits."""
import hashlib
import random
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

from .common import ROOT,read_json,write_json,export
from .prepare import contact_sheet

def groups(records):
    parent=list(range(len(records)))
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(i,j):parent[find(j)]=find(i)
    hashes=[]
    for r in records:
        arr=np.asarray(Image.open(r['file_name']).convert('L').resize((32,32)),dtype=float)
        basis=np.cos(np.pi*(np.arange(32)[None,:]+.5)*np.arange(8)[:,None]/32)
        coeff=(basis @ arr @ basis.T).flatten()[1:]
        hashes.append(coeff>np.median(coeff))
    for i in range(len(records)):
        for j in range(i):
            if records[i]['sha256']==records[j]['sha256'] or np.count_nonzero(hashes[i]!=hashes[j])<=6:union(i,j)
    for i,r in enumerate(records):r['group']=r['source']+'_view_'+str(find(i))
    return records

def split_groups(records):
    bygroup=defaultdict(list)
    for r in records:bygroup[r['group']].append(r)
    if len(bygroup)<3:raise ValueError('Fewer than 3 view groups; refuse to split near duplicates')
    # Seeded, size-balanced assignment; no use of model predictions.
    rng=random.Random(42);items=list(bygroup.items());rng.shuffle(items);items.sort(key=lambda x:-len(x[1]))
    target=dict(train=.7*len(records),val=.15*len(records),test=.15*len(records));counts=Counter()
    for k,rs in items:
        split=max(target,key=lambda s:target[s]-counts[s])
        for r in rs:r['split']=split
        counts[split]+=len(rs)
    return records

def finalize_awir():
    path=ROOT/'data/awir';rs=read_json(path/'manifest.json')
    for r in rs:
        r['modality']='thermal' if r['source_id'].endswith('_R') else 'rgb'
        r['modality_review']='Visually inspected all 106 thumbnails; _R thermal pattern verified for this package only'
        r['status']='reviewed';date=r['source_id'].split('_')[1]
        r['split']='val' if date=='022521' else 'test' if date=='020221' else 'train'
    write_json(path/'manifest.json',rs);export([r for r in rs if r['modality']=='rgb'],path/'prepared')

def finalize_decoys():
    path=ROOT/'data/decoys';rs=read_json(path/'manifest.json');byid={r['source_id']:r for r in rs}
    crops=read_json(path/'review/crop_index.json');review=read_json(ROOT/'reports/decoy_crop_boxes.json')
    flat=[b for sheet in review['sheets'] for b in sheet]
    if len(flat)!=len(crops):raise ValueError('Review count mismatch')
    excluded=set()
    for r in rs:r['boxes']=[]
    for crop,b in zip(crops,flat):
        sid=crop['source_id'].rsplit('_',1)[0];r=byid[sid]
        if b is None:excluded.add(sid);continue
        im=Image.open(crop['file_name']);scale=min(296/im.width,150/im.height,1)
        x1,y1,x2,y2=[z/scale for z in b]
        x1=max(0,x1)+crop['left'];y1=max(0,y1)+crop['top'];x2=min(im.width,x2)+crop['left'];y2=min(im.height,y2)+crop['top']
        r['boxes'].append([x1,y1,x2-x1,y2-y1])
    for r in rs:
        expected=sum(p['species']=='deer' for p in r['points'])
        r['status']='excluded' if r['source_id'] in excluded else 'reviewed'
        r['annotation_origin']='Agent-created boxes from visually inspected point crops; source-key-derived negatives; not human reviewed'
        if r['status']=='reviewed' and len(r['boxes'])!=expected:raise ValueError('Deer count mismatch')
    accepted=[r for r in rs if r['status']=='reviewed']
    groupfile=ROOT/'reports/decoy_view_groups.json'
    if groupfile.exists():
        mapping=read_json(groupfile)['groups']
        for r in accepted:r['group']=mapping[r['source_id']]
    else:groups(accepted)
    split_groups(accepted)
    write_json(path/'manifest.json',rs);export(accepted,path/'prepared')
    # Overlay QA sheets show actual saved annotations, including clipped animals.
    overlay=[]
    for r in accepted:
        if not r['boxes']:continue
        im=Image.open(r['file_name']).convert('RGB');d=ImageDraw.Draw(im)
        for x,y,w,h in r['boxes']:d.rectangle((x,y,x+w,y+h),outline='red',width=3)
        f=path/'review/boxed'/Path(r['file_name']).name;f.parent.mkdir(exist_ok=True);im.save(f)
        overlay.append(dict(source_id=r['source_id'],file_name=str(f)))
    contact_sheet(overlay,path/'review/boxed_sheets')
    negatives=[r for r in accepted if not r['boxes']];contact_sheet(negatives,path/'review/negative_sheets')
    write_json(ROOT/'reports/decoy_split_summary.json',dict(images=len(accepted),boxes=sum(len(r['boxes']) for r in accepted),
        excluded=sorted(excluded),splits=dict(Counter(r['split'] for r in accepted)),
        positive_images_by_split=dict(Counter(r['split'] for r in accepted if r['boxes'])),
        view_groups=len(set(r['group'] for r in accepted)),
        limitation='Same field and stationary decoy arrangement. SIFT/pHash grouping reduces near-duplicate leakage but does NOT establish cross-flight/site independence. Same-site diagnostic only.'))

def adaptation():
    awir=read_json(ROOT/'data/awir/manifest.json');decoy=read_json(ROOT/'data/decoys/manifest.json')
    rs=[r for r in awir if r['modality']=='rgb' and r['split'] in ['train','val']]
    rs += [r for r in decoy if r['status']=='reviewed' and r.get('split') in ['train','val']]
    # Use the exact decoy test set for model comparison; AWIR test remains a separate diagnostic.
    rs += [r for r in decoy if r['status']=='reviewed' and r.get('split')=='test']
    export(rs,ROOT/'data/adaptation')
    write_json(ROOT/'reports/data_summary.json',dict(awir_rgb=sum(r['modality']=='rgb' for r in awir),
        awir_thermal_excluded=sum(r['modality']=='thermal' for r in awir),adaptation_splits=dict(Counter(r['split'] for r in rs)),
        split_frozen=True,test_never_used_for_selection=True))
