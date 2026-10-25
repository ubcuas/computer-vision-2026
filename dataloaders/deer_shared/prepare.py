"""Source-preserving extraction, download and audited annotation conversion."""
import csv
import hashlib
import io
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import requests
from PIL import Image, ImageDraw
from .common import ROOT, digest, read_json, write_json, export

API='https://borealisdata.ca/api/'
DOI='doi:10.5683/SP3/TELIYD'

def get(url):
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s=requests.Session();s.mount('https://',HTTPAdapter(max_retries=Retry(total=5,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
    r=s.get(url,timeout=(20,180));r.raise_for_status();return r

def awir(source, output):
    source=Path(source); output=Path(output); records=[]
    for f in sorted(source.glob('*.JPG')):
        im=Image.open(f).convert('RGB');w,h=im.size
        # File suffix is NOT a reliable modality indicator. All require review.
        boxes=[]
        for line in f.with_suffix('.txt').read_text().splitlines():
            cl,x,y,bw,bh=map(float,line.split())
            if cl!=1:raise ValueError('Unexpected AWIR class '+str(cl))
            boxes.append([(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h])
        target=output/'images'/('awir_'+f.name);target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():im.save(target,quality=95)
        records.append(dict(source_id='awir_'+f.stem,source='awir',file_name=str(target.resolve()),width=w,height=h,
                            boxes=boxes,group='awir_'+f.name.split('_')[0],status='pending',modality='unknown',
                            original_file=str(f.resolve()),original_md5=digest(f,'md5'),sha256=digest(target)))
    write_json(output/'manifest.json',records)
    contact_sheet(records,output/'review',limit=120)
    return records

def contact_sheet(records, output, limit=300):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for offset in range(0,min(len(records),limit),24):
        page=Image.new('RGB',(1200,6*180),'white');d=ImageDraw.Draw(page)
        for j,r in enumerate(records[offset:offset+24]):
            im=Image.open(r['file_name']).convert('RGB');im.thumbnail((296,150))
            x=(j%4)*300;y=(j//4)*180;page.paste(im,(x,y));d.text((x+3,y+153),r['source_id'][:42],fill='black')
        page.save(output/f'contact_{offset//24:02}.jpg',quality=90)

def decoys(pdf,key,output):
    import fitz
    output=Path(output);points=defaultdict(list)
    for row in csv.DictReader(open(key)):
        slide=int(re.search(r'(\d+)',row['image']).group())
        points[slide].append(dict(species=row['class'],x=float(row['x']),y=float(row['y'])))
    doc=fitz.open(pdf);records=[]
    for n,page in enumerate(doc):
        # Large original slide ID at upper left; small footer is the shuffled page number.
        spans=[(s['text'].strip(),s['bbox']) for b in page.get_text('dict')['blocks'] if 'lines'in b for l in b['lines'] for s in l['spans']]
        ids=[int(t) for t,b in spans if t.isdigit() and b[0]<page.rect.width*.15 and b[1]<page.rect.height*.4]
        if len(ids)!=1:raise ValueError(f'Ambiguous original slide ID on page {n+1}: {spans}')
        original_slide=ids[0];slide=n+1
        info=page.get_image_info(xrefs=True)
        if len(info)!=1:raise ValueError(f'Expected one aerial image on page {n+1}')
        imageinfo=info[0];bbox=fitz.Rect(imageinfo['bbox'])
        im=Image.open(io.BytesIO(doc.extract_image(imageinfo['xref'])['image'])).convert('RGB')
        target=output/'images'/f'decoy_{slide:03}.jpg';target.parent.mkdir(parents=True,exist_ok=True)
        im.save(target,quality=95)
        # Key SlideN is SHUFFLED PDF PAGE N, not the large original ID printed top-left.
        # Its points refer to a 300 dpi full-slide canvas. Remove the image's PDF offset.
        pp=[dict(species=p['species'],x=(p['x']*72/300-bbox.x0)*im.width/bbox.width,
                 y=(p['y']*72/300-bbox.y0)*im.height/bbox.height) for p in points.get(slide,[])]
        if any(not (0<=p['x']<im.width and 0<=p['y']<im.height) for p in pp):
            raise ValueError(f'Key/image alignment outside image on page {n+1}')
        records.append(dict(source_id=f'decoy_{slide:03}',source='decoy',file_name=str(target.resolve()),
            width=im.width,height=im.height,boxes=[],points=pp,original_slide=original_slide,pdf_page=n+1,
            group='decoy_unresolved',status='pending',modality='rgb',sha256=digest(target),
            annotation_origin='point-assisted agent annotation; pending box review'))
    write_json(output/'manifest.json',records);contact_sheet(records,output/'review')
    crops=[]
    for r in records:
        im=Image.open(r['file_name']);deer=[p for p in r['points'] if p['species']=='deer']
        for j,p in enumerate(deer):
            x,y=p['x'],p['y'];left=max(0,int(x)-130);top=max(0,int(y)-100)
            crop=im.crop((left,top,min(im.width,left+260),min(im.height,top+200)))
            draw=ImageDraw.Draw(crop);cx=x-left;cy=y-top
            draw.ellipse((cx-3,cy-3,cx+3,cy+3),outline='red',width=1)
            path=output/'review'/'crops'/f'{r["source_id"]}_{j}.jpg';path.parent.mkdir(exist_ok=True)
            crop.save(path);crops.append(dict(source_id=f'{r["source_id"]}_{j}',file_name=str(path),left=left,top=top,point=p))
    write_json(output/'review'/'crop_index.json',crops)
    contact_sheet(crops,output/'review'/'crop_sheets')
    return records

def sherbrooke(output, workers=4, limit=None, validation='full'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    metadata=get(API+'datasets/:persistentId/?persistentId='+DOI).json()['data']['latestVersion']
    write_json(output/'source_metadata.json',metadata)
    files={f['dataFile']['filename']:f['dataFile'] for f in metadata['files']}
    allrecords=[]
    for split,suffix in [('train','Train'),('val','Val')]:
        annfile=files[f'Ann_S_wds_{suffix}.json'];ann=get(API+'access/datafile/'+str(annfile['id'])).json()
        write_json(output/f'original_{split}.json',ann)
        boxes=defaultdict(list)
        for a in ann['annotations']:boxes[a['image_id']].append(a['bbox'])
        chosen=[i for i in ann['images'] if split!='train' or boxes[i['id']]]
        if validation=='positive' and split=='val':chosen=[i for i in chosen if boxes[i['id']]]
        if limit is not None:chosen=chosen[:limit]
        def download(i):
            f=files[i['file_name']];target=output/'images'/i['file_name'];target.parent.mkdir(parents=True,exist_ok=True)
            receipt=output/'receipts'/(i['file_name']+'.json')
            if receipt.exists() and target.exists():return read_json(receipt)
            blob=get(API+'access/datafile/'+str(f['id'])).content
            md5=hashlib.md5(blob).hexdigest();expected=f.get('checksum',{}).get('value')
            if f.get('checksum',{}).get('type')=='MD5' and expected and md5!=expected:raise ValueError('Download checksum mismatch')
            im=Image.open(io.BytesIO(blob)).convert('RGB');ow,oh=im.size
            im.thumbnail((2650,1768),Image.Resampling.LANCZOS);w,h=im.size
            outboxes=[[x*w/ow,y*h/oh,bw*w/ow,bh*h/oh] for x,y,bw,bh in boxes[i['id']]]
            im.save(target,quality=95)
            r=dict(source_id='sherbrooke_'+i['file_name'].rsplit('.',1)[0],source='sherbrooke',file_name=str(target.resolve()),
                width=w,height=h,original_width=ow,original_height=oh,boxes=outboxes,group='sherbrooke_official_'+split,
                split=split,status='reviewed',modality='rgb',sha256=digest(target),original_md5=md5,
                original_sha256=hashlib.sha256(blob).hexdigest(),dataverse_file_id=f['id'],
                annotation_origin='published COCO annotations',download_bytes=len(blob))
            write_json(receipt,r);return r
        with ThreadPoolExecutor(max_workers=workers) as ex:
            jobs=[ex.submit(download,i) for i in chosen]
            for j,future in enumerate(as_completed(jobs),1):
                allrecords.append(future.result())
                if j%10==0 or j==len(jobs):print(f'{split}: {j}/{len(jobs)}',flush=True)
        write_json(output/'manifest.json',sorted(allrecords,key=lambda r:r['source_id']))
    export(allrecords,output/'prepared')
    return allrecords
