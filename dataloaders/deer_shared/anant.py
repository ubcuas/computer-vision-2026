"""Lossless format adapter for Anant's committed dataset; no SAM rerun or resplit."""
import csv,json,hashlib
from pathlib import Path
from PIL import Image
from .common import write_json

def build(repo,output):
    repo=Path(repo).resolve();output=Path(output).resolve()
    base=repo/'prototyping/yolo-fine-tune/dataset'
    with (base/'manifest.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    if len(rows)!=264:raise ValueError('Expected the pinned 264-image dataset')
    inventory=[]
    for split in ('train','val'):
        images=[];anns=[]
        for row in rows:
            if row['split']!=split:continue
            name=row['file'];impath=base/'images'/split/(name+'.jpg');lab=base/'labels'/split/(name+'.txt')
            with Image.open(impath) as im:w,h=im.size
            assert (w,h)==(int(row['width']),int(row['height']))
            image_id=int(row['pdf_page']);images.append(dict(id=image_id,file_name=str(impath),width=w,height=h,source_id=name))
            lines=lab.read_text().splitlines()
            assert len(lines)==int(row['n_boxes'])
            for line in lines:
                cls,x,y,bw,bh=map(float,line.split());assert cls==0 and bw>0 and bh>0
                box=[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h]
                assert box[0]>=-.01 and box[1]>=-.01 and box[0]+box[2]<=w+.01 and box[1]+box[3]<=h+.01
                anns.append(dict(id=len(anns)+1,image_id=image_id,category_id=1,bbox=box,area=box[2]*box[3],iscrowd=0))
            inventory.append(dict(image=name,split=split,image_sha256=hashlib.sha256(impath.read_bytes()).hexdigest(),label_sha256=hashlib.sha256(lab.read_bytes()).hexdigest()))
        write_json(output/'annotations'/f'{split}.json',dict(images=images,annotations=anns,categories=[dict(id=1,name='deer')],info={'description':'Exact Anant dataset, format conversion only'}))
    assert sum(x['split']=='train' for x in inventory)==211
    write_json(output/'inventory.json',inventory)
    return inventory

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--repo',default='.');p.add_argument('--output',required=True);a=p.parse_args();build(a.repo,a.output)
