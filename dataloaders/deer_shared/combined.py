"""Add exact committed AWIR split to the decoy adapter for Anant Run #2."""
from pathlib import Path
import json,hashlib,os
from PIL import Image
from .common import write_json

def combine(repo,workspace,decoy):
    root=Path(os.environ.get('ANANT_AWIR_ROOT',Path(repo)/'prototyping/yolo-fine-tune/dataset_awir'));out=Path(workspace)/'data/anant_combined'
    inventory=json.loads((Path(decoy)/'inventory.json').read_text())
    for split,expected in [('train',42),('val',11)]:
        d=json.loads((Path(decoy)/'annotations'/f'{split}.json').read_text())
        files=sorted((root/'images'/split).glob('*'));files=[f for f in files if f.suffix.lower() in ['.jpg','.jpeg','.png']]
        assert len(files)==expected
        for index,f in enumerate(files,10000):
            label=root/'labels'/split/(f.stem+'.txt')
            with Image.open(f) as im:w,h=im.size
            d['images'].append(dict(id=index,file_name=str(f.resolve()),width=w,height=h,source_id='awir_'+f.stem))
            for line in label.read_text().splitlines():
                c,x,y,bw,bh=map(float,line.split());assert c==0 and bw>0 and bh>0
                box=[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h]
                assert min(box[:2])>=-.01 and box[0]+box[2]<=w+.01 and box[1]+box[3]<=h+.01
                d['annotations'].append(dict(id=len(d['annotations'])+1,image_id=index,category_id=1,bbox=box,area=box[2]*box[3],iscrowd=0))
            inventory.append(dict(image=f.name,source='awir',split=split,image_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),label_sha256=hashlib.sha256(label.read_bytes()).hexdigest()))
        write_json(out/'annotations'/f'{split}.json',d)
    write_json(out/'inventory.json',inventory)
    return out
