import hashlib
import json
from pathlib import Path

import os
ROOT = Path(os.environ.get('DEER_WORKSPACE', Path(__file__).resolve().parents[2] / 'runtime/gfl_deer')).resolve()

def read_json(path):
    return json.loads(Path(path).read_text())

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def coco(records):
    images, annotations = [], []
    for i, r in enumerate(records, 1):
        images.append(dict(id=i, file_name=r['file_name'], width=r['width'], height=r['height'],
                           source_id=r['source_id'], group=r['group'], source=r['source']))
        for box in r['boxes']:
            x, y, w, h = box
            annotations.append(dict(id=len(annotations)+1, image_id=i, category_id=1,
                                    bbox=box, area=w*h, iscrowd=0))
    return dict(info={'description':'UBC UAS GFL research; see provenance manifest'},
                licenses=[], images=images, annotations=annotations,
                categories=[dict(id=1, name='deer', supercategory='animal')])

def validate(records, check_files=False):
    seen = set()
    groups = {}
    hashes = {}
    for r in records:
        if r['source_id'] in seen:
            raise ValueError('Repeated source ID: ' + r['source_id'])
        seen.add(r['source_id'])
        if check_files:
            from PIL import Image
            with Image.open(r['file_name']) as im:
                if im.size!=(r['width'],r['height']):raise ValueError('Image dimension mismatch: '+r['source_id'])
        if r.get('split'):
            old=groups.setdefault(r['group'],r['split'])
            if old!=r['split']: raise ValueError('Group split leakage: '+r['group'])
            if r.get('sha256'):
                old=hashes.setdefault(r['sha256'],r['split'])
                if old!=r['split']: raise ValueError('Identical-image split leakage')
        for b in r['boxes']:
            x,y,w,h=b
            import math
            if not all(math.isfinite(v) for v in b) or w<=0 or h<=0 or min(x,y)<0 or x+w>r['width']+0.01 or y+h>r['height']+0.01:
                raise ValueError('Invalid bbox: '+r['source_id']+str(b))
    return True

def export(records, output):
    output=Path(output); validate(records, check_files=True)
    for split in ['train','val','test']:
        selected=[r for r in records if r.get('split')==split and r.get('status')=='reviewed']
        write_json(output/'annotations'/f'{split}.json',coco(selected))
        labeldir=output/'labels'/split; labeldir.mkdir(parents=True, exist_ok=True)
        paths=[]
        for r in selected:
            p=Path(r['file_name']); paths.append(str(p.resolve()))
            lines=[]
            for x,y,w,h in r['boxes']:
                lines.append(f'0 {(x+w/2)/r["width"]:.8f} {(y+h/2)/r["height"]:.8f} {w/r["width"]:.8f} {h/r["height"]:.8f}')
            (labeldir/(p.stem+'.txt')).write_text('\n'.join(lines)+ ('\n' if lines else ''))
        (output/f'{split}.txt').write_text('\n'.join(paths)+'\n' if paths else '')
    # YOLO locates labels by replacing /images/ with /labels/; provide a complete portable layout.
    import shutil
    for r in records:
        if r.get('status')!='reviewed' or not r.get('split'):continue
        target=output/'images'/r['split']/Path(r['file_name']).name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():shutil.copy2(r['file_name'],target)
    (output/'dataset.yaml').write_text(f'path: {output.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: deer\n')
    write_json(output/'manifest.json',records)
