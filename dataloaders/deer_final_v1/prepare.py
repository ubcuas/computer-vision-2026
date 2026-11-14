"""Materialize identical YOLO/COCO splits against a local external dataset root."""
from pathlib import Path
import argparse,json,hashlib
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
source=json.loads((root/'annotations/deer.coco.json').read_text());images={i['file_name']:i for i in source['images']};annotations={}
for ann in source['annotations']:annotations.setdefault(ann['image_id'],[]).append(ann)
summary=json.loads((HERE/'summary.json').read_text());sets={}
for split in ['train','val','test','train_decoy_4x']:
 file=HERE/(split+'.txt');assert hashlib.sha256(file.read_bytes()).hexdigest()==summary['split_file_sha256'][file.name]
 names=file.read_text().splitlines();paths=[root/n for n in names]
 assert all(x.exists() and (root/'labels'/x.with_suffix('.txt').name).exists() for x in paths),'Missing image/label files'
 (out/(split+'.txt')).write_text(''.join(str(x)+'\n' for x in paths))
 if split=='train_decoy_4x':continue
 sets[split]=set(names);ims=[images[Path(n).name] for n in names];anns=[v for i in ims for v in annotations.get(i['id'],[])]
 (out/(split+'.coco.json')).write_text(json.dumps({'images':ims,'annotations':anns,'categories':source['categories']}))
assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
# COCO image root is ROOT/images. Weighted COCO must use a training sampler; never duplicate COCO IDs.
(out/'data.yaml').write_text(f'path: {json.dumps(str(root))}\ntrain: {json.dumps(str(out/"train_decoy_4x.txt"))}\nval: {json.dumps(str(out/"val.txt"))}\ntest: {json.dumps(str(out/"test.txt"))}\nnames:\n  0: deer\n')
print('Prepared', {s:len(v) for s,v in sets.items()},'at',out)
