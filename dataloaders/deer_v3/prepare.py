"""Prepare portable YOLO paths and MMDetection dataset configs. Standard library only."""
from pathlib import Path
import argparse,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);a=p.parse_args();root=a.root.resolve();runtime=root/'runtime';runtime.mkdir(exist_ok=True);sets={}
for split in ['train','val','test','train_decoy_4x']:
 rel=(root/'splits'/f'{split}.txt').read_text().splitlines();sets[split]=set(rel)
 for n in rel:
  f=root/n;assert f.exists() and (root/'labels'/f.parent.name/f.with_suffix('.txt').name).exists(),n
 (runtime/f'{split}.txt').write_text(''.join(str(root/n)+'\n' for n in rel))
assert not(sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test'])
if a.verify:
 for n,h in json.loads((root/'checksums.json').read_text()).items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
(runtime/'data.yaml').write_text(f'path: {json.dumps(str(root))}\ntrain: {json.dumps(str(runtime/"train_decoy_4x.txt"))}\nval: {json.dumps(str(runtime/"val.txt"))}\ntest: {json.dumps(str(runtime/"test.txt"))}\nnames:\n  0: deer\n')
# Equivalent COCO slices for MMDetection; no duplicated IDs within any annotation file.
train=json.loads((root/'annotations/train.coco.json').read_text());di=[i for i in train['images'] if Path(i['file_name']).name.startswith('mississippi_decoy__')];ids={i['id'] for i in di};dc=dict(images=di,annotations=[b for b in train['annotations'] if b['image_id'] in ids],categories=train['categories']);(runtime/'train_decoys.coco.json').write_text(json.dumps(dc))
# MMDetection 2.x dataset config fragment: caller supplies identical desired train/eval pipelines.
(root_config:=runtime/'mmdet_datasets.py').write_text('''# Generated locally. Call training_dataset(train_pipeline) and evaluation_dataset(split, test_pipeline).
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def coco(ann,pipeline,test_mode=False):
    return dict(type='CocoDataset',ann_file=str(ann),img_prefix=str(ROOT/'images')+'/',classes=('deer',),pipeline=pipeline,test_mode=test_mode,filter_empty_gt=False)

def training_dataset(pipeline):
    # Base epoch once + three extra copies of the decoy SUBSET = exactly 4x decoys.
    return dict(type='ConcatDataset',datasets=[coco(ROOT/'annotations/train.coco.json',pipeline),dict(type='RepeatDataset',times=3,dataset=coco(ROOT/'runtime/train_decoys.coco.json',pipeline))])

def evaluation_dataset(split,pipeline):
    assert split in ('val','test')
    return coco(ROOT/'annotations'/f'{split}.coco.json',pipeline,True)
''')
print('Prepared:',runtime/'data.yaml');print('MMDetection helper:',root_config)
