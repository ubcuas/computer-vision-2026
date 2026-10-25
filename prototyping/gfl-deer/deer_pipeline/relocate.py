"""Relocate prepared manifests after copying a research bundle to Colab."""
from pathlib import Path
from .common import ROOT,read_json,write_json

def relocate():
    changed=0
    for p in (ROOT/'data').rglob('*.json'):
        if 'source_metadata' in p.name or p.name.startswith('original_'):continue
        try:d=read_json(p)
        except Exception:continue
        def walk(x):
            if isinstance(x,list):
                for i in x:walk(i)
            elif isinstance(x,dict):
                name=x.get('file_name')
                if isinstance(name,str) and '/deer_research/' in name:
                    x['file_name']=str(ROOT/name.split('/deer_research/',1)[1])
                for v in x.values():
                    if isinstance(v,(dict,list)):walk(v)
        walk(d);write_json(p,d);changed+=1
    for p in (ROOT/'data').rglob('dataset.yaml'):
        lines=p.read_text().splitlines();lines[0]='path: '+str(p.parent.resolve());p.write_text('\n'.join(lines)+'\n')
    print('Relocated',changed,'JSON files to',ROOT)
if __name__=='__main__':relocate()
