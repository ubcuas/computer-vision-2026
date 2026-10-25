"""Bundle code + prepared source images, excluding environments and original large downloads."""
import argparse,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',default='gfl_research_bundle.zip');p.add_argument('--include-sherbrooke',action='store_true');a=p.parse_args()
with zipfile.ZipFile(ROOT/a.output,'w',zipfile.ZIP_DEFLATED) as z:
    for folder in ['deer_pipeline','scripts','tests','reports','notebooks']:
        for f in (ROOT/folder).rglob('*'):
            if f.is_file() and '__pycache__' not in str(f) and f.suffix not in ['.zip']:
                z.write(f,'deer_research/'+str(f.relative_to(ROOT)))
    for f in ROOT.glob('*.md'):z.write(f,'deer_research/'+f.name)
    for f in ROOT.glob('requirements*.txt'):z.write(f,'deer_research/'+f.name)
    for f in (ROOT/'data').rglob('*'):
        if not f.is_file() or '/review/' in str(f) or '/receipts/' in str(f) or '/prepared/images/' in str(f) or '/adaptation/images/' in str(f):continue
        if '/sherbrooke/' in str(f) and not a.include_sherbrooke:continue
        z.write(f,'deer_research/'+str(f.relative_to(ROOT)))
print(ROOT/a.output)
