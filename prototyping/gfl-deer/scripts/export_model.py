"""Create a portable inference bundle, without images or optimizer state."""
import argparse,hashlib,json,zipfile,tempfile
from pathlib import Path

def export(run):
 import torch
 run=Path(run).resolve();weights=run/'best_f1.pth';config=run/'config.py'
 if not weights.exists() or not config.exists():raise FileNotFoundError('Training must produce best_f1.pth and config.py first')
 target=run/'gfl_model_bundle.zip'
 with tempfile.TemporaryDirectory() as tmp:
  tmp=Path(tmp);saved=torch.load(weights,map_location='cpu')
  torch.save({'state_dict':saved['state_dict'],'meta':saved.get('meta',{})},tmp/'model.pth')
  (tmp/'README.txt').write_text('GFL deer inference bundle. Install the pinned MMDetection environment from pip-freeze.txt.\nmodel.pth contains trained weights without optimizer state. config.py specifies architecture/preprocessing; its original training dataset paths are provenance and are not needed for inference.\nUse: from mmdet.apis import init_detector, inference_detector\nmodel=init_detector("config.py","model.pth",device="cuda:0")\nresult=inference_detector(model,"YOUR_IMAGE.jpg")\nRead validation_calibration.json for the selected confidence threshold.\nFor training continuation retain the ORIGINAL epoch_N.pth separately; this bundle is for inference, not optimizer resume.\n')
  files={'model.pth':tmp/'model.pth','README.txt':tmp/'README.txt','config.py':config}
  for name in ['validation_calibration.json','validation_metrics.json','environment.json','pip-freeze.txt']:
   if (run/name).exists():files[name]=run/name
  for p in (run/'report').glob('*'):
   if p.is_file():files['report/'+p.name]=p
  manifest={name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for name,p in files.items()}
  with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_STORED) as archive:
   for name,p in files.items():archive.write(p,name)
   archive.writestr('manifest.json',json.dumps(manifest,indent=2))
 with zipfile.ZipFile(target) as archive:
  assert archive.testzip() is None
  for name,entry in manifest.items():assert hashlib.sha256(archive.read(name)).hexdigest()==entry['sha256']
 print('Verified portable model bundle:',target)
 return target
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();export(a.run)
