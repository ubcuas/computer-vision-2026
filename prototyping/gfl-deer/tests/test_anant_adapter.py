import unittest,tempfile,json
from pathlib import Path
from dataloaders.deer_shared.anant import build
class ExactAdapterTest(unittest.TestCase):
 def test_committed_data_preserved(self):
  repo=Path(__file__).resolve().parents[3]
  with tempfile.TemporaryDirectory() as folder:
   inventory=build(repo,folder)
   self.assertEqual(len(inventory),264)
   self.assertEqual(len({r['image'] for r in inventory}),264)
   total=0
   for split,n in [('train',211),('val',53)]:
    d=json.loads((Path(folder)/'annotations'/f'{split}.json').read_text());self.assertEqual(len(d['images']),n);total+=len(d['annotations'])
    byid={i['id']:i for i in d['images']}
    for ann in d['annotations']:
     im=byid[ann['image_id']];x,y,w,h=ann['bbox']
     lines=(repo/'prototyping/yolo-fine-tune/dataset/labels'/split/(im['source_id']+'.txt')).read_text().splitlines()
     actual=[(x+w/2)/im['width'],(y+h/2)/im['height'],w/im['width'],h/im['height']]
     self.assertTrue(any(all(abs(a-b)<1e-9 for a,b in zip(actual,map(float,l.split()[1:]))) for l in lines))
   self.assertEqual(total,146)
