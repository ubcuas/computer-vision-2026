import copy
import tempfile
import unittest
from pathlib import Path
from deer_pipeline.common import coco,validate,export,read_json
from deer_pipeline.metrics import detection_metrics,calibrate,ap_metrics
from deer_pipeline.train import results_to_coco

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.gt=dict(info={},images=[dict(id=1,width=100,height=100),dict(id=2,width=100,height=100)],categories=[dict(id=1,name='deer')],annotations=[dict(id=1,image_id=1,category_id=1,bbox=[10,10,20,20],area=400,iscrowd=0)])
        self.pred=[dict(image_id=1,category_id=1,bbox=[10,10,20,20],score=.9)]
    def test_perfect_including_empty_images(self):
        r=detection_metrics(self.gt,self.pred);self.assertEqual(r['f1'],1);self.assertEqual(r['exact_count_accuracy'],1)
    def test_duplicate_prediction_only_matches_once(self):
        r=detection_metrics(self.gt,self.pred*2);self.assertEqual((r['tp'],r['fp'],r['fn']),(1,1,0));self.assertEqual(r['count_mae'],.5)
    def test_background_false_alarm(self):
        r=detection_metrics(self.gt,self.pred+[dict(image_id=2,bbox=[1,1,10,10],score=.8)]);self.assertEqual(r['fp'],1)
    def test_low_score_not_counted(self):
        r=detection_metrics(self.gt,[dict(image_id=1,bbox=[10,10,20,20],score=.1)]);self.assertEqual(r['fn'],1)
    def test_unknown_image_and_nan_rejected(self):
        with self.assertRaises(ValueError):detection_metrics(self.gt,[dict(image_id=3,bbox=[1,1,2,2],score=.5)])
        with self.assertRaises(ValueError):detection_metrics(self.gt,[dict(image_id=1,bbox=[1,1,2,2],score=float('nan'))])
    def test_coco_ap(self):
        self.assertAlmostEqual(ap_metrics(self.gt,self.pred)['ap50'],1)
        self.assertEqual(ap_metrics(self.gt,[])['ap50'],0)
    def test_calibration(self):
        pred=self.pred+[dict(image_id=2,bbox=[1,1,2,2],score=.2)]
        c=calibrate(self.gt,pred);self.assertGreater(c['threshold'],.2);self.assertLessEqual(c['threshold'],.9)
    def test_group_leakage(self):
        r=dict(source_id='a',source='test',file_name='a.jpg',width=100,height=100,boxes=[],group='same',split='train')
        with self.assertRaises(ValueError):validate([r,dict(r,source_id='b',split='test')])
    def test_hash_leakage(self):
        a=dict(source_id='a',width=100,height=100,boxes=[],group='a',split='train',sha256='same')
        with self.assertRaises(ValueError):validate([a,dict(a,source_id='b',group='b',split='test')])
    def test_box_validation(self):
        with self.assertRaises(ValueError):validate([dict(source_id='a',width=10,height=10,boxes=[[0,0,11,1]],group='x')])
    def test_results_clipped_and_ordered(self):
        r=results_to_coco([[[[-1,-1,101,101,.7]]], [[]]],self.gt)
        self.assertEqual(r[0]['bbox'],[0,0,100,100]);self.assertEqual(r[0]['image_id'],1)
    def test_real_manifest_integrity(self):
        root=Path(__file__).resolve().parents[1]
        for source in ['awir/prepared','decoys/prepared','adaptation']:
            p=root/'data'/source/'manifest.json'
            if p.exists():validate(read_json(p))
    def test_point_mapping_regression(self):
        root=Path(__file__).resolve().parents[1];p=root/'data/decoys/manifest.json'
        if not p.exists():self.skipTest('Source data not prepared')
        rs=read_json(p);r=next(x for x in rs if x['pdf_page']==167)
        self.assertEqual(r['original_slide'],218);self.assertEqual(len(r['boxes']),2)
        self.assertTrue(all(0<=p['x']<r['width'] for p in r['points']))

if __name__=='__main__':unittest.main()
