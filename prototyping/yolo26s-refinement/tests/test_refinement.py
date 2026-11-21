"""CPU checks of annotation safety and validation-only review calibration."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('refinement', Path(__file__).parents[1] / 'run.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class RefinementTests(unittest.TestCase):
    def test_crop_contains_every_intersecting_deer(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; run = root / 'out'
            (data / 'annotations').mkdir(parents=True)
            (data / 'images/train').mkdir(parents=True)
            run.mkdir()
            Image.new('RGB', (1000, 1000), 'white').save(data / 'images/train/bambi__example.jpg')
            gt = dict(images=[dict(id=1, file_name='train/bambi__example.jpg', width=1000, height=1000)],
                      annotations=[dict(image_id=1, bbox=[480, 480, 40, 40]),
                                   dict(image_id=1, bbox=[650, 450, 160, 100])])
            m.write(data / 'annotations/train.coco.json', gt)
            paths, counts = m.tiles(data, run, dict(seed=42, tiles_per_source=1,
                              tile_fraction=.6, tile_max_box_area_fraction=.002))
            self.assertEqual(counts, {'bambi': 1})
            row = m.read(run / 'prepared/tiles.json')[0]
            self.assertEqual(row['crop_xywh'], [200, 200, 600, 600])
            self.assertEqual(row['boxes'], [[280, 280, 40, 40], [450, 250, 150, 100]])
            labels = (run / 'prepared/labels/train/bambi__example__tile.txt').read_text().splitlines()
            self.assertEqual(len(labels), 2)
            for line, box in zip(labels, row['boxes']):
                cls, cx, cy, w, h = map(float, line.split())
                self.assertEqual(cls, 0)
                decoded = [(cx-w/2)*600, (cy-h/2)*600, w*600, h*600]
                for actual, expected in zip(decoded, box):
                    self.assertAlmostEqual(actual, expected, places=5)
            self.assertEqual(len(paths), 1)

    def test_infeasible_recall_target_is_reported_and_test_reuses_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); data = root / 'data'; run = root / 'out'
            gt = dict(images=[dict(id=1, file_name='val/bambi__one.jpg', width=100, height=100)],
                      annotations=[dict(image_id=1, bbox=[10, 10, 10, 10]),
                                   dict(image_id=1, bbox=[70, 70, 10, 10])], categories=[dict(id=1, name='deer')])
            pred = [dict(image_id=1, category_id=1, bbox=[10, 10, 10, 10], score=.5)]
            m.write(data / 'annotations/val.coco.json', gt)
            m.write(run / 'val/predictions.json', pred)
            m.write(run / 'recipe.json', dict(review_recall_target=.95, review_max_fp_per_image=2))
            result = m.review_report(data, run / 'val')
            cal = m.read(run / 'review_calibration.json')
            self.assertFalse(cal['target_met'])
            self.assertEqual(result['recall'], .5)
            m.write(data / 'annotations/test.coco.json', gt)
            m.write(run / 'test/predictions.json', pred)
            test = m.review_report(data, run / 'test', cal)
            self.assertEqual(test['threshold'], cal['threshold'])
            self.assertFalse((run / 'test/review_calibration.json').exists())


if __name__ == '__main__':
    unittest.main()
