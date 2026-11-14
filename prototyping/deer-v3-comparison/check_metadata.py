"""Check the committed v3 inventory and notebook without data, weights or GPUs."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'dataloaders/deer_v3'


def read_split(name):
    return (DATA / 'splits' / f'{name}.txt').read_text().splitlines()


def check_inventory():
    checksum, filename = (DATA / 'CHECKSUMS.sha256').read_text().split()
    assert filename == 'checksums.json'
    checksums = (DATA / filename).read_bytes()
    assert hashlib.sha256(checksums).hexdigest() == checksum
    files = json.loads(checksums)
    expected_sizes = {'train': 3193, 'val': 358, 'test': 238}
    seen_hashes, seen_groups = {}, {}
    expected_paths = {split: set() for split in expected_sizes}
    decoys = set()
    with (DATA / 'frozen_manifest.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            split = row['split']
            path = f"images/{split}/{row['name']}"
            label = f"labels/{split}/{Path(row['name']).stem}.txt"
            assert path not in expected_paths[split], path
            assert files[path] == row['sha256'] and label in files, path
            assert seen_hashes.setdefault(row['sha256'], split) == split, path
            group = row['dataset'], row['group']
            assert seen_groups.setdefault(group, split) == split, group
            expected_paths[split].add(path)
            if split == 'train' and row['dataset'] == 'mississippi_decoy':
                decoys.add(path)
    for split, count in expected_sizes.items():
        paths = read_split(split)
        assert len(paths) == count and set(paths) == expected_paths[split], split
    assert len(decoys) == 139
    weighted = Counter({path: 4 if path in decoys else 1 for path in expected_paths['train']})
    assert Counter(read_split('train_decoy_4x')) == weighted


def check_notebook():
    notebook_path = Path(__file__).with_name('Deer_v3_Comparison.ipynb')
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells, 1):
        if cell.cell_type == 'code':
            compile(cell.source, f'{notebook_path}:cell{index}', 'exec')


if __name__ == '__main__':
    check_inventory()
    check_notebook()
    print('Frozen v3 inventory, decoy weighting and notebook structure verified.')
