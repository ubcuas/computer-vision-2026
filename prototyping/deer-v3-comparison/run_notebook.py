"""Execute the reviewed notebook and save its outputs after each cell."""
from pathlib import Path
import os

import nbformat
from nbclient import NotebookClient

folder = Path(__file__).resolve().parent
workspace = Path(os.environ.get('DEER_COMPARE_WORKSPACE', Path.home() / 'deer_v3_compare'))
suffix = 'executed' if any(os.environ.get(flag) == '1' for flag in ('DEER_COMPARE_RUN', 'DEER_COMPARE_REPORT')) else 'preview'
output = workspace / f'Deer_v3_Comparison.{suffix}.ipynb'
nb = nbformat.read(folder / 'Deer_v3_Comparison.ipynb', as_version=4)


def save(**_):
    pending = output.with_suffix('.tmp')
    nbformat.write(nb, pending)
    pending.replace(output)


try:
    NotebookClient(nb, timeout=None, kernel_name='gfl-server',
                   resources={'metadata': {'path': str(folder)}},
                   on_cell_executed=save).execute()
finally:
    save()
