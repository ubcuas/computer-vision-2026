"""Apply the size-table edit after an already running notebook has closed."""
import argparse
import csv
from pathlib import Path

import nbformat


def find_cell(notebook, prefix):
    return next(cell for cell in notebook.cells if cell.source.startswith(prefix))


def refresh(source, executed, comparison):
    template = nbformat.read(source, as_version=4)
    notebook = nbformat.read(executed, as_version=4)
    heading = '## 1080p camera, different model input sizes'
    find_cell(notebook, heading).source = find_cell(template, heading).source

    prefix = 'rows = summarize(output)'
    summary = find_cell(notebook, prefix)
    summary.source = find_cell(template, prefix).source
    if comparison.is_file():
        with comparison.open() as stream:
            rows = {row['model']: row for row in csv.DictReader(stream) if row['split'] == 'test'}
        for output in summary.outputs:
            markdown = output.get('data', {}).get('text/markdown', '')
            if not markdown.startswith('| Run | Input | Test AP50–95 |') or 'Checkpoint MiB' in markdown:
                continue
            lines = markdown.splitlines()
            lines[0] += ' Inference params (M) | Checkpoint MiB |'
            lines[1] += '---:|---:|'
            for i, line in enumerate(lines[2:], 2):
                key = line.split('|')[1].strip()
                row = rows[key]
                lines[i] += f" {float(row['parameters'])/1e6:.2f} | {int(row['checkpoint_bytes'])/2**20:.1f} |"
            output.data['text/markdown'] = '\n'.join(lines)

    pending = executed.with_name(executed.name + '.size-table.tmp')
    nbformat.write(notebook, pending)
    pending.replace(executed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'executed', 'comparison'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    refresh(args.source, args.executed, args.comparison)
