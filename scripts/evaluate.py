"""Print the VAL-selected TEST metrics for the relation-aware Stage-II arms."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('UBT_RUNS', str(ROOT / 'runs' / 'reproduction'))
sys.path.insert(0, str(ROOT / 'ubt'))

from train_relation_rl import DATASETS, report_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=(*DATASETS, 'all'), default='all')
    args = parser.parse_args()
    selected = DATASETS if args.dataset == 'all' else (args.dataset,)
    for dataset in selected:
        print(f'[{dataset}]', flush=True)
        print(json.dumps(report_dataset(dataset), indent=2), flush=True)


if __name__ == '__main__':
    main()
