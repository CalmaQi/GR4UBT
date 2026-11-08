"""Train Stage I, relation-aware Stage II, and evaluate selected checkpoints."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
RUN_ROOT = Path(os.environ.get('UBT_RUNS', ROOT / 'runs' / 'reproduction'))


def config(name):
    return json.loads((ROOT / 'configs' / name).read_text())


def run(*args):
    command = [PYTHON, '-u', *map(str, args)]
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


def beauty():
    cfg = config('beauty.json')
    if not (RUN_ROOT / 'beauty/stage1_complete.json').exists():
        s = cfg['stage1']
        run('ubt/runner.py', '--dataset', 'beauty', '--k', s['K'],
            '--mode', 'base', '--variant', s['variant'],
            '--dim', s['dim'], '--batch', s['batch'])


def cellphones():
    cfg = config('cellphones.json')
    result = RUN_ROOT / 'cellphones/full' / cfg['name'] / 'result.json'
    if not result.exists():
        run('ubt/train_stage1.py', '--config', 'configs/cellphones.json', '--phase', 'full')


def ml1m():
    cfg = config('ml1m.json')
    root = RUN_ROOT / 'ml1m'
    warm = root / 'warmup' / cfg['warmup']['name'] / 'result.json'
    full = root / 'full' / cfg['full']['name'] / 'result.json'
    if not warm.exists():
        run('ubt/train_ml1m.py', '--mode', 'warmup')
    if not full.exists():
        run('ubt/train_ml1m.py', '--mode', 'full')


def lastfm():
    cfg = config('lastfm.json')
    result = RUN_ROOT / 'lastfm/full' / cfg['name'] / 'result.json'
    if not result.exists():
        run('ubt/train_lastfm.py', '--mode', 'train', '--config', 'configs/lastfm.json')


TRAINERS = {'beauty': beauty, 'cellphones': cellphones, 'ml1m': ml1m, 'lastfm': lastfm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=(*TRAINERS, 'all'), default='all')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--cuda-visible-devices', help='optional local GPU selector')
    args = parser.parse_args()
    os.environ.setdefault('UBT_RUNS', str(RUN_ROOT))
    if args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    selected = TRAINERS if args.dataset == 'all' else {args.dataset: TRAINERS[args.dataset]}
    if not args.evaluate_only:
        for train_stage1 in selected.values():
            train_stage1()
        run('ubt/train_relation_rl.py', '--dataset', args.dataset)
    command = [PYTHON, '-u', 'scripts/evaluate.py', '--dataset', args.dataset]
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


if __name__ == '__main__':
    main()
