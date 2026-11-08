"""Train and evaluate the selected best UBT configuration(s)."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def config(name):
    return json.loads((ROOT / 'configs' / name).read_text())


def run(*args):
    command = [PYTHON, '-u', *map(str, args)]
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def beauty():
    cfg = config('beauty.json')
    stage = ROOT / 'runs/beauty/stage1_complete.json'
    rl = ROOT / 'runs/beauty_rl/beauty' / cfg['rl']['tag'] / 'result.json'
    if not stage.exists():
        run('ubt/runner.py', '--dataset', 'beauty', '--k', cfg['stage1']['K'],
            '--mode', 'base', '--variant', cfg['stage1']['variant'],
            '--dim', cfg['stage1']['dim'], '--batch', cfg['stage1']['batch'])
    if not (ROOT / 'runs/beauty_rl/beauty/diagnostic.json').exists():
        run('ubt/diagnose_rl.py', '--dataset', 'beauty')
    if not rl.exists():
        r = cfg['rl']
        run('ubt/train_rl.py', '--dataset', 'beauty', '--arm', r['arm'],
            '--epochs', r['epochs'], '--batch', r['batch'], '--updates', r['updates'],
            '--lr', r['learning_rate'], '--rl-scale', r['rl_scale'],
            '--tag', r['tag'], '--group', r['group'])


def cellphones():
    cfg = config('cellphones.json')
    result = ROOT / 'runs/cellphones/full' / cfg['name'] / 'result.json'
    if not result.exists():
        run('ubt/train_stage1.py', '--config', 'configs/cellphones.json', '--phase', 'full')


def ml1m():
    root = ROOT / 'runs/ml1m'
    warm = root / 'warmup/t5_lastfm/result.json'
    full = root / 'full/t5_lastfm/result.json'
    if not warm.exists():
        run('ubt/train_ml1m.py', '--mode', 'warmup')
    if not full.exists():
        run('ubt/train_ml1m.py', '--mode', 'full')


def lastfm():
    cfg = config('lastfm.json')
    result = ROOT / 'runs/lastfm/full' / cfg['name'] / 'result.json'
    if not result.exists():
        run('ubt/train_lastfm.py', '--mode', 'train', '--config', 'configs/lastfm.json')


TRAINERS = {'beauty': beauty, 'cellphones': cellphones, 'ml1m': ml1m, 'lastfm': lastfm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=(*TRAINERS, 'all'), default='all')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--cuda-visible-devices', help='optional local GPU selector')
    args = parser.parse_args()
    if args.cuda_visible_devices is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    selected = TRAINERS if args.dataset == 'all' else {args.dataset: TRAINERS[args.dataset]}
    if not args.evaluate_only:
        for train in selected.values():
            train()
    command = [PYTHON, '-u', 'scripts/evaluate.py', '--dataset', args.dataset]
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


if __name__ == '__main__':
    main()
