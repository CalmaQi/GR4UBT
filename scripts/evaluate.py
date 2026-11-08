"""Evaluate the selected checkpoint for one or all datasets."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ubt'))


def beauty():
    import runner as model
    from paths import RUNS

    folder = RUNS / 'beauty_rl/beauty/grpo_hit_s025'
    data = dict(np.load(ROOT / 'data/processed/beauty/assets.npz', allow_pickle=False))
    actor, _ = model.load_actor(data, 3, folder / 'selected.pt', 448)
    metrics = model.evaluate(actor, data, 'test', 3, path=folder / 'test_predictions.npz')
    model.save(folder / 'test_metrics.json', {
        'status': 'complete', 'dataset': 'beauty', 'K': 3,
        'recipe': 'grpo_hit_s025', 'metrics': metrics, 'TEST_used_for_selection': False,
    })
    return metrics


def cellphones():
    import train_stage1 as model
    from paths import RUNS, save

    name = 'combined_no_extra_users_tails'
    folder = RUNS / 'cellphones/full' / name
    actor, data, state = model.load_tuned_checkpoint(folder / 'stage1.pt')
    cfg = state['config']
    metrics = model.evaluate(actor, data, 'test', cfg['K'], bs=cfg['eval_batch'],
                             path=folder / 'test_predictions.npz')
    save(folder / 'test_metrics.json', {
        'status': 'complete', 'dataset': 'cellphones', 'K': cfg['K'],
        'config': cfg, 'metrics': metrics, 'TEST_used_for_selection': False,
    })
    return metrics


def ml1m():
    import train_ml1m as model
    return model.evaluate()


def lastfm():
    import train_lastfm as model
    return model.evaluate()


EVALUATORS = {'beauty': beauty, 'cellphones': cellphones, 'ml1m': ml1m, 'lastfm': lastfm}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=(*EVALUATORS, 'all'), default='all')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-capable GPU is required')
    selected = EVALUATORS if args.dataset == 'all' else {args.dataset: EVALUATORS[args.dataset]}
    for dataset, evaluate in selected.items():
        print(f'[{dataset}]', flush=True)
        print(json.dumps(evaluate(), indent=2), flush=True)


if __name__ == '__main__':
    main()
