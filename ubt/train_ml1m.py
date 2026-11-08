"""Train and evaluate the selected ML-1M recipe without report side effects."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import stage1_core as core
from paths import DATA, ROOT, RUNS

OUT = RUNS / 'ml1m'
core.OUT = OUT
core.REFERENCE = DATA / 'ml1m/reference_val.json'


def load_data(cfg):
    if cfg.get('dataset') != 'ml1m' or cfg.get('K') != 5:
        raise ValueError('Expected the registered ML-1M K=5 configuration')
    with np.load(DATA / 'ml1m/assets.npz', allow_pickle=False) as f:
        data = dict(f)
    with np.load(DATA / 'ml1m/tail50.npz', allow_pickle=False) as f:
        data.update(dict(f))
    with np.load(DATA / f"ml1m/{cfg['asset']}.npz", allow_pickle=False) as f:
        data.update(text=f['text'], codes=f['codes'])
    data['dense_y'] = np.ones((len(data['dense_h']), 5), np.int32)
    return data


core.load_data = load_data


def seed():
    torch.set_num_threads(4)
    torch.manual_seed(2024)
    np.random.seed(2024)
    torch.backends.cuda.matmul.allow_tf32 = True


def evaluate():
    cfg = json.loads((ROOT / 'configs/ml1m.json').read_text())['full']
    data = load_data(cfg)
    checkpoint = OUT / 'full/t5_lastfm/selected.pt'
    state = torch.load(checkpoint, map_location='cuda', weights_only=False)
    actor = core.make_actor(cfg, data, state)
    metrics = core.evaluate(actor, data, 'test', 5, bs=128,
                            path=OUT / 'test_predictions.npz')
    core.save(OUT / 'test_metrics.json', {
        'status': 'complete', 'dataset': 'ml1m', 'K': 5,
        'config': cfg, 'metrics': metrics, 'TEST_used_for_selection': False,
    })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('warmup', 'full', 'evaluate'), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-capable GPU is required')
    seed()
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'configs/ml1m.json').read_text())
    if args.mode == 'warmup':
        core.warmup(config['warmup'])
    elif args.mode == 'full':
        cfg = dict(config['full'])
        cfg['reuse_dense'] = str(OUT / 'warmup/t5_lastfm/dense.pt')
        core.train(cfg, 'full')
    else:
        print(json.dumps(evaluate(), indent=2))


if __name__ == '__main__':
    main()
