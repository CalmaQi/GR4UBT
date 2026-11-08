"""Train and evaluate the selected Last-FM recipe."""
import argparse
import json
from pathlib import Path
import torch
import stage1_core as core
from paths import ROOT, RUNS

OUT = RUNS / 'lastfm'
core.OUT = OUT

def seed():
    torch.set_num_threads(4)
    torch.manual_seed(2024)
    import numpy as np
    np.random.seed(2024)
    torch.backends.cuda.matmul.allow_tf32 = True

def train(config_path):
    cfg = json.loads(Path(config_path).read_text())
    core.train(cfg, 'full')

def evaluate():
    cfg = json.loads((ROOT / 'configs/lastfm.json').read_text())
    folder = OUT / 'full' / cfg['name']
    result = core.read(folder / 'result.json')
    state = torch.load(result['checkpoint'], map_location='cuda', weights_only=False)
    cfg = result['config']
    data = core.load_data(cfg)
    actor = core.make_actor(cfg, data, state)
    metrics = core.evaluate(actor, data, 'test', 5, bs=128, path=OUT / 'test_predictions.npz')
    core.save(OUT / 'test_metrics.json', {'status':'complete','dataset':'lastfm','K':5,
        'config':cfg,'metrics':metrics,'TEST_used_for_selection':False})
    return metrics

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('train','evaluate'), required=True)
    parser.add_argument('--config', default=str(ROOT/'configs/lastfm.json'))
    args=parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('A CUDA-capable GPU is required')
    seed(); OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='train': train(args.config)
    else: print(json.dumps(evaluate(),indent=2))

if __name__=='__main__': main()
