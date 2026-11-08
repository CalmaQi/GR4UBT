# UBT reproduction

This is a standalone copy of the code and inputs needed to retrain and evaluate the selected UBT recipe for Beauty, Cell Phones, ML-1M, and Last-FM. The original repository files are unchanged. The package contains prepared feature and split assets, but no model checkpoints; training starts from scratch.

## Setup

Use Linux, Python 3.10 or 3.11, and a CUDA-enabled PyTorch build that matches the machine's CUDA driver. Install PyTorch first, then the remaining packages:

```bash
git clone https://github.com/CalmaQi/GR4UBT.git ubtgit
cd ubtgit
git lfs install
git lfs pull
```

The processed `.npz` inputs are stored with Git LFS. Run `git lfs pull` after cloning if they are still LFS pointer files. Install a CUDA-enabled PyTorch build that matches the machine's driver, then install the remaining packages:

```bash
python -m pip install -r requirements.txt
```

If the repository is already cloned, enter it before pulling the assets:

```bash
cd ubtgit
git lfs pull
```

By default, PyTorch uses its first visible GPU. To choose a local GPU explicitly:

```bash
python scripts/run.py --dataset all --cuda-visible-devices 0
```

The GPU selector is optional and should match the new machine.

## Run

Retrain and evaluate all four datasets:

```bash
python scripts/run.py --dataset all
```

Or select one dataset:

```bash
python scripts/run.py --dataset beauty
python scripts/run.py --dataset cellphones
python scripts/run.py --dataset ml1m
python scripts/run.py --dataset lastfm
```

To evaluate checkpoints that are already present under `runs/`:

```bash
python scripts/run.py --dataset all --evaluate-only
```

The script runs datasets sequentially. ML-1M includes its dense NTP warmup. Checkpoints, logs, progress, and final metrics are written under `runs/`; no output is written to the source repository outside `ubtgit/`.

## Data and configs

The processed `.npz` files are included so a new machine does not need the original absolute paths, local model caches, or a separate feature-generation step. `data/ASSET_MANIFEST.json` records each input file's size and SHA-256. Keep each dataset's base asset, item features, codes, and tail windows together because their item-row mapping is shared.

| Dataset | Included inputs | Best recipe config |
|---|---|---|
| Beauty | `data/processed/beauty/assets.npz` | `configs/beauty.json` |
| Cell Phones | `data/processed/cellphones/{assets,bge_u16,tail50}.npz` | `configs/cellphones.json` |
| ML-1M | `data/processed/ml1m/{assets,t5_u16,tail50}.npz` | `configs/ml1m.json` |
| Last-FM | `data/processed/lastfm/{assets,t5_text_bge_code,tail50}.npz` | `configs/lastfm.json` |

The underlying public sources are Amazon Reviews 2014 Beauty, Amazon Reviews 2014 Cell Phones, MovieLens 1M, and Last.fm 1K. Replacing the packaged `.npz` inputs requires rebuilding the same item mapping, split, text features, and SID codes as a matched set.

## Selected recipes

- **Beauty, K=3:** Full Stage I followed by `grpo_hit_s025`; GRPO group 16, 3 epochs, batch 32, 2 updates, learning rate `1e-6`, reward scale `0.25`. Reward scaling uses TRAIN-only probes.
- **Cell Phones, K=3:** `combined_no_extra_users_tails`; BGE item features and codes, one layer, history 50, tail windows, no extra users, no alignment.
- **ML-1M, K=5:** `t5_lastfm`; dense warmup then joint trajectory training, T5 codes, history 50, eta `0.99`.
- **Last-FM, K=5:** `bge_tails_direct_lr001`; T5 text features with BGE codes, direct joint training at learning rate `1e-3`, live input feedback, no separate warmup.

Checkpoint selection uses validation metrics. The final selected checkpoints are evaluated on TEST by `scripts/evaluate.py`. The entry scripts are in `scripts/`, the required model components are in `ubt/`, and per-dataset settings are in `configs/`.
