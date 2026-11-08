# UBT reproduction

This standalone package trains UBT on Beauty, Cell Phones, ML-1M, and Last-FM. It includes the prepared split and feature assets; model checkpoints are trained from scratch. The training entry point runs Stage I and relation-aware Stage II for every selected dataset.

## Setup

Use Linux, Python 3.10 or 3.11, and a CUDA-enabled PyTorch build that matches the machine's driver. Install PyTorch first, then clone the repository and install the remaining requirements:

```bash
git clone https://github.com/CalmaQi/GR4UBT.git ubtgit
cd ubtgit
git lfs install
git lfs pull
python -m pip install -r requirements.txt
```

The processed `.npz` files use Git LFS. Run `git lfs pull` after cloning if any of them are still LFS pointer files.

## Run

Train and evaluate all four datasets:

```bash
python scripts/run.py --dataset all
```

Or run one dataset:

```bash
python scripts/run.py --dataset beauty
python scripts/run.py --dataset cellphones
python scripts/run.py --dataset ml1m
python scripts/run.py --dataset lastfm
```

To choose a visible GPU, for example GPU 0:

```bash
python scripts/run.py --dataset all --cuda-visible-devices 0
```

The run order is Stage I followed by all four relation-aware Stage-II arms for each dataset. The arms cross the directed TRAIN transition reward on/off with continuation credit on/off. Each arm selects its checkpoint by validation SNDCG@10 and evaluates TEST once after selection. ML-1M includes its dense warmup. Training outputs go under `runs/reproduction/`.

To print metrics for completed Stage-II runs without training again:

```bash
python scripts/run.py --dataset all --evaluate-only
```

## Data and configs

Every dataset uses Sentence-T5 base text vectors (768 dimensions). The included feature assets and their hashes are listed in `data/ASSET_MANIFEST.json`; item rows are aligned to each packaged split. The manifest records the SID code source separately from the text encoder. No BGE text-vector asset is included.

| Dataset | Split and feature assets | Config |
|---|---|---|
| Beauty | `data/processed/beauty/{assets,sentence_t5_u16}.npz` | `configs/beauty.json` |
| Cell Phones | `data/processed/cellphones/{assets,sentence_t5_u16,tail50}.npz` | `configs/cellphones.json` |
| ML-1M | `data/processed/ml1m/{assets,sentence_t5_u16,tail50}.npz` | `configs/ml1m.json` |
| Last-FM | `data/processed/lastfm/{assets,sentence_t5_bge_sid,tail50}.npz` | `configs/lastfm.json` |

The public sources are Amazon Reviews 2014 Beauty, Amazon Reviews 2014 Cell Phones, MovieLens 1M, and Last.fm 1K. If replacing a processed asset, preserve the item-row order shared by the split, text vectors, and SID codes.

Dataset-specific Stage-I settings and the common Stage-II hyperparameters are in `configs/`. Core training, model, relation-reward, and evaluation code is in `ubt/`; `scripts/` contains the supported entry points.
