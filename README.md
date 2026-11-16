<div align="center">

# TraGR

### From Tokens to Behaviors: Rethinking Generative Recommendation for User Trajectory Prediction

[![Paper](https://img.shields.io/badge/arXiv-2611.16159-b31b1b.svg)](https://arxiv.org/abs/2611.16159)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg)](#installation)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C.svg)](#installation)

**Behavior-level generation · Probability feedback · Relation-aware trajectory learning**

[Paper](https://arxiv.org/abs/2611.16159) · [Quick start](#quick-start) · [Datasets](#datasets) · [SID construction](#sid-construction) · [Citation](#citation)

</div>

TraGR predicts a sequence of future user interactions from observed history. It generates the SID tokens of each interaction in parallel, passes probability feedback to subsequent behavior states, and learns from model-generated trajectories with relation-aware rewards and continuation credit.

This repository includes the model, dataset configurations, prepared training inputs, and a CPU pipeline for constructing semantic IDs. Model checkpoints are trained from scratch. Internal Python modules retain the `ubt` package name.

## Installation

Use Linux, Python 3.10 or 3.11, and a CUDA-enabled PyTorch build compatible with your GPU driver. Install PyTorch for your machine, then:

```bash
git clone https://github.com/CalmaQi/GR4UBT.git ubtgit
cd ubtgit

git lfs install
git lfs pull
python -m pip install -r requirements.txt
```

Processed `.npz` files are stored with Git LFS. The prepared inputs include Sentence-T5 base text vectors, SID assignments, and chronological splits, so the default training workflow requires no feature-generation step or text-encoder download.

## Quick start

Train Stage I, run relation-aware Stage II, and report TEST metrics on all four datasets:

```bash
python scripts/run.py --dataset all --cuda-visible-devices 0
```

Select a single dataset:

```bash
python scripts/run.py --dataset beauty
python scripts/run.py --dataset cellphones
python scripts/run.py --dataset ml1m
python scripts/run.py --dataset lastfm
```

The GPU selector is optional; without it, PyTorch uses the first visible GPU.

### Training and evaluation

1. **Stage I:** Train the supervised behavior model using the dataset configuration. ML-1M includes dense next-item warmup.
2. **Stage II:** Run the four combinations of transition reward on/off and continuation credit on/off for every selected dataset.
3. **Selection:** Select each Stage-II arm's checkpoint by validation SNDCG@10, then evaluate it on TEST.

For `--dataset all`, the script completes Stage I on all datasets before starting Stage II. Outputs are saved under `runs/reproduction/`. Completed stages are reused when the command is repeated. The complete relation-aware arm is `transition_1_continuation_1`; the remaining arms are ablations.

Print the stored results of completed Stage-II runs:

```bash
python scripts/run.py --dataset all --evaluate-only
```

This command reads the saved metrics; it does not rerun model inference.

## Datasets

| Dataset | Public source | Horizon | Configuration | Prepared inputs |
|:--|:--|:--:|:--|:--|
| Beauty | Amazon Reviews 2014 | 3 | [beauty.json](configs/beauty.json) | [beauty/](data/processed/beauty/) |
| Cell Phones | Amazon Reviews 2014 | 3 | [cellphones.json](configs/cellphones.json) | [cellphones/](data/processed/cellphones/) |
| ML-1M | MovieLens 1M | 5 | [ml1m.json](configs/ml1m.json) | [ml1m/](data/processed/ml1m/) |
| Last-FM | Last.fm 1K | 5 | [lastfm.json](configs/lastfm.json) | [lastfm/](data/processed/lastfm/) |

Each feature asset contains `text` (`N × 768`, float32 Sentence-T5 vectors) and `codes` (`N × 16`, integer SID tokens). Row `r` corresponds to item ID `r + 1`; ID `0` is reserved for padding. Split assets and feature arrays must share this item ordering.

[data/ASSET_MANIFEST.json](data/ASSET_MANIFEST.json) records the included files and checksums. The default training commands use these packaged assets for reproducibility.

## SID construction

The SID pipeline fits **16 codebooks with 256 entries each** on static catalog features. It reads no interaction histories, training targets, or held-out labels.

```text
Catalog features in fixed item order
                │
                ▼
       Optimized product quantization
                │
                ▼
       Product quantization: 16 × 8 bits
                │
                ▼
       Dataset-specific collision handling
                │
                ▼
       features.npz + encoder + manifest.json
```

Install the optional CPU construction dependency:

```bash
python -m pip install -r requirements-sid.txt
```

### Build from the included text vectors

```bash
python scripts/build_sid.py --dataset beauty
python scripts/build_sid.py --dataset cellphones
python scripts/build_sid.py --dataset ml1m
python scripts/build_sid.py --dataset lastfm
```

These commands fit new encoders from the packaged Sentence-T5 vectors. Outputs go to `runs/sid/<dataset>/`, and training continues to use the packaged assets. A fresh fit can produce different SID assignments because encoder inputs, fitted parameters, and FAISS versions affect the result. The generated manifest reports `matches_packaged_codes` explicitly; use the packaged assets for the reported experiment settings.

The construction settings are in [configs/sid.json](configs/sid.json): seed `2024`, four CPU threads, OPQ/PQ iteration counts, normalization, and collision policy. Beauty uses separate OPQ and PQ stages and retains the original collision policy. The other datasets use the complete `OPQ16,IVF1,PQ16x8` encoder and resolve duplicate SIDs with the least-cost unused one-token alternative in item-row order.

### Use your own features or replay an encoder

Supply an ordered `.npy` matrix or an `.npz` file with a `text` array:

```bash
python scripts/build_sid.py --dataset beauty \
  --embeddings features.npy \
  --output runs/sid/custom_beauty
```

If quantization uses a separate feature matrix, supply it with `--sid-embeddings sid_features.npy`; it must follow the same item-row order. Both source matrices are fingerprinted in the output manifest.

Replay a previously saved encoder with the same ordered feature inputs:

```bash
python scripts/build_sid.py --dataset beauty \
  --encoder-dir runs/sid/beauty \
  --output runs/sid/beauty_replay
```

For custom inputs, repeat the original `--embeddings` and `--sid-embeddings` arguments. The replay command checks the quantization-input fingerprint before encoding. Output directories must be new, which keeps previous builds available for comparison.

<details>
<summary><strong>Generated files</strong></summary>

| File | Contents |
|:--|:--|
| `features.npz` | Input text vectors and constructed SID codes in the original item order |
| `encoder.faiss` | Full OPQ/IVF/PQ encoder for the combined recipe |
| `rotation.faiss`, `quantizer.faiss` | Separate OPQ and PQ parameters for the static recipe |
| `manifest.json` | Parameters, input/code fingerprints, library versions, collision changes, and comparison with packaged SID assignments |

</details>

## Repository layout

```text
configs/                 Dataset training settings and SID construction recipes
data/processed/          Prepared splits and feature assets (Git LFS)
scripts/run.py           Stage I → relation-aware Stage II → results
scripts/evaluate.py      Read completed Stage-II metrics
scripts/build_sid.py     Fit or replay SID encoders
ubt/                     Model, training, evaluation, and SID construction code
runs/                    Generated outputs; excluded from Git
```

## Citation

If you use this work, please cite:

```bibtex
@article{wang2026tragr,
  title   = {From Tokens to Behaviors: Rethinking Generative Recommendation for User Trajectory Prediction},
  author  = {Wang, Jiaqi and Yang, Hanchen and Zhang, Hengyu and Ma, Hongxu and
             Rao, Wenqi and Wang, Hongyang and Sun, Zhongbo and Qian, Renliang and
             Chen, Weixin and Li, Wengen and Guan, Jihong and Zhou, Shuigeng and
             Cai, Qingpeng and Jiang, Peng and Cheng, Hong},
  journal = {arXiv preprint arXiv:2611.16159},
  year    = {2026},
  doi     = {10.48550/arXiv.2611.16159},
  url     = {https://arxiv.org/abs/2611.16159}
}
```
