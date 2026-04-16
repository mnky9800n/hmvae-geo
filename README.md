# LilyVAE

Heteroscedastic masked variational autoencoder for the LILY ocean drilling
database. Trained on six routinely-collected physical-property measurements
(bulk density, magnetic susceptibility, NGR, RGB reflectance) from 238,767
observations across 296 IODP boreholes, the model produces 10-dimensional
embeddings that (a) reconstruct the input variables, (b) cluster along
geologist-assigned lithology, and (c) enable zero-shot prediction of
dozens of other geophysical variables never seen during training.

Our central finding: for this sediment-dominated dataset, the continuous
measurements contain enough structure that supervision with human-assigned
lithology labels adds no value and, at high weight, actively degrades
reconstruction.

## Getting started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management
- ~15 GB free disk (10.8 GB raw data + processed tensors + checkpoints)
- GPU recommended (CUDA or Apple MPS). CPU-only works but the full pipeline
  takes over a day.

### Setup

```bash
git clone https://github.com/mnky9800n/hmvae-geo.git
cd hmvae-geo
uv sync                         # install dependencies
uv run lilyvae-download         # fetch LILY data from Zenodo (~10.8 GB)
```

### Reproduce every paper number and figure

```bash
uv run lilyvae-pipeline
```

This is idempotent: each step is skipped if its output already exists.
Use `--force` to re-run from scratch.

### Quick sanity check without the full pipeline

```bash
uv run pytest tests/ -m "not slow"                 # unit tests, no data needed
uv run lilyvae-bootstrap --iterations 2            # ~5 min per fold instead of 30 folds
```

### Working on the sparse autoencoder (SAE)

The SAE code lives at `src/lilyvae/evaluation/sae.py`. It trains on the
10-dimensional VAE latent space and exposes:

- `SparseAutoencoder` -- small nn.Module with configurable hidden width
- `train_sae(z, cfg)` -- train on a precomputed tensor of VAE latents
- `score_features_against_targets(...)` -- per-feature scoring against four
  geological hypotheses (depth, carbonate-vs-clay, volcanic, organic-rich)
- `steer(...)` -- amplify a single SAE feature and decode back through the
  VAE to see how reconstructions change

The SAE is not wired into `lilyvae-pipeline`; it's a standalone exploration
tool. See the module docstring for details on its limitations (the 10-dim
VAE latent is much smaller than the 768+ dim residual streams SAEs are
typically applied to, see https://arxiv.org/abs/2512.24440).

Checkpoints and preprocessed tensors the SAE needs:

- `data/checkpoints/unsupervised_embeddings.pt` (10-dim latents)
- `data/processed/train_labels.pt`, `label_encoder.pkl`, `metadata.pt`

Minimal example:

```python
import torch
from lilyvae.config import CHECKPOINT_DIR
from lilyvae.evaluation.sae import train_sae, SAETrainConfig

z = torch.load(CHECKPOINT_DIR / "unsupervised_embeddings.pt", weights_only=True)
sae, diagnostics = train_sae(z, SAETrainConfig(d_hidden=64, num_epochs=200))
```

## What gets evaluated

The paper's main results table has three row groups:

**Baselines** (self-supervised, no labels), evaluated single-point on the
full 238k-sample dataset:
- PCA on the 6 input features
- TabNet pretraining (Arik & Pfister 2021) -- masked-feature prediction
- SCARF (Bahri et al. 2022) -- contrastive with feature corruption

**Ablations** (VAE with one component changed), also evaluated single-point:
- Fixed-variance (MSE) decoder instead of heteroscedastic
- Zero masking instead of learnable mask tokens
- No masking at all
- Fixed β instead of annealing schedule

**Full models**, evaluated with a 30-fold borehole-level bootstrap to
produce training-stability CIs:
- Unsupervised VAE (ours)
- Semi-supervised VAE with lithology classifier head

Every model is scored on reconstruction R², clustering ARI (against true
lithology), and classification AUC (one-vs-rest SVM on embeddings).
Zero-shot R² (CatBoost regressing ~65 unseen geophysical variables from
the embeddings) is reported on the full-data models only.

## Repository layout

```
src/lilyvae/
├── config.py                    # All hyperparameters, paths, constants
├── data/                        # Zenodo download, CSV preprocessing, PyTorch dataset
├── model/                       # VAE (heteroscedastic), classification head
├── training/                    # Loss functions, learnable masking, training loop
├── evaluation/
│   ├── bootstrap.py             # Bootstrap evaluator for the two VAE variants
│   ├── single_point.py          # Full-data evaluator for baselines + ablations
│   ├── baselines.py             # PCA, TabNet, SCARF
│   ├── ablations.py             # FixedVarianceVAE, ZeroMasking, NoMasking
│   ├── reconstruction.py        # Per-variable R²
│   ├── clustering.py            # GMM ARI
│   ├── classification.py        # SVM macro AUC
│   ├── zeroshot.py              # CatBoost zero-shot prediction
│   └── sae.py                   # Sparse autoencoder on latent space (future work)
├── plotting/                    # Paper figure generators
└── scripts/                     # CLI entry points
    ├── download.py              # lilyvae-download
    ├── pipeline.py              # lilyvae-pipeline (the main one)
    ├── train.py                 # lilyvae-train
    ├── bootstrap.py             # lilyvae-bootstrap
    ├── single_point.py          # lilyvae-single-point
    └── plot.py                  # lilyvae-plot

data/                            # All generated artifacts (gitignored)
├── raw/                         # Downloaded LILY CSVs
├── processed/                   # train_features.pt, train_labels.pt, metadata.pt, zeroshot_targets.pt
├── checkpoints/                 # {unsupervised,semi_supervised}_vae.pt, *_embeddings.pt
└── results/
    ├── bootstrap_results.json   # Canonical VAE numbers
    ├── single_point_results.json # Baseline + ablation numbers
    └── zeroshot_fulldata.json   # Per-variable zero-shot R²
```

## CLI reference

| Command | Purpose |
|---|---|
| `lilyvae-download` | Download the LILY database from Zenodo (~10.8 GB) |
| `lilyvae-pipeline` | Run everything end-to-end (the main command) |
| `lilyvae-train` | Train just the full-data VAE checkpoints |
| `lilyvae-bootstrap` | Run just the bootstrap evaluation of the two VAE variants |
| `lilyvae-single-point` | Baselines + ablations on full data |
| `lilyvae-plot` | Regenerate paper figures |

## Runtime

On an AMD EPYC 7995 workstation with 64 cores, 256 GB RAM, and 2× NVIDIA
RTX 6000 Ada, the full pipeline (download excluded) takes roughly X.X
hours. On a modern laptop CPU the bootstrap evaluation alone takes over 24
hours and is not recommended; use `lilyvae-bootstrap --iterations 2` for a
quick sanity check during development.

## Data

Raw data is downloaded from Zenodo via `lilyvae-download`:

- LILY database (main): https://zenodo.org/records/8408297
- LILY density (supplement): https://zenodo.org/records/10001855

The preprocessing step bins observations into 20 cm depth intervals to
co-locate measurements from different instruments on the same core
section, then retains only rows where all six training variables are
present.

## Paper

Paper in preparation.
