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

## Quick start

```bash
# One-time setup
uv sync                         # install dependencies
uv run lilyvae-download         # fetch LILY data from Zenodo (~10.8 GB)

# Reproduce every paper number and figure
uv run lilyvae-pipeline
```

`lilyvae-pipeline` runs the entire workflow end-to-end:

1. Preprocess raw CSVs into PyTorch tensors
2. Train unsupervised and semi-supervised VAE checkpoints on the full dataset
3. Run the canonical bootstrap evaluation: 30 borehole-level splits,
   every model (11 total) retrained on each split, per-fold metrics
   aggregated into `data/results/bootstrap_results.json`
4. Generate paper figures

Each step is idempotent and will be skipped if its expected output already
exists. Use `--force` to re-run from scratch.

## What gets evaluated

Every bootstrap iteration holds out 10% of boreholes and trains all 11 of
these models on the remaining 90%. Within a fold, all models see the exact
same train/test split so the comparison is apples-to-apples.

**Baselines (self-supervised / unsupervised, no labels):**
- PCA on the 6 input features
- Linear autoencoder (no KL term, no nonlinearities)
- Deterministic autoencoder (same architecture, no variational objective)
- TabNet pretraining (Arik & Pfister 2021) — masked-feature prediction
- SCARF (Bahri et al. 2022) — contrastive with feature corruption

**Ablations (VAE with one component changed):**
- Fixed-variance (MSE) decoder instead of heteroscedastic
- Zero masking instead of learnable mask tokens
- No masking at all
- Fixed β instead of annealing schedule

**Full models:**
- Unsupervised VAE (ours)
- Semi-supervised VAE with lithology classifier head

Each model is scored on reconstruction R², clustering ARI (against true
lithology), classification AUC (one-vs-rest SVM on embeddings), and
zero-shot R² (CatBoost regressing unseen geophysical variables from the
embeddings, trained and evaluated within the same held-out fold).

## Repository layout

```
src/lilyvae/
├── config.py                    # All hyperparameters, paths, constants
├── data/                        # Zenodo download, CSV preprocessing, PyTorch dataset
├── model/                       # VAE (heteroscedastic), classification head
├── training/                    # Loss functions, learnable masking, training loop
├── evaluation/
│   ├── bootstrap.py             # Canonical bootstrap runner (THE source of truth)
│   ├── baselines.py             # PCA, linear/det AE, TabNet, SCARF
│   ├── ablations.py             # Fixed-variance decoder, alt masking strategies
│   ├── reconstruction.py        # Per-variable R²
│   ├── clustering.py            # GMM ARI
│   ├── classification.py        # SVM macro AUC
│   ├── zeroshot.py              # CatBoost zero-shot prediction
│   └── bootstrap_zeroshot.py    # Legacy bootstrap zero-shot (kept for quick eval)
├── plotting/                    # Paper figure generators
└── scripts/                     # CLI entry points
    ├── download.py              # lilyvae-download
    ├── pipeline.py              # lilyvae-pipeline (the main one)
    ├── train.py                 # lilyvae-train
    ├── bootstrap.py             # lilyvae-bootstrap
    ├── evaluate.py              # lilyvae-evaluate (quick debug, single run)
    ├── zeroshot.py              # lilyvae-zeroshot (Monte Carlo exploration)
    └── plot.py                  # lilyvae-plot

data/                            # All generated artifacts (gitignored)
├── raw/                         # Downloaded LILY CSVs
├── processed/                   # train_features.pt, train_labels.pt, metadata.pt, zeroshot_targets.pt
├── checkpoints/                 # {unsupervised,semi_supervised}_vae.pt, *_embeddings.pt
└── results/
    └── bootstrap_results.json   # Canonical paper numbers
```

## CLI reference

| Command | Purpose |
|---|---|
| `lilyvae-download` | Download the LILY database from Zenodo (~10.8 GB) |
| `lilyvae-pipeline` | Run everything end-to-end (the main command) |
| `lilyvae-train` | Train just the full-data VAE checkpoints |
| `lilyvae-bootstrap` | Run just the bootstrap evaluation |
| `lilyvae-evaluate` | Quick debug: evaluate a checkpoint on full data (NOT what paper reports) |
| `lilyvae-zeroshot` | Zero-shot with Monte Carlo hyperparameter search (exploration) |
| `lilyvae-plot` | Regenerate paper figures |

### `lilyvae-evaluate` vs `lilyvae-bootstrap`

`lilyvae-evaluate` runs a single-shot evaluation of the saved checkpoints on
the full dataset. It's useful for quick debugging but its numbers are
**not** what the paper reports.

`lilyvae-bootstrap` runs the canonical 30-fold borehole-level bootstrap
used in the paper. Every number in the main results table comes from
`data/results/bootstrap_results.json`.

## Runtime

On an AMD EPYC 7995 workstation with 64 cores, 256 GB RAM, and 2× NVIDIA
RTX 6000 Ada, the full pipeline (download excluded) takes roughly X.X
hours. On a modern laptop CPU the bootstrap evaluation alone takes over 24
hours and is not recommended; use the `--no-zeroshot` flag or a smaller
`--iterations` value for development.

## Data

Raw data is downloaded from Zenodo via `lilyvae-download`:

- LILY database (main): https://zenodo.org/records/8408297
- LILY density (supplement): https://zenodo.org/records/10001855

The preprocessing step bins observations into 20 cm depth intervals to
co-locate measurements from different instruments on the same core
section, then retains only rows where all six training variables are
present.

## Paper

*Heteroscedastic Masked Variational Autoencoders for Sparse Geophysical
Data*, NeurIPS 2026 AI for Science track (submission in preparation).

## Citation

```bibtex
@inproceedings{aiken2026lilyvae,
  title={Heteroscedastic Masked Variational Autoencoders for Sparse Geophysical Data},
  author={Aiken, John M. and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```
