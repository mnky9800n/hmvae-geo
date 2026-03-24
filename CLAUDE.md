# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LilyVAE is a variational autoencoder for reconstructing boreholes in the LILY database (an IODP ocean drilling core dataset). The goal is to learn embeddings from a subset of variables (RGB values, bulk density, magnetic susceptibility, NGR) that enable zero-shot prediction of all other variables in the dataset.

**Key references:**
- Paper: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GC011287
- Data: https://zenodo.org/records/8408297 and https://zenodo.org/records/10001855

## Commands

Uses `uv` for project management. Full reproducible pipeline:

```bash
uv sync                      # install dependencies
uv run lilyvae-download      # download LILY data from Zenodo (~10.8 GB)
uv run lilyvae-train         # train both unsupervised + semi-supervised VAEs
uv run lilyvae-evaluate      # run all evaluation metrics
uv run lilyvae-zeroshot      # bootstrap zero-shot evaluation
uv run lilyvae-zeroshot --monte-carlo  # Monte Carlo exploration
uv run lilyvae-plot          # generate all paper figures
```

Train a single model variant: `uv run lilyvae-train --model unsupervised`
Generate a single figure: `uv run lilyvae-plot --figure umap_lithology`

## Architecture

`src/lilyvae/` is a Python package with these modules:

- **config.py** — all hyperparameters, paths, and constants in one place
- **data/** — download from Zenodo (`download.py`), CSV preprocessing + 20cm binning (`preprocessing.py`), PyTorch Dataset (`dataset.py`)
- **model/** — VAE with heteroscedastic Gaussian decoder (`vae.py`), lithology classification head (`heads.py`)
- **training/** — loss functions with beta/alpha annealing (`loss.py`), learnable mask tokens (`masking.py`), training loop (`trainer.py`)
- **evaluation/** — reconstruction R² (`reconstruction.py`), GMM clustering ARI (`clustering.py`), SVM classification AUC (`classification.py`), CatBoost zero-shot prediction (`zeroshot.py`), bootstrap CIs (`bootstrap.py`), Monte Carlo search (`monte_carlo.py`)
- **plotting/** — all paper figure generation, PNGs saved to the paper directory
- **scripts/** — CLI entry points for each pipeline stage

## Key Design Decisions

- **Heteroscedastic decoder**: Decoder outputs per-variable mean AND log-variance (not fixed MSE). This prevents noisy variables from dominating the latent space.
- **Learnable mask tokens**: Masked features are replaced with learned parameters (not zeros), so the model distinguishes "missing" from "average".
- **No train/test split**: Trained on full dataset; validated via downstream zero-shot transfer with bootstrap CIs over borehole splits.
- **Evaluation priority**: Zero-shot R² is the primary metric, not reconstruction.

## Data

Raw data lives in `data/raw/` (gitignored). Preprocessed tensors in `data/processed/`. Checkpoints in `data/checkpoints/`. Results in `data/results/`.

The paper LaTeX and figures are in `an-unsupervised-embeddings-model-for-ocean-crust-lithology/`.

## Custom Agents

- `.claude/agents/geology-expert.md` — domain expert on ocean crust lithology, IODP data, petrophysical variables
- `.claude/agents/vae-expert.md` — expert on VAE architecture, representation learning, training strategies
