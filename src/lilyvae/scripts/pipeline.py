"""End-to-end reproducibility pipeline.

One command that runs every step needed to reproduce the paper's results:

1. Preprocess the LILY database (skipped if processed tensors already exist).
2. Train full-data VAE checkpoints (skipped if already on disk). These are
   "the" deployable models and provide embeddings for the zero-shot eval.
3. Bootstrap ONLY the two full VAE variants across 30 borehole splits to
   get training-stability confidence intervals for reconstruction R^2,
   ARI, AUC. Writes data/results/bootstrap_results.json. Baselines and
   ablations are NOT bootstrapped -- see step 4.
4. Single-point full-data evaluation of the 5 baselines (PCA, linear AE,
   deterministic AE, TabNet, SCARF) and 4 ablation variants (fixed-variance
   decoder, zero masking, no masking, fixed beta), plus their zero-shot R^2.
   Writes data/results/single_point_results.json.
5. Zero-shot evaluation on the full-data VAE embeddings via CatBoost.
   Writes data/results/zeroshot_fulldata.json.
6. Generate paper figures.

Each step is idempotent: if its expected output already exists, it is
skipped with a message. Use `--force` to re-run everything.

Usage
-----
    uv run lilyvae-download              # first, to fetch raw data
    uv run lilyvae-pipeline              # then, to produce results
    uv run lilyvae-pipeline --force      # or, to re-run everything
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from lilyvae.config import (
    CHECKPOINT_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
)


def _exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _make_serializable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    return obj


def _step_preprocess(force: bool) -> None:
    features_path = PROCESSED_DIR / "train_features.pt"
    if _exists(features_path) and not force:
        print(f"[skip] preprocessed tensors already present at {PROCESSED_DIR}")
        return
    print(f"[run]  preprocessing raw LILY data -> {PROCESSED_DIR}")
    from lilyvae.data.preprocessing import preprocess_and_save
    preprocess_and_save()


def _step_train(force: bool, device: str) -> None:
    unsup_ckpt = CHECKPOINT_DIR / "unsupervised_vae.pt"
    semi_ckpt = CHECKPOINT_DIR / "semi_supervised_vae.pt"
    if _exists(unsup_ckpt) and _exists(semi_ckpt) and not force:
        print(f"[skip] VAE checkpoints already present at {CHECKPOINT_DIR}")
        return

    from lilyvae.data.dataset import get_dataloader
    from lilyvae.training.trainer import (
        train_model, save_checkpoint, extract_embeddings,
    )

    for model_type in ("unsupervised", "semi-supervised"):
        print(f"\n[run]  training full-data {model_type} VAE")
        dataloader = get_dataloader()
        result = train_model(dataloader=dataloader, model_type=model_type, device=device)
        save_checkpoint(result, model_type)

        eval_loader = get_dataloader(shuffle=False)
        embeddings = extract_embeddings(result["vae"], eval_loader, device)
        prefix = model_type.replace("-", "_")
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, CHECKPOINT_DIR / f"{prefix}_embeddings.pt")


def _step_bootstrap(
    force: bool,
    device: str,
    iterations: int,
) -> None:
    out_path = RESULTS_DIR / "bootstrap_results.json"
    if _exists(out_path) and not force:
        print(f"[skip] bootstrap_results.json already at {out_path}")
        return

    from lilyvae.evaluation.bootstrap import run_bootstrap

    print(f"\n[run]  canonical bootstrap evaluation ({iterations} iterations)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_bootstrap(
        n_iterations=iterations,
        device=device,
    )
    with open(out_path, "w") as f:
        json.dump(_make_serializable(result), f, indent=2)
    print(f"[done] bootstrap written to {out_path}")


def _step_single_point(force: bool, device: str) -> None:
    out_path = RESULTS_DIR / "single_point_results.json"
    if _exists(out_path) and not force:
        print(f"[skip] single_point_results.json already at {out_path}")
        return

    from lilyvae.evaluation.single_point import run_single_point_eval

    print(f"\n[run]  single-point full-data evaluation (baselines + ablations)")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_single_point_eval(device=device, include_zeroshot=True)
    with open(out_path, "w") as f:
        json.dump(_make_serializable(result), f, indent=2)
    print(f"[done] single-point written to {out_path}")


def _step_zeroshot(force: bool) -> None:
    """Zero-shot evaluation on full-data VAE embeddings.

    Reuses the embeddings extracted in _step_train. For each zero-shot
    target variable, trains a CatBoost regressor on the embeddings of
    samples where that variable is observed and reports R^2. Matches the
    paper's original zero-shot methodology (in-sample CatBoost on the
    full 238k-sample dataset).
    """
    out_path = RESULTS_DIR / "zeroshot_fulldata.json"
    if _exists(out_path) and not force:
        print(f"[skip] zeroshot_fulldata.json already at {out_path}")
        return

    from lilyvae.evaluation.zeroshot import evaluate_all_zeroshot

    print(f"\n[run]  zero-shot evaluation on full-data VAE embeddings")
    zeroshot_targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)
    all_results = {}
    for model_type in ("unsupervised", "semi_supervised"):
        embeddings_path = CHECKPOINT_DIR / f"{model_type}_embeddings.pt"
        if not _exists(embeddings_path):
            print(f"  [skip] {model_type}: {embeddings_path} not found")
            continue
        print(f"\n  {model_type}:")
        embeddings = torch.load(embeddings_path, weights_only=True)
        per_variable = evaluate_all_zeroshot(embeddings, zeroshot_targets)
        r2s = [v["r2"] for v in per_variable.values()
               if isinstance(v.get("r2"), float) and not (v["r2"] != v["r2"])]
        avg = float(sum(r2s) / len(r2s)) if r2s else float("nan")
        all_results[model_type] = {
            "r2_avg": avg,
            "n_variables": len(r2s),
            "per_variable": per_variable,
        }
        print(f"    average R^2 across {len(r2s)} variables: {avg:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_make_serializable(all_results), f, indent=2)
    print(f"[done] zero-shot written to {out_path}")


def _step_plot(force: bool) -> None:
    print(f"\n[run]  generating paper figures")
    # plot.main() uses argparse on sys.argv, so invoke it with a clean argv
    import sys
    from lilyvae.scripts import plot as plot_module
    saved_argv = sys.argv
    sys.argv = ["lilyvae-plot", "--figure", "all"]
    try:
        plot_module.main()
    except SystemExit:
        pass
    finally:
        sys.argv = saved_argv


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end reproducibility pipeline for LilyVAE paper results."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every step even if outputs already exist.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--iterations",
        type=int,
        default=30,
        help="Bootstrap iterations (default: 30, matches paper).",
    )
    parser.add_argument(
        "--skip-zeroshot",
        action="store_true",
        help="Skip the post-training zero-shot evaluation step.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip paper figure regeneration at the end.",
    )
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("LilyVAE reproducibility pipeline")
    print("=" * 70)

    _step_preprocess(force=args.force)
    _step_train(force=args.force, device=args.device)
    _step_bootstrap(
        force=args.force,
        device=args.device,
        iterations=args.iterations,
    )
    _step_single_point(force=args.force, device=args.device)
    if not args.skip_zeroshot:
        _step_zeroshot(force=args.force)
    if not args.skip_plot:
        _step_plot(force=args.force)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"Pipeline complete in {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print("=" * 70)
    print(f"\nResults: {RESULTS_DIR / 'bootstrap_results.json'}")


if __name__ == "__main__":
    main()
