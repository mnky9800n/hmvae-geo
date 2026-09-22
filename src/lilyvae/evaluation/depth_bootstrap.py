"""Is the transfer signal just depth, acting through compaction?

The objection is specific and it is the right one to press. Sediment
compacts with burial, so density rises, porosity falls, and velocity
increases, all together and all as a function of depth. A model given
six properties that each track compaction could predict a seventh
property that also tracks compaction while knowing nothing about rock.
Anyone who has looked at a core could do the same with a depth scale.

Answering it needs more than noting that the embedding beats depth.
Four arms, all scored on held-out boreholes over the shared folds:

``depth_only``       depth alone predicts the target. The proxy story in
                     its purest form, and the number that says how much
                     of each property is simply burial.
``embedding``        the ten-dimensional latent, as reported elsewhere.
``embedding_resid``  the latent predicts the target after depth has been
                     regressed out of it. What survives here cannot be
                     compaction, because compaction has been removed.
``embedding_depth``  the latent plus depth together, which bounds how
                     much the two carry jointly.

A separate question is whether depth belongs in the model at all.
`run_depth_input_ablation` trains a seven-variable VAE that sees depth
as an input alongside the six measurements. If depth carried useful
structure the model was missing, that variant should transfer better.

Depth is quantised to the 20 cm binning used throughout, so it is the
bin centre rather than the exact sample depth.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from lilyvae.config import (
    BIN_SIZE_CM,
    HIDDEN_DIMS,
    LATENT_DIM,
    MASK_FRACTION,
    PROCESSED_DIR,
    SEED,
)
from lilyvae.evaluation.folds import borehole_splits
from lilyvae.evaluation.targets import (
    HEADLINE_TIERS,
    TARGET_REGISTRY,
    apply_valid_range,
    select_targets,
    winsorize_values,
)
from lilyvae.evaluation.zeroshot_bootstrap import (
    MIN_TEST_ROWS,
    MIN_TRAIN_ROWS,
    _distribution,
    _fit_catboost,
)
from lilyvae.training.trainer import get_device

#: The arms compared. Kept to four because each is a CatBoost fit per
#: target per fold, and the full sweep already costs hours.
ARMS = ("depth_only", "embedding", "embedding_resid", "embedding_depth")


def _prepare(targets: dict, names: list[str]) -> dict:
    out = {}
    for name in names:
        spec = TARGET_REGISTRY[name]
        values = np.asarray(targets[name]["values"], dtype=np.float64)
        mask, _ = apply_valid_range(
            values, np.asarray(targets[name]["valid_mask"], dtype=bool), spec
        )
        out[name] = (values, mask, bool(spec.winsorize))
    return out


def _arm_features(
    arm: str, embeddings: np.ndarray, depths: np.ndarray
) -> np.ndarray:
    if arm == "depth_only":
        return depths.reshape(-1, 1)
    if arm in ("embedding", "embedding_resid"):
        return embeddings
    if arm == "embedding_depth":
        return np.column_stack([embeddings, depths])
    raise ValueError(f"unknown arm {arm!r}")


def evaluate_depth_arms(
    embeddings: np.ndarray,
    depths: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    winsorize: bool = False,
) -> dict:
    """Score every arm for one target on one fold."""
    fit_rows = valid & train_mask
    score_rows = valid & test_mask
    n_train, n_test = int(fit_rows.sum()), int(score_rows.sum())
    if n_train < MIN_TRAIN_ROWS or n_test < MIN_TEST_ROWS:
        return {"n_train": n_train, "n_test": n_test}

    y_train = values[fit_rows].copy()
    y_test = values[score_rows].copy()
    if winsorize:
        _, _ = winsorize_values(y_train)
        lo, hi = np.percentile(y_train, [1.0, 99.0])
        y_train, y_test = np.clip(y_train, lo, hi), np.clip(y_test, lo, hi)
    if np.ptp(y_test) == 0:
        return {"n_train": n_train, "n_test": n_test}

    out = {"n_train": n_train, "n_test": n_test}
    for arm in ARMS:
        x = _arm_features(arm, embeddings, depths)
        if arm == "embedding_resid":
            # Regress depth out of the target on the training rows, then
            # ask the embedding to predict what is left. The depth model
            # is fitted on training rows only, so the residual on the
            # held-out side uses no information from it.
            depth_model = _fit_catboost(
                depths[fit_rows].reshape(-1, 1), y_train
            )
            r_train = y_train - depth_model.predict(depths[fit_rows].reshape(-1, 1))
            r_test = y_test - depth_model.predict(depths[score_rows].reshape(-1, 1))
            if np.ptp(r_test) == 0 or np.var(r_train) < 1e-12:
                out[arm] = float("nan")
                continue
            model = _fit_catboost(x[fit_rows], r_train)
            out[arm] = float(r2_score(r_test, model.predict(x[score_rows])))
        else:
            model = _fit_catboost(x[fit_rows], y_train)
            out[arm] = float(r2_score(y_test, model.predict(x[score_rows])))
    return out


def run_depth_bootstrap(
    n_iterations: int = 30,
    test_fraction: float = 0.1,
    tiers: tuple[str, ...] = HEADLINE_TIERS,
    device: str = "auto",
) -> dict:
    """Bootstrap the depth arms across borehole folds."""
    from lilyvae.evaluation.bootstrap import ALL_TRAINERS

    dev = get_device(device)
    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)

    borehole_ids = np.asarray(metadata["borehole_ids"])
    depths = np.asarray(metadata["depth_bins"], dtype=np.float64) * (BIN_SIZE_CM / 100.0)
    names = select_targets(targets, tiers)
    prepared = _prepare(targets, names)
    trainer = dict(ALL_TRAINERS)["unsupervised_vae"]

    print(f"Depth bootstrap: {n_iterations} folds, {len(names)} targets, "
          f"arms {', '.join(ARMS)}")
    print(f"  depth range {depths.min():.1f}-{depths.max():.1f} m")

    per_iteration, start = [], time.time()
    for fold in borehole_splits(borehole_ids, n_iterations, test_fraction, seed=SEED):
        t0 = time.time()
        print(f"\n{'='*70}\nFold {fold.iteration+1}/{n_iterations}\n{'='*70}")
        trained = trainer(
            features[fold.train_mask], labels[fold.train_mask], features, dev
        )
        z = trained.embeddings.numpy().astype(np.float64)

        per_variable = {}
        for name, (values, valid, wins) in prepared.items():
            try:
                res = evaluate_depth_arms(
                    z, depths, values, valid,
                    fold.train_mask, fold.test_mask, winsorize=wins,
                )
            except Exception as e:
                res = {"error": str(e)}
            per_variable[name] = res
            if "embedding" in res:
                print(f"  {TARGET_REGISTRY[name].label[:34]:<36} "
                      f"depth {res.get('depth_only', float('nan')):+.3f}  "
                      f"emb {res.get('embedding', float('nan')):+.3f}  "
                      f"resid {res.get('embedding_resid', float('nan')):+.3f}")

        per_iteration.append({
            "iteration": fold.iteration,
            "per_variable": per_variable,
            "elapsed_seconds": time.time() - t0,
        })

    summary = _summarise(per_iteration, names)
    _print_summary(summary, names)
    return {
        "config": {
            "n_iterations": n_iterations, "test_fraction": test_fraction,
            "tiers": list(tiers), "arms": list(ARMS), "targets": names, "seed": SEED,
        },
        "per_iteration": per_iteration,
        "summary": summary,
        "elapsed_seconds": time.time() - start,
    }


def _summarise(per_iteration: list[dict], names: list[str]) -> dict:
    summary = {"per_target": {}, "overall": {}}
    for name in names:
        entry = {}
        for arm in ARMS:
            vals = [
                it["per_variable"].get(name, {}).get(arm)
                for it in per_iteration
            ]
            vals = np.array([v for v in vals if v is not None and np.isfinite(v)])
            if vals.size:
                entry[arm] = _distribution(vals)
        summary["per_target"][name] = entry

    for arm in ARMS:
        medians = [
            summary["per_target"][n][arm]["median"]
            for n in names if arm in summary["per_target"].get(n, {})
        ]
        if medians:
            summary["overall"][arm] = {
                "median_over_targets": float(np.median(medians)),
                "mean_over_targets": float(np.mean(medians)),
                "n_targets": len(medians),
            }
    return summary


def _print_summary(summary: dict, names: list[str]) -> None:
    print(f"\n{'='*86}")
    print(f"{'target':<34} " + "".join(f"{a[:15]:>13}" for a in ARMS))
    print("=" * 86)
    for name in names:
        e = summary["per_target"].get(name, {})
        label = TARGET_REGISTRY[name].label if name in TARGET_REGISTRY else name
        row = "".join(
            f"{e[a]['median']:>13.3f}" if a in e else f"{'--':>13}" for a in ARMS
        )
        print(f"{label[:33]:<34} {row}")
    print("-" * 86)
    row = "".join(
        f"{summary['overall'][a]['median_over_targets']:>13.3f}"
        if a in summary["overall"] else f"{'--':>13}" for a in ARMS
    )
    print(f"{'median over targets':<34} {row}")


def run_depth_input_ablation(
    n_iterations: int = 10,
    test_fraction: float = 0.1,
    tiers: tuple[str, ...] = HEADLINE_TIERS,
    device: str = "auto",
) -> dict:
    """Does giving the model depth as a seventh input help or hurt?

    The six-variable model is compared against one trained on the same
    six plus standardised depth. This is the counterpart to the arms
    above: those ask whether depth explains the transfer, this asks
    whether depth is information the model was missing.

    A geological prediction is worth stating before the result. Depth is
    the strongest single ordering in the dataset, and the six inputs
    already correlate with it through compaction, so an encoder given
    depth directly can satisfy the reconstruction term by encoding
    position in the hole rather than the properties of the rock. If that
    happens, transfer to a new borehole should get worse rather than
    better, because a depth in one hole means something different in
    another.
    """
    from lilyvae.evaluation.ablations import _train_ablation_model
    from lilyvae.evaluation.bootstrap import _make_vae_dataloader
    from lilyvae.config import LEARNING_RATE, MAX_GRAD_NORM, NUM_EPOCHS, WEIGHT_DECAY
    from lilyvae.model.vae import LilyVAE
    from lilyvae.training.masking import LearnableMasking

    dev = get_device(device)
    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)

    borehole_ids = np.asarray(metadata["borehole_ids"])
    depths = np.asarray(metadata["depth_bins"], dtype=np.float64) * (BIN_SIZE_CM / 100.0)
    names = select_targets(targets, tiers)
    prepared = _prepare(targets, names)

    results = {"six_variable": {}, "seven_variable_with_depth": {}}
    print(f"Depth-as-input ablation: {n_iterations} folds, {len(names)} targets")

    for fold in borehole_splits(borehole_ids, n_iterations, test_fraction, seed=SEED):
        print(f"\nFold {fold.iteration+1}/{n_iterations}")
        for variant in ("six_variable", "seven_variable_with_depth"):
            if variant == "six_variable":
                x = features
            else:
                # Depth standardised on the training rows only, so the
                # held-out boreholes contribute nothing to the scaling.
                train_depth = depths[fold.train_mask]
                scaled = (depths - train_depth.mean()) / max(train_depth.std(), 1e-9)
                x = torch.cat(
                    [features, torch.tensor(scaled, dtype=torch.float32).unsqueeze(1)],
                    dim=1,
                )

            torch.manual_seed(SEED)
            vae = LilyVAE(input_dim=x.shape[1], latent_dim=LATENT_DIM,
                          hidden_dims=list(HIDDEN_DIMS))
            masking = LearnableMasking(x.shape[1], MASK_FRACTION)
            vae = _train_ablation_model(
                dataloader=_make_vae_dataloader(x[fold.train_mask],
                                                labels[fold.train_mask]),
                vae=vae, masking=masking, decoder_type="heteroscedastic",
                beta_schedule_type="annealing", alpha_weight=0.0,
                num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY, max_grad_norm=MAX_GRAD_NORM, device=dev,
            )
            vae.eval()
            with torch.no_grad():
                z = vae.get_embeddings(x.to(dev)).cpu().numpy().astype(np.float64)

            scores = []
            for name, (values, valid, wins) in prepared.items():
                res = evaluate_depth_arms(
                    z, depths, values, valid, fold.train_mask, fold.test_mask,
                    winsorize=wins,
                )
                r2 = res.get("embedding")
                results[variant].setdefault(name, []).append(r2)
                if r2 is not None and np.isfinite(r2):
                    scores.append(r2)
            print(f"  {variant:<28} median R2 over targets "
                  f"{np.median(scores) if scores else float('nan'):.4f}")

    summary = {}
    for variant, per_target in results.items():
        medians = []
        for name, vals in per_target.items():
            vals = np.array([v for v in vals if v is not None and np.isfinite(v)])
            if vals.size:
                medians.append(float(np.median(vals)))
        summary[variant] = {
            "median_over_targets": float(np.median(medians)) if medians else float("nan"),
            "n_targets": len(medians),
        }

    print(f"\n{'='*70}")
    for variant, s in summary.items():
        print(f"  {variant:<30} median over targets {s['median_over_targets']:.4f}")
    delta = (summary["seven_variable_with_depth"]["median_over_targets"]
             - summary["six_variable"]["median_over_targets"])
    print(f"  adding depth as an input changes transfer by {delta:+.4f}")

    return {
        "config": {"n_iterations": n_iterations, "tiers": list(tiers),
                   "targets": names, "seed": SEED},
        "per_target": results,
        "summary": summary,
    }
