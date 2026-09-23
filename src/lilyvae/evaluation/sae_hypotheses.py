"""Do the sparse features line up with couplings the literature predicts?

Reading an embedding usually means finding a direction, looking at what
activates it, and naming it. That is post hoc, and on this dataset it is
also circular: the lithology labels used to name a direction are
assigned from the same six measurements the encoder was given, so a
feature will always seem to correspond to something.

This inverts the order. Section~4.3 lists the physical couplings that
make the held-out variables predictable at all, each with a published
source. Those couplings say which axes a representation of this data
ought to contain. Here they are written down first, as specific
predictions, and then tested.

Two choices keep the test honest.

Each hypothesis is scored against a **measured held-out variable**, not
against a lithology label. Porosity, thermal conductivity, magnetisation
and velocity were never shown to the encoder, so a feature tracking one
of them is saying something the model was not told.

Every alignment is compared against a **null built by rotating the
latent space**. A sparse feature is a direction through a
ten-dimensional space that already encodes the six inputs, and the
inputs already correlate with the targets, so some alignment arrives for
free. The null says how much, and only the excess is reported.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr

from lilyvae.config import PROCESSED_DIR, SEED
from lilyvae.evaluation.folds import borehole_splits
from lilyvae.evaluation.sae import SAETrainConfig, train_sae
from lilyvae.evaluation.targets import apply_valid_range, winsorize_values


@dataclass(frozen=True)
class Hypothesis:
    """One prediction from the literature about what the space should hold."""

    name: str
    target: str          # the held-out column that measures it
    citation: str        # bibtex key of the source making the prediction
    statement: str       # what that source says, in one line


#: Written before looking at any feature. Each names a coupling from
#: Section 4.3 and the variable that measures it.
HYPOTHESES: tuple[Hypothesis, ...] = (
    Hypothesis(
        "compaction", "MAD_Porosity (vol%)", "blum2003",
        "Bulk density fixes porosity through the mass balance between "
        "grain, pore fluid and bulk volume, so an axis of compaction "
        "state should exist.",
    ),
    Hypothesis(
        "thermal", "TCON_Thermal conductivity mean (W/(m*K))", "goto2009relations",
        "Conductivity mixes the solid framework with the pore water, so "
        "it should follow the same axis as porosity but not identically.",
    ),
    Hypothesis(
        "acoustic", "PWC_P-wave velocity x (m/s)", "hamilton1982sound",
        "Compressional velocity in unconsolidated marine sediment "
        "follows porosity and grain size.",
    ),
    Hypothesis(
        "magnetic", "SRM_Intensity raw (A/m)", "ellwood1980induced",
        "Remanence and susceptibility both scale with ferrimagnetic "
        "mineral concentration, so a magnetic-concentration axis should "
        "exist separately from the compaction one.",
    ),
    Hypothesis(
        "grain_framework", "MAD_Grain density (g/cm^3)", "brigaud1989mineralogy",
        "Mineralogy controls conductivity independently of porosity, so "
        "the mineral framework should be separable from pore space.",
    ),
)


def _prepare(name: str, targets: dict) -> tuple[np.ndarray, np.ndarray]:
    from lilyvae.evaluation.targets import TARGET_REGISTRY

    spec = TARGET_REGISTRY.get(name)
    values = np.asarray(targets[name]["values"], dtype=np.float64)
    mask, _ = apply_valid_range(
        values, np.asarray(targets[name]["valid_mask"], dtype=bool), spec
    )
    if spec is not None and spec.winsorize and mask.sum():
        clipped, _ = winsorize_values(values[mask])
        values = values.copy()
        values[mask] = clipped
    return values, mask


def _selectivity(acts: np.ndarray, per_target: dict[str, tuple],
                 fold_mask: np.ndarray) -> dict[str, float]:
    """How specifically does any one feature pick out one target?

    The maximum correlation over many features is a poor statistic here.
    Every target correlates with the six inputs, the latent encodes those
    inputs, so almost any direction correlates with almost everything and
    a maximum over 32 features is high by construction.

    An interpretable axis should be *specific*: strongly aligned with one
    property and weakly with the rest. For each feature this measures the
    gap between its best and second-best target, then takes the largest
    gap across features. A diffuse decomposition, where every feature
    tracks the same shared compaction signal, scores near zero however
    high its individual correlations are.
    """
    names = list(per_target)
    rho = np.zeros((acts.shape[1], len(names)))
    for j in range(acts.shape[1]):
        for t, name in enumerate(names):
            values, valid = per_target[name]
            m = valid & fold_mask
            a = acts[m, j]
            if m.sum() < 50 or np.ptp(a) == 0:
                continue
            r = spearmanr(a, values[m]).statistic
            rho[j, t] = abs(r) if np.isfinite(r) else 0.0

    out = {}
    for t, name in enumerate(names):
        gaps = []
        for j in range(acts.shape[1]):
            others = np.delete(rho[j], t)
            gaps.append(rho[j, t] - others.max() if others.size else rho[j, t])
        out[name] = float(max(gaps))
    return out


def _alignment(activations: np.ndarray, values: np.ndarray,
               mask: np.ndarray) -> float:
    """Strongest rank alignment between any one feature and the target."""
    best = 0.0
    for j in range(activations.shape[1]):
        a = activations[mask, j]
        if np.ptp(a) == 0:
            continue
        rho = spearmanr(a, values[mask]).statistic
        if np.isfinite(rho) and abs(rho) > abs(best):
            best = float(rho)
    return best


def run_hypothesis_test(
    n_folds: int = 3,
    d_hidden: int = 32,
    n_null: int = 50,
    device: str = "auto",
) -> dict:
    """Score each literature hypothesis against a rotation null.

    The SAE is fitted on each fold's training boreholes and the alignment
    measured on the held-out ones, so a feature that only describes the
    boreholes it was fitted to does not count.

    The null rotates the latent space by a random orthogonal matrix
    before fitting, which preserves every distance and correlation in the
    embedding while destroying the particular axes the SAE found. If a
    hypothesis scores no better than that, the sparse decomposition has
    added nothing to what the latent already carried.
    """
    rng = np.random.RandomState(SEED)
    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    labels = torch.load(PROCESSED_DIR / "train_labels.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)
    borehole_ids = np.asarray(metadata["borehole_ids"])

    from lilyvae.evaluation.bootstrap import ALL_TRAINERS
    from lilyvae.training.trainer import get_device

    dev = get_device(device)
    trainer = dict(ALL_TRAINERS)["unsupervised_vae"]
    prepared = {h.target: _prepare(h.target, targets) for h in HYPOTHESES}

    observed: dict[str, list[float]] = {h.name: [] for h in HYPOTHESES}
    null: dict[str, list[float]] = {h.name: [] for h in HYPOTHESES}
    obs_sel: dict[str, list[float]] = {h.name: [] for h in HYPOTHESES}
    null_sel: dict[str, list[float]] = {h.name: [] for h in HYPOTHESES}
    by_name = {h.name: h.target for h in HYPOTHESES}
    start = time.time()

    for fold in borehole_splits(borehole_ids, n_folds, 0.1, seed=SEED):
        print(f"\nFold {fold.iteration + 1}/{n_folds}", flush=True)
        trained = trainer(
            features[fold.train_mask], labels[fold.train_mask], features, dev
        )
        z = trained.embeddings

        cfg = SAETrainConfig(d_hidden=d_hidden, num_epochs=120, device=device)
        sae, _ = train_sae(z[fold.train_mask], cfg)
        with torch.no_grad():
            acts = sae.encode(z).cpu().numpy()

        for h in HYPOTHESES:
            values, valid = prepared[h.target]
            m = valid & fold.test_mask
            if m.sum() < 50:
                continue
            observed[h.name].append(_alignment(acts, values, m))
            print(f"  {h.name:<16} rho {observed[h.name][-1]:+.3f} "
                  f"(n={int(m.sum())})", flush=True)

        sel = _selectivity(acts, {n: prepared[t] for n, t in by_name.items()},
                           fold.test_mask)
        for name, v in sel.items():
            obs_sel[name].append(v)

        # Null: the same procedure on a randomly rotated latent space.
        for _ in range(max(1, n_null // n_folds)):
            q, _ = np.linalg.qr(rng.normal(size=(z.shape[1], z.shape[1])))
            zr = torch.tensor(z.numpy() @ q, dtype=torch.float32)
            sae_r, _ = train_sae(zr[fold.train_mask],
                                 SAETrainConfig(d_hidden=d_hidden, num_epochs=40,
                                                device=device))
            with torch.no_grad():
                acts_r = sae_r.encode(zr).cpu().numpy()
            for h in HYPOTHESES:
                values, valid = prepared[h.target]
                m = valid & fold.test_mask
                if m.sum() >= 50:
                    null[h.name].append(_alignment(acts_r, values, m))
            sel_r = _selectivity(acts_r,
                                 {n: prepared[t] for n, t in by_name.items()},
                                 fold.test_mask)
            for name, v in sel_r.items():
                null_sel[name].append(v)

    results = []
    for h in HYPOTHESES:
        obs = np.abs(observed[h.name])
        nul = np.abs(null[h.name])
        if not len(obs) or not len(nul):
            continue
        p = float((nul >= np.median(obs)).mean())
        results.append({
            "hypothesis": h.name, "target": h.target, "citation": h.citation,
            "statement": h.statement,
            "observed_rho": float(np.median(obs)),
            "null_rho": float(np.median(nul)),
            "null_p95": float(np.percentile(nul, 95)),
            "excess": float(np.median(obs) - np.median(nul)),
            "selectivity": float(np.median(obs_sel[h.name])) if obs_sel[h.name] else float("nan"),
            "selectivity_null": float(np.median(null_sel[h.name])) if null_sel[h.name] else float("nan"),
            "selectivity_null_p95": float(np.percentile(null_sel[h.name], 95)) if null_sel[h.name] else float("nan"),
            "selective": bool(
                obs_sel[h.name] and null_sel[h.name]
                and np.median(obs_sel[h.name]) > np.percentile(null_sel[h.name], 95)
            ),
            "p_value": p,
            "supported": bool(np.median(obs) > np.percentile(nul, 95)),
        })

    return {
        "config": {"n_folds": n_folds, "d_hidden": d_hidden,
                   "n_null": n_null, "seed": SEED},
        "results": results,
        "elapsed_seconds": time.time() - start,
    }
