"""Does depth transfer once it is put in a comparable coordinate?

Holes reach anywhere from a few metres to 1800 m, median 213 m, so a
sample at 50 m sits near the bottom of one hole and near the top of
another. Absolute depth is therefore not the same quantity from hole to
hole, and its failure to transfer may be a failure of the coordinate
instead of a fact about burial.

Three coordinates, each given to the same regressor and scored on
held-out boreholes:
  absolute    depth in metres
  fractional  position within the hole's own cored interval, 0 to 1
  log         log10 of depth, which compresses the deep holes

Fractional depth uses the hole's own top and bottom, which are known at
prediction time for a hole that has been drilled, so it is not leakage.
"""
import numpy as np, torch
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from lilyvae.config import BIN_SIZE_CM, PROCESSED_DIR, SEED
from lilyvae.evaluation.folds import borehole_splits
from lilyvae.evaluation.targets import (
    TARGET_REGISTRY, apply_valid_range, select_targets, winsorize_values)

meta = torch.load(PROCESSED_DIR/"metadata.pt", weights_only=False)
tg = torch.load(PROCESSED_DIR/"zeroshot_targets.pt", weights_only=False)
bh = np.asarray(meta["borehole_ids"])
depth = np.asarray(meta["depth_bins"], float) * (BIN_SIZE_CM/100.0)

frac = np.zeros_like(depth)
for b in np.unique(bh):
    k = bh == b
    lo, hi = depth[k].min(), depth[k].max()
    frac[k] = (depth[k] - lo) / (hi - lo) if hi > lo else 0.0
logd = np.log10(np.maximum(depth, 0.1))

COORDS = {"absolute": depth, "fractional": frac, "log": logd}
names = select_targets(tg, ("independent",))
res = {c: {} for c in COORDS}

for fold in borehole_splits(bh, 5, 0.1, seed=SEED):
    for name in names:
        spec = TARGET_REGISTRY[name]
        v = np.asarray(tg[name]["values"], float)
        mask, _ = apply_valid_range(v, np.asarray(tg[name]["valid_mask"], bool), spec)
        fit, sc = mask & fold.train_mask, mask & fold.test_mask
        if fit.sum() < 50 or sc.sum() < 25 or np.ptp(v[sc]) == 0:
            continue
        y_fit, y_sc = v[fit].copy(), v[sc].copy()
        if spec.winsorize:
            # Thresholds from the training rows only, as in the main
            # pipeline; without this the corrupt SRM values dominate.
            lo, hi = np.percentile(y_fit, [1.0, 99.0])
            y_fit, y_sc = np.clip(y_fit, lo, hi), np.clip(y_sc, lo, hi)
        if np.ptp(y_sc) == 0:
            continue
        for cname, coord in COORDS.items():
            m = CatBoostRegressor(iterations=300, depth=4, learning_rate=0.1,
                                  verbose=0, random_seed=SEED,
                                  bootstrap_type="Bernoulli", subsample=0.8)
            m.fit(coord[fit].reshape(-1,1), y_fit)
            res[cname].setdefault(name, []).append(
                r2_score(y_sc, m.predict(coord[sc].reshape(-1,1))))

print(f"{'property':<30}" + "".join(f"{c:>13}" for c in COORDS))
print("-"*70)
for name in names:
    if name not in res["absolute"]: continue
    print(f"{TARGET_REGISTRY[name].label[:29]:<30}" +
          "".join(f"{np.median(res[c][name]):>13.3f}" for c in COORDS))
print("-"*70)
print(f"{'median over properties':<30}" +
      "".join(f"{np.median([np.median(v) for v in res[c].values()]):>13.3f}" for c in COORDS))
