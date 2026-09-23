"""Depth should dominate. Why doesn't it?

Sediment accumulates by superposition and compacts with burial, so a
geologist expects depth to be the strongest single predictor of almost
any physical property. Across held-out boreholes it predicts essentially
nothing. The obvious resolution is that depth orders a hole beautifully
and means something different in the next hole along.

This measures both: how much of a property depth explains inside a
single borehole, against how much it explains when boreholes are pooled.
"""
import numpy as np, torch
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from lilyvae.config import BIN_SIZE_CM, PROCESSED_DIR, SEED
from lilyvae.evaluation.targets import TARGET_REGISTRY, apply_valid_range, select_targets

meta = torch.load(PROCESSED_DIR/"metadata.pt", weights_only=False)
tg = torch.load(PROCESSED_DIR/"zeroshot_targets.pt", weights_only=False)
bh = np.asarray(meta["borehole_ids"])
depth = np.asarray(meta["depth_bins"], float) * (BIN_SIZE_CM/100.0)

def fit(X, y):
    m = CatBoostRegressor(iterations=300, depth=4, learning_rate=0.1, verbose=0,
                          random_seed=SEED, bootstrap_type="Bernoulli", subsample=0.8)
    m.fit(X, y); return r2_score(y, m.predict(X))

print(f"{'property':<30} {'within hole':>12} {'pooled':>9} {'between-hole':>13}")
print(f"{'':<30} {'(median)':>12} {'':>9} {'share of var':>13}")
print("-"*68)
rows=[]
for name in select_targets(tg, ("independent",)):
    spec = TARGET_REGISTRY[name]
    vals = np.asarray(tg[name]["values"], float)
    mask, _ = apply_valid_range(vals, np.asarray(tg[name]["valid_mask"], bool), spec)
    if mask.sum() < 200: continue
    d, y, g = depth[mask], vals[mask], bh[mask]

    # Within one hole: fit depth -> property separately per borehole.
    within=[]
    for b in np.unique(g):
        k = g == b
        if k.sum() >= 60 and np.ptp(y[k]) > 0 and np.ptp(d[k]) > 0:
            within.append(fit(d[k].reshape(-1,1), y[k]))
    if not within: continue

    pooled = fit(d.reshape(-1,1), y)

    # How much of the property's variance is differences between holes?
    gm = y.mean(); between = 0.0
    for b in np.unique(g):
        k = g == b
        between += k.sum() * (y[k].mean() - gm)**2
    frac = between / ((y - gm)**2).sum()

    rows.append((spec.label, np.median(within), pooled, frac, len(within)))
    print(f"{spec.label[:29]:<30} {np.median(within):>12.3f} {pooled:>9.3f} {frac:>13.2f}")

a=np.array([(r[1], r[2]) for r in rows])
print("-"*68)
print(f"{'median over properties':<30} {np.median(a[:,0]):>12.3f} {np.median(a[:,1]):>9.3f}")
