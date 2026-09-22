"""Is held-out-borehole R2 pessimistically biased, and by how much?

The objection is real: a held-out borehole can carry rock the model
never saw, so its score is not what the model would achieve on a new
borehole drawn from the same population. The answer is not to argue
about it but to measure it, which is why each fold recorded how far its
held-out boreholes sit from its training set.

Two independent distance measures per fold:
  lithology_jaccard  overlap of the lithology sets, 1.0 = no new rock
  energy_distance    two-sample statistic in the six input features

If R2 rises as the held-out set gets closer to training, the holdout is
costing us a measurable amount and the intercept at zero distance
estimates what a "typical" new borehole would score.
"""
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr

d = json.load(open("data/results/zeroshot_bootstrap.json"))
rows = []
for it in d["per_iteration"]:
    for m in it["models"]:
        if m["model"] != "unsupervised_vae":
            continue
        rows.append((
            it["lithology_jaccard"], it["energy_distance"],
            m["zeroshot_r2_headline"], it["n_test_boreholes"],
        ))

a = np.array(rows, dtype=float)
jac, energy, r2 = a[:, 0], a[:, 1], a[:, 2]
ok = np.isfinite(jac) & np.isfinite(energy) & np.isfinite(r2)
jac, energy, r2 = jac[ok], energy[ok], r2[ok]

print(f"{len(r2)} folds")
print(f"  headline R2   : median {np.median(r2):.4f}  range [{r2.min():.4f}, {r2.max():.4f}]")
print(f"  lithology overlap: median {np.median(jac):.3f}  range [{jac.min():.3f}, {jac.max():.3f}]")
print(f"  energy distance  : median {np.median(energy):.4f}  range [{energy.min():.4f}, {energy.max():.4f}]")
print()
print("Does fold difficulty explain fold score?")
for name, x in (("lithology overlap", jac), ("energy distance", energy)):
    pr, pp = pearsonr(x, r2)
    sr, sp = spearmanr(x, r2)
    print(f"  R2 vs {name:<18} pearson {pr:+.3f} (p={pp:.3f})   spearman {sr:+.3f} (p={sp:.3f})")

# Extrapolate to a held-out set indistinguishable from training.
print()
print("Linear fit, extrapolated to a held-out borehole set that looks like training:")
for name, x, target in (("lithology overlap", jac, 1.0),
                        ("energy distance", energy, 0.0)):
    if np.ptp(x) == 0:
        continue
    slope, intercept = np.polyfit(x, r2, 1)
    pred = slope * target + intercept
    print(f"  by {name:<18} slope {slope:+.4f}  ->  R2 at "
          f"{'overlap=1' if target else 'distance=0'}: {pred:.4f}"
          f"   (observed median {np.median(r2):.4f})")

# Split folds by difficulty.
print()
med_j = np.median(jac)
easy, hard = r2[jac >= med_j], r2[jac < med_j]
print(f"folds with above-median lithology overlap: median R2 {np.median(easy):.4f} (n={len(easy)})")
print(f"folds with below-median lithology overlap: median R2 {np.median(hard):.4f} (n={len(hard)})")
