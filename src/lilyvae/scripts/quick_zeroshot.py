"""Quick out-of-sample zero-shot comparison: unsup VAE vs semi VAE vs raw features."""

import numpy as np
import torch
import json
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score
from lilyvae.config import PROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR, SEED


def main():
    features = torch.load(PROCESSED_DIR / "train_features.pt", weights_only=True)
    metadata = torch.load(PROCESSED_DIR / "metadata.pt", weights_only=False)
    zs_targets = torch.load(PROCESSED_DIR / "zeroshot_targets.pt", weights_only=False)
    bh_ids = metadata["borehole_ids"]

    # Only well-behaved variables with enough data
    targets = {}
    for k, v in zs_targets.items():
        valid = v["valid_mask"].numpy().astype(bool)
        vals = v["values"].numpy()
        if valid.sum() > 5000 and np.isfinite(vals[valid]).all() and np.abs(vals[valid]).max() < 1e6:
            targets[k] = v
    print(f"Using {len(targets)} target variables")

    # Shared borehole splits
    unique_bh = np.unique(bh_ids)
    n_test = max(1, int(len(unique_bh) * 0.2))
    rng = np.random.RandomState(SEED)
    splits = []
    for _ in range(20):
        perm = rng.permutation(unique_bh)
        test_bh = set(perm[:n_test])
        test_mask = np.array([bh in test_bh for bh in bh_ids])
        splits.append((~test_mask, test_mask))

    def run_model(emb, label):
        per_iter = []
        for i, (tr_m, te_m) in enumerate(splits):
            r2s = []
            for t in targets.values():
                valid = t["valid_mask"].numpy().astype(bool)
                vals = t["values"].numpy()
                tr = tr_m & valid
                te = te_m & valid
                if tr.sum() < 20 or te.sum() < 10:
                    continue
                try:
                    m = CatBoostRegressor(
                        iterations=500, depth=6, learning_rate=0.1,
                        verbose=0, random_seed=i,
                        bootstrap_type="Bernoulli", subsample=0.8,
                    )
                    m.fit(emb[tr], vals[tr])
                    r2 = r2_score(vals[te], m.predict(emb[te]))
                    if r2 > -10:
                        r2s.append(r2)
                except Exception:
                    continue
            if r2s:
                med = float(np.median(r2s))
                per_iter.append(med)
                print(f"  {label} iter {i+1}/20: {med:.4f}")
        arr = np.array(per_iter)
        return {
            "median": float(np.median(arr)),
            "iqr_low": float(np.percentile(arr, 25)),
            "iqr_high": float(np.percentile(arr, 75)),
        }

    unsup = torch.load(CHECKPOINT_DIR / "unsupervised_embeddings.pt", weights_only=True).numpy()
    semi = torch.load(CHECKPOINT_DIR / "semi_supervised_embeddings.pt", weights_only=True).numpy()
    raw = features.numpy()

    print("\nUnsupervised VAE:")
    u = run_model(unsup, "unsup")

    print("\nSemi-supervised VAE:")
    s = run_model(semi, "semi")

    print("\nRaw features:")
    r = run_model(raw, "raw")

    print(f"\n{'='*60}")
    print("OUT-OF-SAMPLE ZERO-SHOT (median R2, 20 bootstrap iterations)")
    print(f"{'='*60}")
    for name, res in [("Raw features", r), ("Unsupervised VAE", u), ("Semi-supervised VAE", s)]:
        print(f"  {name:25s}  median={res['median']:.4f}  IQR=[{res['iqr_low']:.4f}, {res['iqr_high']:.4f}]")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "zeroshot_comparison.json", "w") as f:
        json.dump({"raw_features": r, "unsupervised_vae": u, "semi_supervised_vae": s}, f, indent=2)
    print("Saved.")


if __name__ == "__main__":
    main()
