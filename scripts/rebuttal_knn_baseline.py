"""kNN imputation baseline for nZbB W3 (add a simple imputation/representation baseline).

kNN on the six raw inputs, same in-sample protocol as the paper:
  - zero-shot: KNeighborsRegressor per held-out target -> mean R^2
  - classification: KNeighborsClassifier -> one-vs-rest macro AUC
Also reports raw-input GMM ARI so the "simple baseline" row is complete.
Writes data/results/rebuttal_knn.json.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import r2_score, roc_auc_score

from lilyvae.evaluation.clustering import evaluate_clustering

PROCESSED = Path("data/processed")
RESULTS = Path("data/results")
K = 10


def knn_zeroshot_mean_r2(features: np.ndarray, targets: dict) -> tuple[float, dict]:
    per = {}
    for name, t in targets.items():
        mask = t["valid_mask"].numpy().astype(bool)
        X, y = features[mask], t["values"].numpy()[mask]
        # LILY release contains a few non-finite / extreme entries; drop them
        finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
        X, y = X[finite], y[finite]
        if len(X) < 50:
            per[name] = float("nan")
            continue
        # k+1 neighbours, drop self (in-sample) to avoid trivial self-prediction
        model = KNeighborsRegressor(n_neighbors=min(K + 1, len(X)))
        model.fit(X, y)
        dist, idx = model.kneighbors(X)
        # drop the first neighbour (itself) then average the rest
        y_pred = y[idx[:, 1:]].mean(axis=1)
        per[name] = float(r2_score(y, y_pred))
    vals = [v for v in per.values() if v == v]
    return (float(np.mean(vals)) if vals else float("nan")), per


def knn_classification_auc(features: np.ndarray, labels: np.ndarray, top_n: int = 10) -> float:
    clf = KNeighborsClassifier(n_neighbors=K)
    clf.fit(features, labels)
    proba = clf.predict_proba(features)  # (N, n_present_classes)
    present = clf.classes_.astype(int)
    per_class = {}
    for col, cls in enumerate(present):
        y_bin = (labels == cls).astype(int)
        if y_bin.sum() < 2 or (1 - y_bin).sum() < 2:
            continue
        try:
            per_class[int(cls)] = float(roc_auc_score(y_bin, proba[:, col]))
        except Exception:
            continue
    return float(np.mean(list(per_class.values()))) if per_class else float("nan")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    features = torch.load(PROCESSED / "train_features.pt")
    labels = torch.load(PROCESSED / "train_labels.pt")
    targets = torch.load(PROCESSED / "zeroshot_targets.pt")
    feats_np = features.numpy()
    labels_np = labels.numpy().astype(int)

    print("kNN zero-shot ...")
    zs_mean, zs_per = knn_zeroshot_mean_r2(feats_np, targets)
    print(f"  kNN zero-shot mean R^2 = {zs_mean:.4f}")

    print("kNN classification AUC ...")
    auc = knn_classification_auc(feats_np, labels_np)
    print(f"  kNN macro AUC = {auc:.4f}")

    print("raw-input GMM ARI ...")
    ari = evaluate_clustering(features, labels)["ari"]
    print(f"  raw-input ARI = {ari:.4f}")

    out = {
        "knn_zeroshot_mean_r2": zs_mean,
        "knn_classification_auc_macro": auc,
        "raw_input_gmm_ari": ari,
        "k": K,
        "knn_zeroshot_per_target": zs_per,
    }
    (RESULTS / "rebuttal_knn.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {RESULTS / 'rebuttal_knn.json'}")
    print("\n=== rows to add to W3 table ===")
    print(f"| kNN (raw inputs, imputation) | (n/a) | {zs_mean:.3f} | {auc:.3f} |")
    print(f"| Raw inputs (GMM/linear/CatBoost) | {ari:.3f} | 0.795 | 0.932 |")


if __name__ == "__main__":
    main()
