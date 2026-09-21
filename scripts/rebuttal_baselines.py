"""Rebuttal baselines E1 + E2 (DQpL Q1 / AC Ask 1a, Ask 3).

E1: purely supervised lithology classification directly from the 6 raw input
    variables, vs. our unsupervised-embedding linear probe (paper AUC ~0.957).
E2: simple regression for zero-shot prediction directly from the 6 raw inputs,
    vs. our embedding-based zero-shot (paper mean R^2 ~0.82).

Same samples, same metric code, in-sample (matching the paper's methodology).
Writes data/results/rebuttal_baselines.json and prints a markdown table.
"""

import json
from pathlib import Path

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from lilyvae.evaluation.classification import evaluate_classification
from lilyvae.evaluation.zeroshot import evaluate_all_zeroshot

PROCESSED = Path("data/processed")
CHECKPOINTS = Path("data/checkpoints")
RESULTS = Path("data/results")


def supervised_catboost_auc(features: np.ndarray, labels: np.ndarray, top_n: int = 10) -> dict:
    """Purely supervised classifier on raw features: one multiclass CatBoost,
    macro one-vs-rest AUC over classes with both pos/neg examples."""
    clf = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.1, verbose=0, random_seed=42,
        loss_function="MultiClass",
    )
    clf.fit(features, labels)
    proba = clf.predict_proba(features)  # (N, n_classes)
    model_classes = clf.classes_.astype(int)

    per_class_auc = {}
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    for col, cls in enumerate(model_classes):
        y_bin = (labels == cls).astype(int)
        if y_bin.sum() < 2 or (1 - y_bin).sum() < 2:
            continue
        try:
            per_class_auc[int(cls)] = float(roc_auc_score(y_bin, proba[:, col]))
        except Exception:
            continue
    macro = float(np.mean(list(per_class_auc.values()))) if per_class_auc else float("nan")
    top_classes = np.argsort(counts)[::-1][:top_n]
    top_auc = {int(c): per_class_auc[int(c)] for c in top_classes if int(c) in per_class_auc}
    return {
        "auc_macro": macro,
        "top_class_auc": top_auc,
        "n_classes_evaluated": len(per_class_auc),
    }


def mean_r2(zeroshot_result: dict) -> float:
    vals = [r["r2"] for r in zeroshot_result.values() if r["r2"] == r["r2"]]  # drop NaN
    return float(np.mean(vals)) if vals else float("nan")


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    features = torch.load(PROCESSED / "train_features.pt")          # (N, 6)
    labels = torch.load(PROCESSED / "train_labels.pt")              # (N,)
    embeddings = torch.load(CHECKPOINTS / "unsupervised_embeddings.pt")  # (N, 10)
    targets = torch.load(PROCESSED / "zeroshot_targets.pt")

    feats_np = features.numpy()
    labels_np = labels.numpy().astype(int)

    print("=== E1: classification (AUC macro) ===")
    emb_cls = evaluate_classification(embeddings, labels)
    print(f"  embeddings + linear probe: {emb_cls['auc_macro']:.4f}")
    raw_linear = evaluate_classification(features, labels)
    print(f"  raw 6 inputs + linear probe: {raw_linear['auc_macro']:.4f}")
    raw_supervised = supervised_catboost_auc(feats_np, labels_np)
    print(f"  raw 6 inputs + supervised CatBoost: {raw_supervised['auc_macro']:.4f}")

    print("\n=== E2: zero-shot (mean R^2 over targets) ===")
    print("  embeddings ->")
    emb_zs = evaluate_all_zeroshot(embeddings, targets)
    print("  raw 6 inputs ->")
    raw_zs = evaluate_all_zeroshot(features, targets)
    emb_zs_mean, raw_zs_mean = mean_r2(emb_zs), mean_r2(raw_zs)

    out = {
        "E1_classification_auc_macro": {
            "embeddings_linear_probe": emb_cls["auc_macro"],
            "raw_inputs_linear_probe": raw_linear["auc_macro"],
            "raw_inputs_supervised_catboost": raw_supervised["auc_macro"],
            "n_classes_evaluated_embeddings": emb_cls["n_classes_evaluated"],
        },
        "E2_zeroshot_mean_r2": {
            "embeddings": emb_zs_mean,
            "raw_inputs": raw_zs_mean,
            "n_targets": len(targets),
            "per_target_embeddings": {k: v["r2"] for k, v in emb_zs.items()},
            "per_target_raw_inputs": {k: v["r2"] for k, v in raw_zs.items()},
        },
    }
    (RESULTS / "rebuttal_baselines.json").write_text(json.dumps(out, indent=2))

    print("\n\n=== SUMMARY (markdown) ===")
    print("| Task | Raw 6 inputs | Our embeddings |")
    print("|---|---|---|")
    print(f"| Lithology classification, macro AUC (supervised CatBoost) | "
          f"{raw_supervised['auc_macro']:.3f} | {emb_cls['auc_macro']:.3f} |")
    print(f"| Lithology classification, macro AUC (linear probe) | "
          f"{raw_linear['auc_macro']:.3f} | {emb_cls['auc_macro']:.3f} |")
    print(f"| Zero-shot prediction, mean R^2 | {raw_zs_mean:.3f} | {emb_zs_mean:.3f} |")
    print(f"\nWrote {RESULTS / 'rebuttal_baselines.json'}")


if __name__ == "__main__":
    main()
