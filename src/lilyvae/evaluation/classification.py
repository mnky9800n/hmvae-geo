"""Lithology classification evaluation using SVM on embeddings."""

import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier


def evaluate_classification(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    top_n: int = 10,
) -> dict:
    """Train SVM on embeddings and compute per-class ROC AUC.

    Args:
        embeddings: (N, latent_dim) tensor
        labels: (N,) tensor of integer labels
        top_n: Number of most common classes to report individual AUC for.
    """
    z = embeddings.numpy()
    y = labels.numpy()
    classes = np.unique(y)

    # One-vs-rest SVM
    svm = OneVsRestClassifier(
        LinearSVC(max_iter=10000, random_state=42),
        n_jobs=-1,
    )
    y_bin = label_binarize(y, classes=classes)
    svm.fit(z, y_bin)
    decision = svm.decision_function(z)

    # Macro AUC
    try:
        macro_auc = roc_auc_score(y_bin, decision, average="macro")
    except ValueError:
        macro_auc = float("nan")

    # Per-class AUC for top N most common classes
    class_counts = np.bincount(y, minlength=len(classes))
    top_classes = np.argsort(class_counts)[::-1][:top_n]

    per_class_auc = {}
    for idx in top_classes:
        cls = classes[idx]
        try:
            auc = roc_auc_score(y_bin[:, idx], decision[:, idx])
            per_class_auc[int(cls)] = float(auc)
        except ValueError:
            per_class_auc[int(cls)] = float("nan")

    return {
        "auc_macro": float(macro_auc),
        "per_class_auc": per_class_auc,
        "svm": svm,
        "classes": classes,
        "decision": decision,
        "y_bin": y_bin,
    }
