"""Lithology classification evaluation using per-class SVM on embeddings."""

import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score


def evaluate_classification(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    top_n: int = 10,
) -> dict:
    """Train per-class one-vs-rest SVMs on embeddings and compute AUC.

    Each lithology class gets its own binary SVM. Only classes with both
    positive and negative examples in the data are evaluated.

    Args:
        embeddings: (N, latent_dim) tensor
        labels: (N,) tensor of integer labels
        top_n: Number of most common classes to report individual AUC for.
    """
    z = embeddings.numpy()
    y = labels.numpy()
    classes = np.unique(y)
    class_counts = np.bincount(y, minlength=max(classes) + 1)

    per_class_auc = {}
    per_class_decision = {}

    for cls in classes:
        y_binary = (y == cls).astype(int)
        # Need both positive and negative examples
        if y_binary.sum() < 2 or (1 - y_binary).sum() < 2:
            continue

        svm = LinearSVC(max_iter=10000, random_state=42, dual="auto")
        try:
            svm.fit(z, y_binary)
            decision = svm.decision_function(z)
            auc = roc_auc_score(y_binary, decision)
            per_class_auc[int(cls)] = float(auc)
            per_class_decision[int(cls)] = decision
        except Exception:
            continue

    # Macro AUC over all evaluated classes
    if per_class_auc:
        macro_auc = float(np.mean(list(per_class_auc.values())))
    else:
        macro_auc = float("nan")

    # Top N most common classes
    top_classes = np.argsort(class_counts)[::-1][:top_n]
    top_class_auc = {
        int(cls): per_class_auc.get(int(cls), float("nan"))
        for cls in top_classes
        if int(cls) in per_class_auc
    }

    return {
        "auc_macro": macro_auc,
        "per_class_auc": per_class_auc,
        "top_class_auc": top_class_auc,
        "n_classes_evaluated": len(per_class_auc),
        "classes": classes,
    }
