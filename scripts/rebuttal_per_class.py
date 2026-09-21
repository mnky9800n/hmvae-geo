"""Per-class F1/recall for DQpL Q2 (does supervision help specific/minority classes?)
and mKnD Q2 (class imbalance).

Same probe (one-vs-rest LinearSVC, matching the paper) on the unsupervised vs
semi-supervised embeddings; compares per-class recall/F1, with attention to
minority classes. In-sample, consistent with the paper's classification eval.
Writes data/results/rebuttal_per_class.json.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support

PROCESSED = Path("data/processed")
CHECKPOINTS = Path("data/checkpoints")
RESULTS = Path("data/results")


def per_class_metrics(emb: np.ndarray, y: np.ndarray) -> dict:
    clf = LinearSVC(max_iter=5000, random_state=42, dual="auto")
    clf.fit(emb, y)
    y_pred = clf.predict(emb)
    labels = np.unique(y)
    prec, rec, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=labels, zero_division=0
    )
    return {
        int(c): {"precision": float(prec[i]), "recall": float(rec[i]),
                 "f1": float(f1[i]), "support": int(support[i])}
        for i, c in enumerate(labels)
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    y = torch.load(PROCESSED / "train_labels.pt").numpy().astype(int)
    emb_unsup = torch.load(CHECKPOINTS / "unsupervised_embeddings.pt").numpy()
    emb_semi = torch.load(CHECKPOINTS / "semi_supervised_embeddings.pt").numpy()

    print("unsupervised probe ...")
    unsup = per_class_metrics(emb_unsup, y)
    print("semi-supervised probe ...")
    semi = per_class_metrics(emb_semi, y)

    classes = sorted(unsup.keys())
    supports = np.array([unsup[c]["support"] for c in classes])
    order = np.argsort(supports)  # ascending: minority first

    def macro_f1(d, subset):
        return float(np.mean([d[c]["f1"] for c in subset]))

    all_c = classes
    minority = [classes[i] for i in order[: max(1, len(classes) // 2)]]   # rarer half
    majority = [classes[i] for i in order[len(classes) // 2:]]

    summary = {
        "macro_f1_all": {"unsup": macro_f1(unsup, all_c), "semi": macro_f1(semi, all_c)},
        "macro_f1_minority_half": {"unsup": macro_f1(unsup, minority), "semi": macro_f1(semi, minority)},
        "macro_f1_majority_half": {"unsup": macro_f1(unsup, majority), "semi": macro_f1(semi, majority)},
        "n_classes": len(classes),
        "min_support": int(supports.min()),
        "max_support": int(supports.max()),
    }
    # how many minority classes does supervision help / hurt on recall?
    help_rec = sum(1 for c in minority if semi[c]["recall"] > unsup[c]["recall"] + 1e-9)
    hurt_rec = sum(1 for c in minority if semi[c]["recall"] < unsup[c]["recall"] - 1e-9)
    summary["minority_recall_semi_vs_unsup"] = {"semi_better": help_rec, "unsup_better": hurt_rec,
                                                "n_minority": len(minority)}

    out = {"summary": summary, "per_class_unsup": unsup, "per_class_semi": semi}
    (RESULTS / "rebuttal_per_class.json").write_text(json.dumps(out, indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
