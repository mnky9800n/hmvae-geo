"""Append-only record of every configuration tried.

The ledger exists so that a reported improvement can be read alongside
how many attempts produced it. A best-of-N maximum is biased upward by
an amount that grows with N, so the count is part of the result, not
bookkeeping. It also makes the loop resumable and stops a configuration
being evaluated twice.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from lilyvae.search.space import SearchConfig


@dataclass
class Trial:
    """One evaluated configuration."""

    index: int
    config: dict
    ari_median: float
    auc_median: float
    ari_iqr: list[float] = field(default_factory=list)
    auc_iqr: list[float] = field(default_factory=list)
    #: Probe fitted on training boreholes and applied to held-out ones.
    #: This is the objective, not `auc_median`.
    auc_transfer_median: float = float("nan")
    auc_transfer_iqr: list[float] = field(default_factory=list)
    #: Per-fold values, kept so the spread can be compared against the
    #: difference between configurations. A gap smaller than the fold
    #: spread is not a result.
    ari_folds: list[float] = field(default_factory=list)
    auc_transfer_folds: list[float] = field(default_factory=list)
    #: Per-lithology transfer AUC, keyed by encoded class id. The macro
    #: average covers only about 45 of 139 classes on a given fold, so a
    #: configuration can be much better on some lithologies while the
    #: average says it is no better at all.
    per_class_transfer: dict = field(default_factory=dict)
    n_classes_scored: int = 0
    r2_median: float = float("nan")
    n_folds: int = 0
    elapsed_seconds: float = 0.0
    proposed_by: str = "random"
    rationale: str = ""
    error: str = ""

    def _spread(self, folds: list[float], iqr: list[float]) -> float:
        """Fold-to-fold standard deviation, estimated from the IQR if the
        per-fold values predate this field."""
        vals = [v for v in folds if v == v]
        if len(vals) >= 2:
            return float(np.std(vals, ddof=1))
        if len(iqr) == 2 and all(v == v for v in iqr):
            # For a normal sample the IQR spans about 1.349 sigma.
            return float((iqr[1] - iqr[0]) / 1.349)
        return float("nan")

    @property
    def transfer_spread(self) -> float:
        return self._spread(self.auc_transfer_folds, self.auc_transfer_iqr)

    @property
    def ari_spread(self) -> float:
        return self._spread(self.ari_folds, self.ari_iqr)

    def sort_key(self, transfer_floor: float) -> tuple:
        """Rank by ARI, but only among configurations that keep transfer.

        Averaging ARI with transfer AUC does not express the goal. ARI
        varies about four times as widely across configurations as
        transfer AUC does (0.126 against 0.030 over the first round), so
        a mean of the two is ARI wearing a disguise, and it will happily
        trade away generalisation for clustering without saying so. The
        first round showed exactly that: the configurations with the
        best ARI had the worst transfer, correlation -0.85, and no
        configuration beat the baseline on transfer at all.

        So the constraint is stated rather than averaged. A
        configuration is admissible when its transfer AUC has not fallen
        more than one fold standard deviation below the baseline's;
        among those, higher ARI wins. Inadmissible configurations sort
        below every admissible one and are still ranked among
        themselves, so a near miss stays visible.
        """
        if self.error:
            return (-1, float("-inf"), float("-inf"))
        feasible = (
            self.auc_transfer_median == self.auc_transfer_median
            and self.auc_transfer_median >= transfer_floor
        )
        return (1 if feasible else 0, self.ari_median, self.auc_transfer_median)

class Ledger:
    """JSONL trial log."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[Trial]:
        if not self.path.exists():
            return []
        trials = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                trials.append(Trial(**json.loads(line)))
        return trials

    def append(self, trial: Trial) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(trial)) + "\n")

    def seen(self) -> set[str]:
        """Config keys already evaluated, so the loop does not repeat work."""
        return {SearchConfig.from_dict(t.config).key() for t in self.all() if not t.error}

    def next_index(self) -> int:
        return len(self.all())

    #: How far below the baseline's transfer AUC a configuration may fall
    #: and still count as keeping generalisation, in fold standard
    #: deviations.
    TRANSFER_TOLERANCE_SD = 1.0

    def baseline(self) -> Trial | None:
        """The reference configuration the constraint is measured against."""
        for t in self.all():
            if t.proposed_by == "baseline" and not t.error:
                return t
        return None

    def transfer_floor(self) -> float:
        """Lowest transfer AUC a configuration may have and stay admissible.

        Returns -inf when no baseline has been recorded, which leaves
        every configuration admissible rather than silently rejecting
        all of them.
        """
        base = self.baseline()
        if base is None:
            return float("-inf")
        sd = base.transfer_spread
        if sd != sd:
            return base.auc_transfer_median
        return base.auc_transfer_median - self.TRANSFER_TOLERANCE_SD * sd

    def ranked(self) -> list[Trial]:
        floor = self.transfer_floor()
        ok = [t for t in self.all() if not t.error]
        return sorted(ok, key=lambda t: t.sort_key(floor), reverse=True)

    def best(self) -> Trial | None:
        ranked = self.ranked()
        return ranked[0] if ranked else None

    def summary(self, top: int = 10) -> str:
        """Human and agent readable digest, used to propose the next config."""
        trials = [t for t in self.all() if not t.error]
        failed = [t for t in self.all() if t.error]
        if not trials:
            return "No successful trials yet."

        floor = self.transfer_floor()
        base = self.baseline()
        lines = [f"{len(trials)} successful trial(s), {len(failed)} failed."]
        if base is not None:
            lines.append(
                f"Objective: maximise ARI subject to transfer AUC >= {floor:.4f} "
                f"(baseline {base.auc_transfer_median:.4f} minus "
                f"{self.TRANSFER_TOLERANCE_SD:g} fold SD of "
                f"{base.transfer_spread:.4f})."
            )
        else:
            lines.append(
                "No baseline trial recorded, so no transfer constraint is "
                "applied yet. Run one with --by baseline."
            )
        lines += [
            "",
            f"{'':>2}{'#':>3} {'ARI':>7} {'trAUC':>7} {'probe':>7} {'R2':>7}  "
            f"{'latent':>6} {'hidden':>14} {'beta':>8} {'mask':>5} "
            f"{'lr':>8} {'ep':>4}  by",
        ]
        for t in self.ranked()[:top]:
            c = t.config
            keeps = (
                t.auc_transfer_median == t.auc_transfer_median
                and t.auc_transfer_median >= floor
            )
            lines.append(
                f"{'  ' if keeps else ' x'}{t.index:>3} {t.ari_median:>7.4f} "
                f"{t.auc_transfer_median:>7.4f} {t.auc_median:>7.4f} "
                f"{t.r2_median:>7.4f}  {c['latent_dim']:>6} "
                f"{str(tuple(c['hidden_dims'])):>14} {c['beta_end']:>8.4f} "
                f"{c['mask_fraction']:>5.2f} {c['learning_rate']:>8.5f} "
                f"{c['num_epochs']:>4}  {t.proposed_by}"
            )
        lines.append("")
        lines.append("  x marks a configuration that gave up too much transfer AUC.")
        if failed:
            lines += ["", "failed:"]
            lines += [f"  #{t.index}: {t.error[:100]}" for t in failed[-5:]]
        return "\n".join(lines)


#: A per-class transfer AUC gain must exceed this to count as an
#: improvement. Set from the per-class fold spread when it is available
#: and used as a floor otherwise, because a class measured on two folds
#: can show a large apparent gain from noise alone.
MIN_CLASS_GAIN = 0.02

#: A drop larger than this counts as breaking a lithology. Chosen from
#: the observed failure it is meant to catch: raising the KL weight to
#: 0.3 took tuff from 0.615 to 0.152 and ash/tuff from 0.483 to 0.131,
#: which no gain elsewhere should be allowed to buy.
MAX_CLASS_LOSS = 0.05


def class_profile(trial: "Trial", baseline: "Trial") -> dict:
    """Compare a trial to the baseline lithology by lithology.

    The macro average cannot express what matters here. It covers a
    shifting subset of about 45 of the 139 classes, it includes classes
    where the probe is anti-predictive out of borehole, and it lets a
    large gain on one rock cancel a collapse on another. Raising the KL
    weight to 0.3 improved very fine sand by 0.29 and interbedded sand
    and mud by 0.20 while destroying tuff and ash/tuff, and the macro
    number moved by 0.014.

    Only classes both runs scored are compared, so the count is not
    partly a change of denominator. A class counts as improved when the
    gain clears both MIN_CLASS_GAIN and its own fold spread.

    Returns counts of improved, broken and unchanged classes, plus the
    per-class deltas.
    """
    a = baseline.per_class_transfer or {}
    b = trial.per_class_transfer or {}
    common = sorted(set(a) & set(b), key=int)

    deltas, improved, broken = {}, 0, 0
    for cls in common:
        ma, mb = a[cls]["auc_median"], b[cls]["auc_median"]
        delta = mb - ma
        deltas[cls] = delta
        spread = max(
            [v for v in (a[cls].get("auc_sd"), b[cls].get("auc_sd")) if v == v]
            or [0.0]
        )
        if delta > max(MIN_CLASS_GAIN, spread):
            improved += 1
        elif delta < -MAX_CLASS_LOSS:
            broken += 1

    return {
        "n_common": len(common),
        "n_improved": improved,
        "n_broken": broken,
        "n_unchanged": len(common) - improved - broken,
        "deltas": deltas,
    }
