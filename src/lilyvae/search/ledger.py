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
    r2_median: float = float("nan")
    n_folds: int = 0
    elapsed_seconds: float = 0.0
    proposed_by: str = "random"
    rationale: str = ""
    error: str = ""

    @property
    def objective(self) -> float:
        """Scalar used to order the ledger: ARI and *transfer* AUC.

        The probe AUC in `auc_median` is not used. A three-epoch model
        scores 0.9535 against 0.9573 for a fully trained one, so its
        entire headroom is smaller than the fold-to-fold interquartile
        width of about 0.01, and optimising it would chase noise. It is
        the same saturation that made reconstruction R^2 useless as a
        discriminator.

        Transfer AUC sits near 0.82 and drops when a configuration
        stops generalising to a new borehole, so it has both room and
        meaning. ARI ranges from about 0.14 to 0.42 across folds.

        Both terms are near enough in scale to average without one
        dominating, but this remains a sorting convenience. The
        trade-off between the two is the thing to look at.
        """
        if self.error:
            return float("-inf")
        transfer = self.auc_transfer_median
        if transfer != transfer:  # nan, from a trial recorded before this change
            return float("-inf")
        return 0.5 * self.ari_median + 0.5 * transfer


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

    def best(self) -> Trial | None:
        ok = [t for t in self.all() if not t.error]
        return max(ok, key=lambda t: t.objective) if ok else None

    def summary(self, top: int = 10) -> str:
        """Human and agent readable digest, used to propose the next config."""
        trials = [t for t in self.all() if not t.error]
        failed = [t for t in self.all() if t.error]
        if not trials:
            return "No successful trials yet."

        lines = [
            f"{len(trials)} successful trial(s), {len(failed)} failed.",
            "",
            f"{'#':>3} {'ARI':>7} {'trAUC':>7} {'probe':>7} {'R2':>7}  "
            f"{'latent':>6} {'hidden':>14} {'beta':>8} {'mask':>5} "
            f"{'lr':>8} {'ep':>4}  by",
        ]
        for t in sorted(trials, key=lambda t: t.objective, reverse=True)[:top]:
            c = t.config
            lines.append(
                f"{t.index:>3} {t.ari_median:>7.4f} "
                f"{t.auc_transfer_median:>7.4f} {t.auc_median:>7.4f} "
                f"{t.r2_median:>7.4f}  {c['latent_dim']:>6} "
                f"{str(tuple(c['hidden_dims'])):>14} {c['beta_end']:>8.4f} "
                f"{c['mask_fraction']:>5.2f} {c['learning_rate']:>8.5f} "
                f"{c['num_epochs']:>4}  {t.proposed_by}"
            )
        if failed:
            lines += ["", "failed:"]
            lines += [f"  #{t.index}: {t.error[:100]}" for t in failed[-5:]]
        return "\n".join(lines)
