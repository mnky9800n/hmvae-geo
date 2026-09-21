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
    r2_median: float = float("nan")
    n_folds: int = 0
    elapsed_seconds: float = 0.0
    proposed_by: str = "random"
    rationale: str = ""
    error: str = ""

    @property
    def objective(self) -> float:
        """Scalar used only for ordering the ledger.

        ARI and AUC live on different scales: ARI sits near 0.26 with an
        interquartile width around 0.08, AUC near 0.96 with a width
        around 0.01. Adding them lets AUC set the level while ARI sets
        the variance, which is why this is a convenience for sorting and
        not the thing to optimise blindly. Report both, and look at the
        trade-off rather than the sum.
        """
        if self.error:
            return float("-inf")
        return 0.5 * self.ari_median + 0.5 * self.auc_median


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
            f"{'#':>3} {'ARI':>7} {'AUC':>7} {'R2':>7}  {'latent':>6} "
            f"{'hidden':>14} {'beta':>8} {'mask':>5} {'lr':>8} {'ep':>4}  by",
        ]
        for t in sorted(trials, key=lambda t: t.objective, reverse=True)[:top]:
            c = t.config
            lines.append(
                f"{t.index:>3} {t.ari_median:>7.4f} {t.auc_median:>7.4f} "
                f"{t.r2_median:>7.4f}  {c['latent_dim']:>6} "
                f"{str(tuple(c['hidden_dims'])):>14} {c['beta_end']:>8.4f} "
                f"{c['mask_fraction']:>5.2f} {c['learning_rate']:>8.5f} "
                f"{c['num_epochs']:>4}  {t.proposed_by}"
            )
        if failed:
            lines += ["", "failed:"]
            lines += [f"  #{t.index}: {t.error[:100]}" for t in failed[-5:]]
        return "\n".join(lines)
