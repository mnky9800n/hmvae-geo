"""Search for configurations that improve clustering and classification.

Three subcommands, designed so an agent can drive the loop:

    uv run lilyvae-search status
        Print the ledger: every configuration tried, ranked, with how
        many attempts produced the current best. Read this before
        proposing.

    uv run lilyvae-search run --config '{"latent_dim": 16, ...}'
        Evaluate one configuration on the search boreholes and append it
        to the ledger. --random draws one instead, for seeding.

    uv run lilyvae-search report --index N
        Score trial N on the reporting boreholes, which the search has
        never seen. Run this once, at the end.

The loop is therefore: status, think, run, repeat; then report. It is
resumable, because the ledger is the only state.

The objective is to maximise ARI subject to transfer AUC staying
within one fold standard deviation of the baseline's. It is a
constraint rather than a weighted sum because ARI varies about four
times as widely across configurations as transfer AUC does, so any
average of the two is ARI in disguise and will trade generalisation
away without saying so.

A warning that belongs in the output rather than a footnote: the best of
N trials is biased upward by an amount that grows with N. `status`
prints the attempt count for that reason, and any improvement worth
keeping has to survive `report` on boreholes the search never touched.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from lilyvae.config import RESULTS_DIR, SEED
from lilyvae.search.ledger import Ledger, Trial
from lilyvae.search.objective import evaluate_config
from lilyvae.search.space import SEARCH_SPACE, SearchConfig, random_config

DEFAULT_LEDGER = RESULTS_DIR / "search_ledger.jsonl"


def _cmd_status(args, ledger: Ledger) -> int:
    print(ledger.summary(top=args.top))
    best = ledger.best()
    if best:
        n = len([t for t in ledger.all() if not t.error])
        floor = ledger.transfer_floor()
        keeps = best.auc_transfer_median >= floor
        print()
        print(f"Best so far is trial #{best.index} out of {n} attempt(s).")
        print(f"  ARI {best.ari_median:.4f}  transfer AUC "
              f"{best.auc_transfer_median:.4f}  (probe AUC {best.auc_median:.4f})")
        if not keeps:
            print("  NOTE: nothing tried yet keeps transfer AUC above the "
                  "floor, so this is the least bad rather than an improvement.")
        print(f"  config: {json.dumps(best.config)}")
        print()
        print("The best of N trials is biased upward by an amount that grows")
        print("with N. Confirm with: uv run lilyvae-search report --index "
              f"{best.index}")
    print()
    print("Search space (the agent may propose outside these bounds):")
    for k, v in SEARCH_SPACE.items():
        print(f"  {k:<18} {v}")
    return 0


def _cmd_run(args, ledger: Ledger) -> int:
    if args.random:
        seen = ledger.seen()
        rng = np.random.RandomState(SEED + ledger.next_index())
        for _ in range(100):
            config = random_config(rng)
            if config.key() not in seen:
                break
        proposed_by, rationale = "random", "unguided draw"
    else:
        if not args.config:
            print("give --config '<json>' or --random")
            return 2
        config = SearchConfig.from_dict(json.loads(args.config))
        proposed_by = args.by
        rationale = args.rationale

    if config.key() in ledger.seen():
        print("This configuration is already in the ledger; nothing to do.")
        return 0

    index = ledger.next_index()
    print(f"trial #{index}: {json.dumps(config.to_dict())}")
    print(f"  proposed by {proposed_by}" + (f": {rationale}" if rationale else ""))

    try:
        res = evaluate_config(
            config, n_folds=args.folds, device=args.device,
            pool="search", report_fraction=args.report_fraction,
        )
        trial = Trial(
            index=index, config=config.to_dict(),
            ari_median=res["ari_median"], auc_median=res["auc_median"],
            ari_iqr=res["ari_iqr"], auc_iqr=res["auc_iqr"],
            auc_transfer_median=res["auc_transfer_median"],
            auc_transfer_iqr=res["auc_transfer_iqr"],
            ari_folds=[f["ari"] for f in res["per_fold"]],
            auc_transfer_folds=[f["auc_transfer"] for f in res["per_fold"]],
            r2_median=res["r2_median"], n_folds=res["n_folds"],
            elapsed_seconds=res["elapsed_seconds"],
            proposed_by=proposed_by, rationale=rationale,
        )
        print(f"  ARI          {res['ari_median']:.4f} IQR {res['ari_iqr']}")
        print(f"  transfer AUC {res['auc_transfer_median']:.4f} "
              f"IQR {res['auc_transfer_iqr']}      <- objective")
        print(f"  probe AUC    {res['auc_median']:.4f} IQR {res['auc_iqr']}"
              "  (saturated, not optimised)")
        print(f"  recon R2     {res['r2_median']:.4f}")
        floor = ledger.transfer_floor()
        if floor > float("-inf"):
            if res["auc_transfer_median"] >= floor:
                print(f"  keeps transfer AUC (floor {floor:.4f})")
            else:
                print(f"  GIVES UP transfer AUC: {res['auc_transfer_median']:.4f} "
                      f"< floor {floor:.4f}; higher ARI here is a trade, "
                      "not an improvement")
        print(f"  {res['elapsed_seconds']/60:.1f} min")
    except Exception as e:
        trial = Trial(
            index=index, config=config.to_dict(),
            ari_median=float("nan"), auc_median=float("nan"),
            proposed_by=proposed_by, rationale=rationale, error=str(e),
        )
        print(f"  FAILED: {e}")

    ledger.append(trial)
    return 0


def _cmd_report(args, ledger: Ledger) -> int:
    trials = {t.index: t for t in ledger.all()}
    if args.index not in trials:
        print(f"no trial #{args.index} in {ledger.path}")
        return 2
    trial = trials[args.index]
    config = SearchConfig.from_dict(trial.config)

    print(f"Scoring trial #{args.index} on the reserved reporting boreholes.")
    print(f"  search ARI {trial.ari_median:.4f}, "
          f"transfer AUC {trial.auc_transfer_median:.4f}")
    res = evaluate_config(
        config, n_folds=args.folds, device=args.device,
        pool="report", report_fraction=args.report_fraction,
    )
    print()
    print(f"  reported ARI          {res['ari_median']:.4f} IQR {res['ari_iqr']}")
    print(f"  reported probe AUC    {res['auc_median']:.4f} IQR {res['auc_iqr']}")
    print(f"  reported transfer AUC {res['auc_transfer_median']:.4f}")
    print(f"  reported recon R2     {res['r2_median']:.4f}")
    drop_ari = trial.ari_median - res["ari_median"]
    drop_auc = trial.auc_transfer_median - res["auc_transfer_median"]
    print()
    print(f"  shrinkage from search to report: ARI {drop_ari:+.4f}, "
          f"transfer AUC {drop_auc:+.4f}")
    print("  A large drop means the search fitted the tuning folds.")

    out = RESULTS_DIR / f"search_report_trial{args.index}.json"
    out.write_text(json.dumps(
        {"trial": trial.config, "search": {
            "ari_median": trial.ari_median, "auc_median": trial.auc_median},
         "report": res}, indent=2, default=float))
    print(f"\nWritten to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ledger", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status", help="print the ledger and the search space")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(fn=_cmd_status)

    p = sub.add_parser("run", help="evaluate one configuration")
    p.add_argument("--config", default=None, help="JSON configuration")
    p.add_argument("--random", action="store_true")
    p.add_argument("--by", default="agent", help="who proposed it")
    p.add_argument("--rationale", default="", help="why, recorded in the ledger")
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--report-fraction", type=float, default=0.2)
    p.set_defaults(fn=_cmd_run)

    p = sub.add_parser("report", help="score one trial on the reserved boreholes")
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--report-fraction", type=float, default=0.2)
    p.set_defaults(fn=_cmd_report)

    args = parser.parse_args()
    ledger = Ledger(args.ledger or DEFAULT_LEDGER)
    return args.fn(args, ledger)


if __name__ == "__main__":
    raise SystemExit(main())
