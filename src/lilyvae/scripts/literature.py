"""CLI for the literature ledger.

    uv run lilyvae-literature check --doi 10.x/y --title "..." \
        --mechanism porosity --note "states the mass-balance relation"
    uv run lilyvae-literature status
    uv run lilyvae-literature bibtex > refs_validated.bib

`check` resolves the DOI at Crossref and compares the returned title
against the one claimed. Only entries where both agree reach the
bibliography.
"""

from __future__ import annotations

import argparse

from lilyvae.config import RESULTS_DIR
from lilyvae.literature.crossref import validate
from lilyvae.literature.ledger import MECHANISMS, LiteratureLedger

DEFAULT_LEDGER = RESULTS_DIR / "literature.json"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--ledger", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="validate one DOI and record it")
    c.add_argument("--doi", required=True)
    c.add_argument("--title", required=True, help="the title you believe it has")
    c.add_argument("--mechanism", default="", choices=[""] + list(MECHANISMS))
    c.add_argument("--note", default="")

    sub.add_parser("status", help="what is validated and what is still missing")
    sub.add_parser("bibtex", help="emit BibTeX for validated entries only")

    args = p.parse_args()
    ledger = LiteratureLedger(args.ledger or DEFAULT_LEDGER)

    if args.cmd == "check":
        ref = validate(args.doi, args.title, args.mechanism, args.note)
        added = ledger.add(ref)
        mark = {"validated": "OK ", "mismatch": "NO ",
                "not_found": "NO ", "error": "?? "}[ref.status]
        print(f"{mark}{ref.doi}")
        print(f"    claimed:  {ref.claimed_title[:78]}")
        print(f"    crossref: {ref.crossref_title[:78] or '(nothing)'}")
        print(f"    status:   {ref.status}  (title similarity "
              f"{ref.title_similarity:.2f})")
        if ref.message:
            print(f"    note:     {ref.message[:90]}")
        if ref.ok:
            print(f"    key:      {ref.citation_key()}")
        if not added:
            print("    (already in the ledger)")
        return 0 if ref.ok else 1

    if args.cmd == "status":
        print(ledger.summary())
        return 0

    print(ledger.bibtex(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
