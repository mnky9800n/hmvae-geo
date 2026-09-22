"""Verify a DOI against Crossref before it is allowed near the bibliography.

A language model asked for a citation will produce something that looks
exactly like a citation. The failure is not that the DOI is malformed,
it is that the DOI resolves to a real paper that is not the one being
cited. Checking only that a DOI exists catches nothing.

So two checks run here. The DOI must resolve at Crossref, and the title
Crossref returns must match the title the citation claims. A resolvable
DOI attached to the wrong title is recorded as a mismatch and kept out
of the bibliography, because that is the failure mode that survives
casual proofreading and reaches print.

No API key is needed. Crossref asks for a contact address in the user
agent so they can get in touch about heavy use, which is provided.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher

CROSSREF_WORK = "https://api.crossref.org/works/"
USER_AGENT = (
    "lilyvae-literature/1.0 (https://github.com/mnky9800n/hmvae-geo; "
    "mailto:johnm.aiken@gmail.com)"
)

#: Title similarity below which a resolvable DOI is treated as pointing
#: at a different paper. Titles differ in punctuation, case, subtitle
#: handling and Unicode, so an exact match is too strict; 0.75 on
#: normalised text separates "same paper, formatted differently" from
#: "different paper" in practice.
TITLE_MATCH_FLOOR = 0.75

#: Crossref asks for no more than 50 requests a second and is a free
#: service run on donations. One request every half second is plenty.
REQUEST_INTERVAL_S = 0.5

_last_request = 0.0


@dataclass
class Reference:
    """One candidate citation and what Crossref says about it."""

    doi: str
    claimed_title: str
    mechanism: str = ""
    note: str = ""
    status: str = "unchecked"     # validated | mismatch | not_found | error
    crossref_title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    container: str = ""
    title_similarity: float = 0.0
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "validated"

    def citation_key(self) -> str:
        """A stable BibTeX key: first author surname, year, first title word."""
        surname = "anon"
        if self.authors:
            surname = re.sub(r"[^a-z]", "", self.authors[0].split()[-1].lower())
        word = "x"
        for token in re.findall(r"[A-Za-z]{4,}", self.crossref_title or self.claimed_title):
            if token.lower() not in {"the", "and", "for", "from", "with", "that",
                                     "this", "into", "their", "using"}:
                word = token.lower()
                break
        return f"{surname}{self.year or ''}{word}"


def _normalise(title: str) -> str:
    title = title.lower()
    title = re.sub(r"<[^>]+>", " ", title)       # Crossref returns some MathML
    title = re.sub(r"[^a-z0-9 ]", " ", title)
    return " ".join(title.split())


def _throttle() -> None:
    global _last_request
    wait = REQUEST_INTERVAL_S - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()


def clean_doi(doi: str) -> str:
    """Strip the things people paste around a DOI."""
    doi = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/",
                   "https://dx.doi.org/", "doi:", "DOI:"):
        if doi.lower().startswith(prefix.lower()):
            doi = doi[len(prefix):]
    return doi.strip().rstrip(".,;")


def validate(doi: str, claimed_title: str, mechanism: str = "",
             note: str = "", timeout: float = 20.0) -> Reference:
    """Resolve a DOI at Crossref and check the title matches the claim."""
    ref = Reference(doi=clean_doi(doi), claimed_title=claimed_title.strip(),
                    mechanism=mechanism, note=note)
    url = CROSSREF_WORK + urllib.parse.quote(ref.doi, safe="")

    try:
        _throttle()
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        ref.status = "not_found" if e.code == 404 else "error"
        ref.message = f"HTTP {e.code}"
        return ref
    except Exception as e:                          # network, timeout, bad JSON
        ref.status = "error"
        ref.message = str(e)[:200]
        return ref

    work = payload.get("message", {})
    titles = work.get("title") or []
    ref.crossref_title = titles[0] if titles else ""
    ref.container = (work.get("container-title") or [""])[0]
    ref.authors = [
        " ".join(filter(None, [a.get("given"), a.get("family")]))
        for a in work.get("author", [])
    ]
    for key in ("published-print", "published-online", "issued", "created"):
        parts = work.get(key, {}).get("date-parts") or [[]]
        if parts[0]:
            ref.year = int(parts[0][0])
            break

    ref.title_similarity = SequenceMatcher(
        None, _normalise(ref.claimed_title), _normalise(ref.crossref_title)
    ).ratio()

    if not ref.crossref_title:
        ref.status = "error"
        ref.message = "Crossref returned no title for this DOI"
    elif ref.title_similarity >= TITLE_MATCH_FLOOR:
        ref.status = "validated"
    else:
        ref.status = "mismatch"
        ref.message = (
            f"DOI resolves, but to {ref.crossref_title!r}, "
            f"not {ref.claimed_title!r}"
        )
    return ref


def clean_title(title: str) -> str:
    """Strip the JATS markup Crossref returns inside titles.

    Publishers send small-caps and similar as tags, so a title arrives as
    "Quantifying <scp>K</scp>, <scp>U</scp>...". Left in place that
    reaches the bibliography verbatim.
    """
    title = re.sub(r"</?(scp|i|b|sub|sup|italic|bold)>", "", title)
    title = re.sub(r"<[^>]+>", "", title)
    return " ".join(title.split())


def to_bibtex(ref: Reference) -> str:
    """Emit a BibTeX entry from the Crossref record, not from the claim."""
    if not ref.ok:
        raise ValueError(f"refusing to emit BibTeX for {ref.doi} ({ref.status})")
    authors = " and ".join(ref.authors) if ref.authors else "Unknown"
    fields = [
        ("title", f"{{{clean_title(ref.crossref_title)}}}"),
        ("author", f"{{{authors}}}"),
        ("journal", f"{{{ref.container}}}"),
        ("year", f"{{{ref.year}}}" if ref.year else "{}"),
        ("doi", f"{{{ref.doi}}}"),
    ]
    body = ",\n  ".join(f"{k:<8}= {v}" for k, v in fields)
    return f"@article{{{ref.citation_key()},\n  {body}\n}}"
