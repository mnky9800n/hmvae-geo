"""Store validated references against the physical link each supports.

The discussion has to explain why six routinely logged measurements
carry information about four properties nobody measured. That is a claim
about physics, and it needs references that state the coupling, not
references that merely mention the words.

Each entry is therefore filed under the mechanism it supports, and the
gaps are as informative as the entries: a mechanism with no validated
reference is one the paper cannot yet justify from the literature.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from lilyvae.literature.crossref import Reference, to_bibtex

#: The couplings the discussion needs to support, phrased as the
#: physical question a reference has to answer.
MECHANISMS: dict[str, str] = {
    "porosity": (
        "Why bulk density constrains porosity: the mass balance between "
        "grain, pore fluid and bulk density in saturated sediment."
    ),
    "thermal_conductivity": (
        "Why thermal conductivity follows porosity and mineralogy: mixing "
        "of the conductivity of the solid framework and the pore water."
    ),
    "p_wave": (
        "Why compressional velocity follows porosity and density in "
        "unconsolidated marine sediment."
    ),
    "magnetisation": (
        "Why remanent magnetisation tracks magnetic susceptibility: both "
        "scale with the concentration of ferrimagnetic minerals."
    ),
    "ngr_clay": (
        "Why natural gamma radiation indexes clay and terrigenous input "
        "through K, Th and U, rather than organic content."
    ),
    "colour_carbonate": (
        "Why core colour tracks carbonate against clay content, which is "
        "what makes reflectance informative about lithology."
    ),
    "drilling_overprint": (
        "Why the drill string imparts a steep downward remanence, which "
        "is why inclination here is not a palaeomagnetic signal."
    ),
    "evaluation_protocol": (
        "Why blocked cross-validation is required when samples carry "
        "spatial or hierarchical structure."
    ),
}


class LiteratureLedger:
    """Append-only JSON record of checked references."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def all(self) -> list[Reference]:
        if not self.path.exists():
            return []
        return [Reference(**r) for r in json.loads(self.path.read_text())]

    def save(self, refs: list[Reference]) -> None:
        self.path.write_text(json.dumps([asdict(r) for r in refs], indent=2))

    def add(self, ref: Reference) -> bool:
        """Record a reference. Returns False if the DOI is already present."""
        refs = self.all()
        if any(r.doi.lower() == ref.doi.lower() for r in refs):
            return False
        refs.append(ref)
        self.save(refs)
        return True

    def validated(self) -> list[Reference]:
        return [r for r in self.all() if r.ok]

    def by_mechanism(self) -> dict[str, list[Reference]]:
        out: dict[str, list[Reference]] = {m: [] for m in MECHANISMS}
        for ref in self.validated():
            out.setdefault(ref.mechanism or "unfiled", []).append(ref)
        return out

    def bibtex(self) -> str:
        entries = [to_bibtex(r) for r in self.validated()]
        return "\n\n".join(entries) + ("\n" if entries else "")

    def summary(self) -> str:
        refs = self.all()
        good = [r for r in refs if r.ok]
        bad = [r for r in refs if not r.ok]
        lines = [f"{len(good)} validated, {len(bad)} rejected, "
                 f"{len(refs)} checked in total.", ""]

        for mechanism, question in MECHANISMS.items():
            found = [r for r in good if r.mechanism == mechanism]
            mark = "  " if found else "!!"
            lines.append(f"{mark} {mechanism}  ({len(found)} reference(s))")
            lines.append(f"     {question}")
            for r in found:
                first = r.authors[0].split()[-1] if r.authors else "?"
                lines.append(f"       - {first} {r.year}: {r.crossref_title[:66]}")
            lines.append("")

        if bad:
            lines.append("rejected:")
            for r in bad:
                lines.append(f"  [{r.status}] {r.doi}  {r.message[:80]}")
        missing = [m for m in MECHANISMS if not any(r.mechanism == m for r in good)]
        if missing:
            lines.append("")
            lines.append("Mechanisms with no validated reference, so not yet "
                         "defensible from the literature:")
            lines += [f"  {m}" for m in missing]
        return "\n".join(lines)
