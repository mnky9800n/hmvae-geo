"""Literature support for the discussion, with every DOI checked."""

from lilyvae.literature.crossref import Reference, to_bibtex, validate
from lilyvae.literature.ledger import MECHANISMS, LiteratureLedger

__all__ = ["Reference", "validate", "to_bibtex", "LiteratureLedger", "MECHANISMS"]
