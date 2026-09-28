"""Check the AGU front-matter limits that the submission system enforces.

Three Key Points of at most 140 characters each, an abstract of at most
250 words, a plain language summary of at most 200 words, and the
presence of Open Research, acknowledgments and a corresponding author.
Exits non-zero if any of those fail, so `check_paper_build.sh` can gate
on it.

Usage:  python scripts/check_agu_frontmatter.py [path/to/main.tex]
"""
import re, sys, pathlib

DEFAULT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "an-unsupervised-embeddings-model-for-ocean-crust-lithology"
    / "jgr_ml.tex"
)
path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
tex = path.read_text()

def block(env):
    m = re.search(r"\\begin\{%s\}(.*?)\\end\{%s\}" % (env, env), tex, re.S)
    return m.group(1) if m else ""

def section_star(name):
    m = re.search(r"\\section\*\{%s\}(.*?)(?=\\section)" % name, tex, re.S)
    return m.group(1) if m else ""

def words(s):
    s = re.sub(r"%.*", "", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$\\]", " ", s)
    return len([w for w in s.split() if any(c.isalnum() for c in w)])

fail = 0

kps = [k.strip() for k in block("keypoints").split(r"\item") if k.strip()]
print(f"Key Points: {len(kps)} (AGU requires 3)")
if len(kps) != 3:
    fail = 1
for i, k in enumerate(kps, 1):
    flat = " ".join(k.split())
    status = "ok" if len(flat) <= 140 else "TOO LONG"
    print(f"  {i}. {len(flat):>3}/140 chars  {status}")
    print(f"     {flat}")
    if len(flat) > 140:
        fail = 1

for label, text, limit in (
    ("Abstract", block("abstract"), 250),
    ("Plain Language Summary", section_star("Plain Language Summary"), 200),
):
    n = words(text)
    status = "ok" if n <= limit else "TOO LONG"
    print(f"{label}: {n}/{limit} words  {status}")
    if n > limit:
        fail = 1

for required in (r"\\section\*\{Open Research\}", r"\\acknowledgments",
                 r"\\correspondingauthor"):
    if not re.search(required, tex):
        print(f"MISSING: {required}")
        fail = 1




def check_sample_counts(tex: str) -> int:
    """Numbers written into prose drift from the data that produced them.

    The observation count appeared twice in this manuscript and both
    were stale after the physical-range filter changed it. A hardcoded
    figure that no longer matches the tensors is the kind of error that
    survives every proofread, so it is checked rather than trusted.
    """
    import re

    try:
        import torch

        from lilyvae.config import PROCESSED_DIR

        actual = len(torch.load(
            PROCESSED_DIR / "train_labels.pt", weights_only=True
        ))
    except Exception as e:                       # no data present, e.g. in CI
        print(f"sample-count check skipped: {e}")
        return 0

    formatted = {f"{actual:,}", str(actual)}
    found = set(re.findall(r"\b\d{3},\d{3}\b", tex))
    stale = {f for f in found if f not in formatted}
    if stale:
        print(f"FAIL: manuscript cites {sorted(stale)} observations; "
              f"the processed data has {actual:,}")
        return 1
    if found:
        print(f"Sample count: {actual:,} matches the manuscript  ok")
    return 0


fail |= check_sample_counts(tex)



def check_floats_are_referenced(tex: str) -> int:
    """Every labelled float must be cited from the prose.

    The previous submission labelled a correlation figure and never
    referred to it, and a reviewer listed that among the reasons the
    paper was hard to follow. LaTeX does not warn about a label nobody
    uses, so it is checked here.
    """
    import re

    labels = set(re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", tex))
    refs = set(re.findall(r"\\(?:ref|autoref|Cref|cref)\{([^}]+)\}", tex))
    orphans = sorted(labels - refs)
    if orphans:
        print("FAIL: labelled floats never referenced from the text: "
              + ", ".join(orphans))
        return 1
    if labels:
        print(f"Floats: all {len(labels)} labelled float(s) referenced  ok")
    return 0


fail |= check_floats_are_referenced(tex)



def check_citations_resolve(tex_path) -> int:
    """Every \\cite key must exist in the bibliography.

    BibTeX reports this too, but only after a full four-pass build, and
    the message is easy to miss among its other warnings. Checking the
    source directly gives the answer immediately and names the key.
    """
    import re
    from pathlib import Path

    bib = Path(tex_path).parent / "references.bib"
    if not bib.exists():
        return 0
    known = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib.read_text()))
    cited = set()
    for group in re.findall(r"\\cite[A-Za-z]*\*?(?:\[[^\]]*\])*\{([^}]+)\}", tex):
        cited.update(k.strip() for k in group.split(","))
    missing = sorted(cited - known)
    if missing:
        print(f"FAIL: cited keys absent from references.bib: {missing}")
        return 1
    if cited:
        print(f"Citations: all {len(cited)} cited key(s) present in the bib  ok")
    return 0


fail |= check_citations_resolve(path)
sys.exit(fail)
