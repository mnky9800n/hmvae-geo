#!/usr/bin/env bash
#
# Build the manuscript and fail on any unresolved citation or reference.
#
# The NeurIPS submission went out with 18 occurrences of `\\citep{...}`
# -- a double backslash, which LaTeX renders as a forced line break
# followed by the literal text `citep{key}`. pdflatex exits 0 on that,
# so nothing caught it and two reviewers reported the result as
# "citations not properly referenced" and "numerous LaTeX formatting
# issues". This script is the check that would have caught it.
#
# Usage:  bash scripts/check_paper_build.sh [main-tex-basename]
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/an-unsupervised-embeddings-model-for-ocean-crust-lithology"
MAIN="${1:-jgr_ml}"

cd "$PAPER_DIR" || { echo "no paper directory at $PAPER_DIR"; exit 1; }
[ -f "$MAIN.tex" ] || { echo "no such file: $PAPER_DIR/$MAIN.tex"; exit 1; }

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

run_tex() {
    pdflatex -interaction=nonstopmode -halt-on-error \
             -output-directory="$BUILD" "$MAIN.tex" >/dev/null 2>&1
}

echo "building $MAIN.tex"
if ! run_tex; then
    echo
    echo "FAIL: pdflatex could not compile $MAIN.tex"
    grep -A4 -m3 '^!' "$BUILD/$MAIN.log" 2>/dev/null || tail -30 "$BUILD/$MAIN.log"
    exit 1
fi

# bibtex resolves the .bib against the .aux; it must run from the
# directory holding references.bib so \bibliography{references} resolves.
( cd "$BUILD" && BIBINPUTS="$PAPER_DIR:" bibtex "$MAIN" >/dev/null 2>&1 )
run_tex; run_tex   # two more passes to settle citations and cross-refs

LOG="$BUILD/$MAIN.log"
BLG="$BUILD/$MAIN.blg"
status=0

report() {   # report <label> <file> <pattern>
    local hits
    hits=$(grep -E "$3" "$2" 2>/dev/null | sort -u)
    if [ -n "$hits" ]; then
        echo
        echo "FAIL: $1"
        echo "$hits" | sed 's/^/    /'
        status=1
    fi
}

report "undefined citations"  "$LOG" "Citation .* undefined"
report "undefined references" "$LOG" "Reference .* undefined"
report "undefined labels"     "$LOG" "LaTeX Warning: There were undefined references"
report "bibtex errors"        "$BLG" "I (was|couldn't)|Warning--(I didn't|empty)"

# A line that runs into the margin is visible on the page, and in draft
# mode the class draws a black rule beside it. Small overshoots are not
# worth chasing, so only report the ones a reader would see.
worst=$(grep -oE 'Overfull \\hbox \(([0-9.]+)pt too wide' "$LOG" 2>/dev/null \
        | grep -oE '[0-9.]+' | sort -gr | head -1)
if [ -n "$worst" ] && [ "$(printf '%.0f' "$worst")" -ge 5 ]; then
    echo
    echo "FAIL: text runs into the margin, worst line ${worst}pt too wide"
    grep -E 'Overfull \\hbox \([0-9]{2,}\.?[0-9]*pt too wide' "$LOG" \
        | sed 's/^/    /' | head -10
    status=1
fi

# A stray `\\cite` renders as a line break plus literal text, which no
# LaTeX warning reports. Catch it in the source instead.
if grep -n '\\\\cite' "$PAPER_DIR/$MAIN.tex" >/dev/null 2>&1; then
    echo
    echo "FAIL: escaped \\\\cite in $MAIN.tex (renders as a line break, not a citation)"
    grep -n '\\\\cite' "$PAPER_DIR/$MAIN.tex" | sed 's/^/    /'
    status=1
fi

# apacite (loaded by agujournal2019.cls) provides \cite and \citeA, not
# the natbib \citep/\citet. Using those against this class is a build
# error waiting to happen.
if grep -nE '\\cite[pt]\{' "$PAPER_DIR/$MAIN.tex" >/dev/null 2>&1; then
    echo
    echo "FAIL: natbib \\citep/\\citet in $MAIN.tex; this class uses apacite (\\cite, \\citeA)"
    grep -nE '\\cite[pt]\{' "$PAPER_DIR/$MAIN.tex" | sed 's/^/    /'
    status=1
fi

# AGU enforces the front-matter limits at submission, so check them here
# rather than discovering them in the submission form.
FRONTMATTER_PY=(python3)
if command -v uv >/dev/null 2>&1 && [ -d "$SCRIPT_DIR/../.venv" ]; then
    FRONTMATTER_PY=(uv --project "$SCRIPT_DIR/.." run python)
fi
if ! "${FRONTMATTER_PY[@]}" "$SCRIPT_DIR/check_agu_frontmatter.py" \
        "$PAPER_DIR/$MAIN.tex"; then
    echo
    echo "FAIL: AGU front-matter limits"
    status=1
fi

if [ "$status" -eq 0 ]; then
    pages=$(tr -d '\n' < "$LOG" | grep -oE '\([0-9]+ pages' \
             | grep -oE '[0-9]+ pages' | tail -1 || echo "? pages")
    cites=$(grep -c '^\\bibitem' "$BUILD/$MAIN.bbl" 2>/dev/null || echo 0)
    echo "OK: $MAIN.tex builds clean, $pages, $cites references resolved"
fi
exit "$status"
