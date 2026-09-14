#!/bin/bash
# Fetch dependencies for the accompaniment-database pipeline:
#   1. MMA (Musical MIDI Accompaniment) → data/tools/
#   2. iReal Pro chart corpora (jazz1460 / pop400 / blues50) → data/ireal/
# Run once from the repo root: bash scripts/fetch_accompaniment_deps.sh

set -e

MMA_VERSION="25.05.3"
TOOLS_DIR="data/tools"
IREAL_DIR="data/ireal"
PYTHON="${PYTHON:-.venv/bin/python}"

mkdir -p "$TOOLS_DIR" "$IREAL_DIR"

# ── 1. MMA ─────────────────────────────────────────────────────────────────────
if [ ! -d "$TOOLS_DIR/mma-bin-$MMA_VERSION" ]; then
    echo "→ Downloading MMA $MMA_VERSION ..."
    curl -sL -o "$TOOLS_DIR/mma.tar.gz" "https://www.mellowood.ca/mma/mma-devl.$MMA_VERSION.tar.gz"
    tar xzf "$TOOLS_DIR/mma.tar.gz" -C "$TOOLS_DIR"
    rm "$TOOLS_DIR/mma.tar.gz"
    echo "→ Building MMA groove database ..."
    (cd "$TOOLS_DIR/mma-bin-$MMA_VERSION" && "$OLDPWD/$PYTHON" mma.py -G | tail -2)
else
    echo "→ MMA already present at $TOOLS_DIR/mma-bin-$MMA_VERSION"
fi

# ── 2. iReal corpora (from ireal-musicxml's public test data) ──────────────────
BASE="https://raw.githubusercontent.com/infojunkie/ireal-musicxml/main/test/data"
for f in jazz1460 pop400 blues50 country dixieland1; do
    if [ ! -f "$IREAL_DIR/$f.txt" ]; then
        echo "→ Downloading $f.txt ..."
        curl -sL -o "$IREAL_DIR/$f.txt" "$BASE/$f.txt"
    else
        echo "→ $f.txt already present"
    fi
done

echo "Done. Build the database with:"
echo "  $PYTHON scripts/build_accompaniment_db.py"
