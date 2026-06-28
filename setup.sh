#!/bin/bash
# Harmonia — environment setup
# Run once: bash setup.sh

set -e

PYTHON=${PYTHON:-python3.11}
VENV_DIR=".venv"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Harmonia — setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Check Python version
echo "→ Checking Python..."
$PYTHON -c "import sys; assert sys.version_info >= (3,10), f'Python 3.10+ required, got {sys.version}'"
echo "  OK: $($PYTHON --version)"

# 2. Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment at $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "→ Virtual environment already exists at $VENV_DIR"
fi

# 3. Activate
source "$VENV_DIR/bin/activate"
echo "→ Activated: $(which python)"

# 4. Upgrade pip
pip install --upgrade pip -q

# 5. Install harmonia + dev deps
echo "→ Installing harmonia[dev] ..."
pip install -e ".[dev]" -q
echo "  Done."

# 6. Check FluidSynth (optional, for MIDI rendering)
echo "→ Checking FluidSynth (optional, needed for data pipeline)..."
if command -v fluidsynth &> /dev/null; then
    echo "  OK: $(fluidsynth --version 2>&1 | head -1)"
else
    echo "  NOT FOUND — install with:"
    echo "    macOS:  brew install fluidsynth && pip install pyfluidsynth"
    echo "    Linux:  apt install fluidsynth && pip install pyfluidsynth"
fi

# 7. Create data directories
echo "→ Creating data directories..."
mkdir -p data/{soundfonts,cache,renders,pop909,maestro}
touch data/soundfonts/.gitkeep

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Setup complete."
echo ""
echo " Next steps:"
echo "   source .venv/bin/activate"
echo "   make download-pop909      # get training data"
echo "   make test                 # run tests"
echo "   make infer FILE=song.wav  # transcribe audio"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
