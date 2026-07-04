.PHONY: install install-dev test lint format check clean

# ── Environment ────────────────────────────────────────────────────────────────

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,render]"

# ── Quality ────────────────────────────────────────────────────────────────────

lint:
	ruff check harmonia/ tests/
	mypy harmonia/

format:
	black harmonia/ tests/
	ruff check --fix harmonia/ tests/

check: lint test

# ── Tests ──────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-theory:
	pytest tests/test_theory.py -v

# ── Data pipeline ──────────────────────────────────────────────────────────────

download-pop909:
	@echo "Cloning POP909 dataset (~500MB)..."
	git clone https://github.com/music-x-lab/POP909-Dataset data/pop909
	@echo "Done. Run: make parse-pop909"

parse-pop909:
	python -c "\
from harmonia.data.pop909_parser import POP909Parser; \
p = POP909Parser('data/pop909'); \
songs = p.parse_all(); \
print(p.chord_statistics(songs))"

accomp-deps:
	bash scripts/fetch_accompaniment_deps.sh

accomp-db:
	.venv/bin/python scripts/build_accompaniment_db.py

# ── Inference ──────────────────────────────────────────────────────────────────

infer:
	@test -n "$(FILE)" || (echo "Usage: make infer FILE=path/to/audio.wav"; exit 1)
	python scripts/process_audio.py $(FILE)

# ── Clean ──────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
