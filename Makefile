.PHONY: serve golden test migrate-state clean

# ── Run ──────────────────────────────────────────────────────────────────────

serve:
	.venv/bin/python -m harmonia.server

# ── Golden report (rebake the library with the live engine, diff a baseline) ─

golden:
	.venv/bin/python -m tools.golden --engine harmonia --out state/cache/golden/run \
		--baseline state/cache/golden/baseline

# ── Tests ────────────────────────────────────────────────────────────────────

test:
	.venv/bin/python -m pytest tests/ -v

# ── State layout migration (harmonia_min/state -> state/{human,cache}) ──────

migrate-state:
	.venv/bin/python -m tools.migrate_state

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
