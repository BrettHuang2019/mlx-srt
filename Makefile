VENV := .venv
PY := $(VENV)/bin/python

.PHONY: install test test-integration lint fix check clean

install:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -e '.[test,dev]'

test:
	$(VENV)/bin/pytest

test-integration:
	$(VENV)/bin/pytest -m integration

# Linter only, no auto-formatter: the compact hand-wrapped style is deliberate.
lint:
	$(VENV)/bin/ruff check src tests

fix:
	$(VENV)/bin/ruff check --fix src tests

check: lint test

clean:
	rm -rf build dist .pytest_cache .ruff_cache src/*.egg-info
	find src tests -name '__pycache__' -type d -exec rm -rf {} +
