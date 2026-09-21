.PHONY: setup test lint

setup:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests