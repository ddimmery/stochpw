main: coverage build_docs examples serve

build_docs:
	PYTHONPATH=src uv run python -m mkdocs build

serve:
	PYTHONPATH=src uv run python -m mkdocs serve

typecheck:
	uv run python -m pyright src/ examples/

lint:
	uv run ruff check --fix && uv run ruff format

test:
	PYTHONPATH=src uv run python -m pytest tests

coverage:
	PYTHONPATH=src uv run python -m coverage run --source=stochpw -m pytest tests && uv run python -m coverage report -m && uv run python -m coverage html

examples:
	PYTHONPATH=src uv run python docs/gen_examples.py

.PHONY: main build_docs serve typecheck lint test coverage examples
