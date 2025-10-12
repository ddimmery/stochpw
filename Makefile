main: coverage build_docs examples serve

build_docs:
	PYTHONPATH=src poetry run python -m mkdocs build

serve:
	PYTHONPATH=src poetry run python -m mkdocs serve

typecheck:
	poetry run python -m pyright src/ examples/

lint:
	poetry run ruff check --fix && poetry run ruff format

test:
	PYTHONPATH=src poetry run python -m pytest tests

coverage:
	PYTHONPATH=src poetry run python -m coverage run --source=stochpw -m pytest tests && poetry run python -m coverage report -m && poetry run python -m coverage html

examples:
	PYTHONPATH=src poetry run python docs/gen_examples.py

.PHONY: main build_docs serve typecheck lint test coverage examples
