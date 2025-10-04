main: coverage build_docs serve

build_docs:
	PYTHONPATH=src poetry run mkdocs build

serve:
	PYTHONPATH=src poetry run mkdocs serve

typecheck:
	poetry run pyright src/

lint:
	poetry run ruff check --fix && poetry run ruff format

test:
	PYTHONPATH=src poetry run pytest tests

coverage:
	PYTHONPATH=src poetry run coverage run --source=stochpw -m pytest tests && poetry run coverage report -m && poetry run coverage html
