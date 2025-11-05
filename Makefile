.DEFAULT_GOAL := help

# ANSI color codes for better readability
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

# Python path configuration
PYTHONPATH := src

##@ General

.PHONY: help
help: ## Display this help message
	@echo ""
	@echo "$(BLUE)stochpw - Permutation Weighting for Causal Inference$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Usage:\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(YELLOW)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@echo ""

##@ Development

.PHONY: install
install: ## Install package and dependencies
	uv sync

.PHONY: lint
lint: ## Lint and format code with ruff
	uv run ruff check --fix
	uv run ruff format

.PHONY: lint-check
lint-check: ## Check linting without making changes
	uv run ruff check
	uv run ruff format --check

.PHONY: typecheck
typecheck: ## Run type checking with basedpyright
	uv run python -m basedpyright src/ examples/

##@ Testing

.PHONY: test
test: ## Run tests with pytest
	PYTHONPATH=$(PYTHONPATH) uv run python -m pytest tests

.PHONY: test-verbose
test-verbose: ## Run tests with verbose output
	PYTHONPATH=$(PYTHONPATH) uv run python -m pytest tests -v

.PHONY: coverage
coverage: ## Run tests with coverage report
	PYTHONPATH=$(PYTHONPATH) uv run python -m coverage run --source=stochpw -m pytest tests
	uv run python -m coverage report -m
	uv run python -m coverage html
	@echo "$(GREEN)Coverage report generated in htmlcov/index.html$(NC)"

.PHONY: coverage-open
coverage-open: coverage ## Generate coverage report and open in browser
	open htmlcov/index.html || xdg-open htmlcov/index.html

##@ Quality Assurance

.PHONY: check
check: lint-check typecheck test ## Run all checks (lint, typecheck, test)

.PHONY: qa
qa: lint typecheck test ## Run full quality assurance (lint with fixes, typecheck, test)

##@ Documentation

.PHONY: examples
examples: ## Generate example outputs
	PYTHONPATH=$(PYTHONPATH) uv run python docs/gen_examples.py

.PHONY: build-docs
build-docs: examples ## Build documentation with mkdocs
	PYTHONPATH=$(PYTHONPATH) uv run python -m mkdocs build
	@echo "$(GREEN)Documentation built in site/$(NC)"

.PHONY: serve
serve: ## Serve documentation locally (with live reload)
	PYTHONPATH=$(PYTHONPATH) uv run python -m mkdocs serve

.PHONY: docs
docs: build-docs ## Alias for build-docs

##@ Build & Release

.PHONY: clean
clean: ## Clean build artifacts and cache files
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .coverage htmlcov/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "$(GREEN)Cleaned build artifacts$(NC)"

.PHONY: build
build: clean ## Build package distribution
	uv build
	@echo "$(GREEN)Package built in dist/$(NC)"

.PHONY: publish
publish: build ## Publish package to PyPI
	uv publish
	@echo "$(GREEN)Package published to PyPI$(NC)"

.PHONY: publish-test
publish-test: build ## Publish package to TestPyPI
	uv publish --publish-url https://test.pypi.org/legacy/
	@echo "$(GREEN)Package published to TestPyPI$(NC)"

##@ Maintenance

.PHONY: update-deps
update-deps: ## Update dependencies to latest versions
	uv lock --upgrade
	@echo "$(GREEN)Dependencies updated$(NC)"

.PHONY: all
all: clean qa coverage build-docs ## Run complete CI/CD pipeline locally
	@echo "$(GREEN)All tasks completed successfully!$(NC)"

# Legacy aliases for backwards compatibility
.PHONY: main
main: all ## Legacy alias for 'all' target
