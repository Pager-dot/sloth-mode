.DEFAULT_GOAL := help
PY ?= python3

.PHONY: help
help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## install in editable mode with dev extras
	$(PY) -m pip install -e ".[dev]"

.PHONY: test
test: ## run the test suite (no model server needed)
	$(PY) -m pytest

.PHONY: lint
lint: ## ruff check + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

.PHONY: fmt
fmt: ## format the code
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

.PHONY: demo
demo: ## build a fake messy folder and tidy it with heuristics only
	$(PY) -m sloth --no-model tidy examples/messy-downloads
	$(PY) -m sloth --no-model brief examples/chats

.PHONY: gui
gui: ## open the web UI with no model (heuristics only)
	$(PY) -m sloth --no-model gui examples/messy-downloads

.PHONY: record
record: ## re-record the README demo (needs the demo extra + a browser)
	$(PY) scripts/record_demo.py --api cloud

.PHONY: doctor
doctor: ## check whether a local model is reachable
	$(PY) -m sloth doctor

.PHONY: clean
clean: ## remove caches and build output
	rm -rf build dist .pytest_cache .ruff_cache **/__pycache__ *.egg-info
