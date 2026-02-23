.PHONY: help install install-dev lint format test test-cov clean train evaluate predict

PYTHON ?= python
PIP    ?= pip

## help: Show this help message
help:
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^## ' Makefile | sed 's/## /  /'

## install: Install production dependencies
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## install-dev: Install all dependencies including dev tools
install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt

## lint: Run flake8, black check, isort check, and mypy
lint:
	flake8 . --max-line-length=120 --exclude=notebooks,__pycache__,.git
	black --check --line-length=120 .
	isort --check-only .
	mypy . --ignore-missing-imports --exclude notebooks

## format: Auto-format code with black and isort
format:
	black --line-length=120 .
	isort .

## test: Run pytest (no coverage)
test:
	pytest tests/ -v

## test-cov: Run pytest with coverage report
test-cov:
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html

## clean: Remove cache files, build artifacts, and coverage reports
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage coverage.xml .mypy_cache

## train: Train the model with the default config
train:
	$(PYTHON) train.py --config configs/train_config.yaml

## evaluate: Evaluate a trained checkpoint
## Usage: make evaluate CKPT=checkpoints/best.pth DATA=data/cats
evaluate:
	$(PYTHON) evaluate.py --checkpoint $(CKPT) --data-dir $(DATA)

## predict: Run inference on a single image
## Usage: make predict IMG=path/to/image.jpg CKPT=checkpoints/best.pth
predict:
	$(PYTHON) predict.py --image $(IMG) --checkpoint $(CKPT)
