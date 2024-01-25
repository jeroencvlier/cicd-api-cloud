.PHONY: setup install test clean help
.ONESHELL:

VENV := venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip3
PROJECT_ROOT := $(shell pwd)
PYTHON_INTERPRETER := $(PROJECT_ROOT)/venv/bin/python
VENV_EXISTS := $(shell test -d $(VENV) && echo 1 || echo 0)

setup:
	@if [ $(VENV_EXISTS) -eq 0 ]; then \
		echo "Creating virtual environment..."; \
		python3.8 -m venv $(VENV); \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment already exists. Skipping creation."; \
	fi

install:
	@echo "Installing dependencies..."
	@echo "Python version: $(shell $(PYTHON) --version)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt -qq
	@$(PIP) install -e .
	@echo "Dependencies installed."

dev: setup install
	@echo "To activate the virtual environment, run:"
	@echo "      >>>   source $(VENV)/bin/activate   <<<     "

test: install
	@echo "Running tests..."
	@$(PYTHON) -m pytest -v

sanity:
	@echo "Running sanity check..."
	@$(PYTHON_INTERPRETER) $(PROJECT_ROOT)/sanitycheck.py $(PROJECT_ROOT)/tests/test_api.py

train:
	@echo "Training model..."
	@$(PYTHON_INTERPRETER) $(PROJECT_ROOT)/src/train_model.py

clean: 
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete
	@rm -rf cicd.egg-info
	@rm -rf .pytest_cache
	@echo "Clean up complete."

help:
	@echo "Available commands:"
	@echo "  setup    - Set up the virtual environment"
	@echo "  install  - Install the project and dependencies"
	@echo "  dev      - Set up the development environment"
	@echo "  test     - Run tests"
	@echo "  sanity   - Run sanity check"
	@echo "  train    - Train model"
	@echo "  clean    - Clean up the project"
	@echo "  help     - Show this message"
