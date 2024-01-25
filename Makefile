.PHONY: setup install test clean help
.ONESHELL:

# Configuration
VENV := venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip3

# Check if the virtual environment already exists
VENV_EXISTS := $(shell test -d $(VENV) && echo 1 || echo 0)

# Setup the virtual environment
setup:
	@if [ $(VENV_EXISTS) -eq 0 ]; then \
		echo "Creating virtual environment..."; \
		python3.8 -m venv $(VENV); \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment already exists. Skipping creation."; \
	fi


# Install the project in editable mode
install:
	@echo "Installing dependencies..."
	@echo "Python version: $(shell $(PYTHON) --version)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt -qq
	@$(PIP) install -e .
	@echo "Dependencies installed."

# run setup and install and activate for development
dev: setup install
	@echo "To activate the virtual environment, run:"
	@echo "      >>>   source $(VENV)/bin/activate   <<<     "

# Run tests
test: install
	@echo "Running tests..."
	@$(PYTHON) -m pytest

# Clean up the project
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -delete
	@echo "Clean up complete."

# Display help for commands
help:
	@echo "Available commands:"
	@echo "  setup    - Set up the virtual environment"
	@echo "  install  - Install the project and dependencies"
	@echo "  dev      - Set up the development environment"
	@echo "  test     - Run tests"
	@echo "  clean    - Clean up the project"
	@echo "  help     - Show this message"
