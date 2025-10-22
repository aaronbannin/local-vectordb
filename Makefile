.PHONY: help install dev test clean format reset load query

# Default target
help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies"
	@echo "  make dev       - Run development server with seed data"
	@echo "  make seed      - Load data and run a sample query"
	@echo "  make reset     - Reset the database"
	@echo "  make load      - Load sample data into database"
	@echo "  make query     - Run a query (usage: make query q='your query string')"
	@echo "  make format    - Format code with black"
	@echo "  make clean     - Remove cache and build files"

# Install dependencies
install:
	@if [ ! -d .venv ]; then \
		echo "Creating virtual environment..."; \
		python -m venv .venv; \
	fi
	@echo "Installing dependencies..."
	@. .venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Dependencies installed! Activate with: source .venv/bin/activate"

# Run development server
dev:
	docker compose up --build

# Run tests
seed:
	$(MAKE) reset
	$(MAKE) load
	$(MAKE) query q="What is the capital of Germany?"

reset:
	python -c "from tests.e2e import reset_db; reset_db()"

load:
	python -c "from tests.e2e import load; load()"

query:
	python -c "from tests.e2e import query; query()"

# Format code
format:
	black src/ tests/
	@echo "Code formatted successfully!"

# Clean cache and build files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "Cleaned up cache and build files!"

# Setup virtual environment
venv:
	python -m venv .venv
	@echo "Virtual environment created! Activate it with: source venv/bin/activate"
