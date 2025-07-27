# Makefile for Causal Eval Bench
# Provides standardized commands for development, testing, and deployment

.PHONY: help install dev test lint format clean build docker run stop logs shell docs deploy

# =============================================================================
# Configuration
# =============================================================================

# Project settings
PROJECT_NAME := causal-eval-bench
PYTHON_VERSION := 3.11
POETRY := poetry
DOCKER_COMPOSE := docker-compose

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)Causal Eval Bench - Development Commands$(NC)"
	@echo ""
	@echo "$(YELLOW)Setup:$(NC)"
	@grep -E '^[a-zA-Z_-]+.*:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(install|dev|clean)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@grep -E '^[a-zA-Z_-]+.*:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(format|lint|test|docs)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Docker:$(NC)"
	@grep -E '^[a-zA-Z_-]+.*:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(docker|run|stop|logs|shell)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Build & Deploy:$(NC)"
	@grep -E '^[a-zA-Z_-]+.*:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(build|deploy|release)' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Setup and Installation
# =============================================================================

install: ## Install project dependencies
	@echo "$(BLUE)Installing project dependencies...$(NC)"
	$(POETRY) install --with dev,test,docs
	@echo "$(GREEN)✓ Dependencies installed$(NC)"

dev: install ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(POETRY) install --with dev,test,docs
	$(POETRY) run pre-commit install
	@echo "$(GREEN)✓ Development environment ready$(NC)"

clean: ## Clean up build artifacts and caches
	@echo "$(BLUE)Cleaning up build artifacts...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .tox/
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# =============================================================================
# Code Quality
# =============================================================================

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(POETRY) run black .
	$(POETRY) run isort .
	@echo "$(GREEN)✓ Code formatted$(NC)"

lint: ## Run linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(POETRY) run ruff check .
	$(POETRY) run mypy causal_eval/
	$(POETRY) run bandit -r causal_eval/ -x tests/
	@echo "$(GREEN)✓ Linting complete$(NC)"

lint-fix: ## Fix linting issues automatically
	@echo "$(BLUE)Fixing linting issues...$(NC)"
	$(POETRY) run ruff check --fix .
	$(POETRY) run black .
	$(POETRY) run isort .
	@echo "$(GREEN)✓ Linting issues fixed$(NC)"

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(POETRY) run pytest tests/ -v --cov=causal_eval --cov-report=html --cov-report=xml
	@echo "$(GREEN)✓ Tests complete$(NC)"

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(POETRY) run pytest tests/unit/ -v -m "unit"
	@echo "$(GREEN)✓ Unit tests complete$(NC)"

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(POETRY) run pytest tests/integration/ -v -m "integration"
	@echo "$(GREEN)✓ Integration tests complete$(NC)"

test-e2e: ## Run end-to-end tests only
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	$(POETRY) run pytest tests/e2e/ -v -m "e2e"
	@echo "$(GREEN)✓ End-to-end tests complete$(NC)"

test-performance: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(POETRY) run pytest tests/ -v -m "performance" --benchmark-only
	@echo "$(GREEN)✓ Performance tests complete$(NC)"

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	$(POETRY) run pytest-watch tests/ -- -v

coverage: test ## Generate and open coverage report
	@echo "$(BLUE)Opening coverage report...$(NC)"
	open htmlcov/index.html || xdg-open htmlcov/index.html

# =============================================================================
# Documentation
# =============================================================================

docs: ## Build and serve documentation
	@echo "$(BLUE)Building and serving documentation...$(NC)"
	$(POETRY) run mkdocs serve

docs-build: ## Build documentation for production
	@echo "$(BLUE)Building documentation...$(NC)"
	$(POETRY) run mkdocs build
	@echo "$(GREEN)✓ Documentation built$(NC)"

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(BLUE)Deploying documentation...$(NC)"
	$(POETRY) run mkdocs gh-deploy
	@echo "$(GREEN)✓ Documentation deployed$(NC)"

# =============================================================================
# Docker Commands
# =============================================================================

docker: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(PROJECT_NAME):latest .
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	docker build --target development -t $(PROJECT_NAME):dev .
	@echo "$(GREEN)✓ Development Docker image built$(NC)"

run: ## Start all services with docker-compose
	@echo "$(BLUE)Starting all services...$(NC)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)✓ Services started$(NC)"
	@echo "$(YELLOW)API available at: http://localhost:8000$(NC)"
	@echo "$(YELLOW)Documentation at: http://localhost:8080$(NC)"

run-dev: ## Start services in development mode
	@echo "$(BLUE)Starting services in development mode...$(NC)"
	$(DOCKER_COMPOSE) up --build
	@echo "$(GREEN)✓ Development services started$(NC)"

stop: ## Stop all services
	@echo "$(BLUE)Stopping all services...$(NC)"
	$(DOCKER_COMPOSE) down
	@echo "$(GREEN)✓ Services stopped$(NC)"

restart: stop run ## Restart all services

logs: ## Show logs from all services
	$(DOCKER_COMPOSE) logs -f

logs-app: ## Show logs from app service only
	$(DOCKER_COMPOSE) logs -f app

shell: ## Open shell in app container
	$(DOCKER_COMPOSE) exec app /bin/bash

shell-db: ## Open database shell
	$(DOCKER_COMPOSE) exec postgres psql -U causal_eval_user -d causal_eval_bench

# =============================================================================
# Testing with Docker
# =============================================================================

test-docker: ## Run tests in Docker
	@echo "$(BLUE)Running tests in Docker...$(NC)"
	$(DOCKER_COMPOSE) --profile testing run --rm test
	@echo "$(GREEN)✓ Docker tests complete$(NC)"

test-performance-docker: ## Run performance tests in Docker
	@echo "$(BLUE)Running performance tests in Docker...$(NC)"
	$(DOCKER_COMPOSE) --profile performance up --build performance-test
	@echo "$(GREEN)✓ Docker performance tests complete$(NC)"

# =============================================================================
# Monitoring
# =============================================================================

monitoring: ## Start monitoring services (Prometheus, Grafana)
	@echo "$(BLUE)Starting monitoring services...$(NC)"
	$(DOCKER_COMPOSE) --profile monitoring up -d prometheus grafana
	@echo "$(GREEN)✓ Monitoring services started$(NC)"
	@echo "$(YELLOW)Prometheus: http://localhost:9090$(NC)"
	@echo "$(YELLOW)Grafana: http://localhost:3000 (admin/admin123)$(NC)"

# =============================================================================
# Build and Release
# =============================================================================

build: clean ## Build distribution packages
	@echo "$(BLUE)Building distribution packages...$(NC)"
	$(POETRY) build
	@echo "$(GREEN)✓ Distribution packages built$(NC)"

release-test: build ## Release to test PyPI
	@echo "$(BLUE)Releasing to test PyPI...$(NC)"
	$(POETRY) publish --repository testpypi
	@echo "$(GREEN)✓ Released to test PyPI$(NC)"

release: build ## Release to PyPI
	@echo "$(BLUE)Releasing to PyPI...$(NC)"
	$(POETRY) publish
	@echo "$(GREEN)✓ Released to PyPI$(NC)"

version: ## Show current version
	@$(POETRY) version

version-patch: ## Bump patch version
	@echo "$(BLUE)Bumping patch version...$(NC)"
	$(POETRY) version patch
	@echo "$(GREEN)✓ Version bumped to $(shell poetry version -s)$(NC)"

version-minor: ## Bump minor version
	@echo "$(BLUE)Bumping minor version...$(NC)"
	$(POETRY) version minor
	@echo "$(GREEN)✓ Version bumped to $(shell poetry version -s)$(NC)"

version-major: ## Bump major version
	@echo "$(BLUE)Bumping major version...$(NC)"
	$(POETRY) version major
	@echo "$(GREEN)✓ Version bumped to $(shell poetry version -s)$(NC)"

# =============================================================================
# Security
# =============================================================================

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	$(POETRY) run bandit -r causal_eval/ -f json -o bandit-report.json
	$(POETRY) run safety check --json --output safety-report.json
	@echo "$(GREEN)✓ Security checks complete$(NC)"

# =============================================================================
# Database Operations
# =============================================================================

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(NC)"
	$(POETRY) run alembic upgrade head
	@echo "$(GREEN)✓ Database migrations complete$(NC)"

db-revision: ## Create new database revision
	@echo "$(BLUE)Creating new database revision...$(NC)"
	@read -p "Enter revision message: " msg; \
	$(POETRY) run alembic revision --autogenerate -m "$$msg"
	@echo "$(GREEN)✓ Database revision created$(NC)"

db-reset: ## Reset database (development only)
	@echo "$(RED)WARNING: This will delete all data!$(NC)"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		$(DOCKER_COMPOSE) exec postgres psql -U causal_eval_user -d causal_eval_bench -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"; \
		$(POETRY) run alembic upgrade head; \
		echo "$(GREEN)✓ Database reset complete$(NC)"; \
	else \
		echo "$(YELLOW)Database reset cancelled$(NC)"; \
	fi

# =============================================================================
# Utility Commands
# =============================================================================

requirements: ## Export requirements.txt from poetry
	@echo "$(BLUE)Exporting requirements.txt...$(NC)"
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes
	$(POETRY) export -f requirements.txt --output requirements-dev.txt --with dev,test,docs --without-hashes
	@echo "$(GREEN)✓ Requirements exported$(NC)"

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	$(POETRY) run pre-commit run --all-files
	@echo "$(GREEN)✓ Pre-commit hooks complete$(NC)"

init-secrets: ## Initialize secrets baseline for detect-secrets
	@echo "$(BLUE)Initializing secrets baseline...$(NC)"
	$(POETRY) run detect-secrets scan --baseline .secrets.baseline
	@echo "$(GREEN)✓ Secrets baseline initialized$(NC)"

# =============================================================================
# Health Checks
# =============================================================================

health: ## Check health of all services
	@echo "$(BLUE)Checking service health...$(NC)"
	@curl -f http://localhost:8000/health || echo "$(RED)✗ API service unhealthy$(NC)"
	@$(DOCKER_COMPOSE) exec postgres pg_isready -U causal_eval_user -d causal_eval_bench > /dev/null 2>&1 && echo "$(GREEN)✓ Database healthy$(NC)" || echo "$(RED)✗ Database unhealthy$(NC)"
	@$(DOCKER_COMPOSE) exec redis redis-cli ping > /dev/null 2>&1 && echo "$(GREEN)✓ Redis healthy$(NC)" || echo "$(RED)✗ Redis unhealthy$(NC)"

# =============================================================================
# Development Utilities
# =============================================================================

jupyter: ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook server...$(NC)"
	$(POETRY) run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

setup-hooks: ## Setup git hooks
	@echo "$(BLUE)Setting up git hooks...$(NC)"
	$(POETRY) run pre-commit install
	$(POETRY) run pre-commit install --hook-type commit-msg
	@echo "$(GREEN)✓ Git hooks installed$(NC)"