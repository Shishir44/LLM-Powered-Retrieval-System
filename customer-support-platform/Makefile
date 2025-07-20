# Customer Support Platform - Makefile
# This Makefile provides convenient commands for development, testing, and deployment

.PHONY: help install dev test lint format clean build deploy docs

# Default target
help: ## Show this help message
	@echo "Customer Support Platform - Available Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Development Setup
install: ## Install dependencies using Poetry
	@echo "Installing dependencies..."
	poetry install
	@echo "Dependencies installed successfully!"

install-dev: ## Install development dependencies
	@echo "Installing development dependencies..."
	poetry install --with dev
	pre-commit install
	@echo "Development environment setup complete!"

# Development
dev: ## Start development environment with Docker Compose
	@echo "Starting development environment..."
	docker-compose up -d
	@echo "Development environment started!"
	@echo "API Gateway: http://localhost:8080"
	@echo "API Documentation: http://localhost:8080/docs"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"

dev-logs: ## Show logs from development environment
	docker-compose logs -f

dev-stop: ## Stop development environment
	@echo "Stopping development environment..."
	docker-compose down
	@echo "Development environment stopped!"

dev-rebuild: ## Rebuild and restart development environment
	@echo "Rebuilding development environment..."
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "Development environment rebuilt!"

# Code Quality
lint: ## Run linting checks
	@echo "Running linting checks..."
	poetry run black --check .
	poetry run isort --check-only .
	poetry run flake8 .
	poetry run mypy .
	@echo "Linting checks completed!"

format: ## Format code
	@echo "Formatting code..."
	poetry run black .
	poetry run isort .
	@echo "Code formatted!"

lint-fix: format ## Fix linting issues

# Testing
test: ## Run all tests
	@echo "Running all tests..."
	poetry run pytest tests/ -v
	@echo "All tests completed!"

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	poetry run pytest tests/unit/ -v
	@echo "Unit tests completed!"

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	poetry run pytest tests/integration/ -v
	@echo "Integration tests completed!"

test-e2e: ## Run end-to-end tests
	@echo "Running end-to-end tests..."
	poetry run pytest tests/e2e/ -v
	@echo "End-to-end tests completed!"

test-coverage: ## Run tests with coverage report
	@echo "Running tests with coverage..."
	poetry run pytest tests/ --cov=services --cov=shared --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/"

test-watch: ## Run tests in watch mode
	@echo "Running tests in watch mode..."
	poetry run ptw tests/

# Security
security: ## Run security checks
	@echo "Running security checks..."
	poetry run bandit -r services/ shared/
	poetry run safety check
	@echo "Security checks completed!"

# Database
db-migrate: ## Run database migrations
	@echo "Running database migrations..."
	poetry run alembic upgrade head
	@echo "Database migrations completed!"

db-reset: ## Reset database (development only)
	@echo "Resetting database..."
	docker-compose down postgres
	docker volume rm customer-support-platform_postgres_data || true
	docker-compose up -d postgres
	sleep 10
	$(MAKE) db-migrate
	@echo "Database reset completed!"

# Building
build: ## Build all Docker images
	@echo "Building Docker images..."
	docker-compose build
	@echo "Docker images built successfully!"

build-production: ## Build production Docker images
	@echo "Building production images..."
	docker build -f services/api-gateway/Dockerfile -t customer-support/api-gateway:latest .
	docker build -f services/conversation-service/Dockerfile -t customer-support/conversation-service:latest .
	docker build -f services/knowledge-base-service/Dockerfile -t customer-support/knowledge-base-service:latest .
	@echo "Production images built successfully!"

# Deployment
deploy-local: ## Deploy to local Kubernetes (minikube/kind)
	@echo "Deploying to local Kubernetes..."
	kubectl apply -f infrastructure/kubernetes/namespace.yaml
	kubectl apply -f infrastructure/kubernetes/secrets.yaml
	kubectl apply -f infrastructure/kubernetes/
	@echo "Deployed to local Kubernetes!"

deploy-staging: ## Deploy to staging environment
	@echo "Deploying to staging..."
	kubectl config use-context staging
	kubectl apply -f infrastructure/kubernetes/namespace.yaml
	kubectl apply -f infrastructure/kubernetes/secrets.yaml
	kubectl apply -f infrastructure/kubernetes/
	kubectl rollout status deployment/api-gateway -n customer-support
	@echo "Deployed to staging!"

deploy-production: ## Deploy to production environment
	@echo "Deploying to production..."
	kubectl config use-context production
	kubectl apply -f infrastructure/kubernetes/namespace.yaml
	kubectl apply -f infrastructure/kubernetes/secrets.yaml
	kubectl apply -f infrastructure/kubernetes/
	kubectl rollout status deployment/api-gateway -n customer-support
	@echo "Deployed to production!"

# Monitoring
logs: ## Show logs from Kubernetes pods
	kubectl logs -f deployment/api-gateway -n customer-support

logs-conversation: ## Show logs from conversation service
	kubectl logs -f deployment/conversation-service -n customer-support

logs-knowledge: ## Show logs from knowledge base service
	kubectl logs -f deployment/knowledge-base-service -n customer-support

status: ## Show status of all deployments
	kubectl get deployments -n customer-support
	kubectl get pods -n customer-support
	kubectl get services -n customer-support

# Documentation
docs: ## Generate documentation
	@echo "Generating documentation..."
	poetry run mkdocs build
	@echo "Documentation generated in site/"

docs-serve: ## Serve documentation locally
	@echo "Serving documentation..."
	poetry run mkdocs serve
	@echo "Documentation available at http://localhost:8000"

# Cleanup
clean: ## Clean up temporary files and containers
	@echo "Cleaning up..."
	docker-compose down --volumes --remove-orphans
	docker system prune -f
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/
	@echo "Cleanup completed!"

clean-all: ## Clean everything including Docker images
	@echo "Cleaning everything..."
	docker-compose down --volumes --remove-orphans
	docker system prune -af
	docker volume prune -f
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .venv/
	@echo "Complete cleanup finished!"

# Performance
load-test: ## Run load tests
	@echo "Running load tests..."
	poetry run locust -f tests/performance/locustfile.py --headless \
		--users 100 --spawn-rate 10 --run-time 2m \
		--host http://localhost:8080
	@echo "Load tests completed!"

benchmark: ## Run performance benchmarks
	@echo "Running benchmarks..."
	poetry run python scripts/benchmark.py
	@echo "Benchmarks completed!"

# Configuration
config-check: ## Check configuration validity
	@echo "Checking configuration..."
	poetry run python -c "from shared.config.settings import get_settings; print('Configuration is valid!')"

secrets-template: ## Generate secrets template
	@echo "Generating secrets template..."
	cp infrastructure/kubernetes/secrets.yaml infrastructure/kubernetes/secrets.template.yaml
	sed -i 's/your-.*-here/REPLACE_ME/g' infrastructure/kubernetes/secrets.template.yaml
	@echo "Secrets template generated!"

# Development Utilities
shell: ## Start interactive shell in development container
	docker-compose exec api-gateway /bin/bash

db-shell: ## Connect to database shell
	docker-compose exec postgres psql -U postgres -d customer_support

redis-shell: ## Connect to Redis shell
	docker-compose exec redis redis-cli

# Health Checks
health: ## Check health of all services
	@echo "Checking service health..."
	@curl -f http://localhost:8080/health && echo "✓ API Gateway healthy"
	@curl -f http://localhost:8001/health && echo "✓ Conversation Service healthy"
	@curl -f http://localhost:8002/health && echo "✓ Knowledge Base Service healthy"

health-k8s: ## Check health of services in Kubernetes
	@echo "Checking Kubernetes service health..."
	kubectl get pods -n customer-support
	kubectl exec -n customer-support deployment/api-gateway -- curl -f http://localhost:8080/health

# CI/CD
ci: lint test security ## Run all CI checks locally
	@echo "All CI checks passed!"

pre-commit: lint test-unit ## Run pre-commit checks
	@echo "Pre-commit checks passed!"

# Release
version: ## Show current version
	@poetry version

version-bump-patch: ## Bump patch version
	poetry version patch
	@echo "Version bumped to $(shell poetry version -s)"

version-bump-minor: ## Bump minor version
	poetry version minor
	@echo "Version bumped to $(shell poetry version -s)"

version-bump-major: ## Bump major version
	poetry version major
	@echo "Version bumped to $(shell poetry version -s)"

# Environment Variables
env-template: ## Generate environment template
	@echo "Generating environment template..."
	cp .env.example .env.template
	@echo "Environment template generated as .env.template"

# Quick Start
quickstart: install dev db-migrate ## Quick start for new developers
	@echo ""
	@echo "🚀 Quick start completed!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Copy .env.example to .env and configure your settings"
	@echo "2. Visit http://localhost:8080/docs for API documentation"
	@echo "3. Run 'make test' to ensure everything works"
	@echo "4. Run 'make help' to see all available commands"
	@echo ""

# Default environment variables
export PYTHONPATH := $(shell pwd)
export ENVIRONMENT := development