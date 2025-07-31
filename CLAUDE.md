# Claude Code Memory for Causal Eval Bench

## Repository Overview

**Causal Eval Bench** is a comprehensive evaluation framework for testing genuine causal reasoning in language models. It's a Python-based project with FastAPI backend, comprehensive testing suite, and advanced SDLC practices.

## Key Project Information

### Technology Stack
- **Language**: Python 3.9-3.12
- **Framework**: FastAPI, Pydantic, SQLAlchemy
- **Database**: PostgreSQL (production), SQLite (testing)  
- **Cache**: Redis
- **Testing**: pytest, coverage, hypothesis
- **Documentation**: MkDocs Material
- **Containers**: Docker, docker-compose
- **Dependency Management**: Poetry

### Project Structure
```
causal-eval-bench/
├── causal_eval/              # Main package (not yet created)
├── tests/                    # Comprehensive test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   ├── e2e/                 # End-to-end tests
│   ├── load/                # Load tests with Locust
│   └── performance/         # Performance benchmarks
├── docs/                    # Comprehensive documentation
├── scripts/                 # Automation and utility scripts
├── docker/                  # Docker configurations
└── .github/                 # GitHub configuration (no workflows yet)
```

### Current Maturity Level
**ADVANCED (85-90%)** - This repository has exceptional SDLC maturity:
- ✅ Comprehensive documentation and architecture
- ✅ Advanced tooling configuration (ruff, mypy, black, poetry)
- ✅ Complete testing infrastructure
- ✅ Security framework (dependabot, pre-commit, security.md)
- ✅ Monitoring setup (Prometheus, Grafana)
- ✅ Container orchestration
- ❌ **Missing**: Active GitHub Actions workflows (templates exist)

## Development Workflow

### Essential Commands
```bash
# Setup development environment
make dev

# Code quality
make lint              # Run all linting checks
make format           # Format code with black/isort
make lint-fix         # Auto-fix linting issues

# Testing
make test             # Run all tests with coverage
make test-unit        # Unit tests only
make test-integration # Integration tests only
make test-e2e         # End-to-end tests only

# Development server
make run              # Start all services with docker-compose
make run-dev          # Start in development mode
make logs             # View service logs

# Database operations
make db-migrate       # Run Alembic migrations
make db-revision      # Create new migration
```

### Quality Standards
- **Test Coverage**: Minimum 80% (enforced)
- **Type Checking**: MyPy strict mode enabled
- **Linting**: Ruff with comprehensive rule set
- **Security**: Bandit security scanning
- **Pre-commit**: Hooks for formatting, linting, security

### Testing Strategy
- **Unit Tests**: Core logic and algorithms
- **Integration Tests**: API endpoints, database operations
- **E2E Tests**: Complete user workflows
- **Performance Tests**: Benchmarking and regression detection  
- **Load Tests**: Locust-based load testing

## Architecture Notes

### Core Domains
1. **Causal Attribution**: Testing cause vs correlation understanding
2. **Counterfactual Reasoning**: "What if" scenario analysis
3. **Intervention Analysis**: Understanding causal interventions
4. **Causal Chain Reasoning**: Multi-step causal relationships
5. **Confounding Variables**: Identifying confounders

### Key Design Patterns
- **Plugin Architecture**: Extensible task and domain system
- **Async Processing**: FastAPI with async/await patterns
- **Event-Driven**: Redis pub/sub for real-time updates
- **Microservices Ready**: Container-based architecture
- **API-First**: RESTful API with OpenAPI/Swagger

## Monitoring & Observability

### Metrics Collection
- **Application Metrics**: Custom Prometheus metrics
- **Infrastructure Metrics**: System and container metrics
- **Business Metrics**: Evaluation success rates, performance
- **Security Metrics**: Vulnerability scan results

### Logging Strategy
- **Structured Logging**: JSON format with correlation IDs
- **Log Levels**: Debug, Info, Warning, Error, Critical
- **Log Aggregation**: Centralized log collection
- **Alerting**: Prometheus AlertManager integration

## Deployment Architecture

### Environments
- **Development**: Local docker-compose stack
- **Staging**: Kubernetes deployment (documented)
- **Production**: Multi-region deployment (planned)

### CI/CD Pipeline Status
⚠️ **Manual Setup Required**: GitHub Actions workflows are documented but not active. Templates exist in `docs/workflows/examples/`:
- `ci.yml` - Comprehensive CI pipeline
- `cd.yml` - Continuous deployment  
- `security-scan.yml` - Security scanning
- `dependency-update.yml` - Automated updates
- `release.yml` - Release automation

## Security Considerations

### Security Measures
- **Dependency Scanning**: Safety, Bandit, Dependabot
- **Secret Management**: Environment variables, no hardcoded secrets
- **Container Security**: Multi-stage builds, vulnerability scanning
- **HTTPS Only**: TLS/SSL enforcement
- **Input Validation**: Pydantic models with strict validation

### Compliance
- **GDPR Ready**: Privacy by design principles
- **SOC 2**: Security controls documented
- **SLSA**: Supply chain security practices

## Performance Optimization

### Current Optimizations
- **Database**: Connection pooling, query optimization
- **Caching**: Redis for frequently accessed data  
- **Async Operations**: Non-blocking I/O throughout
- **Container Optimization**: Multi-stage builds, minimal base images

### Performance Targets
- **API Response**: <200ms 95th percentile
- **Test Execution**: <5 minutes full suite
- **Memory Usage**: <512MB per container
- **Database Queries**: <50ms average

## Common Issues & Solutions

### Development Setup
```bash
# If poetry installation fails
pip install --upgrade pip setuptools wheel
pip install poetry

# If pre-commit hooks fail
poetry run pre-commit clean
poetry run pre-commit install

# Database connection issues
docker-compose down && docker-compose up -d postgres
make db-migrate
```

### Testing Issues
```bash
# Flaky test debugging
pytest -xvs tests/specific_test.py --lf

# Performance test baseline updates
pytest tests/performance/ --benchmark-autosave
```

### Docker Issues
```bash
# Clear Docker cache
docker system prune -a

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
```

## Future Enhancements

### Short Term (Next Sprint)
- [ ] Activate GitHub Actions workflows
- [ ] Complete core evaluation engine implementation
- [ ] Add real-time leaderboard functionality

### Medium Term (Next Quarter)
- [ ] Multi-language support for evaluations
- [ ] Advanced visualization dashboard
- [ ] Automated report generation

### Long Term (Next Year)
- [ ] Machine learning model integration
- [ ] Enterprise SSO integration
- [ ] Advanced analytics and insights

## Important Files & Locations

### Configuration
- `pyproject.toml` - Project dependencies and tool configuration
- `Makefile` - Development workflow automation
- `docker-compose.yml` - Local development environment
- `.pre-commit-config.yaml` - Git hooks configuration

### Documentation
- `README.md` - Project overview and quick start
- `ARCHITECTURE.md` - System architecture documentation
- `docs/` - Comprehensive project documentation
- `docs/workflows/` - CI/CD workflow templates

### Testing
- `tests/conftest.py` - Pytest configuration and fixtures
- `tests/*/` - Organized test suites by type
- `locustfile.py` - Load testing configuration

## Team Conventions

### Commit Message Format
- feat: New features
- fix: Bug fixes  
- docs: Documentation changes
- test: Test additions/modifications
- refactor: Code refactoring
- style: Formatting changes
- chore: Maintenance tasks

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Code Review Requirements
- [ ] All tests passing
- [ ] Code coverage maintained
- [ ] Security scan passed
- [ ] Documentation updated
- [ ] Performance impact assessed

---

**Last Updated**: 2025-07-30
**Repository Maturity**: ADVANCED (85-90%)
**Primary Contact**: daniel@terragon-labs.com