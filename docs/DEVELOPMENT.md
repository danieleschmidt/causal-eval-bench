# 🛠️ Development Guide

This guide covers everything you need to know to develop and contribute to the Causal Eval Bench project.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ 
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- [Docker](https://docs.docker.com/get-docker/) for containerized development
- [Git](https://git-scm.com/) for version control

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/causal-eval-bench
   cd causal-eval-bench
   ```

2. **Install dependencies**
   ```bash
   poetry install --with dev,test,docs
   ```

3. **Set up pre-commit hooks**
   ```bash
   poetry run pre-commit install
   ```

4. **Copy environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start development services**
   ```bash
   docker compose -f docker-compose.dev.yml up -d
   ```

6. **Run the application**
   ```bash
   poetry run uvicorn causal_eval.api.main:app --reload
   ```

## 🏗️ Project Structure

```
causal-eval-bench/
├── causal_eval/           # Main package
│   ├── __init__.py
│   ├── api/              # FastAPI application
│   ├── cli/              # Command line interface
│   ├── tasks/            # Evaluation tasks
│   ├── evaluation/       # Evaluation engine
│   ├── generation/       # Test generation
│   ├── analysis/         # Analysis tools
│   ├── models/           # Data models
│   └── utils/            # Utilities
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   ├── e2e/             # End-to-end tests
│   ├── performance/     # Performance tests
│   └── load/            # Load tests
├── docs/                # Documentation
├── scripts/             # Utility scripts
├── docker/              # Docker configurations
├── .github/             # GitHub workflows
└── ...
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run specific test types
poetry run pytest tests/unit/          # Unit tests
poetry run pytest tests/integration/   # Integration tests
poetry run pytest tests/e2e/          # End-to-end tests

# Run with coverage
poetry run pytest --cov=causal_eval --cov-report=html

# Run performance benchmarks
poetry run pytest tests/performance/ --benchmark-only
```

### Writing Tests

- **Unit tests**: Test individual functions and classes in isolation
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete user workflows
- **Performance tests**: Benchmark critical code paths

Example unit test:
```python
import pytest
from causal_eval.tasks.attribution import CausalAttribution

def test_causal_attribution_scoring():
    task = CausalAttribution()
    response = "The relationship is causal because..."
    ground_truth = {"is_causal": True, "explanation": "..."}
    
    score = task.evaluate_response(response, ground_truth)
    assert 0.0 <= score <= 1.0
```

## 🔧 Code Quality

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **Bandit**: Security linting

Run all quality checks:
```bash
poetry run pre-commit run --all-files
```

### Pre-commit Hooks

Pre-commit hooks automatically run quality checks before commits:
- Code formatting
- Import sorting
- Linting
- Type checking
- Security scanning
- Tests

## 📦 Dependencies

### Adding Dependencies

```bash
# Production dependency
poetry add package_name

# Development dependency
poetry add --group dev package_name

# Optional dependency group
poetry add --group test package_name
```

### Updating Dependencies

```bash
# Update all dependencies
poetry update

# Update specific package
poetry update package_name

# Check for outdated packages
poetry show --outdated
```

## 🐳 Docker Development

### Development Container

Use the development container for a consistent environment:

```bash
# Start development environment
docker compose -f docker-compose.dev.yml up -d

# Access the container
docker compose -f docker-compose.dev.yml exec app bash

# View logs
docker compose -f docker-compose.dev.yml logs -f app
```

### Dev Container (VS Code)

The repository includes a dev container configuration for VS Code:

1. Install the "Dev Containers" extension
2. Open the repository in VS Code
3. When prompted, click "Reopen in Container"

## 🗃️ Database

### Local Development

The development setup includes a PostgreSQL database:

```bash
# Database is started with docker-compose
# Connection string: postgresql://causal_eval_user:causal_eval_password@localhost:5432/causal_eval_bench_dev

# Run migrations
poetry run alembic upgrade head

# Create a new migration
poetry run alembic revision --autogenerate -m "Description"
```

### Database Schema

Key tables:
- `models`: LLM configurations
- `tasks`: Evaluation questions
- `evaluations`: Individual test results
- `evaluation_sessions`: Grouped evaluation runs

## 🔍 Debugging

### Local Debugging

1. Set up your IDE to use the Poetry virtual environment
2. Set breakpoints in your code
3. Run tests or the application in debug mode

### VS Code Configuration

The repository includes VS Code launch configurations:
- Debug FastAPI server
- Debug CLI commands
- Debug specific test files

### Logging

Configure logging levels in your `.env`:
```bash
LOG_LEVEL=DEBUG
```

View logs:
```bash
# Application logs
tail -f logs/causal_eval_bench.log

# Docker logs
docker compose -f docker-compose.dev.yml logs -f
```

## 🚀 API Development

### FastAPI Application

The API is built with FastAPI and includes:
- Automatic OpenAPI documentation
- Request/response validation
- Authentication middleware
- Error handling

Access the API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Adding New Endpoints

1. Define models in `causal_eval/models/`
2. Implement endpoints in `causal_eval/api/routes/`
3. Add tests in `tests/integration/api/`

Example endpoint:
```python
from fastapi import APIRouter, Depends
from causal_eval.models.evaluation import EvaluationRequest, EvaluationResponse

router = APIRouter()

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_model(
    request: EvaluationRequest,
    db: Session = Depends(get_db)
):
    # Implementation
    pass
```

## 🎯 Task Development

### Creating New Evaluation Tasks

1. Implement the task class in `causal_eval/tasks/`
2. Inherit from `CausalTask` base class
3. Implement required methods
4. Add comprehensive tests

Example task:
```python
from causal_eval.tasks.base import CausalTask

class NewCausalTask(CausalTask):
    def __init__(self):
        super().__init__(
            task_type="new_causal_task",
            description="Description of the new task"
        )
    
    def generate_question(self, domain: str, difficulty: str) -> Dict:
        # Generate a question for the task
        pass
    
    def evaluate_response(self, response: str, ground_truth: Dict) -> float:
        # Score the model's response
        pass
```

### Task Registration

Register new tasks in the plugin system:
```toml
# pyproject.toml
[tool.poetry.plugins."causal_eval.tasks"]
new_task = "causal_eval.tasks.new_task:NewCausalTask"
```

## 📊 Performance

### Monitoring Performance

- Use `pytest-benchmark` for micro-benchmarks
- Monitor API response times with Prometheus metrics
- Profile memory usage with `memray`

### Optimization Guidelines

1. **Database queries**: Use SQLAlchemy efficiently
2. **API responses**: Implement response caching
3. **Model evaluation**: Parallelize where possible
4. **Memory usage**: Stream large datasets

## 🔐 Security

### Security Best Practices

- Never commit secrets or API keys
- Use environment variables for configuration
- Validate all inputs
- Implement rate limiting
- Use HTTPS in production

### Security Scanning

Automated security scans run on every commit:
- Bandit for Python security issues
- Safety for vulnerable dependencies
- Trivy for container vulnerabilities

## 📚 Documentation

### Writing Documentation

- Use Markdown for documentation
- Follow the existing style and structure
- Include code examples
- Update docs when adding features

### Building Documentation

```bash
# Install docs dependencies
poetry install --with docs

# Serve documentation locally
poetry run mkdocs serve

# Build documentation
poetry run mkdocs build
```

### Documentation Structure

- `docs/`: Main documentation
- `docs/api/`: API reference
- `docs/guides/`: User guides
- `docs/development/`: Development guides

## 🚀 Contributing

### Contribution Workflow

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Add** tests for new functionality
5. **Run** the test suite
6. **Submit** a pull request

### Pull Request Guidelines

- Write clear commit messages
- Include tests for new features
- Update documentation if needed
- Ensure all CI checks pass
- Request review from maintainers

### Code Review Process

1. Automated checks must pass
2. At least one approving review required
3. All conversations must be resolved
4. Branch must be up to date with main

## 🐛 Troubleshooting

### Common Issues

**Issue**: Poetry installation fails
```bash
# Solution: Update pip and try again
pip install --upgrade pip
curl -sSL https://install.python-poetry.org | python3 -
```

**Issue**: Database connection fails
```bash
# Solution: Ensure PostgreSQL is running
docker compose -f docker-compose.dev.yml up postgres -d
```

**Issue**: Tests fail with import errors
```bash
# Solution: Install in development mode
poetry install --with dev,test
```

**Issue**: Pre-commit hooks fail
```bash
# Solution: Run manual formatting
poetry run black .
poetry run isort .
poetry run ruff check . --fix
```

### Getting Help

- Check the [troubleshooting guide](troubleshooting.md)
- Search existing [GitHub issues](https://github.com/your-org/causal-eval-bench/issues)
- Join our [Discord community](https://discord.gg/causal-eval)
- Ask questions in [GitHub Discussions](https://github.com/your-org/causal-eval-bench/discussions)

## 📈 Performance Monitoring

### Local Monitoring

Access monitoring dashboards:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

### Metrics

Key metrics to monitor:
- Evaluation response time
- Database query performance
- Memory usage
- Error rates

## 🎯 Release Process

### Development Cycle

1. **Feature development** on feature branches
2. **Integration** via pull requests to `develop`
3. **Testing** on the develop branch
4. **Release preparation** via release branches
5. **Production release** to `main`

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Commands

```bash
# Create a release (automated via GitHub Actions)
git tag v1.2.3
git push origin v1.2.3

# Manual release
poetry version patch  # or minor, major
poetry build
poetry publish
```

## 🏆 Best Practices

### Code Organization

- Keep functions small and focused
- Use type hints throughout
- Write docstrings for public APIs
- Organize imports properly

### Testing Strategy

- Write tests first (TDD)
- Aim for high test coverage (>80%)
- Test edge cases and error conditions
- Use fixtures for common test data

### Performance

- Profile before optimizing
- Use async/await for I/O operations
- Cache expensive computations
- Monitor production performance

### Security

- Validate all inputs
- Use parameterized queries
- Keep dependencies updated
- Follow OWASP guidelines

## 📞 Support

Need help? Reach out through:

- 📧 Email: dev@causal-eval-bench.org
- 💬 Discord: [Development Channel](https://discord.gg/causal-eval-dev)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/causal-eval-bench/issues)
- 💡 Discussions: [GitHub Discussions](https://github.com/your-org/causal-eval-bench/discussions)