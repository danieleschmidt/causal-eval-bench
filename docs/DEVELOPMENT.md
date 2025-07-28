# Development Guide

## Quick Setup
```bash
# Clone and setup
git clone <repo-url>
make dev

# Run tests
make test
```

## Architecture
- See [ARCHITECTURE.md](../ARCHITECTURE.md) for system design
- Core modules: evaluation engine, task framework, API layer

## Development Process
1. Create feature branch from `main`
2. Follow [CONTRIBUTING.md](../CONTRIBUTING.md) guidelines
3. Run tests: `make test lint`
4. Submit PR with clear description

## Key Commands
- `make dev` - Setup development environment
- `make test` - Run full test suite
- `make lint` - Code quality checks
- `make docs` - Build documentation

## Resources
- [Contributing Guide](../CONTRIBUTING.md)
- [Code Examples](getting-started/installation.md)
- [API Docs](index.md)

## Getting Help
- GitHub Issues for bugs
- GitHub Discussions for questions
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for communication channels