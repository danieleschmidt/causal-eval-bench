# Contributing Guide

Welcome! We appreciate all contributions to advance causal reasoning evaluation.

## Quick Start

```bash
# Fork & clone repository
git clone https://github.com/YOUR_USERNAME/repo.git
make dev && make test
```

## Contribution Types
- **Code**: New tasks, features, bug fixes
- **Research**: Domain expertise, evaluation methods
- **Documentation**: Guides, examples, API docs
- **Community**: Issue triage, reviews, discussions

## Development Workflow

1. Create feature branch: `git checkout -b feature/name`
2. Follow coding standards (see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md))
3. Write tests and run: `make test lint`
4. Use conventional commits
5. Submit PR with clear description

## Standards
- **Python**: Follow [PEP 8](https://pep8.org/), use Black formatting
- **Testing**: Write tests with pytest, aim for >80% coverage
- **Docs**: Google-style docstrings, update CHANGELOG.md

## Resources
- **Documentation**: See [docs/](docs/) directory
- **Examples**: [docs/getting-started/](docs/getting-started/)
- **API Reference**: [docs/index.md](docs/index.md)
- **Architecture**: [ARCHITECTURE.md](ARCHITECTURE.md)

## Getting Help
- **Issues**: Bug reports and feature requests
- **Discussions**: Questions and design discussions
- **Security**: security@causal-eval-bench.org

**Questions?** Email contribute@causal-eval-bench.org