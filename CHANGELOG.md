# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project structure and documentation
- Comprehensive SDLC automation framework
- Development environment setup with devcontainer support
- Testing framework with unit, integration, and e2e tests
- Docker containerization with multi-stage builds
- Code quality tools (linting, formatting, pre-commit hooks)
- Security scanning and vulnerability management
- Monitoring and observability with Prometheus and Grafana
- Comprehensive documentation with MkDocs

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- Implemented secrets scanning with detect-secrets
- Added security policy and vulnerability reporting process
- Container security scanning with vulnerability detection
- Secure defaults for all configuration options

## [0.1.0] - 2025-01-27

### Added
- Initial release of Causal Eval Bench
- Core evaluation framework for causal reasoning
- Support for multiple model providers (OpenAI, Anthropic, Google AI, Hugging Face)
- Basic causal reasoning tasks:
  - Causal Attribution
  - Counterfactual Reasoning
  - Causal Intervention
  - Causal Chain Reasoning
  - Confounding Analysis
- Domain coverage for medical, social, economic, and scientific domains
- Python SDK with comprehensive API
- CLI interface for command-line usage
- Basic test generation capabilities
- SQLite database support for local development
- Redis caching for performance optimization
- FastAPI-based REST API
- Docker support for containerized deployment
- Basic documentation and examples
- MIT license

### Technical Details
- Python 3.9+ support
- FastAPI web framework
- SQLAlchemy ORM with async support
- Redis for caching and session management
- Poetry for dependency management
- Pytest for testing framework
- Black and isort for code formatting
- Ruff for linting
- MyPy for type checking
- Pre-commit hooks for code quality
- Docker multi-stage builds
- Docker Compose for development environment

### Known Limitations
- Limited to text-based causal reasoning (no multi-modal support)
- Basic error analysis capabilities
- No public leaderboard yet
- Limited domain-specific customization
- Performance testing framework in early stages

---

## Release Notes Format

Each release will include:

### Version Types
- **Major** (X.0.0): Breaking changes, major new features
- **Minor** (0.X.0): New features, backwards compatible
- **Patch** (0.0.X): Bug fixes, security updates

### Categories
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

### Migration Guides
For breaking changes, we provide:
- Migration instructions
- Backwards compatibility timeline
- Code examples for updates

## Upcoming Releases

### v0.2.0 (Planned: Q2 2025)
- Extended task suite with advanced causal reasoning
- Domain-specific generators for 10+ domains
- Web interface for interactive evaluation
- Enhanced error analysis and profiling
- Performance optimizations
- Multi-language support

### v0.3.0 (Planned: Q3 2025)
- Multi-modal causal reasoning support
- Advanced statistical analysis
- Custom domain builder toolkit
- Longitudinal evaluation tracking
- Enhanced reporting capabilities

### v1.0.0 (Planned: Q4 2025)
- Production-ready public leaderboard
- Enterprise API with SLA guarantees
- Comprehensive benchmarking suite
- Academic dataset publication
- Platform integrations (HuggingFace, OpenAI, etc.)

## Support Policy

### Long-Term Support (LTS)
- **v1.0.x**: 2 years of security updates (until Q4 2027)
- **v0.x.x**: 6 months of critical security fixes

### Security Updates
- **Critical**: Immediate patch release
- **High**: Within 72 hours
- **Medium**: Next scheduled release
- **Low**: Next minor version

### Backwards Compatibility
- **API**: Semantic versioning with deprecation warnings
- **Data formats**: Forward and backward compatibility
- **Configuration**: Migration tools for breaking changes

## Contributing to Releases

### Release Process
1. Feature development in feature branches
2. Pull request review and testing
3. Integration testing on staging
4. Release candidate creation
5. Community testing period
6. Final release and announcement

### Release Criteria
- All tests passing
- Documentation updated
- Security scan clean
- Performance benchmarks met
- Migration guide prepared (if needed)

### Beta Testing
Join our beta testing program:
- Email: beta@causal-eval-bench.org
- Discord: #beta-testing channel
- Early access to new features
- Provide feedback and bug reports

---

**Note**: This changelog is automatically updated as part of our release process. For real-time updates, follow our [GitHub releases](https://github.com/your-org/causal-eval-bench/releases).