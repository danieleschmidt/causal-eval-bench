# CI/CD Workflows Documentation

This document provides comprehensive documentation for the CI/CD workflows required for the Causal Eval Bench project.

## ⚠️ Manual Setup Required

**Important**: Due to GitHub App permission limitations, the workflow files in this documentation must be manually created by repository maintainers. All workflow templates are provided in the `examples/` directory.

## Required Workflows

The following GitHub Actions workflows must be created manually:

### 1. CI Pipeline (`ci.yml`)
- **Location**: `.github/workflows/ci.yml`
- **Purpose**: Validate pull requests with comprehensive testing and quality checks
- **Triggers**: Pull requests, pushes to main branch

### 2. Continuous Deployment (`cd.yml`)
- **Location**: `.github/workflows/cd.yml`  
- **Purpose**: Automated deployment to staging and production environments
- **Triggers**: Pushes to main branch, manual dispatch

### 3. Dependency Updates (`dependency-update.yml`)
- **Location**: `.github/workflows/dependency-update.yml`
- **Purpose**: Automated dependency updates with security scanning
- **Triggers**: Schedule (weekly), manual dispatch

### 4. Security Scanning (`security-scan.yml`)
- **Location**: `.github/workflows/security-scan.yml`
- **Purpose**: Comprehensive security scanning and vulnerability assessment  
- **Triggers**: Schedule (daily), pull requests

### 5. Release Automation (`release.yml`)
- **Location**: `.github/workflows/release.yml`
- **Purpose**: Automated release creation and package publishing
- **Triggers**: Version tags, manual dispatch

## Workflow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pull Request  │───▶│  CI Pipeline    │───▶│  Code Quality   │
│   (Developer)   │    │  (ci.yml)       │    │  Gates          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Main Branch   │───▶│  CD Pipeline    │───▶│   Production    │
│   (Merge)       │    │  (cd.yml)       │    │   Deployment    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Security Scan  │
                       │  (security.yml) │
                       └─────────────────┘
```

## Setup Instructions

### Step 1: Create Workflow Files

Copy the workflow templates from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy workflow templates
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/cd.yml .github/workflows/
cp docs/workflows/examples/dependency-update.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/release.yml .github/workflows/
```

### Step 2: Configure Secrets

Add the following secrets in GitHub repository settings:

#### Required Secrets
- `OPENAI_API_KEY`: OpenAI API key for testing
- `ANTHROPIC_API_KEY`: Anthropic API key for testing
- `POSTGRES_PASSWORD`: Database password for testing
- `JWT_SECRET_KEY`: JWT secret for authentication

#### Optional Secrets (for deployment)
- `DOCKER_REGISTRY_TOKEN`: Container registry access
- `AWS_ACCESS_KEY_ID`: AWS deployment credentials
- `AWS_SECRET_ACCESS_KEY`: AWS deployment credentials
- `SLACK_WEBHOOK_URL`: Notifications
- `SENTRY_DSN`: Error tracking

### Step 3: Configure Environments

Create the following GitHub environments:

#### Staging Environment
- **Reviewers**: Development team
- **Deployment branches**: `main`, `develop`
- **Secrets**: Staging-specific configuration

#### Production Environment  
- **Reviewers**: Senior developers, DevOps team
- **Deployment branches**: `main` only
- **Protection rules**: Required reviews, deployment windows
- **Secrets**: Production configuration

### Step 4: Branch Protection Rules

Configure branch protection for `main` branch:

```yaml
Protection Rules:
  - Require pull request reviews: 2 reviewers
  - Dismiss stale reviews: true
  - Require review from CODEOWNERS: true
  - Require status checks: true
    - CI Pipeline / Tests
    - CI Pipeline / Security Scan
    - CI Pipeline / Build
  - Require branches to be up to date: true
  - Require conversation resolution: true
  - Include administrators: true
```

## Workflow Details

### CI Pipeline (`ci.yml`)

**Purpose**: Validate all code changes with comprehensive testing and quality checks.

**Stages**:
1. **Setup**: Checkout code, setup Python, install dependencies
2. **Code Quality**: Linting (Ruff, MyPy, Bandit), formatting (Black, isort)
3. **Testing**: Unit tests, integration tests, coverage reporting
4. **Security**: Security scanning, dependency vulnerability checks
5. **Build**: Docker image build and vulnerability scanning
6. **Documentation**: Build and validate documentation

**Matrix Strategy**:
- **Python Versions**: 3.9, 3.10, 3.11, 3.12
- **Operating Systems**: Ubuntu, macOS, Windows
- **Test Types**: Unit, integration, e2e

**Artifacts**:
- Test coverage reports
- Security scan results
- Docker images (for testing)
- Documentation builds

### CD Pipeline (`cd.yml`)

**Purpose**: Automated deployment to staging and production environments.

**Stages**:
1. **Build**: Create production Docker images
2. **Staging Deploy**: Deploy to staging environment
3. **Integration Tests**: Run tests against staging
4. **Production Deploy**: Deploy to production (with approval)
5. **Monitoring**: Post-deployment verification
6. **Rollback**: Automatic rollback on failure

**Deployment Strategy**:
- **Blue/Green**: Zero-downtime deployments
- **Health Checks**: Verify service health before traffic switch
- **Rollback**: Automatic rollback on health check failures

**Notifications**:
- Slack notifications for deployment status
- Email notifications for production deployments
- Status updates to pull requests

### Security Scanning (`security-scan.yml`)

**Purpose**: Comprehensive security analysis and vulnerability detection.

**Scans**:
1. **SAST**: Static Application Security Testing with Bandit
2. **Dependency Scan**: Vulnerability scanning with Safety
3. **Container Scan**: Docker image vulnerability scanning
4. **Secret Detection**: Scan for hardcoded secrets
5. **SBOM Generation**: Software Bill of Materials
6. **License Compliance**: License compatibility checking

**Tools**:
- **Bandit**: Python security linting
- **Safety**: Python dependency vulnerability scanning
- **Trivy**: Container and filesystem vulnerability scanning
- **GitLeaks**: Secret detection
- **FOSSA**: License compliance (optional)

**Reporting**:
- Security scan results uploaded to GitHub Security tab
- SARIF format reports for integration
- Slack notifications for critical vulnerabilities

### Dependency Updates (`dependency-update.yml`)

**Purpose**: Automated dependency updates with security and compatibility checks.

**Process**:
1. **Scan**: Check for available updates
2. **Update**: Create pull requests for updates
3. **Test**: Run full test suite against updates
4. **Security**: Verify updates don't introduce vulnerabilities
5. **Merge**: Auto-merge minor/patch updates if tests pass

**Update Strategy**:
- **Security Updates**: Immediate processing
- **Minor Updates**: Weekly batch processing
- **Major Updates**: Manual review required

**Tools**:
- **Dependabot**: GitHub native dependency updates
- **Poetry**: Python dependency management
- **Custom Scripts**: Specialized update logic

## Quality Gates

### Code Quality Requirements

All workflows enforce the following quality gates:

#### Code Coverage
- **Minimum Coverage**: 80% overall
- **New Code Coverage**: 100% for new files
- **Coverage Types**: Line, branch, function coverage

#### Code Quality
- **Linting**: Zero Ruff violations
- **Type Checking**: Zero MyPy errors  
- **Security**: Zero high/critical Bandit issues
- **Formatting**: Code must be formatted with Black

#### Testing
- **Unit Tests**: Must pass 100%
- **Integration Tests**: Must pass 100%
- **Performance Tests**: Must meet baseline requirements

#### Security
- **Vulnerability Scan**: Zero high/critical vulnerabilities
- **Secret Detection**: No secrets in code
- **License Compliance**: All dependencies approved

### Failure Handling

#### Automatic Retries
- Network-related failures: 3 retries with exponential backoff
- Flaky tests: Individual test retry mechanism
- Deployment failures: Automatic rollback

#### Manual Intervention
- Security vulnerabilities: Require manual review
- Major dependency updates: Require approval
- Production deployment failures: Alert on-call team

## Environment Configuration

### Development Environment
```yaml
Environment: development
Deployment: Feature branches, pull requests
Database: In-memory SQLite
Cache: Local Redis
Monitoring: Basic logging
```

### Staging Environment
```yaml
Environment: staging
Deployment: Main branch auto-deploy
Database: PostgreSQL (staging instance)
Cache: Redis cluster
Monitoring: Full observability stack
Load Testing: Automated performance tests
```

### Production Environment
```yaml
Environment: production
Deployment: Manual approval required
Database: PostgreSQL (production cluster)
Cache: Redis cluster with failover
Monitoring: Full observability + alerting
Backup: Automated daily backups
Security: Enhanced security controls
```

## Monitoring and Alerting

### Workflow Monitoring

#### Metrics Tracked
- **Success Rate**: Percentage of successful workflow runs
- **Duration**: Average and 95th percentile execution times
- **Failure Rate**: Categorized by failure type
- **Resource Usage**: CPU, memory, storage usage

#### Alerts
- **Workflow Failures**: Immediate notification for CI/CD failures
- **Performance Degradation**: Alert when workflows exceed baseline times
- **Security Issues**: Immediate notification for security scan failures

### Dashboard Integration

#### GitHub Actions Dashboard
- Workflow success/failure rates
- Duration trends over time
- Resource utilization metrics
- Security scan results

#### External Monitoring
- **Grafana**: Workflow metrics visualization
- **Slack**: Real-time notifications
- **Email**: Summary reports for stakeholders

## Performance Optimization

### Build Optimization

#### Caching Strategy
```yaml
Cache Layers:
  - Poetry dependencies
  - pip cache
  - Docker layers
  - Node modules (if applicable)
  - Test databases
```

#### Parallelization
```yaml
Parallel Execution:
  - Matrix builds across Python versions
  - Parallel test execution
  - Concurrent security scans
  - Multi-stage Docker builds
```

#### Resource Optimization
```yaml
Resource Allocation:
  - CI: Standard GitHub runners
  - CD: Self-hosted runners for production
  - Large Test Suites: Enhanced compute
  - Security Scans: Dedicated security runners
```

### Performance Baselines

#### Target Metrics
- **CI Pipeline**: <10 minutes end-to-end
- **Test Suite**: <5 minutes full execution
- **Docker Build**: <3 minutes including layers
- **Security Scan**: <2 minutes comprehensive scan

## Security Considerations

### Secrets Management

#### Principles
- **Least Privilege**: Minimal required permissions
- **Rotation**: Regular secret rotation
- **Encryption**: All secrets encrypted at rest
- **Audit**: Secret access logging

#### Implementation
```yaml
Secret Management:
  - GitHub Secrets for CI/CD
  - Azure Key Vault (alternative)
  - AWS Secrets Manager (alternative)
  - HashiCorp Vault (enterprise)
```

### Security Scanning

#### SLSA Compliance
- **Level 1**: Basic provenance tracking
- **Level 2**: Signed provenance
- **Level 3**: Tamper-resistant builds
- **Level 4**: Two-person reviews

#### Supply Chain Security
- **SBOM Generation**: Software Bill of Materials
- **Dependency Pinning**: Exact version pinning
- **Signature Verification**: Verify package signatures
- **Vulnerability Scanning**: Continuous monitoring

## Troubleshooting

### Common Issues

#### Workflow Failures

**Authentication Issues**
```bash
# Check secret configuration
# Verify GitHub token permissions
# Confirm API key validity
```

**Build Failures**
```bash
# Check dependency versions
# Verify Docker base image availability
# Review build logs for specific errors
```

**Test Failures**
```bash
# Check for flaky tests
# Verify test data setup
# Review environment configuration
```

#### Performance Issues

**Slow Builds**
```bash
# Review cache hit rates
# Optimize Docker layer ordering
# Parallelize build steps
```

**Test Timeouts**
```bash
# Increase timeout values
# Optimize slow tests
# Add test parallelization
```

### Debug Commands

```bash
# Local workflow testing with act
act -j ci

# Docker build debugging
docker build --progress=plain .

# Dependency resolution issues
poetry lock --verbose

# Test debugging
pytest -xvs tests/specific_test.py
```

## Migration Guide

### From Existing CI/CD

If migrating from another CI/CD system:

1. **Inventory Current Workflows**
   - Document existing build steps
   - Identify deployment procedures
   - Map quality gates

2. **Create Migration Plan**
   - Parallel run period
   - Feature-by-feature migration
   - Rollback procedures

3. **Update Documentation**
   - Update README instructions
   - Modify contributor guidelines
   - Update deployment procedures

### Best Practices

#### Workflow Design
- **Idempotent**: Workflows should be repeatable
- **Fast Feedback**: Quick failure for obvious issues
- **Comprehensive**: Cover all quality requirements
- **Maintainable**: Easy to understand and modify

#### Security
- **Secret Rotation**: Regular credential updates
- **Least Privilege**: Minimal required permissions
- **Audit Logging**: Track all workflow executions
- **Vulnerability Monitoring**: Continuous security scanning

## Contributing

### Workflow Changes

When modifying workflows:

1. **Test Locally**: Use `act` to test workflow changes
2. **Incremental Changes**: Small, testable modifications
3. **Documentation**: Update this guide with changes
4. **Review Process**: Require review from DevOps team

### Adding New Workflows

For new workflow requirements:

1. **Proposal**: Create RFC for new workflow
2. **Template**: Use existing workflows as templates
3. **Testing**: Thorough testing in development
4. **Documentation**: Add to this guide
5. **Monitoring**: Add appropriate monitoring

## Quick Reference

### Essential Commands
```bash
# Setup workflows
mkdir -p .github/workflows
cp docs/workflows/examples/*.yml .github/workflows/

# Local testing
act -j ci
```

### Key Links
- [Setup Guide](../SETUP_REQUIRED.md) - Manual setup requirements
- [Examples](examples/) - Workflow templates
- [GitHub Actions Docs](https://docs.github.com/en/actions)

---

**Manual Setup Required**: Repository maintainers must create workflow files from the templates in `docs/workflows/examples/` and configure the required secrets and environments as documented above.