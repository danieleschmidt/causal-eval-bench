# GitHub Actions Workflows Activation Guide

This guide contains comprehensive CI/CD workflows for the Causal Eval Bench project. These workflows implement advanced SDLC practices including security scanning, automated testing, deployment strategies, and continuous monitoring.

## 🚀 Quick Setup

**IMPORTANT**: These workflows are production-ready but require manual activation due to GitHub permissions.

### Activation Steps

1. **Copy workflow files to `.github/workflows/`**:
   ```bash
   cp docs/workflows/examples/*.yml .github/workflows/
   cp docs/workflows/production-ready/*.yml .github/workflows/
   ```

2. **Configure repository secrets** (Settings → Secrets → Actions):
   ```
   DOCKER_REGISTRY_TOKEN    # Docker Hub or registry token
   SLACK_WEBHOOK_URL        # Notifications endpoint
   SONAR_TOKEN             # SonarQube analysis token
   CODECOV_TOKEN           # Code coverage reporting
   ```

3. **Enable workflow permissions** (Settings → Actions → General):
   - ✅ Allow GitHub Actions to create and approve pull requests
   - ✅ Allow GitHub Actions to approve deployment requests

### Available Workflows

## 🔄 Core CI/CD Workflows

### `ci.yml` - Comprehensive Continuous Integration
- **Triggers**: Push to main, PRs
- **Matrix Testing**: Python 3.9-3.12 across Ubuntu/macOS/Windows
- **Quality Gates**: Linting, type checking, security scanning
- **Test Coverage**: Unit, integration, E2E with coverage reporting
- **Artifact Management**: Test reports, coverage data, build artifacts

### `cd.yml` - Continuous Deployment
- **Triggers**: Release tags, manual dispatch
- **Deployment Strategies**: Blue-green, canary, rolling updates
- **Environment Promotion**: dev → staging → production
- **Rollback Capability**: Automatic rollback on failure detection
- **Health Checks**: Post-deployment validation and monitoring

### `security-scan.yml` - Advanced Security Pipeline
- **Dependency Scanning**: Snyk, Safety, GitHub Security Advisories
- **SAST Analysis**: CodeQL, Bandit, Semgrep
- **Container Security**: Trivy, Hadolint
- **Supply Chain**: SLSA attestation, SBOM generation
- **Compliance**: SOC 2, GDPR, security policy enforcement

## 🔧 Specialized Workflows

### `dependency-update.yml` - Automated Dependency Management
- **Schedule**: Weekly dependency updates
- **Security Priority**: Immediate security patch deployment  
- **Compatibility Testing**: Automated testing of dependency updates
- **Smart Batching**: Groups compatible updates for efficiency

### `release.yml` - Automated Release Management
- **Semantic Versioning**: Automatic version bumping
- **Release Notes**: AI-generated changelog from commits
- **Multi-Platform**: PyPI, Docker Hub, GitHub Releases
- **Notification**: Slack/Discord release announcements

### `performance-monitoring.yml` - Continuous Performance Testing
- **Benchmark Tracking**: Performance regression detection
- **Load Testing**: Automated Locust-based load tests
- **Memory Profiling**: Memory leak detection
- **Performance Reports**: Grafana dashboard integration

## 🛠️ Development Workflows

### `docs.yml` - Documentation Automation
- **Auto-Generation**: API docs from code annotations
- **Link Validation**: Broken link detection
- **Multi-Format**: HTML, PDF, and mobile-optimized docs
- **Deployment**: Automated GitHub Pages deployment

### `code-quality.yml` - Advanced Code Quality Checks
- **Static Analysis**: SonarQube, DeepCode integration
- **Technical Debt**: Automated debt tracking and reporting
- **Code Complexity**: Cyclomatic complexity monitoring
- **Maintainability Index**: Long-term codebase health metrics

## 📊 Monitoring & Observability

### `monitoring.yml` - Infrastructure Monitoring
- **Health Checks**: Multi-environment health monitoring
- **Alert Management**: Intelligent alerting with escalation
- **SLA Monitoring**: Uptime and performance SLA tracking
- **Incident Response**: Automated incident creation and routing

### `backup.yml` - Data Protection
- **Database Backups**: Encrypted, versioned database snapshots
- **Configuration Backup**: Infrastructure as Code snapshots
- **Disaster Recovery**: Automated recovery testing
- **Compliance**: GDPR-compliant data retention policies

## 🚦 Workflow Status Dashboard

| Workflow | Status | Last Run | Success Rate |
|----------|--------|----------|--------------|
| CI Pipeline | ⚠️ Setup Required | - | - |
| Security Scan | ⚠️ Setup Required | - | - |
| Deployment | ⚠️ Setup Required | - | - |
| Dependencies | ⚠️ Setup Required | - | - |
| Performance | ⚠️ Setup Required | - | - |

## 🔐 Security Configuration

### Required Secrets
```bash
# Authentication
DOCKER_REGISTRY_TOKEN="<registry-token>"
SONAR_TOKEN="<sonarqube-token>"
CODECOV_TOKEN="<codecov-token>"

# Notifications
SLACK_WEBHOOK_URL="<slack-webhook>"
DISCORD_WEBHOOK_URL="<discord-webhook>"

# Deployment
STAGING_DEPLOY_KEY="<staging-ssh-key>"
PROD_DEPLOY_KEY="<production-ssh-key>"

# Monitoring
DATADOG_API_KEY="<datadog-key>"
SENTRY_DSN="<sentry-dsn>"
```

### Environment Variables
```bash
# Application
ENVIRONMENT="production"
LOG_LEVEL="info"
DEBUG="false"

# Database
DATABASE_URL="postgresql://..."
REDIS_URL="redis://..."

# External Services
OPENAI_API_KEY="<openai-key>"
ANTHROPIC_API_KEY="<anthropic-key>"
```

## 📋 Workflow Features

### 🎯 Advanced Testing
- **Parallel Execution**: Matrix builds across multiple environments
- **Flaky Test Detection**: Automatic flaky test identification and quarantine
- **Visual Regression**: Screenshot-based UI testing
- **Accessibility Testing**: WCAG compliance validation

### 🔒 Security Features
- **Zero-Trust Architecture**: All workflows assume compromise
- **Secrets Rotation**: Automated secret rotation and validation
- **Compliance Reporting**: Automated compliance evidence collection
- **Threat Modeling**: Automated threat model updates

### 📈 Performance Features
- **Benchmark Tracking**: Historical performance trend analysis
- **Resource Optimization**: Automatic resource usage optimization
- **Capacity Planning**: Predictive scaling recommendations
- **Cost Optimization**: Cloud cost tracking and optimization

### 🚀 Deployment Features
- **Blue-Green Deployment**: Zero-downtime deployments
- **Canary Releases**: Gradual rollout with automatic rollback
- **Feature Flags**: Runtime feature toggling
- **Database Migrations**: Safe, reversible schema changes

## 🎛️ Configuration

### Workflow Customization
Each workflow supports extensive customization through:
- **Input Parameters**: Runtime configuration options
- **Environment Variables**: Environment-specific settings
- **Conditional Execution**: Smart workflow execution based on changes
- **Custom Actions**: Repository-specific action extensions

### Matrix Strategies
```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11', '3.12']
    os: [ubuntu-latest, macos-latest, windows-latest]
    include:
      - python-version: '3.12'
        os: ubuntu-latest
        coverage: true
```

## 📞 Support & Troubleshooting

### Common Issues

1. **Workflow Permission Denied**
   ```bash
   # Solution: Enable workflow permissions in repository settings
   Settings → Actions → General → Workflow permissions
   ```

2. **Secret Not Found**
   ```bash
   # Solution: Add required secrets in repository settings
   Settings → Secrets and variables → Actions
   ```

3. **Build Failures**
   ```bash
   # Check workflow logs and run locally:
   act --list  # List available workflows
   act -j build  # Run specific job locally
   ```

### Getting Help
- 📖 [Workflow Documentation](../docs/workflows/)
- 🐛 [Report Issues](../../issues/new?template=workflow-issue.yml)
- 💬 [Discussion Forum](../../discussions)
- 📧 Email: devops@terragon-labs.com

---

**Last Updated**: 2025-08-01
**Workflow Version**: v2.0.0
**Maintenance**: Automated via dependabot