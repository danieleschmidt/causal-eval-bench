# Repository Configuration Guide

This document provides comprehensive guidance for configuring the Causal Eval Bench repository, including settings, integrations, and automation.

## Table of Contents

- [Repository Settings](#repository-settings)
- [Branch Protection](#branch-protection)
- [GitHub Integrations](#github-integrations)
- [Security Configuration](#security-configuration)
- [Automation Setup](#automation-setup)
- [Monitoring & Alerts](#monitoring--alerts)
- [Manual Setup Requirements](#manual-setup-requirements)

## Repository Settings

### Basic Configuration

**Repository Details:**
- **Description**: "Comprehensive evaluation framework for causal reasoning in language models"
- **Homepage**: `https://causal-eval-bench.com` (when documentation site is available)
- **Topics**: `causal-reasoning`, `language-models`, `evaluation`, `benchmarking`, `machine-learning`, `ai`, `nlp`, `python`, `research`

**Features:**
- ✅ Wiki enabled (for community documentation)
- ✅ Issues enabled
- ✅ Projects enabled (for roadmap management)
- ✅ Discussions enabled (for community Q&A)
- ❌ Sponsorships disabled (enable when ready for funding)

**Pull Requests:**
- ✅ Allow merge commits
- ✅ Allow squash merging
- ✅ Allow rebase merging
- ✅ Always suggest updating pull request branches
- ✅ Allow auto-merge
- ✅ Automatically delete head branches after merge

### Advanced Settings

**Merge Queue:**
- Enable merge queue for `main` branch
- Require merge queue for protected branches
- Set merge method: `squash`

**Security & Analysis:**
- ✅ Private vulnerability reporting
- ✅ Dependency graph
- ✅ Dependabot alerts
- ✅ Dependabot security updates
- ✅ Code scanning alerts
- ✅ Secret scanning alerts

## Branch Protection

### Main Branch Protection Rules

Configure the following protection rules for the `main` branch:

```yaml
# .github/branch-protection.yml (for automation)
main:
  protection:
    required_status_checks:
      strict: true
      contexts:
        - "ci/tests"
        - "ci/security-scan"
        - "ci/quality-checks"
        - "ci/docker-build"
    enforce_admins: false
    required_pull_request_reviews:
      required_approving_review_count: 2
      dismiss_stale_reviews: true
      require_code_owner_reviews: true
      require_last_push_approval: true
    restrictions:
      users: []
      teams: ["causal-eval-maintainers"]
    required_linear_history: true
    allow_force_pushes: false
    allow_deletions: false
```

### Development Branch Protection

For `develop` branch (if using GitFlow):

```yaml
develop:
  protection:
    required_status_checks:
      strict: true
      contexts:
        - "ci/tests"
        - "ci/security-scan"
    enforce_admins: false
    required_pull_request_reviews:
      required_approving_review_count: 1
      dismiss_stale_reviews: true
      require_code_owner_reviews: true
    allow_force_pushes: false
    allow_deletions: false
```

## GitHub Integrations

### Required Secrets

Configure the following secrets in repository settings:

**CI/CD Secrets:**
```bash
# Docker Registry
DOCKER_HUB_USERNAME      # Docker Hub username
DOCKER_HUB_TOKEN         # Docker Hub access token

# Package Publishing
PYPI_API_TOKEN           # PyPI publishing token
TEST_PYPI_API_TOKEN      # Test PyPI token for validation

# Security Scanning
SNYK_TOKEN              # Snyk security scanning token
GITLEAKS_LICENSE        # GitLeaks license key (if premium)

# Notifications
SLACK_WEBHOOK_URL       # Slack notifications webhook
EMAIL_USERNAME          # SMTP email username
EMAIL_PASSWORD          # SMTP email password
RELEASE_EMAIL_LIST      # Comma-separated release notification emails

# Documentation
DOCS_DEPLOY_TOKEN       # Documentation deployment token
```

**Environment Secrets:**
```bash
# Production environment
PROD_DATABASE_URL       # Production database connection
PROD_REDIS_URL          # Production cache connection
PROD_S3_BUCKET          # Production file storage

# Staging environment  
STAGING_DATABASE_URL    # Staging database connection
STAGING_REDIS_URL       # Staging cache connection
STAGING_S3_BUCKET       # Staging file storage
```

### Repository Variables

Configure the following variables:

```bash
# Build Configuration
PYTHON_VERSION=3.11
POETRY_VERSION=1.7.1
NODE_VERSION=18
DOCKER_REGISTRY=ghcr.io

# Testing Configuration
PYTEST_WORKERS=auto
COVERAGE_THRESHOLD=80
PERFORMANCE_THRESHOLD=20

# Security Configuration
SECURITY_SCAN_SCHEDULE="0 2 * * *"
DEPENDENCY_UPDATE_SCHEDULE="0 6 * * 1"

# Notification Configuration
SLACK_CHANNEL_GENERAL=#general
SLACK_CHANNEL_RELEASES=#releases
SLACK_CHANNEL_SECURITY=#security
SLACK_CHANNEL_DEPENDENCIES=#dependencies
```

### GitHub Apps Integration

**Recommended Apps:**

1. **Dependabot** (Built-in)
   - Automated dependency updates
   - Security vulnerability alerts
   - Configuration: `.github/dependabot.yml`

2. **CodeQL** (Built-in)
   - Code security analysis
   - Configuration: `.github/workflows/codeql.yml`

3. **Lighthouse CI**
   - Performance monitoring
   - Web vitals tracking

4. **Codecov**
   - Code coverage reporting
   - Coverage change tracking

5. **SonarCloud**
   - Code quality analysis
   - Technical debt tracking

## Security Configuration

### Secret Scanning

**Enabled Patterns:**
- API keys and tokens
- Database credentials
- Private keys and certificates
- Cloud provider credentials
- Third-party service tokens

**Custom Patterns:**
```regex
# Custom API key pattern
[a-zA-Z0-9_-]*api[_-]?key[a-zA-Z0-9_-]*\s*[:=]\s*['\"][a-zA-Z0-9_-]{20,}['\"]

# Custom database URL pattern
(postgres|mysql|mongodb)://[a-zA-Z0-9_-]+:[a-zA-Z0-9_-]+@[a-zA-Z0-9.-]+:[0-9]+/[a-zA-Z0-9_-]+
```

### Vulnerability Alerts

**Configuration:**
- Enable Dependabot alerts
- Auto-dismiss low severity alerts after 90 days
- Require manual review for medium+ severity
- Auto-create issues for high/critical vulnerabilities

### Code Scanning

**CodeQL Configuration:**
```yaml
# .github/codeql-config.yml
name: "CodeQL Config"
queries:
  - uses: security-and-quality
  - uses: security-extended
paths-ignore:
  - "tests/**"
  - "docs/**"
  - "scripts/dev/**"
```

## Automation Setup

### GitHub Actions Workflows

**Required Workflows:**

1. **Continuous Integration** (`ci.yml`)
   - Run on: push, pull_request
   - Jobs: test, lint, security-scan, build

2. **Release Automation** (`release.yml`)
   - Run on: tag push, manual trigger
   - Jobs: build, test, publish, notify

3. **Dependency Updates** (`dependency-update.yml`)
   - Run on: schedule (weekly)
   - Jobs: analyze, update, test, PR creation

4. **Security Scanning** (`security-scan.yml`)
   - Run on: schedule (daily), push to main
   - Jobs: SAST, dependency scan, container scan

### Automated Checks

**Pre-merge Checks:**
- ✅ All tests pass
- ✅ Code coverage ≥ 80%
- ✅ Security scan passes
- ✅ Code quality score ≥ B
- ✅ No high/critical vulnerabilities
- ✅ Required reviews obtained

**Post-merge Actions:**
- 🔄 Deploy to staging environment
- 📊 Update project metrics
- 🔍 Run comprehensive security scan
- 📈 Update performance benchmarks

## Monitoring & Alerts

### Health Checks

**Repository Health Monitoring:**
```yaml
# Repository health metrics
health_checks:
  - code_coverage: ">= 80%"
  - security_vulnerabilities: "== 0"
  - test_success_rate: ">= 95%"
  - build_success_rate: ">= 90%"
  - dependency_freshness: "<= 30 days"
```

**Alert Thresholds:**
- 🚨 Critical: Security vulnerability, build failures
- ⚠️ Warning: Coverage drop >5%, dependency age >60 days
- ℹ️ Info: New contributors, milestone progress

### Notification Channels

**Slack Integration:**
```yaml
notifications:
  slack:
    channels:
      general: "#causal-eval-general"
      releases: "#causal-eval-releases"
      security: "#causal-eval-security"
      dependencies: "#causal-eval-deps"
    events:
      - release_published
      - security_alert
      - build_failure
      - dependency_update
```

**Email Notifications:**
- Release announcements
- Security alerts
- Weekly project summaries
- Milestone achievements

## Manual Setup Requirements

Due to GitHub App permission limitations, the following must be configured manually:

### 1. Repository Settings

Navigate to **Settings** → **General**:

1. Set repository description and homepage URL
2. Add repository topics (see [topics list](#basic-configuration))
3. Configure features (Wiki, Issues, Projects, Discussions)
4. Set up pull request settings

### 2. Branch Protection Rules

Navigate to **Settings** → **Branches**:

1. Add protection rule for `main` branch
2. Configure required status checks
3. Set up review requirements
4. Enable code owner reviews

### 3. Security & Analysis

Navigate to **Settings** → **Security & analysis**:

1. Enable all security features
2. Configure Dependabot settings
3. Set up secret scanning patterns
4. Configure CodeQL analysis

### 4. Secrets and Variables

Navigate to **Settings** → **Secrets and variables**:

1. Add all required secrets (see [Required Secrets](#required-secrets))
2. Configure environment variables
3. Set up environment-specific secrets

### 5. GitHub Apps Installation

Visit the **GitHub Marketplace** and install:

1. Dependabot (usually pre-installed)
2. CodeQL (usually pre-installed)  
3. Additional recommended apps (see [GitHub Apps Integration](#github-apps-integration))

### 6. Team Configuration

Navigate to your organization settings:

1. Create teams referenced in CODEOWNERS
2. Assign team members appropriate permissions
3. Configure team notification settings

### 7. Webhook Configuration

For external integrations:

1. Configure Slack webhook for notifications
2. Set up monitoring system webhooks
3. Configure documentation deployment hooks

## Validation Checklist

After completing the manual setup, verify the configuration:

- [ ] Repository description and topics are set
- [ ] Branch protection rules are active
- [ ] All required secrets are configured
- [ ] GitHub Apps are installed and configured
- [ ] Teams and CODEOWNERS are properly set up
- [ ] Webhooks are functioning
- [ ] Test PR can be created and merged following the rules
- [ ] Notifications are working correctly
- [ ] Security scanning is active
- [ ] Dependency updates are scheduled

## Troubleshooting

### Common Issues

**1. Branch Protection Not Working**
- Verify status check names match workflow job names
- Ensure teams have correct permissions
- Check if enforce admins is appropriately set

**2. Secrets Not Available in Workflows**
- Verify secret names match exactly (case-sensitive)
- Check if running in forked PR (secrets not available)
- Ensure secret is set at repository level, not environment

**3. CODEOWNERS Not Working**
- Verify file is in repository root
- Check team names exist in organization
- Ensure "Require review from code owners" is enabled

**4. Notifications Not Sending**
- Verify webhook URLs are correct
- Check webhook payload format
- Test webhook connectivity

### Getting Help

For configuration issues:

1. Check the [GitHub Documentation](https://docs.github.com)
2. Review workflow run logs for specific errors
3. Create an issue in this repository for project-specific problems
4. Contact the DevOps team for organization-level settings

---

**Note**: This configuration guide should be reviewed and updated quarterly to ensure it remains current with GitHub feature updates and project requirements.