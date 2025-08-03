# GitHub Actions Workflow Setup Guide

This guide provides step-by-step instructions for setting up GitHub Actions workflows for the Causal Eval Bench project.

## Prerequisites

Before setting up workflows, ensure you have:

1. **Admin access** to the repository
2. **GitHub Actions enabled** in repository settings
3. **Required secrets** configured
4. **Branch protection rules** (optional but recommended)

## Required Secrets

Configure the following secrets in your repository settings (`Settings > Secrets and variables > Actions`):

### Essential Secrets

```bash
# Container Registry
GITHUB_TOKEN                 # Automatically provided by GitHub

# External API Keys (for testing)
OPENAI_API_KEY              # OpenAI API key for model testing
ANTHROPIC_API_KEY           # Anthropic API key for Claude testing
HUGGINGFACE_API_TOKEN       # Hugging Face API token

# Security Scanning
SNYK_TOKEN                  # Snyk token for vulnerability scanning
CODECOV_TOKEN               # Codecov token for coverage reporting

# Deployment (if using)
DEPLOY_KEY                  # SSH key for deployment
KUBECONFIG                  # Kubernetes config for K8s deployments
```

### Optional Secrets

```bash
# Monitoring
SENTRY_DSN                  # Sentry DSN for error tracking
PROMETHEUS_WEBHOOK          # Prometheus webhook for alerts

# Notifications
SLACK_WEBHOOK               # Slack webhook for notifications
DISCORD_WEBHOOK             # Discord webhook for notifications
```

## Step-by-Step Setup

### 1. Create Workflows Directory

```bash
mkdir -p .github/workflows
```

### 2. Copy Workflow Templates

Copy the desired workflow files from `docs/workflows/examples/` to `.github/workflows/`:

#### Essential Workflows (Start with these)

```bash
# Core CI/CD workflows
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/docker-build.yml .github/workflows/

# Security and dependencies
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/auto-update.yml .github/workflows/
```

#### Advanced Workflows (Add later)

```bash
# Performance and benchmarking
cp docs/workflows/examples/benchmark-suite.yml .github/workflows/

# Production deployment
cp docs/workflows/production-ready/production-deployment.yml .github/workflows/

# Advanced security
cp docs/workflows/production-ready/advanced-security.yml .github/workflows/
```

### 3. Configure Repository Settings

#### Enable GitHub Actions

1. Go to `Settings > Actions > General`
2. Set "Actions permissions" to "Allow all actions and reusable workflows"
3. Set "Workflow permissions" to "Read and write permissions"
4. Check "Allow GitHub Actions to create and approve pull requests"

#### Configure Environments (Optional)

For production deployments, create environments:

1. Go to `Settings > Environments`
2. Create environments: `staging`, `production`
3. Add protection rules and secrets per environment

### 4. Set Up Branch Protection

Protect your main branch with required status checks:

1. Go to `Settings > Branches`
2. Add rule for `main` branch
3. Enable:
   - "Require a pull request before merging"
   - "Require status checks to pass before merging"
   - Select required checks: CI, Security Scan, Tests

### 5. Configure Workflow Files

#### Basic CI Workflow Configuration

Edit `.github/workflows/ci.yml`:

```yaml
# Adjust Python versions if needed
strategy:
  matrix:
    python-version: ["3.9", "3.10", "3.11", "3.12"]

# Customize test commands
- name: Run tests
  run: |
    poetry run pytest tests/ --cov=causal_eval --cov-report=xml
```

#### Docker Build Configuration

Edit `.github/workflows/docker-build.yml`:

```yaml
# Update registry and image name
env:
  REGISTRY: ghcr.io
  IMAGE_NAME: your-org/causal-eval-bench  # Update this

# Customize build targets
strategy:
  matrix:
    target: [development, production]  # Add/remove as needed
```

#### Security Scan Configuration

Edit `.github/workflows/security-scan.yml`:

```yaml
# Configure security tools
- name: Run Bandit
  run: poetry run bandit -r causal_eval/ -f json -o bandit-report.json

# Add custom security checks
- name: Custom security checks
  run: |
    # Add your custom security validation here
```

### 6. Test Workflow Setup

#### Initial Test

1. **Create a test branch**: `git checkout -b test-workflows`
2. **Add workflow files**: `git add .github/workflows/`
3. **Commit changes**: `git commit -m "Add GitHub Actions workflows"`
4. **Push branch**: `git push origin test-workflows`
5. **Create PR**: Open PR to trigger workflows

#### Verify Workflows

Check that workflows run successfully:

1. Go to `Actions` tab in GitHub
2. Verify all workflows trigger on PR
3. Check for any errors or failures
4. Review logs for each workflow step

### 7. Customize for Your Environment

#### Update Repository-Specific Values

Search and replace these placeholders in workflow files:

```bash
# Repository name
your-org/causal-eval-bench → your-actual-org/repo-name

# API endpoints
https://api.your-domain.com → your-actual-api-url

# Container registry
ghcr.io/your-org → your-registry/your-org

# Email addresses
admin@your-domain.com → your-actual-email
```

#### Adjust Resource Limits

For larger repositories, you may need to adjust:

```yaml
# Increase timeout for long-running jobs
timeout-minutes: 60  # Default is 30

# Use more powerful runners for heavy workloads
runs-on: ubuntu-latest-8-cores  # Instead of ubuntu-latest

# Optimize caching for faster builds
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/poetry.lock') }}
```

## Workflow Management

### Monitoring Workflow Performance

#### Set up notifications for workflow failures:

```yaml
# Add to any workflow
- name: Notify on failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

#### Track workflow metrics:

1. Use GitHub's built-in Actions insights
2. Set up alerts for workflow failures
3. Monitor workflow duration trends

### Troubleshooting Common Issues

#### 1. Permission Errors

```yaml
# Ensure correct permissions
permissions:
  contents: read
  packages: write
  security-events: write
```

#### 2. Secret Not Found

```bash
# Check secret name matches exactly (case-sensitive)
${{ secrets.OPENAI_API_KEY }}  # Correct
${{ secrets.openai_api_key }}  # Wrong
```

#### 3. Workflow Not Triggering

```yaml
# Check trigger conditions
on:
  push:
    branches: [main]  # Only triggers on main branch
  pull_request:       # Triggers on all PRs
```

#### 4. Docker Build Failures

```yaml
# Add debugging
- name: Debug Docker build
  run: |
    docker system df
    docker buildx ls
    docker version
```

### Advanced Configurations

#### Matrix Builds

Test across multiple environments:

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ["3.9", "3.10", "3.11"]
    include:
      - os: ubuntu-latest
        python-version: "3.12"
        experimental: true
  fail-fast: false
```

#### Conditional Workflows

Run workflows based on conditions:

```yaml
# Only run on specific file changes
on:
  push:
    paths:
      - 'src/**'
      - 'tests/**'
      - 'pyproject.toml'

# Skip CI for documentation changes
if: "!contains(github.event.head_commit.message, '[skip ci]')"
```

#### Workflow Dependencies

Chain workflows together:

```yaml
# In deployment workflow
needs: [test, security-scan, build]
if: ${{ needs.test.result == 'success' && needs.security-scan.result == 'success' }}
```

## Security Best Practices

### 1. Secret Management

- Use GitHub Secrets for sensitive data
- Never hardcode secrets in workflow files
- Use environment-specific secrets when needed
- Regularly rotate secrets

### 2. Workflow Security

```yaml
# Limit token permissions
permissions:
  contents: read  # Only what's needed

# Pin action versions
uses: actions/checkout@v4  # Specific version
# Not: uses: actions/checkout@main  # Avoid @main
```

### 3. Dependency Security

```yaml
# Scan dependencies
- name: Scan dependencies
  uses: securecodewarrior/github-action-add-sarif@v1
  with:
    sarif-file: dependency-scan.sarif
```

## Maintenance

### Regular Updates

1. **Monthly**: Update action versions
2. **Quarterly**: Review and update workflow logic
3. **Yearly**: Audit all secrets and permissions

### Performance Optimization

1. **Use caching** for dependencies
2. **Parallelize** independent jobs
3. **Optimize** Docker builds with multi-stage builds
4. **Monitor** workflow execution times

### Documentation

1. Keep workflow documentation updated
2. Document any custom modifications
3. Maintain changelog for workflow updates

## Support

If you encounter issues:

1. Check the [GitHub Actions documentation](https://docs.github.com/en/actions)
2. Review workflow logs in the Actions tab
3. Search for similar issues in the repository
4. Create an issue with workflow logs attached

## Example Commit Message

When committing workflow changes:

```bash
git commit -m "ci: add GitHub Actions workflows

- Add CI workflow with testing and linting
- Add Docker build workflow with multi-arch support
- Add security scanning workflow
- Configure automatic dependency updates

Closes #123"
```