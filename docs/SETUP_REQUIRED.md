# Manual Setup Requirements

## Overview
Due to permission limitations, the following items require manual setup by repository maintainers.

## GitHub Actions Workflows
- **Location**: `.github/workflows/`
- **Templates**: Available in `docs/workflows/examples/`
- **Required**: CI, CD, security scanning, dependency updates, release automation
- **Setup**: Copy templates and configure secrets (see [workflows/README.md](workflows/README.md))

## Repository Settings
### Branch Protection Rules
- **Target**: `main` branch
- **Requirements**: 2 reviewers, status checks, up-to-date branches
- **Details**: See [workflows/README.md](workflows/README.md#step-4-branch-protection-rules)

### GitHub Environments
- **Staging**: Auto-deploy from main, development team reviewers
- **Production**: Manual approval, senior developers/DevOps reviewers
- **Configuration**: Environment-specific secrets and protection rules

## External Integrations
### Required for Full SDLC
- **Container Registry**: Docker Hub, AWS ECR, or Azure ACR
- **Monitoring**: Grafana, Prometheus for observability
- **Notifications**: Slack webhooks for deployment status
- **Security Scanning**: Enhanced tools beyond GitHub native scanning

### Optional Enhancements
- **Error Tracking**: Sentry integration
- **Performance Monitoring**: DataDog, New Relic
- **License Compliance**: FOSSA or similar

## Contact
For setup assistance, contact the DevOps team or repository maintainers.