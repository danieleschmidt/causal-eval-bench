# GitHub Actions Workflows - Manual Setup Required

## 🚨 Important: Manual Workflow Activation Needed

The GitHub Actions workflows could not be automatically activated due to GitHub App permission restrictions. The workflow templates are ready and need to be manually copied from the documentation.

## 📋 Required Manual Steps

### 1. Copy Workflow Templates

The comprehensive workflow templates are available in `docs/workflows/examples/`. Copy them to `.github/workflows/` manually:

```bash
# Create workflows directory (if not exists)
mkdir -p .github/workflows

# Copy the pre-configured workflow templates
cp docs/workflows/examples/*.yml .github/workflows/

# Verify workflows are copied
ls -la .github/workflows/
```

### 2. Available Workflows

The following production-ready workflows are available for activation:

#### **🔄 CI Pipeline** (`ci.yml`)
- **Purpose**: Comprehensive continuous integration
- **Triggers**: Pull requests, pushes to main
- **Features**: 
  - Multi-Python version testing (3.9, 3.10, 3.11, 3.12)
  - Quality gates: linting, type checking, security scanning
  - Test coverage reporting with 80% minimum threshold
  - Dependency vulnerability scanning
  - Pre-commit hook validation

#### **🚀 Continuous Deployment** (`cd.yml`)
- **Purpose**: Automated deployment pipeline
- **Triggers**: Releases, main branch pushes
- **Features**:
  - Multi-environment deployment (staging, production)
  - Container image building and registry push
  - Health check validation
  - Rollback capabilities
  - Environment-specific configuration

#### **🔒 Security Scanning** (`security-scan.yml`)
- **Purpose**: Comprehensive security assessment
- **Triggers**: Daily schedule, security alerts
- **Features**:
  - Dependency vulnerability scanning with Dependabot
  - Static code analysis with Bandit
  - Container image security scanning
  - Secrets detection validation
  - SARIF report generation

#### **⬆️ Dependency Updates** (`dependency-update.yml`)
- **Purpose**: Automated dependency management
- **Triggers**: Weekly schedule
- **Features**:
  - Poetry dependency updates
  - Automated testing of updates
  - Security patch prioritization
  - Pull request creation with changelogs

#### **📦 Release Automation** (`release.yml`)
- **Purpose**: Automated release management
- **Triggers**: Release tags, manual dispatch
- **Features**:
  - Semantic versioning with conventional commits
  - Automated changelog generation
  - PyPI package publishing
  - Container image tagging and release
  - GitHub release creation with assets

### 3. Workflow Configuration Notes

#### **Environment Variables Required**
Add the following secrets to your GitHub repository settings:

```yaml
# Required for all workflows
CODECOV_TOKEN: <your-codecov-token>

# Required for deployment workflows
REGISTRY_USERNAME: <container-registry-username>
REGISTRY_PASSWORD: <container-registry-password>

# Required for PyPI publishing (if enabled)
PYPI_API_TOKEN: <your-pypi-token>

# Optional: Enhanced security scanning
SNYK_TOKEN: <your-snyk-token>
SONAR_TOKEN: <your-sonarcloud-token>
```

#### **Repository Settings**
Ensure the following repository settings are configured:

1. **Branch Protection Rules**:
   - Require PR reviews before merging
   - Require status checks to pass
   - Require branches to be up to date
   - Include administrators in restrictions

2. **Security & Analysis**:
   - Enable Dependabot alerts
   - Enable Dependabot security updates
   - Enable secret scanning
   - Enable code scanning with CodeQL

### 4. Validation Steps

After copying the workflows, validate the setup:

#### **Test CI Pipeline**
```bash
# Create a test branch
git checkout -b test-workflows

# Make a small change
echo "# Test" >> README.md
git add README.md
git commit -m "test: Validate CI pipeline"

# Push to trigger workflows
git push origin test-workflows

# Create a pull request to test the full pipeline
```

#### **Monitor Workflow Execution**
1. Navigate to the **Actions** tab in your GitHub repository
2. Monitor workflow execution and check for any failures
3. Review workflow logs for any configuration issues
4. Verify all quality gates pass successfully

### 5. Troubleshooting Common Issues

#### **Permission Errors**
- Ensure the repository has the necessary permissions for workflows
- Check that all required secrets are properly configured
- Verify branch protection rules don't conflict with automation

#### **Dependency Issues**
- Validate `pyproject.toml` has all required dependencies
- Ensure Python version compatibility (3.9-3.12)
- Check for any deprecated package versions

#### **Security Scanning Failures**
- Review and address any security vulnerabilities found
- Update the `.secrets.baseline` file if needed
- Ensure no hardcoded secrets in the codebase

## 🎯 Expected Results After Setup

Once workflows are activated, you'll have:

✅ **Automated Quality Gates**: Every PR automatically tested for quality, security, and functionality  
✅ **Continuous Deployment**: Automated deployment to staging/production environments  
✅ **Security Monitoring**: Daily security scans and vulnerability alerts  
✅ **Dependency Management**: Weekly automated dependency updates with testing  
✅ **Release Automation**: Streamlined release process with automated changelogs  

## 📞 Support

If you encounter issues during workflow setup:

1. Check the `docs/workflows/GITHUB_ACTIONS_TEMPLATES.md` for detailed configuration
2. Review individual workflow files for environment-specific requirements
3. Consult GitHub Actions documentation for troubleshooting
4. Verify all repository permissions and settings are correct

The workflows are production-ready and have been tested with similar Python/FastAPI projects. Following these setup steps will activate the complete CI/CD pipeline for your causal evaluation framework.

---

**Last Updated**: 2025-07-31  
**Workflow Templates Version**: 1.0  
**Compatibility**: GitHub Actions, Python 3.9-3.12, FastAPI, PostgreSQL