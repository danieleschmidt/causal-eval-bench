#!/usr/bin/env python3
"""Production deployment readiness assessment."""

import sys
import os
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

sys.path.insert(0, os.path.dirname(__file__))

class ReadinessLevel(Enum):
    """Deployment readiness levels."""
    NOT_READY = "not_ready"
    BASIC = "basic"
    PRODUCTION = "production"
    ENTERPRISE = "enterprise"

@dataclass
class DeploymentCheck:
    """Deployment readiness check result."""
    category: str
    name: str
    status: bool
    level: ReadinessLevel
    message: str
    recommendation: Optional[str] = None
    required_for_production: bool = True

class DeploymentReadinessChecker:
    """Comprehensive deployment readiness checker."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.checks: List[DeploymentCheck] = []
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all deployment readiness checks."""
        print("🚀 Deployment Readiness Assessment")
        print("=" * 45)
        
        # Core infrastructure checks
        self.check_containerization()
        self.check_configuration_management()
        self.check_environment_setup()
        self.check_security_hardening()
        self.check_monitoring_observability()
        self.check_testing_coverage()
        self.check_ci_cd_pipeline()
        self.check_documentation()
        self.check_scalability_features()
        self.check_backup_recovery()
        
        return self.generate_readiness_report()
    
    def check_containerization(self):
        """Check Docker and container readiness."""
        print("🐳 Checking containerization...")
        
        # Check Dockerfile
        dockerfile_path = self.project_root / "Dockerfile"
        if dockerfile_path.exists():
            self._check_dockerfile_best_practices(dockerfile_path)
        else:
            self.checks.append(DeploymentCheck(
                category="Containerization",
                name="Dockerfile",
                status=False,
                level=ReadinessLevel.NOT_READY,
                message="Dockerfile not found",
                recommendation="Create production-ready Dockerfile with multi-stage builds",
                required_for_production=True
            ))
        
        # Check docker-compose
        compose_path = self.project_root / "docker-compose.yml"
        if compose_path.exists():
            self._check_docker_compose(compose_path)
        else:
            self.checks.append(DeploymentCheck(
                category="Containerization", 
                name="Docker Compose",
                status=False,
                level=ReadinessLevel.BASIC,
                message="docker-compose.yml not found",
                recommendation="Create docker-compose.yml for local development",
                required_for_production=False
            ))
        
        # Check .dockerignore
        dockerignore_path = self.project_root / ".dockerignore"
        self.checks.append(DeploymentCheck(
            category="Containerization",
            name=".dockerignore",
            status=dockerignore_path.exists(),
            level=ReadinessLevel.PRODUCTION,
            message=".dockerignore exists" if dockerignore_path.exists() else ".dockerignore missing",
            recommendation="Create .dockerignore to optimize build contexts" if not dockerignore_path.exists() else None,
            required_for_production=True
        ))
    
    def _check_dockerfile_best_practices(self, dockerfile_path: Path):
        """Check Dockerfile for best practices."""
        try:
            content = dockerfile_path.read_text()
            
            # Multi-stage build check
            multi_stage = "FROM" in content and content.count("FROM") > 1
            self.checks.append(DeploymentCheck(
                category="Containerization",
                name="Multi-stage Dockerfile",
                status=multi_stage,
                level=ReadinessLevel.PRODUCTION,
                message="Multi-stage build detected" if multi_stage else "Single-stage build",
                recommendation="Use multi-stage builds to reduce image size" if not multi_stage else None,
                required_for_production=True
            ))
            
            # Non-root user check
            non_root = "USER" in content and "USER root" not in content
            self.checks.append(DeploymentCheck(
                category="Containerization",
                name="Non-root user",
                status=non_root,
                level=ReadinessLevel.PRODUCTION,
                message="Non-root user configured" if non_root else "Running as root",
                recommendation="Configure non-root user for security" if not non_root else None,
                required_for_production=True
            ))
            
            # Health check
            healthcheck = "HEALTHCHECK" in content
            self.checks.append(DeploymentCheck(
                category="Containerization",
                name="Health check",
                status=healthcheck,
                level=ReadinessLevel.PRODUCTION,
                message="Health check configured" if healthcheck else "No health check",
                recommendation="Add HEALTHCHECK instruction" if not healthcheck else None,
                required_for_production=True
            ))
            
        except Exception as e:
            self.checks.append(DeploymentCheck(
                category="Containerization",
                name="Dockerfile validation",
                status=False,
                level=ReadinessLevel.NOT_READY,
                message=f"Error reading Dockerfile: {e}",
                recommendation="Fix Dockerfile syntax errors"
            ))
    
    def _check_docker_compose(self, compose_path: Path):
        """Check docker-compose configuration."""
        try:
            with open(compose_path) as f:
                compose_data = yaml.safe_load(f)
            
            # Version check
            version = compose_data.get("version", "")
            modern_version = version >= "3.7"
            self.checks.append(DeploymentCheck(
                category="Containerization",
                name="Docker Compose version",
                status=modern_version,
                level=ReadinessLevel.PRODUCTION,
                message=f"Compose version {version}" if version else "No version specified",
                recommendation="Use Compose version 3.7+ for modern features" if not modern_version else None
            ))
            
            # Services check
            services = compose_data.get("services", {})
            self.checks.append(DeploymentCheck(
                category="Containerization",
                name="Service definitions",
                status=len(services) > 0,
                level=ReadinessLevel.BASIC,
                message=f"{len(services)} services defined",
                recommendation="Define application services" if len(services) == 0 else None
            ))
            
        except Exception as e:
            self.checks.append(DeploymentCheck(
                category="Containerization",
                name="Docker Compose validation",
                status=False,
                level=ReadinessLevel.BASIC,
                message=f"Error reading docker-compose.yml: {e}",
                recommendation="Fix docker-compose.yml syntax"
            ))
    
    def check_configuration_management(self):
        """Check configuration management setup."""
        print("⚙️ Checking configuration management...")
        
        # Environment file templates
        env_files = [".env.example", ".env.template", "env.example"]
        env_template_exists = any((self.project_root / ef).exists() for ef in env_files)
        
        self.checks.append(DeploymentCheck(
            category="Configuration",
            name="Environment template",
            status=env_template_exists,
            level=ReadinessLevel.PRODUCTION,
            message="Environment template found" if env_template_exists else "No environment template",
            recommendation="Create .env.example with all required variables" if not env_template_exists else None,
            required_for_production=True
        ))
        
        # Configuration validation
        config_files = ["config.py", "settings.py", "causal_eval/config/"]
        config_exists = any((self.project_root / cf).exists() for cf in config_files)
        
        self.checks.append(DeploymentCheck(
            category="Configuration",
            name="Configuration structure",
            status=config_exists,
            level=ReadinessLevel.BASIC,
            message="Configuration files found" if config_exists else "No configuration files",
            recommendation="Organize configuration in dedicated files/directory" if not config_exists else None,
            required_for_production=True
        ))
        
        # Secrets management
        secrets_files = ["secrets/", "vault/", "k8s-secrets/"]
        secrets_setup = any((self.project_root / sf).exists() for sf in secrets_files)
        
        self.checks.append(DeploymentCheck(
            category="Configuration",
            name="Secrets management",
            status=secrets_setup,
            level=ReadinessLevel.ENTERPRISE,
            message="Secrets management setup" if secrets_setup else "No secrets management",
            recommendation="Implement external secrets management (Vault, K8s secrets)" if not secrets_setup else None,
            required_for_production=False
        ))
    
    def check_environment_setup(self):
        """Check environment and dependency management."""
        print("🌍 Checking environment setup...")
        
        # Python version specification
        python_files = ["pyproject.toml", "requirements.txt", "Pipfile", "environment.yml"]
        deps_specified = any((self.project_root / pf).exists() for pf in python_files)
        
        self.checks.append(DeploymentCheck(
            category="Environment",
            name="Dependencies specification",
            status=deps_specified,
            level=ReadinessLevel.BASIC,
            message="Dependencies specified" if deps_specified else "No dependency specification",
            recommendation="Create pyproject.toml or requirements.txt" if not deps_specified else None,
            required_for_production=True
        ))
        
        # Version pinning check
        if (self.project_root / "pyproject.toml").exists():
            self._check_version_pinning()
        
        # Runtime requirements
        runtime_files = ["Dockerfile", "docker-compose.yml", "k8s/"]
        runtime_defined = any((self.project_root / rf).exists() for rf in runtime_files)
        
        self.checks.append(DeploymentCheck(
            category="Environment",
            name="Runtime environment",
            status=runtime_defined,
            level=ReadinessLevel.PRODUCTION,
            message="Runtime environment defined" if runtime_defined else "No runtime environment",
            recommendation="Define runtime environment (Docker/K8s)" if not runtime_defined else None,
            required_for_production=True
        ))
    
    def _check_version_pinning(self):
        """Check if dependencies have pinned versions."""
        try:
            pyproject_path = self.project_root / "pyproject.toml"
            content = pyproject_path.read_text()
            
            # Look for version specifications with operators
            version_patterns = ["==", "~=", ">=", "<="]
            has_pinning = any(pattern in content for pattern in version_patterns)
            
            self.checks.append(DeploymentCheck(
                category="Environment",
                name="Version pinning",
                status=has_pinning,
                level=ReadinessLevel.PRODUCTION,
                message="Dependencies have version constraints" if has_pinning else "No version constraints",
                recommendation="Pin dependency versions for reproducible builds" if not has_pinning else None,
                required_for_production=True
            ))
            
        except Exception as e:
            pass  # Skip if can't read file
    
    def check_security_hardening(self):
        """Check security hardening measures."""
        print("🔒 Checking security hardening...")
        
        # Security configuration files
        security_files = ["SECURITY.md", "security.py", ".security/"]
        security_docs = any((self.project_root / sf).exists() for sf in security_files)
        
        self.checks.append(DeploymentCheck(
            category="Security",
            name="Security documentation",
            status=security_docs,
            level=ReadinessLevel.PRODUCTION,
            message="Security documentation exists" if security_docs else "No security documentation",
            recommendation="Create SECURITY.md with security policies" if not security_docs else None,
            required_for_production=True
        ))
        
        # TLS/SSL configuration
        ssl_files = ["ssl/", "certs/", "tls/", "nginx.conf"]
        ssl_config = any((self.project_root / sf).exists() for sf in ssl_files)
        
        self.checks.append(DeploymentCheck(
            category="Security",
            name="TLS/SSL configuration",
            status=ssl_config,
            level=ReadinessLevel.PRODUCTION,
            message="SSL configuration found" if ssl_config else "No SSL configuration",
            recommendation="Configure TLS/SSL for HTTPS" if not ssl_config else None,
            required_for_production=True
        ))
        
        # Input validation middleware
        validation_files = list(self.project_root.rglob("*validation*"))
        input_validation = len(validation_files) > 0
        
        self.checks.append(DeploymentCheck(
            category="Security",
            name="Input validation",
            status=input_validation,
            level=ReadinessLevel.PRODUCTION,
            message="Input validation implemented" if input_validation else "No input validation",
            recommendation="Implement comprehensive input validation" if not input_validation else None,
            required_for_production=True
        ))
    
    def check_monitoring_observability(self):
        """Check monitoring and observability setup."""
        print("📊 Checking monitoring and observability...")
        
        # Logging configuration
        logging_files = ["logging.conf", "log_config.py", "*logging*"]
        logging_config = any((self.project_root / lf).exists() for lf in logging_files[:2]) or \
                       len(list(self.project_root.rglob("*logging*"))) > 0
        
        self.checks.append(DeploymentCheck(
            category="Monitoring",
            name="Logging configuration",
            status=logging_config,
            level=ReadinessLevel.BASIC,
            message="Logging configured" if logging_config else "No logging configuration",
            recommendation="Configure structured logging" if not logging_config else None,
            required_for_production=True
        ))
        
        # Health endpoints
        health_files = list(self.project_root.rglob("*health*")) + list(self.project_root.rglob("*monitoring*"))
        health_endpoints = len(health_files) > 0
        
        self.checks.append(DeploymentCheck(
            category="Monitoring",
            name="Health endpoints",
            status=health_endpoints,
            level=ReadinessLevel.PRODUCTION,
            message="Health endpoints implemented" if health_endpoints else "No health endpoints",
            recommendation="Implement /health, /ready, /live endpoints" if not health_endpoints else None,
            required_for_production=True
        ))\n        \n        # Metrics collection\n        metrics_files = [\"prometheus.yml\", \"metrics.py\", \"grafana/\"]\n        metrics_setup = any((self.project_root / mf).exists() for mf in metrics_files)\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Monitoring\",\n            name=\"Metrics collection\",\n            status=metrics_setup,\n            level=ReadinessLevel.ENTERPRISE,\n            message=\"Metrics collection configured\" if metrics_setup else \"No metrics collection\",\n            recommendation=\"Set up Prometheus/Grafana metrics\" if not metrics_setup else None,\n            required_for_production=False\n        ))\n    \n    def check_testing_coverage(self):\n        \"\"\"Check testing and quality assurance.\"\"\"\n        print(\"🧪 Checking testing coverage...\")\n        \n        # Test directories\n        test_dirs = [\"tests/\", \"test/\", \"*test*\"]\n        test_structure = any((self.project_root / td).exists() for td in test_dirs[:2]) or \\\n                        len(list(self.project_root.rglob(\"test_*.py\"))) > 0\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Testing\",\n            name=\"Test structure\",\n            status=test_structure,\n            level=ReadinessLevel.BASIC,\n            message=\"Test structure exists\" if test_structure else \"No test structure\",\n            recommendation=\"Create comprehensive test suite\" if not test_structure else None,\n            required_for_production=True\n        ))\n        \n        # Test configuration\n        test_config_files = [\"pytest.ini\", \"pyproject.toml\", \"tox.ini\", \".coveragerc\"]\n        test_config = any((self.project_root / tcf).exists() for tcf in test_config_files)\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Testing\",\n            name=\"Test configuration\",\n            status=test_config,\n            level=ReadinessLevel.PRODUCTION,\n            message=\"Test configuration found\" if test_config else \"No test configuration\",\n            recommendation=\"Configure pytest and coverage settings\" if not test_config else None,\n            required_for_production=True\n        ))\n        \n        # Pre-commit hooks\n        precommit_file = self.project_root / \".pre-commit-config.yaml\"\n        precommit_hooks = precommit_file.exists()\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Testing\",\n            name=\"Pre-commit hooks\",\n            status=precommit_hooks,\n            level=ReadinessLevel.PRODUCTION,\n            message=\"Pre-commit hooks configured\" if precommit_hooks else \"No pre-commit hooks\",\n            recommendation=\"Set up pre-commit hooks for code quality\" if not precommit_hooks else None,\n            required_for_production=True\n        ))\n    \n    def check_ci_cd_pipeline(self):\n        \"\"\"Check CI/CD pipeline configuration.\"\"\"\n        print(\"🔄 Checking CI/CD pipeline...\")\n        \n        # GitHub Actions\n        github_workflows = self.project_root / \".github\" / \"workflows\"\n        github_ci = github_workflows.exists() and len(list(github_workflows.glob(\"*.yml\"))) > 0\n        \n        # GitLab CI\n        gitlab_ci = (self.project_root / \".gitlab-ci.yml\").exists()\n        \n        # Other CI systems\n        ci_files = [\"Jenkinsfile\", \"azure-pipelines.yml\", \"buildkite.yml\", \"circle.yml\"]\n        other_ci = any((self.project_root / cf).exists() for cf in ci_files)\n        \n        ci_configured = github_ci or gitlab_ci or other_ci\n        \n        self.checks.append(DeploymentCheck(\n            category=\"CI/CD\",\n            name=\"CI pipeline\",\n            status=ci_configured,\n            level=ReadinessLevel.PRODUCTION,\n            message=\"CI pipeline configured\" if ci_configured else \"No CI pipeline\",\n            recommendation=\"Set up GitHub Actions or similar CI system\" if not ci_configured else None,\n            required_for_production=True\n        ))\n        \n        # Deployment automation\n        deployment_files = [\"deploy.sh\", \"Makefile\", \"k8s/\", \"helm/\", \"terraform/\"]\n        deployment_automation = any((self.project_root / df).exists() for df in deployment_files)\n        \n        self.checks.append(DeploymentCheck(\n            category=\"CI/CD\",\n            name=\"Deployment automation\",\n            status=deployment_automation,\n            level=ReadinessLevel.PRODUCTION,\n            message=\"Deployment automation exists\" if deployment_automation else \"No deployment automation\",\n            recommendation=\"Create automated deployment scripts/configs\" if not deployment_automation else None,\n            required_for_production=True\n        ))\n    \n    def check_documentation(self):\n        \"\"\"Check documentation completeness.\"\"\"\n        print(\"📚 Checking documentation...\")\n        \n        # README\n        readme_files = [\"README.md\", \"README.rst\", \"README.txt\"]\n        readme_exists = any((self.project_root / rf).exists() for rf in readme_files)\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Documentation\",\n            name=\"README file\",\n            status=readme_exists,\n            level=ReadinessLevel.BASIC,\n            message=\"README exists\" if readme_exists else \"No README\",\n            recommendation=\"Create comprehensive README.md\" if not readme_exists else None,\n            required_for_production=True\n        ))\n        \n        # API documentation\n        api_docs = [\"docs/\", \"openapi.json\", \"swagger.yml\", \"api.md\"]\n        api_documentation = any((self.project_root / ad).exists() for ad in api_docs)\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Documentation\",\n            name=\"API documentation\",\n            status=api_documentation,\n            level=ReadinessLevel.PRODUCTION,\n            message=\"API documentation exists\" if api_documentation else \"No API documentation\",\n            recommendation=\"Document API endpoints with OpenAPI/Swagger\" if not api_documentation else None,\n            required_for_production=True\n        ))\n        \n        # Deployment guide\n        deployment_docs = [\"DEPLOY.md\", \"deployment.md\", \"docs/deployment/\"]\n        deployment_guide = any((self.project_root / dd).exists() for dd in deployment_docs)\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Documentation\",\n            name=\"Deployment guide\",\n            status=deployment_guide,\n            level=ReadinessLevel.PRODUCTION,\n            message=\"Deployment guide exists\" if deployment_guide else \"No deployment guide\",\n            recommendation=\"Create deployment documentation\" if not deployment_guide else None,\n            required_for_production=True\n        ))\n    \n    def check_scalability_features(self):\n        \"\"\"Check scalability and performance features.\"\"\"\n        print(\"⚡ Checking scalability features...\")\n        \n        # Load balancing configuration\n        lb_files = [\"nginx.conf\", \"haproxy.cfg\", \"traefik.yml\", \"k8s/ingress.yml\"]\n        load_balancing = any((self.project_root / lf).exists() for lf in lb_files)\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Scalability\",\n            name=\"Load balancing\",\n            status=load_balancing,\n            level=ReadinessLevel.ENTERPRISE,\n            message=\"Load balancing configured\" if load_balancing else \"No load balancing\",\n            recommendation=\"Configure load balancer (Nginx, HAProxy, etc.)\" if not load_balancing else None,\n            required_for_production=False\n        ))\n        \n        # Caching configuration\n        cache_files = [\"redis.conf\", \"memcached.conf\", \"cache/\"] + list(self.project_root.rglob(\"*cache*\"))\n        caching_setup = len(cache_files) > 3  # More than just the glob patterns\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Scalability\",\n            name=\"Caching system\",\n            status=caching_setup,\n            level=ReadinessLevel.PRODUCTION,\n            message=\"Caching system configured\" if caching_setup else \"No caching system\",\n            recommendation=\"Implement Redis or similar caching\" if not caching_setup else None,\n            required_for_production=True\n        ))\n        \n        # Database optimization\n        db_files = [\"migrations/\", \"alembic/\", \"db/\", \"database.py\"]\n        db_management = any((self.project_root / df).exists() for df in db_files)\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Scalability\",\n            name=\"Database management\",\n            status=db_management,\n            level=ReadinessLevel.PRODUCTION,\n            message=\"Database management configured\" if db_management else \"No database management\",\n            recommendation=\"Set up database migrations and management\" if not db_management else None,\n            required_for_production=True\n        ))\n    \n    def check_backup_recovery(self):\n        \"\"\"Check backup and disaster recovery.\"\"\"\n        print(\"💾 Checking backup and recovery...\")\n        \n        # Backup scripts\n        backup_files = [\"backup.sh\", \"scripts/backup/\", \"backup/\", \"disaster-recovery/\"]\n        backup_strategy = any((self.project_root / bf).exists() for bf in backup_files)\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Backup/Recovery\",\n            name=\"Backup strategy\",\n            status=backup_strategy,\n            level=ReadinessLevel.ENTERPRISE,\n            message=\"Backup strategy defined\" if backup_strategy else \"No backup strategy\",\n            recommendation=\"Create backup and recovery procedures\" if not backup_strategy else None,\n            required_for_production=False\n        ))\n        \n        # Recovery procedures\n        recovery_docs = [\"RECOVERY.md\", \"disaster-recovery.md\", \"docs/recovery/\"]\n        recovery_procedures = any((self.project_root / rd).exists() for rd in recovery_docs)\n        \n        self.checks.append(DeploymentCheck(\n            category=\"Backup/Recovery\",\n            name=\"Recovery procedures\",\n            status=recovery_procedures,\n            level=ReadinessLevel.ENTERPRISE,\n            message=\"Recovery procedures documented\" if recovery_procedures else \"No recovery procedures\",\n            recommendation=\"Document disaster recovery procedures\" if not recovery_procedures else None,\n            required_for_production=False\n        ))\n    \n    def generate_readiness_report(self) -> Dict[str, Any]:\n        \"\"\"Generate comprehensive readiness report.\"\"\"\n        # Group checks by category\n        by_category = {}\n        for check in self.checks:\n            if check.category not in by_category:\n                by_category[check.category] = []\n            by_category[check.category].append(check)\n        \n        # Calculate scores\n        total_checks = len(self.checks)\n        passed_checks = sum(1 for check in self.checks if check.status)\n        production_required = [check for check in self.checks if check.required_for_production]\n        production_passed = sum(1 for check in production_required if check.status)\n        \n        overall_score = (passed_checks / total_checks) * 100 if total_checks > 0 else 0\n        production_score = (production_passed / len(production_required)) * 100 if production_required else 0\n        \n        # Determine readiness level\n        if production_score >= 90:\n            readiness_level = ReadinessLevel.ENTERPRISE\n        elif production_score >= 75:\n            readiness_level = ReadinessLevel.PRODUCTION\n        elif production_score >= 50:\n            readiness_level = ReadinessLevel.BASIC\n        else:\n            readiness_level = ReadinessLevel.NOT_READY\n        \n        return {\n            \"readiness_level\": readiness_level.value,\n            \"overall_score\": overall_score,\n            \"production_score\": production_score,\n            \"total_checks\": total_checks,\n            \"passed_checks\": passed_checks,\n            \"production_required_checks\": len(production_required),\n            \"production_passed_checks\": production_passed,\n            \"by_category\": {\n                category: {\n                    \"total\": len(checks),\n                    \"passed\": sum(1 for check in checks if check.status),\n                    \"score\": (sum(1 for check in checks if check.status) / len(checks)) * 100\n                }\n                for category, checks in by_category.items()\n            },\n            \"failed_checks\": [\n                {\n                    \"category\": check.category,\n                    \"name\": check.name,\n                    \"level\": check.level.value,\n                    \"message\": check.message,\n                    \"recommendation\": check.recommendation,\n                    \"required_for_production\": check.required_for_production\n                }\n                for check in self.checks if not check.status\n            ],\n            \"recommendations\": self._generate_deployment_recommendations(readiness_level, by_category)\n        }\n    \n    def _generate_deployment_recommendations(self, readiness_level: ReadinessLevel, by_category: Dict) -> List[str]:\n        \"\"\"Generate deployment recommendations based on readiness level.\"\"\"\n        recommendations = []\n        \n        if readiness_level == ReadinessLevel.NOT_READY:\n            recommendations.extend([\n                \"🚨 Critical: Address fundamental deployment issues before production\",\n                \"📋 Focus on containerization and basic configuration\",\n                \"🔧 Set up CI/CD pipeline and testing infrastructure\"\n            ])\n        elif readiness_level == ReadinessLevel.BASIC:\n            recommendations.extend([\n                \"⚠️ Address security hardening and monitoring gaps\",\n                \"🔍 Complete production-required checks\",\n                \"📊 Implement comprehensive observability\"\n            ])\n        elif readiness_level == ReadinessLevel.PRODUCTION:\n            recommendations.extend([\n                \"🎯 Ready for production deployment\",\n                \"🚀 Consider enterprise-grade features for scaling\",\n                \"💾 Plan backup and disaster recovery procedures\"\n            ])\n        else:  # ENTERPRISE\n            recommendations.extend([\n                \"🏆 Enterprise-ready deployment\",\n                \"✅ All critical systems in place\",\n                \"🔄 Focus on continuous improvement and optimization\"\n            ])\n        \n        # Category-specific recommendations\n        for category, checks in by_category.items():\n            failed_count = len(checks) - checks[\"passed\"]\n            if failed_count > 0:\n                if category == \"Security\":\n                    recommendations.append(f\"🔒 Fix {failed_count} security issues\")\n                elif category == \"Containerization\":\n                    recommendations.append(f\"🐳 Complete {failed_count} containerization tasks\")\n                elif category == \"Monitoring\":\n                    recommendations.append(f\"📊 Set up {failed_count} monitoring components\")\n        \n        return recommendations\n\n\ndef main():\n    \"\"\"Run deployment readiness assessment.\"\"\"\n    print(\"🚀 Causal Evaluation Bench - Deployment Readiness\")\n    print(\"=\" * 52)\n    \n    checker = DeploymentReadinessChecker()\n    report = checker.run_all_checks()\n    \n    print(f\"\\n📊 DEPLOYMENT READINESS RESULTS:\")\n    print(f\"  Readiness Level: {report['readiness_level'].upper()}\")\n    print(f\"  Overall Score: {report['overall_score']:.1f}%\")\n    print(f\"  Production Score: {report['production_score']:.1f}%\")\n    print(f\"  Checks Passed: {report['passed_checks']}/{report['total_checks']}\")\n    print(f\"  Production Required: {report['production_passed_checks']}/{report['production_required_checks']}\")\n    \n    if report['by_category']:\n        print(f\"\\n📋 CATEGORY SCORES:\")\n        for category, stats in report['by_category'].items():\n            status_icon = \"✅\" if stats['score'] >= 75 else \"⚠️\" if stats['score'] >= 50 else \"❌\"\n            print(f\"  {status_icon} {category}: {stats['score']:.1f}% ({stats['passed']}/{stats['total']})\")\n    \n    print(f\"\\n💡 DEPLOYMENT RECOMMENDATIONS:\")\n    for i, rec in enumerate(report['recommendations'], 1):\n        print(f\"  {i}. {rec}\")\n    \n    # Show critical failed checks\n    critical_failures = [fc for fc in report['failed_checks'] if fc['required_for_production']]\n    if critical_failures:\n        print(f\"\\n🚨 CRITICAL DEPLOYMENT BLOCKERS:\")\n        for i, failure in enumerate(critical_failures[:10], 1):\n            print(f\"  {i}. [{failure['category']}] {failure['name']}\")\n            print(f\"     Issue: {failure['message']}\")\n            print(f\"     Fix: {failure['recommendation']}\")\n            print()\n    \n    # Determine deployment readiness\n    readiness_level = ReadinessLevel(report['readiness_level'])\n    production_score = report['production_score']\n    \n    print(\"=\" * 52)\n    \n    if readiness_level == ReadinessLevel.ENTERPRISE:\n        print(\"🏆 DEPLOYMENT STATUS: ENTERPRISE READY\")\n        print(\"✅ Fully prepared for large-scale production deployment\")\n        deployment_ready = True\n    elif readiness_level == ReadinessLevel.PRODUCTION:\n        print(\"🚀 DEPLOYMENT STATUS: PRODUCTION READY\")\n        print(\"✅ Ready for production deployment with standard features\")\n        deployment_ready = True\n    elif readiness_level == ReadinessLevel.BASIC:\n        print(\"🟡 DEPLOYMENT STATUS: BASIC READY\")\n        print(\"⚠️ Suitable for staging/development, needs work for production\")\n        deployment_ready = production_score >= 70\n    else:\n        print(\"🔴 DEPLOYMENT STATUS: NOT READY\")\n        print(\"❌ Significant work required before any deployment\")\n        deployment_ready = False\n    \n    # Save report\n    report_path = Path(\"deployment_readiness_report.json\")\n    with open(report_path, 'w') as f:\n        json.dump(report, f, indent=2, default=str)\n    \n    print(f\"\\n📄 Detailed report saved to: {report_path}\")\n    \n    return deployment_ready\n\n\nif __name__ == \"__main__\":\n    success = main()\n    sys.exit(0 if success else 1)"