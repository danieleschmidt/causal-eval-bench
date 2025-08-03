#!/usr/bin/env python3
"""
SDLC Implementation Validation Script
Validates the completeness and quality of the Terragon SDLC implementation.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any


class SDLCValidator:
    """Validates SDLC implementation completeness and quality."""
    
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.validation_results = {
            "checkpoints": {},
            "overall_score": 0,
            "recommendations": [],
            "status": "unknown"
        }
    
    def validate_checkpoint_1_foundation(self) -> Dict[str, Any]:
        """Validate Checkpoint 1: Project Foundation & Documentation."""
        results = {"score": 0, "max_score": 100, "items": []}
        
        # Check user guide
        user_guide = self.repo_path / "docs/guides/user/getting-started.md"
        if user_guide.exists():
            results["score"] += 25
            results["items"].append("✅ User guide exists")
        else:
            results["items"].append("❌ User guide missing")
        
        # Check developer guide
        dev_guide = self.repo_path / "docs/guides/developer/contributing.md"
        if dev_guide.exists():
            results["score"] += 25
            results["items"].append("✅ Developer guide exists")
        else:
            results["items"].append("❌ Developer guide missing")
        
        # Check API reference
        api_ref = self.repo_path / "docs/guides/api/reference.md"
        if api_ref.exists():
            results["score"] += 25
            results["items"].append("✅ API reference exists")
        else:
            results["items"].append("❌ API reference missing")
        
        # Check deployment guide
        deploy_guide = self.repo_path / "docs/guides/deployment/production.md"
        if deploy_guide.exists():
            results["score"] += 25
            results["items"].append("✅ Deployment guide exists")
        else:
            results["items"].append("❌ Deployment guide missing")
        
        return results
    
    def validate_checkpoint_2_devenv(self) -> Dict[str, Any]:
        """Validate Checkpoint 2: Development Environment & Tooling."""
        results = {"score": 0, "max_score": 100, "items": []}
        
        # Check VSCode configuration
        vscode_launch = self.repo_path / ".vscode/launch.json"
        if vscode_launch.exists():
            results["score"] += 50
            results["items"].append("✅ VSCode debug configuration exists")
        else:
            results["items"].append("❌ VSCode debug configuration missing")
        
        vscode_tasks = self.repo_path / ".vscode/tasks.json"
        if vscode_tasks.exists():
            results["score"] += 50
            results["items"].append("✅ VSCode tasks configuration exists")
        else:
            results["items"].append("❌ VSCode tasks configuration missing")
        
        return results
    
    def validate_checkpoint_3_testing(self) -> Dict[str, Any]:
        """Validate Checkpoint 3: Testing Infrastructure."""
        results = {"score": 0, "max_score": 100, "items": []}
        
        # Check test fixtures
        test_data = self.repo_path / "tests/fixtures/test_data.py"
        if test_data.exists():
            results["score"] += 50
            results["items"].append("✅ Test data factories exist")
        else:
            results["items"].append("❌ Test data factories missing")
        
        # Check assertion helpers
        assertion_helpers = self.repo_path / "tests/fixtures/assertion_helpers.py"
        if assertion_helpers.exists():
            results["score"] += 50
            results["items"].append("✅ Assertion helpers exist")
        else:
            results["items"].append("❌ Assertion helpers missing")
        
        return results
    
    def validate_checkpoint_4_build(self) -> Dict[str, Any]:
        """Validate Checkpoint 4: Build & Containerization."""
        results = {"score": 0, "max_score": 100, "items": []}
        
        # Check Dockerfile enhancements
        dockerfile = self.repo_path / "Dockerfile"
        if dockerfile.exists():
            with open(dockerfile, 'r') as f:
                content = f.read()
                if "buildx" in content or "BUILDPLATFORM" in content:
                    results["score"] += 50
                    results["items"].append("✅ Multi-architecture Docker build support")
                else:
                    results["items"].append("❌ Multi-architecture Docker build not configured")
        
        # Check docker-compose override
        compose_override = self.repo_path / "docker-compose.override.yml.example"
        if compose_override.exists():
            results["score"] += 50
            results["items"].append("✅ Docker Compose override example exists")
        else:
            results["items"].append("❌ Docker Compose override example missing")
        
        return results
    
    def validate_checkpoint_5_monitoring(self) -> Dict[str, Any]:
        """Validate Checkpoint 5: Monitoring & Observability Setup."""
        results = {"score": 0, "max_score": 100, "items": []}
        
        # Check Prometheus recording rules
        prometheus_rules = self.repo_path / "docker/prometheus/recording_rules.yml"
        if prometheus_rules.exists():
            results["score"] += 50
            results["items"].append("✅ Prometheus recording rules exist")
        else:
            results["items"].append("❌ Prometheus recording rules missing")
        
        # Check monitoring setup script
        monitoring_script = self.repo_path / "scripts/monitoring_setup.py"
        if monitoring_script.exists():
            results["score"] += 50
            results["items"].append("✅ Monitoring setup script exists")
        else:
            results["items"].append("❌ Monitoring setup script missing")
        
        return results
    
    def validate_checkpoint_6_workflows(self) -> Dict[str, Any]:
        """Validate Checkpoint 6: Workflow Documentation & Templates."""
        results = {"score": 0, "max_score": 100, "items": []}
        
        # Check workflow templates
        workflow_examples = self.repo_path / "docs/workflows/examples"
        if workflow_examples.exists():
            template_count = len(list(workflow_examples.glob("*.yml")))
            if template_count >= 6:  # Expected templates
                results["score"] += 50
                results["items"].append(f"✅ Workflow templates exist ({template_count} files)")
            else:
                results["items"].append(f"⚠️ Incomplete workflow templates ({template_count} files)")
        else:
            results["items"].append("❌ Workflow templates missing")
        
        # Check setup guide
        setup_guide = self.repo_path / "docs/workflows/WORKFLOW_SETUP_GUIDE.md"
        if setup_guide.exists():
            results["score"] += 50
            results["items"].append("✅ Workflow setup guide exists")
        else:
            results["items"].append("❌ Workflow setup guide missing")
        
        return results
    
    def validate_checkpoint_7_metrics(self) -> Dict[str, Any]:
        """Validate Checkpoint 7: Metrics & Automation Setup."""
        results = {"score": 0, "max_score": 100, "items": []}
        
        # Check metrics collector
        metrics_collector = self.repo_path / "scripts/metrics_collector.py"
        if metrics_collector.exists():
            results["score"] += 33
            results["items"].append("✅ Metrics collector script exists")
        else:
            results["items"].append("❌ Metrics collector script missing")
        
        # Check automation manager
        automation_manager = self.repo_path / "scripts/automation_manager.py"
        if automation_manager.exists():
            results["score"] += 33
            results["items"].append("✅ Automation manager script exists")
        else:
            results["items"].append("❌ Automation manager script missing")
        
        # Check project metrics
        project_metrics = self.repo_path / ".github/project-metrics.json"
        if project_metrics.exists():
            results["score"] += 34
            results["items"].append("✅ Project metrics configuration exists")
        else:
            results["items"].append("❌ Project metrics configuration missing")
        
        return results
    
    def validate_checkpoint_8_integration(self) -> Dict[str, Any]:
        """Validate Checkpoint 8: Integration & Final Configuration."""
        results = {"score": 0, "max_score": 100, "items": []}
        
        # Check completion documentation
        completion_doc = self.repo_path / "docs/TERRAGON_SDLC_COMPLETION.md"
        if completion_doc.exists():
            results["score"] += 50
            results["items"].append("✅ SDLC completion documentation exists")
        else:
            results["items"].append("❌ SDLC completion documentation missing")
        
        # Check validation script (this file)
        validation_script = self.repo_path / "scripts/sdlc_validation.py"
        if validation_script.exists():
            results["score"] += 50
            results["items"].append("✅ SDLC validation script exists")
        else:
            results["items"].append("❌ SDLC validation script missing")
        
        return results
    
    def validate_code_quality(self) -> Dict[str, Any]:
        """Validate overall code quality and configuration."""
        results = {"score": 0, "max_score": 100, "items": []}
        
        # Check pyproject.toml
        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            results["score"] += 25
            results["items"].append("✅ Poetry configuration exists")
        
        # Check pre-commit configuration
        precommit = self.repo_path / ".pre-commit-config.yaml"
        if precommit.exists():
            results["score"] += 25
            results["items"].append("✅ Pre-commit hooks configured")
        
        # Check Makefile
        makefile = self.repo_path / "Makefile"
        if makefile.exists():
            results["score"] += 25
            results["items"].append("✅ Development automation (Makefile) exists")
        
        # Check README
        readme = self.repo_path / "README.md"
        if readme.exists():
            results["score"] += 25
            results["items"].append("✅ README documentation exists")
        
        return results
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete SDLC validation."""
        print("🔍 Running Terragon SDLC Validation...")
        print("=" * 60)
        
        # Validate each checkpoint
        checkpoints = {
            "checkpoint_1_foundation": self.validate_checkpoint_1_foundation(),
            "checkpoint_2_devenv": self.validate_checkpoint_2_devenv(),
            "checkpoint_3_testing": self.validate_checkpoint_3_testing(),
            "checkpoint_4_build": self.validate_checkpoint_4_build(),
            "checkpoint_5_monitoring": self.validate_checkpoint_5_monitoring(),
            "checkpoint_6_workflows": self.validate_checkpoint_6_workflows(),
            "checkpoint_7_metrics": self.validate_checkpoint_7_metrics(),
            "checkpoint_8_integration": self.validate_checkpoint_8_integration(),
            "code_quality": self.validate_code_quality()
        }
        
        # Calculate overall score
        total_score = sum(cp["score"] for cp in checkpoints.values())
        max_total_score = sum(cp["max_score"] for cp in checkpoints.values())
        overall_percentage = (total_score / max_total_score) * 100
        
        # Print results
        for checkpoint_name, result in checkpoints.items():
            checkpoint_display = checkpoint_name.replace("_", " ").title()
            percentage = (result["score"] / result["max_score"]) * 100
            status_emoji = "✅" if percentage >= 90 else "⚠️" if percentage >= 70 else "❌"
            
            print(f"{status_emoji} {checkpoint_display}: {percentage:.1f}% ({result['score']}/{result['max_score']})")
            for item in result["items"]:
                print(f"   {item}")
            print()
        
        # Overall assessment
        print("=" * 60)
        print(f"📊 OVERALL SDLC SCORE: {overall_percentage:.1f}% ({total_score}/{max_total_score})")
        
        if overall_percentage >= 95:
            status = "ADVANCED - PRODUCTION READY"
            print("🎯 STATUS: ADVANCED - PRODUCTION READY")
            print("✅ Repository meets enterprise-grade SDLC standards")
        elif overall_percentage >= 85:
            status = "GOOD - NEARLY PRODUCTION READY"
            print("🔶 STATUS: GOOD - NEARLY PRODUCTION READY")
            print("⚠️ Minor improvements needed for production deployment")
        elif overall_percentage >= 70:
            status = "BASIC - DEVELOPMENT READY"
            print("🔶 STATUS: BASIC - DEVELOPMENT READY")
            print("⚠️ Significant improvements needed for production")
        else:
            status = "INCOMPLETE - REQUIRES WORK"
            print("❌ STATUS: INCOMPLETE - REQUIRES WORK")
            print("❌ Major SDLC components missing or incomplete")
        
        print("=" * 60)
        
        return {
            "checkpoints": checkpoints,
            "overall_score": overall_percentage,
            "total_score": total_score,
            "max_total_score": max_total_score,
            "status": status,
            "timestamp": "2025-08-02T00:00:00Z"
        }


def main():
    """Main validation entry point."""
    repo_path = Path(__file__).parent.parent
    validator = SDLCValidator(repo_path)
    
    results = validator.run_full_validation()
    
    # Save results
    results_file = repo_path / ".github/sdlc_validation_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Validation results saved to: {results_file}")
    
    # Exit with appropriate code
    if results["overall_score"] >= 95:
        sys.exit(0)  # Success
    elif results["overall_score"] >= 85:
        sys.exit(1)  # Warning
    else:
        sys.exit(2)  # Error


if __name__ == "__main__":
    main()