#!/usr/bin/env python3
"""
Repository automation manager for Causal Eval Bench.
Handles automated maintenance tasks, dependency updates, and system health checks.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import tempfile
import requests


class AutomationManager:
    """Manage automated repository maintenance tasks."""

    def __init__(self, repo_path: str = ".", dry_run: bool = False):
        self.repo_path = Path(repo_path).resolve()
        self.dry_run = dry_run
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.repo_path / 'automation.log')
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_command(self, command: List[str], cwd: Optional[Path] = None) -> Tuple[str, str, int]:
        """Run a command and return stdout, stderr, return code."""
        try:
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would run: {' '.join(command)}")
                return "", "", 0
            
            result = subprocess.run(
                command,
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                timeout=300,
            )
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", 1
        except Exception as e:
            return "", str(e), 1

    def cleanup_old_branches(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up old merged branches."""
        self.logger.info(f"🧹 Cleaning up branches older than {days_old} days...")
        
        result = {
            "branches_deleted": [],
            "branches_kept": [],
            "errors": []
        }
        
        # Get merged branches
        stdout, stderr, code = self.run_command([
            "git", "branch", "--merged", "main", "--format=%(refname:short)%09%(committerdate:iso)"
        ])
        
        if code != 0:
            result["errors"].append(f"Failed to list branches: {stderr}")
            return result
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        for line in stdout.split('\n'):
            if not line.strip():
                continue
            
            try:
                branch_name, date_str = line.split('\t')
                branch_date = datetime.fromisoformat(date_str.replace(' ', 'T'))
                
                # Skip main and develop branches
                if branch_name in ['main', 'develop', 'master']:
                    continue
                
                if branch_date < cutoff_date:
                    self.logger.info(f"Deleting old branch: {branch_name}")
                    stdout, stderr, code = self.run_command([
                        "git", "branch", "-d", branch_name
                    ])
                    
                    if code == 0:
                        result["branches_deleted"].append(branch_name)
                    else:
                        result["errors"].append(f"Failed to delete {branch_name}: {stderr}")
                else:
                    result["branches_kept"].append(branch_name)
                    
            except Exception as e:
                result["errors"].append(f"Error processing branch {line}: {e}")
        
        self.logger.info(f"Deleted {len(result['branches_deleted'])} old branches")
        return result

    def cleanup_docker_resources(self) -> Dict[str, Any]:
        """Clean up Docker images, containers, and volumes."""
        self.logger.info("🐳 Cleaning up Docker resources...")
        
        result = {
            "images_removed": 0,
            "containers_removed": 0,
            "volumes_removed": 0,
            "space_freed": "0MB",
            "errors": []
        }
        
        # Remove unused containers
        stdout, stderr, code = self.run_command([
            "docker", "container", "prune", "-f"
        ])
        if code == 0:
            # Parse output to get count
            if "Total reclaimed space" in stderr:
                result["space_freed"] = stderr.split("Total reclaimed space: ")[1].strip()
        else:
            result["errors"].append(f"Container cleanup failed: {stderr}")
        
        # Remove unused images
        stdout, stderr, code = self.run_command([
            "docker", "image", "prune", "-f"
        ])
        if code != 0:
            result["errors"].append(f"Image cleanup failed: {stderr}")
        
        # Remove unused volumes
        stdout, stderr, code = self.run_command([
            "docker", "volume", "prune", "-f"
        ])
        if code != 0:
            result["errors"].append(f"Volume cleanup failed: {stderr}")
        
        # Remove build cache
        stdout, stderr, code = self.run_command([
            "docker", "builder", "prune", "-f"
        ])
        if code != 0:
            result["errors"].append(f"Build cache cleanup failed: {stderr}")
        
        return result

    def update_dependencies(self, update_type: str = "minor") -> Dict[str, Any]:
        """Update project dependencies."""
        self.logger.info(f"📦 Updating dependencies ({update_type})...")
        
        result = {
            "updates_applied": [],
            "updates_failed": [],
            "security_updates": [],
            "breaking_changes": [],
            "errors": []
        }
        
        # Backup current lock file
        lock_file = self.repo_path / "poetry.lock"
        if lock_file.exists():
            shutil.copy2(lock_file, lock_file.with_suffix(".lock.backup"))
        
        try:
            # Check for outdated packages
            stdout, stderr, code = self.run_command([
                "poetry", "show", "--outdated", "--format", "json"
            ])
            
            if code == 0 and stdout:
                outdated_packages = json.loads(stdout) if stdout else []
                
                for package in outdated_packages:
                    name = package.get("name", "")
                    current = package.get("version", "")
                    latest = package.get("latest", "")
                    
                    # Determine if it's a security update
                    is_security = self.is_security_update(name, current, latest)
                    if is_security:
                        result["security_updates"].append(f"{name}: {current} -> {latest}")
                    
                    # Apply updates based on type
                    if update_type == "all" or is_security:
                        success = self.update_single_dependency(name, latest)
                        if success:
                            result["updates_applied"].append(f"{name}: {current} -> {latest}")
                        else:
                            result["updates_failed"].append(f"{name}: {current} -> {latest}")
            
            # Run security audit
            self.run_security_audit(result)
            
        except Exception as e:
            result["errors"].append(f"Dependency update failed: {e}")
            # Restore backup if something went wrong
            backup_file = lock_file.with_suffix(".lock.backup")
            if backup_file.exists():
                shutil.copy2(backup_file, lock_file)
        
        return result

    def is_security_update(self, package_name: str, current_version: str, latest_version: str) -> bool:
        """Check if an update is security-related (simplified check)."""
        # This is a simplified implementation
        # In practice, you'd check against vulnerability databases
        security_keywords = ["security", "vulnerability", "cve", "patch"]
        
        try:
            # Check PyPI API for release information
            response = requests.get(f"https://pypi.org/pypi/{package_name}/{latest_version}/json", timeout=10)
            if response.status_code == 200:
                data = response.json()
                description = data.get("info", {}).get("description", "").lower()
                return any(keyword in description for keyword in security_keywords)
        except Exception:
            pass
        
        return False

    def update_single_dependency(self, package_name: str, version: str) -> bool:
        """Update a single dependency."""
        stdout, stderr, code = self.run_command([
            "poetry", "update", package_name
        ])
        return code == 0

    def run_security_audit(self, result: Dict[str, Any]):
        """Run security audit on dependencies."""
        stdout, stderr, code = self.run_command([
            "poetry", "run", "safety", "check", "--json"
        ])
        
        if code == 0:
            try:
                if stdout:
                    vulnerabilities = json.loads(stdout)
                    if vulnerabilities:
                        result["security_vulnerabilities"] = len(vulnerabilities)
                        for vuln in vulnerabilities[:5]:  # Limit to 5 most critical
                            result["security_updates"].append(
                                f"SECURITY: {vuln.get('package_name', 'unknown')} - {vuln.get('advisory', 'No details')}"
                            )
            except json.JSONDecodeError:
                pass

    def check_code_quality(self) -> Dict[str, Any]:
        """Run code quality checks and fix issues where possible."""
        self.logger.info("🔍 Running code quality checks...")
        
        result = {
            "formatting_fixes": 0,
            "linting_issues": 0,
            "type_errors": 0,
            "security_issues": 0,
            "auto_fixes_applied": [],
            "manual_fixes_needed": [],
            "errors": []
        }
        
        # Run Black formatter
        stdout, stderr, code = self.run_command([
            "poetry", "run", "black", "--check", "--diff", "."
        ])
        
        if code != 0:
            # Apply formatting fixes
            stdout, stderr, code = self.run_command([
                "poetry", "run", "black", "."
            ])
            if code == 0:
                result["formatting_fixes"] = len(stderr.split('\n')) if stderr else 0
                result["auto_fixes_applied"].append("Code formatting (Black)")
        
        # Run isort
        stdout, stderr, code = self.run_command([
            "poetry", "run", "isort", "--check-only", "."
        ])
        
        if code != 0:
            # Apply import sorting fixes
            stdout, stderr, code = self.run_command([
                "poetry", "run", "isort", "."
            ])
            if code == 0:
                result["auto_fixes_applied"].append("Import sorting (isort)")
        
        # Run Ruff linting
        stdout, stderr, code = self.run_command([
            "poetry", "run", "ruff", "check", "--fix", "."
        ])
        
        if stdout:
            result["linting_issues"] = len(stdout.split('\n'))
            if "--fix" in stdout:
                result["auto_fixes_applied"].append("Linting fixes (Ruff)")
        
        # Run MyPy type checking
        stdout, stderr, code = self.run_command([
            "poetry", "run", "mypy", "causal_eval/"
        ])
        
        if code != 0:
            result["type_errors"] = len(stderr.split('\n')) if stderr else 0
            if result["type_errors"] > 0:
                result["manual_fixes_needed"].append(f"Type errors: {result['type_errors']} issues")
        
        # Run Bandit security check
        stdout, stderr, code = self.run_command([
            "poetry", "run", "bandit", "-r", "causal_eval/", "-f", "json"
        ])
        
        if code != 0 and stdout:
            try:
                bandit_results = json.loads(stdout)
                issues = bandit_results.get("results", [])
                result["security_issues"] = len(issues)
                if result["security_issues"] > 0:
                    result["manual_fixes_needed"].append(f"Security issues: {result['security_issues']} found")
            except json.JSONDecodeError:
                pass
        
        return result

    def run_tests_and_coverage(self) -> Dict[str, Any]:
        """Run test suite and check coverage."""
        self.logger.info("🧪 Running tests and coverage analysis...")
        
        result = {
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "coverage_percentage": 0.0,
            "coverage_below_threshold": [],
            "test_duration": 0.0,
            "errors": []
        }
        
        # Run pytest with coverage
        stdout, stderr, code = self.run_command([
            "poetry", "run", "pytest", "tests/", "-v", 
            "--cov=causal_eval", "--cov-report=json", "--cov-report=term",
            "--json-report", "--json-report-file=test-report.json"
        ])
        
        # Parse test results
        test_report_file = self.repo_path / "test-report.json"
        if test_report_file.exists():
            try:
                with open(test_report_file, "r") as f:
                    test_data = json.load(f)
                
                summary = test_data.get("summary", {})
                result["tests_passed"] = summary.get("passed", 0)
                result["tests_failed"] = summary.get("failed", 0)
                result["tests_skipped"] = summary.get("skipped", 0)
                result["test_duration"] = test_data.get("duration", 0.0)
                
                # Clean up
                test_report_file.unlink()
                
            except Exception as e:
                result["errors"].append(f"Failed to parse test results: {e}")
        
        # Parse coverage results
        coverage_file = self.repo_path / "coverage.json"
        if coverage_file.exists():
            try:
                with open(coverage_file, "r") as f:
                    coverage_data = json.load(f)
                
                totals = coverage_data.get("totals", {})
                result["coverage_percentage"] = totals.get("percent_covered", 0.0)
                
                # Check for files below coverage threshold
                threshold = 80.0
                files = coverage_data.get("files", {})
                for file_path, file_data in files.items():
                    file_coverage = file_data.get("summary", {}).get("percent_covered", 0.0)
                    if file_coverage < threshold:
                        result["coverage_below_threshold"].append({
                            "file": file_path,
                            "coverage": file_coverage
                        })
                
                # Clean up
                coverage_file.unlink()
                
            except Exception as e:
                result["errors"].append(f"Failed to parse coverage results: {e}")
        
        return result

    def update_documentation(self) -> Dict[str, Any]:
        """Update and validate documentation."""
        self.logger.info("📚 Updating documentation...")
        
        result = {
            "docs_built": False,
            "broken_links": [],
            "missing_docs": [],
            "api_docs_updated": False,
            "errors": []
        }
        
        # Build documentation
        stdout, stderr, code = self.run_command([
            "poetry", "run", "mkdocs", "build", "--strict"
        ])
        
        if code == 0:
            result["docs_built"] = True
        else:
            result["errors"].append(f"Documentation build failed: {stderr}")
        
        # Check for missing API documentation
        api_files = list(self.repo_path.rglob("causal_eval/**/*.py"))
        for api_file in api_files:
            if api_file.name != "__init__.py":
                # Simple check for docstrings
                try:
                    with open(api_file, "r") as f:
                        content = f.read()
                    
                    if 'def ' in content and '"""' not in content:
                        result["missing_docs"].append(str(api_file.relative_to(self.repo_path)))
                except Exception:
                    pass
        
        return result

    def monitor_system_health(self) -> Dict[str, Any]:
        """Monitor overall system health."""
        self.logger.info("🏥 Monitoring system health...")
        
        result = {
            "disk_usage": {},
            "memory_usage": {},
            "service_status": {},
            "alerts": [],
            "recommendations": []
        }
        
        # Check disk usage
        try:
            total, used, free = shutil.disk_usage(self.repo_path)
            result["disk_usage"] = {
                "total_gb": total // (1024**3),
                "used_gb": used // (1024**3),
                "free_gb": free // (1024**3),
                "usage_percentage": (used / total) * 100
            }
            
            if result["disk_usage"]["usage_percentage"] > 80:
                result["alerts"].append("High disk usage detected")
                result["recommendations"].append("Clean up old files and Docker resources")
        except Exception as e:
            result["alerts"].append(f"Could not check disk usage: {e}")
        
        # Check Docker services (if running)
        stdout, stderr, code = self.run_command([
            "docker-compose", "ps", "--format", "json"
        ])
        
        if code == 0:
            try:
                services = json.loads(stdout) if stdout else []
                for service in services:
                    name = service.get("Service", "unknown")
                    state = service.get("State", "unknown")
                    result["service_status"][name] = state
                    
                    if state != "running":
                        result["alerts"].append(f"Service {name} is not running")
            except json.JSONDecodeError:
                pass
        
        return result

    def generate_automation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive automation report."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        report = f"""
# 🤖 Automation Report

**Generated**: {timestamp}
**Repository**: causal-eval-bench
**Mode**: {'Dry Run' if self.dry_run else 'Live Run'}

## 📊 Summary

"""
        
        # Add summary for each task
        for task_name, task_result in results.items():
            if isinstance(task_result, dict):
                report += f"### {task_name.replace('_', ' ').title()}\n"
                
                # Count successes and errors
                successes = sum(1 for key, value in task_result.items() 
                              if isinstance(value, list) and 'error' not in key.lower() and value)
                errors = len(task_result.get('errors', []))
                
                if errors == 0:
                    report += "✅ **Status**: Completed successfully\n"
                else:
                    report += f"⚠️ **Status**: Completed with {errors} error(s)\n"
                
                # Add key metrics
                for key, value in task_result.items():
                    if key != 'errors' and isinstance(value, (int, float, str)):
                        report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
                
                if task_result.get('errors'):
                    report += "**Errors**:\n"
                    for error in task_result['errors'][:3]:  # Limit to 3 errors
                        report += f"- {error}\n"
                
                report += "\n"
        
        # Add recommendations
        report += "## 💡 Recommendations\n\n"
        
        all_recommendations = []
        for task_result in results.values():
            if isinstance(task_result, dict):
                all_recommendations.extend(task_result.get('recommendations', []))
                if task_result.get('errors'):
                    all_recommendations.append("Review and fix automation errors")
        
        if all_recommendations:
            for rec in set(all_recommendations):  # Remove duplicates
                report += f"- {rec}\n"
        else:
            report += "- No specific recommendations at this time\n"
        
        report += f"\n---\n\n*Report generated by automation_manager.py*\n"
        
        return report

    def run_full_automation(self, tasks: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all automation tasks."""
        if tasks is None:
            tasks = [
                "cleanup_old_branches",
                "cleanup_docker_resources", 
                "update_dependencies",
                "check_code_quality",
                "run_tests_and_coverage",
                "update_documentation",
                "monitor_system_health"
            ]
        
        self.logger.info(f"🚀 Starting automation tasks: {', '.join(tasks)}")
        
        results = {}
        
        for task in tasks:
            try:
                self.logger.info(f"Running task: {task}")
                
                if task == "cleanup_old_branches":
                    results[task] = self.cleanup_old_branches()
                elif task == "cleanup_docker_resources":
                    results[task] = self.cleanup_docker_resources()
                elif task == "update_dependencies":
                    results[task] = self.update_dependencies("minor")
                elif task == "check_code_quality":
                    results[task] = self.check_code_quality()
                elif task == "run_tests_and_coverage":
                    results[task] = self.run_tests_and_coverage()
                elif task == "update_documentation":
                    results[task] = self.update_documentation()
                elif task == "monitor_system_health":
                    results[task] = self.monitor_system_health()
                else:
                    self.logger.warning(f"Unknown task: {task}")
                    continue
                
                self.logger.info(f"✅ Task {task} completed")
                
            except Exception as e:
                self.logger.error(f"❌ Task {task} failed: {e}")
                results[task] = {"errors": [str(e)]}
        
        return results


def main():
    """Main entry point for automation manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository automation manager")
    parser.add_argument("--repo-path", default=".", help="Path to repository root")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    parser.add_argument("--tasks", nargs="+", help="Specific tasks to run")
    parser.add_argument("--report", action="store_true", help="Generate automation report")
    parser.add_argument("--report-file", default="automation-report.md", help="Report output file")
    
    args = parser.parse_args()
    
    try:
        manager = AutomationManager(args.repo_path, args.dry_run)
        results = manager.run_full_automation(args.tasks)
        
        # Generate report if requested
        if args.report:
            report = manager.generate_automation_report(results)
            with open(args.report_file, "w") as f:
                f.write(report)
            print(f"📊 Automation report saved to {args.report_file}")
        
        # Print summary
        total_errors = sum(len(result.get('errors', [])) for result in results.values() if isinstance(result, dict))
        
        if total_errors == 0:
            print("✅ All automation tasks completed successfully!")
        else:
            print(f"⚠️ Automation completed with {total_errors} error(s). Check the logs for details.")
            
    except Exception as e:
        print(f"❌ Automation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()