#!/usr/bin/env python3
"""
Maintenance automation script for Causal Eval Bench.

This script provides automated maintenance tasks including:
- Dependency updates
- Security scanning
- Performance monitoring
- Health checks
- Cleanup operations
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import requests


class MaintenanceManager:
    """Automated maintenance operations for the project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.config_file = project_root / "maintenance.json"
        self.log_file = project_root / "logs" / "maintenance.log"
        
        # Ensure logs directory exists
        self.log_file.parent.mkdir(exist_ok=True)
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Log a message with timestamp."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
    
    def run_command(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and log the result."""
        self.log(f"Running command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=check,
                cwd=self.project_root
            )
            
            if result.stdout:
                self.log(f"STDOUT: {result.stdout.strip()}")
            if result.stderr:
                self.log(f"STDERR: {result.stderr.strip()}", "WARNING")
                
            return result
            
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e}", "ERROR")
            raise
    
    def check_dependencies(self) -> Dict[str, any]:
        """Check for outdated dependencies."""
        self.log("Checking for outdated dependencies...")
        
        # Check Python dependencies
        try:
            result = self.run_command(["poetry", "show", "--outdated", "--format", "json"])
            outdated_packages = json.loads(result.stdout) if result.stdout.strip() else []
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            self.log("Failed to check Python dependencies", "ERROR")
            outdated_packages = []
        
        # Check for security vulnerabilities
        try:
            security_result = self.run_command(["poetry", "run", "safety", "check", "--json"])
            vulnerabilities = json.loads(security_result.stdout) if security_result.stdout.strip() else []
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            self.log("Failed to check security vulnerabilities", "ERROR")
            vulnerabilities = []
        
        return {
            "outdated_packages": outdated_packages,
            "vulnerabilities": vulnerabilities,
            "checked_at": datetime.now().isoformat()
        }
    
    def update_dependencies(self, auto_update: bool = False) -> bool:
        """Update dependencies if safe to do so."""
        self.log("Updating dependencies...")
        
        if not auto_update:
            self.log("Manual approval required for dependency updates")
            return False
        
        try:
            # Update patch and minor versions only
            self.run_command(["poetry", "update"])
            
            # Run tests to ensure updates don't break anything
            test_result = self.run_command(["poetry", "run", "pytest", "tests/"], check=False)
            
            if test_result.returncode != 0:
                self.log("Tests failed after dependency update, reverting...", "ERROR")
                self.run_command(["git", "checkout", "poetry.lock"])
                return False
            
            self.log("Dependencies updated successfully")
            return True
            
        except subprocess.CalledProcessError:
            self.log("Failed to update dependencies", "ERROR")
            return False
    
    def run_security_scan(self) -> Dict[str, any]:
        """Run comprehensive security scans."""
        self.log("Running security scans...")
        
        results = {
            "bandit": None,
            "safety": None,
            "secrets": None,
            "scanned_at": datetime.now().isoformat()
        }
        
        # Bandit - Python security linter
        try:
            bandit_result = self.run_command([
                "poetry", "run", "bandit", "-r", "causal_eval/", 
                "-f", "json", "-o", "bandit-report.json"
            ], check=False)
            
            if Path("bandit-report.json").exists():
                with open("bandit-report.json") as f:
                    results["bandit"] = json.load(f)
                    
        except Exception as e:
            self.log(f"Bandit scan failed: {e}", "ERROR")
        
        # Safety - Dependency vulnerability scanner
        try:
            safety_result = self.run_command([
                "poetry", "run", "safety", "check", "--json"
            ], check=False)
            
            if safety_result.stdout.strip():
                results["safety"] = json.loads(safety_result.stdout)
                
        except Exception as e:
            self.log(f"Safety scan failed: {e}", "ERROR")
        
        # detect-secrets - Secret scanner
        try:
            secrets_result = self.run_command([
                "poetry", "run", "detect-secrets", "scan", "--baseline", ".secrets.baseline"
            ], check=False)
            
            results["secrets"] = {
                "exit_code": secrets_result.returncode,
                "new_secrets_found": secrets_result.returncode != 0
            }
            
        except Exception as e:
            self.log(f"Secrets scan failed: {e}", "ERROR")
        
        return results
    
    def cleanup_old_files(self, days_old: int = 30) -> None:
        """Clean up old temporary files and logs."""
        self.log(f"Cleaning up files older than {days_old} days...")
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0
        
        # Clean up patterns
        cleanup_patterns = [
            "**/*.pyc",
            "**/__pycache__",
            "**/.*cache",
            "logs/*.log",
            "test_results/*",
            "coverage_reports/*"
        ]
        
        for pattern in cleanup_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    file_date = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_date < cutoff_date:
                        file_path.unlink()
                        cleaned_count += 1
                elif file_path.is_dir() and not any(file_path.iterdir()):
                    # Remove empty directories
                    file_path.rmdir()
                    cleaned_count += 1
        
        self.log(f"Cleaned up {cleaned_count} old files/directories")
    
    def check_service_health(self) -> Dict[str, any]:
        """Check health of external services and dependencies."""
        self.log("Checking service health...")
        
        health_results = {
            "checked_at": datetime.now().isoformat(),
            "services": {}
        }
        
        # Check if API endpoints are accessible
        services_to_check = [
            ("GitHub API", "https://api.github.com/rate_limit"),
            ("PyPI", "https://pypi.org/pypi/pip/json"),
            ("Docker Hub", "https://hub.docker.com/v2/repositories/library/python/"),
        ]
        
        for service_name, url in services_to_check:
            try:
                response = requests.get(url, timeout=10)
                health_results["services"][service_name] = {
                    "status": "healthy" if response.status_code == 200 else "unhealthy",
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
            except Exception as e:
                health_results["services"][service_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return health_results
    
    def generate_maintenance_report(self) -> Dict[str, any]:
        """Generate comprehensive maintenance report."""
        self.log("Generating maintenance report...")
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "project_info": {
                "name": "causal-eval-bench",
                "version": self._get_project_version(),
                "python_version": sys.version,
            }
        }
        
        # Add all maintenance check results
        report["dependencies"] = self.check_dependencies()
        report["security"] = self.run_security_scan()
        report["health"] = self.check_service_health()
        
        # Add system information
        report["system"] = {
            "disk_usage": self._get_disk_usage(),
            "git_status": self._get_git_status(),
        }
        
        # Save report
        report_file = self.project_root / f"maintenance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Maintenance report saved to {report_file}")
        return report
    
    def _get_project_version(self) -> str:
        """Get current project version from pyproject.toml."""
        try:
            result = self.run_command(["poetry", "version", "--short"])
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _get_disk_usage(self) -> Dict[str, any]:
        """Get disk usage information."""
        try:
            result = self.run_command(["df", "-h", str(self.project_root)])
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                parts = lines[1].split()
                return {
                    "total": parts[1] if len(parts) > 1 else "unknown",
                    "used": parts[2] if len(parts) > 2 else "unknown",
                    "available": parts[3] if len(parts) > 3 else "unknown",
                    "use_percent": parts[4] if len(parts) > 4 else "unknown"
                }
        except:
            pass
        return {"error": "Unable to get disk usage"}
    
    def _get_git_status(self) -> Dict[str, any]:
        """Get git repository status."""
        try:
            status_result = self.run_command(["git", "status", "--porcelain"])
            branch_result = self.run_command(["git", "branch", "--show-current"])
            
            return {
                "current_branch": branch_result.stdout.strip(),
                "uncommitted_changes": len(status_result.stdout.strip().split('\n')) if status_result.stdout.strip() else 0,
                "is_clean": not bool(status_result.stdout.strip())
            }
        except:
            return {"error": "Unable to get git status"}


def main():
    """Main entry point for maintenance script."""
    parser = argparse.ArgumentParser(description="Automated maintenance for Causal Eval Bench")
    parser.add_argument("--check-deps", action="store_true", help="Check for outdated dependencies")
    parser.add_argument("--update-deps", action="store_true", help="Update dependencies")
    parser.add_argument("--auto-update", action="store_true", help="Auto-update safe dependencies")
    parser.add_argument("--security-scan", action="store_true", help="Run security scans")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old files")
    parser.add_argument("--cleanup-days", type=int, default=30, help="Days threshold for cleanup")
    parser.add_argument("--health-check", action="store_true", help="Check service health")
    parser.add_argument("--full-report", action="store_true", help="Generate full maintenance report")
    parser.add_argument("--all", action="store_true", help="Run all maintenance tasks")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    manager = MaintenanceManager(project_root)
    
    try:
        if args.all or args.check_deps:
            deps_info = manager.check_dependencies()
            print(f"Found {len(deps_info['outdated_packages'])} outdated packages")
            print(f"Found {len(deps_info['vulnerabilities'])} security vulnerabilities")
        
        if args.all or args.update_deps:
            if manager.update_dependencies(auto_update=args.auto_update):
                print("Dependencies updated successfully")
            else:
                print("Dependency update skipped or failed")
        
        if args.all or args.security_scan:
            security_results = manager.run_security_scan()
            print("Security scan completed")
        
        if args.all or args.cleanup:
            manager.cleanup_old_files(days_old=args.cleanup_days)
            print("Cleanup completed")
        
        if args.all or args.health_check:
            health_results = manager.check_service_health()
            healthy_services = sum(1 for s in health_results["services"].values() if s.get("status") == "healthy")
            total_services = len(health_results["services"])
            print(f"Service health: {healthy_services}/{total_services} services healthy")
        
        if args.all or args.full_report:
            report = manager.generate_maintenance_report()
            print("Full maintenance report generated")
        
        if not any(vars(args).values()):
            parser.print_help()
            
    except Exception as e:
        manager.log(f"Maintenance script failed: {e}", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()