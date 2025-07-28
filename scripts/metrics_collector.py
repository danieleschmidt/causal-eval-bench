#!/usr/bin/env python3
"""
Comprehensive metrics collection and automation for Causal Eval Bench.
Collects project health, code quality, security, and performance metrics.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
import tempfile
import shutil


class MetricsCollector:
    """Collect and analyze project metrics."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.metrics = {
            "project": "causal-eval-bench",
            "collection_timestamp": datetime.now(timezone.utc).isoformat(),
            "version": self._get_project_version(),
            "commit_hash": self._get_commit_hash(),
        }

    def _run_command(self, command: List[str], cwd: Optional[Path] = None) -> Tuple[str, str, int]:
        """Run a shell command and return stdout, stderr, return code."""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.repo_path,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            return result.stdout.strip(), result.stderr.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "", "Command timed out", 1
        except Exception as e:
            return "", str(e), 1

    def _get_project_version(self) -> str:
        """Get project version from pyproject.toml."""
        try:
            import toml
            with open(self.repo_path / "pyproject.toml", "r") as f:
                data = toml.load(f)
            return data.get("tool", {}).get("poetry", {}).get("version", "unknown")
        except Exception:
            return "unknown"

    def _get_commit_hash(self) -> str:
        """Get current git commit hash."""
        stdout, _, code = self._run_command(["git", "rev-parse", "HEAD"])
        return stdout if code == 0 else "unknown"

    def collect_code_metrics(self) -> Dict[str, Any]:
        """Collect code-related metrics."""
        print("📊 Collecting code metrics...")
        
        metrics = {
            "total_files": 0,
            "lines_of_code": 0,
            "python_files": 0,
            "test_files": 0,
            "documentation_files": 0,
            "configuration_files": 0,
        }

        # Count files and lines
        file_extensions = {
            ".py": "python_files",
            ".md": "documentation_files",
            ".yml": "configuration_files",
            ".yaml": "configuration_files",
            ".toml": "configuration_files",
            ".json": "configuration_files",
        }

        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file() and not any(exclude in str(file_path) for exclude in [
                ".git", "__pycache__", ".pytest_cache", ".mypy_cache", 
                ".ruff_cache", "node_modules", ".venv", "venv"
            ]):
                metrics["total_files"] += 1
                
                # Count lines of code
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        lines = len(f.readlines())
                        metrics["lines_of_code"] += lines
                except Exception:
                    pass
                
                # Categorize files
                suffix = file_path.suffix.lower()
                if suffix in file_extensions:
                    metrics[file_extensions[suffix]] += 1
                
                # Identify test files
                if "test" in file_path.name.lower() and suffix == ".py":
                    metrics["test_files"] += 1

        # Calculate code complexity (if radon is available)
        try:
            stdout, _, code = self._run_command([
                "python", "-m", "radon", "cc", "--json", "causal_eval/"
            ])
            if code == 0:
                complexity_data = json.loads(stdout)
                total_complexity = 0
                function_count = 0
                
                for file_data in complexity_data.values():
                    for item in file_data:
                        if item["type"] in ["function", "method"]:
                            total_complexity += item["complexity"]
                            function_count += 1
                
                metrics["average_complexity"] = (
                    total_complexity / function_count if function_count > 0 else 0
                )
                metrics["total_functions"] = function_count
        except Exception:
            metrics["average_complexity"] = 0
            metrics["total_functions"] = 0

        return metrics

    def collect_test_metrics(self) -> Dict[str, Any]:
        """Collect testing-related metrics."""
        print("🧪 Collecting test metrics...")
        
        metrics = {
            "test_coverage": 0.0,
            "tests_total": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "tests_skipped": 0,
            "test_duration": 0.0,
            "coverage_by_module": {},
        }

        # Run pytest with coverage
        stdout, stderr, code = self._run_command([
            "python", "-m", "pytest", "tests/", "--tb=no", "-v",
            "--cov=causal_eval", "--cov-report=json", "--cov-report=term",
            "--json-report", "--json-report-file=test-report.json"
        ])

        # Parse test results
        try:
            if (self.repo_path / "test-report.json").exists():
                with open(self.repo_path / "test-report.json", "r") as f:
                    test_data = json.load(f)
                
                summary = test_data.get("summary", {})
                metrics["tests_total"] = summary.get("total", 0)
                metrics["tests_passed"] = summary.get("passed", 0)
                metrics["tests_failed"] = summary.get("failed", 0)
                metrics["tests_skipped"] = summary.get("skipped", 0)
                metrics["test_duration"] = test_data.get("duration", 0.0)
        except Exception as e:
            print(f"⚠️  Could not parse test results: {e}")

        # Parse coverage results
        try:
            if (self.repo_path / "coverage.json").exists():
                with open(self.repo_path / "coverage.json", "r") as f:
                    coverage_data = json.load(f)
                
                totals = coverage_data.get("totals", {})
                metrics["test_coverage"] = totals.get("percent_covered", 0.0)
                
                # Module-level coverage
                files = coverage_data.get("files", {})
                for file_path, file_data in files.items():
                    module_name = file_path.replace("/", ".").replace(".py", "")
                    metrics["coverage_by_module"][module_name] = file_data.get("summary", {}).get("percent_covered", 0.0)
        except Exception as e:
            print(f"⚠️  Could not parse coverage results: {e}")

        return metrics

    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        print("🔒 Collecting security metrics...")
        
        metrics = {
            "vulnerabilities_total": 0,
            "vulnerabilities_high": 0,
            "vulnerabilities_medium": 0,
            "vulnerabilities_low": 0,
            "last_security_scan": datetime.now(timezone.utc).isoformat(),
            "security_tools": {},
            "dependencies_scanned": 0,
        }

        # Run Bandit (SAST)
        stdout, stderr, code = self._run_command([
            "python", "-m", "bandit", "-r", "causal_eval/", "-f", "json", "-o", "bandit-report.json"
        ])
        
        try:
            if (self.repo_path / "bandit-report.json").exists():
                with open(self.repo_path / "bandit-report.json", "r") as f:
                    bandit_data = json.load(f)
                
                results = bandit_data.get("results", [])
                metrics["security_tools"]["bandit"] = {
                    "issues_found": len(results),
                    "high": len([r for r in results if r.get("issue_severity") == "HIGH"]),
                    "medium": len([r for r in results if r.get("issue_severity") == "MEDIUM"]),
                    "low": len([r for r in results if r.get("issue_severity") == "LOW"]),
                }
        except Exception as e:
            print(f"⚠️  Could not parse Bandit results: {e}")

        # Run Safety (dependency vulnerabilities)
        stdout, stderr, code = self._run_command([
            "python", "-m", "safety", "check", "--json", "--output", "safety-report.json"
        ])
        
        try:
            if (self.repo_path / "safety-report.json").exists():
                with open(self.repo_path / "safety-report.json", "r") as f:
                    safety_data = json.load(f)
                
                if isinstance(safety_data, list):
                    metrics["security_tools"]["safety"] = {
                        "vulnerabilities_found": len(safety_data),
                        "packages_affected": len(set(v.get("package_name", "") for v in safety_data)),
                    }
                    
                    for vuln in safety_data:
                        metrics["vulnerabilities_total"] += 1
                        # Safety doesn't provide severity levels, assume medium
                        metrics["vulnerabilities_medium"] += 1
        except Exception as e:
            print(f"⚠️  Could not parse Safety results: {e}")

        # Get dependency count
        try:
            stdout, stderr, code = self._run_command(["python", "-m", "pip", "list", "--format=json"])
            if code == 0:
                pip_list = json.loads(stdout)
                metrics["dependencies_scanned"] = len(pip_list)
        except Exception:
            pass

        return metrics

    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics."""
        print("📝 Collecting Git metrics...")
        
        metrics = {
            "total_commits": 0,
            "contributors": 0,
            "branches": 0,
            "days_since_last_commit": 0,
            "commits_last_30_days": 0,
            "most_active_contributors": [],
        }

        # Total commits
        stdout, stderr, code = self._run_command(["git", "rev-list", "--count", "HEAD"])
        if code == 0:
            metrics["total_commits"] = int(stdout)

        # Contributors
        stdout, stderr, code = self._run_command([
            "git", "shortlog", "-sn", "--all", "--no-merges"
        ])
        if code == 0:
            contributors = stdout.strip().split("\n") if stdout else []
            metrics["contributors"] = len(contributors)
            metrics["most_active_contributors"] = [
                {"name": line.split("\t")[1], "commits": int(line.split("\t")[0])}
                for line in contributors[:5] if "\t" in line
            ]

        # Branches
        stdout, stderr, code = self._run_command(["git", "branch", "-a"])
        if code == 0:
            metrics["branches"] = len(stdout.strip().split("\n")) if stdout else 0

        # Days since last commit
        stdout, stderr, code = self._run_command([
            "git", "log", "-1", "--format=%ct"
        ])
        if code == 0:
            last_commit_time = int(stdout)
            current_time = datetime.now(timezone.utc).timestamp()
            metrics["days_since_last_commit"] = int((current_time - last_commit_time) / 86400)

        # Commits in last 30 days
        stdout, stderr, code = self._run_command([
            "git", "rev-list", "--count", "--since='30 days ago'", "HEAD"
        ])
        if code == 0:
            metrics["commits_last_30_days"] = int(stdout)

        return metrics

    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance-related metrics."""
        print("⚡ Collecting performance metrics...")
        
        metrics = {
            "docker_image_size": 0,
            "build_duration": 0.0,
            "test_suite_duration": 0.0,
            "dependency_install_time": 0.0,
            "performance_benchmarks": {},
        }

        # Docker image size
        try:
            stdout, stderr, code = self._run_command([
                "docker", "images", "--format", "table {{.Size}}", "causal-eval:latest"
            ])
            if code == 0 and stdout:
                size_str = stdout.split("\n")[-1] if "\n" in stdout else stdout
                # Convert size to MB (simplified)
                if "MB" in size_str:
                    metrics["docker_image_size"] = float(size_str.replace("MB", "").strip())
                elif "GB" in size_str:
                    metrics["docker_image_size"] = float(size_str.replace("GB", "").strip()) * 1024
        except Exception:
            pass

        # Run performance benchmarks if available
        stdout, stderr, code = self._run_command([
            "python", "-m", "pytest", "tests/performance/", "--benchmark-only", 
            "--benchmark-json=benchmark-results.json"
        ])
        
        try:
            if (self.repo_path / "benchmark-results.json").exists():
                with open(self.repo_path / "benchmark-results.json", "r") as f:
                    benchmark_data = json.load(f)
                
                benchmarks = benchmark_data.get("benchmarks", [])
                for benchmark in benchmarks:
                    name = benchmark.get("name", "unknown")
                    stats = benchmark.get("stats", {})
                    metrics["performance_benchmarks"][name] = {
                        "mean": stats.get("mean", 0),
                        "stddev": stats.get("stddev", 0),
                        "min": stats.get("min", 0),
                        "max": stats.get("max", 0),
                    }
        except Exception as e:
            print(f"⚠️  Could not collect performance benchmarks: {e}")

        return metrics

    def collect_documentation_metrics(self) -> Dict[str, Any]:
        """Collect documentation-related metrics."""
        print("📚 Collecting documentation metrics...")
        
        metrics = {
            "documentation_files": 0,
            "documentation_lines": 0,
            "api_documentation_coverage": 0.0,
            "readme_quality_score": 0,
            "documentation_sections": [],
        }

        # Count documentation files
        doc_files = list(self.repo_path.rglob("*.md"))
        metrics["documentation_files"] = len(doc_files)
        
        # Count documentation lines
        total_lines = 0
        for doc_file in doc_files:
            try:
                with open(doc_file, "r", encoding="utf-8", errors="ignore") as f:
                    total_lines += len(f.readlines())
            except Exception:
                pass
        metrics["documentation_lines"] = total_lines

        # Analyze README.md
        readme_path = self.repo_path / "README.md"
        if readme_path.exists():
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
                
                # Simple quality score based on sections
                sections = [
                    "installation", "usage", "example", "api", "contributing",
                    "license", "test", "documentation", "feature"
                ]
                
                score = 0
                found_sections = []
                for section in sections:
                    if section.lower() in readme_content.lower():
                        score += 1
                        found_sections.append(section)
                
                metrics["readme_quality_score"] = (score / len(sections)) * 100
                metrics["documentation_sections"] = found_sections
            except Exception:
                pass

        # Check for API documentation (simplified check for docstrings)
        python_files = list(self.repo_path.rglob("causal_eval/**/*.py"))
        documented_functions = 0
        total_functions = 0
        
        for py_file in python_files:
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Simple function counting (not perfect but gives an idea)
                import re
                functions = re.findall(r'def\s+\w+\s*\([^)]*\):', content)
                total_functions += len(functions)
                
                # Count functions with docstrings (simplified)
                documented = re.findall(r'def\s+\w+\s*\([^)]*\):\s*"""', content)
                documented_functions += len(documented)
                
            except Exception:
                pass
        
        if total_functions > 0:
            metrics["api_documentation_coverage"] = (documented_functions / total_functions) * 100

        return metrics

    def calculate_health_scores(self, all_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall health scores based on collected metrics."""
        print("🏥 Calculating health scores...")
        
        scores = {}
        
        # Code Quality Score (0-100)
        code_metrics = all_metrics.get("code_metrics", {})
        test_metrics = all_metrics.get("test_metrics", {})
        
        quality_factors = []
        
        # Test coverage factor
        coverage = test_metrics.get("test_coverage", 0)
        quality_factors.append(min(coverage, 100))
        
        # Code complexity factor (lower is better)
        complexity = code_metrics.get("average_complexity", 10)
        complexity_score = max(0, 100 - (complexity - 5) * 10)  # Penalty after complexity 5
        quality_factors.append(complexity_score)
        
        # Test pass rate factor
        total_tests = test_metrics.get("tests_total", 1)
        passed_tests = test_metrics.get("tests_passed", 0)
        test_pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        quality_factors.append(test_pass_rate)
        
        scores["code_quality"] = sum(quality_factors) / len(quality_factors) if quality_factors else 0
        
        # Security Score (0-100)
        security_metrics = all_metrics.get("security_metrics", {})
        high_vulns = security_metrics.get("vulnerabilities_high", 0)
        medium_vulns = security_metrics.get("vulnerabilities_medium", 0)
        low_vulns = security_metrics.get("vulnerabilities_low", 0)
        
        # Penalties for vulnerabilities
        security_score = 100
        security_score -= high_vulns * 20  # -20 points per high vuln
        security_score -= medium_vulns * 10  # -10 points per medium vuln
        security_score -= low_vulns * 2  # -2 points per low vuln
        
        scores["security"] = max(0, security_score)
        
        # Documentation Score (0-100)
        doc_metrics = all_metrics.get("documentation_metrics", {})
        doc_factors = []
        
        # README quality
        readme_score = doc_metrics.get("readme_quality_score", 0)
        doc_factors.append(readme_score)
        
        # API documentation coverage
        api_doc_coverage = doc_metrics.get("api_documentation_coverage", 0)
        doc_factors.append(api_doc_coverage)
        
        # Documentation completeness (based on file count)
        doc_files = doc_metrics.get("documentation_files", 0)
        completeness_score = min(100, doc_files * 10)  # 10 points per doc file, max 100
        doc_factors.append(completeness_score)
        
        scores["documentation"] = sum(doc_factors) / len(doc_factors) if doc_factors else 0
        
        # Maintenance Score (0-100)
        git_metrics = all_metrics.get("git_metrics", {})
        maintenance_factors = []
        
        # Recent activity factor
        days_since_commit = git_metrics.get("days_since_last_commit", 365)
        activity_score = max(0, 100 - days_since_commit * 2)  # -2 points per day
        maintenance_factors.append(activity_score)
        
        # Contributor diversity factor
        contributors = git_metrics.get("contributors", 1)
        diversity_score = min(100, contributors * 20)  # 20 points per contributor, max 100
        maintenance_factors.append(diversity_score)
        
        scores["maintenance"] = sum(maintenance_factors) / len(maintenance_factors) if maintenance_factors else 0
        
        # Overall Score
        scores["overall"] = sum(scores.values()) / len(scores)
        
        return scores

    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a human-readable metrics report."""
        health_scores = metrics.get("health_scores", {})
        
        report = f"""
# 📊 Project Metrics Report

**Project**: {metrics['project']}
**Version**: {metrics['version']}
**Generated**: {metrics['collection_timestamp']}
**Commit**: {metrics['commit_hash'][:8]}

## 🏆 Health Scores

| Category | Score | Status |
|----------|-------|--------|
| Overall | {health_scores.get('overall', 0):.1f}% | {'🟢 Excellent' if health_scores.get('overall', 0) >= 80 else '🟡 Good' if health_scores.get('overall', 0) >= 60 else '🔴 Needs Improvement'} |
| Code Quality | {health_scores.get('code_quality', 0):.1f}% | {'🟢 Excellent' if health_scores.get('code_quality', 0) >= 80 else '🟡 Good' if health_scores.get('code_quality', 0) >= 60 else '🔴 Needs Improvement'} |
| Security | {health_scores.get('security', 0):.1f}% | {'🟢 Excellent' if health_scores.get('security', 0) >= 80 else '🟡 Good' if health_scores.get('security', 0) >= 60 else '🔴 Needs Improvement'} |
| Documentation | {health_scores.get('documentation', 0):.1f}% | {'🟢 Excellent' if health_scores.get('documentation', 0) >= 80 else '🟡 Good' if health_scores.get('documentation', 0) >= 60 else '🔴 Needs Improvement'} |
| Maintenance | {health_scores.get('maintenance', 0):.1f}% | {'🟢 Excellent' if health_scores.get('maintenance', 0) >= 80 else '🟡 Good' if health_scores.get('maintenance', 0) >= 60 else '🔴 Needs Improvement'} |

## 📈 Key Metrics

### Code Metrics
- **Total Files**: {metrics.get('code_metrics', {}).get('total_files', 0):,}
- **Lines of Code**: {metrics.get('code_metrics', {}).get('lines_of_code', 0):,}
- **Python Files**: {metrics.get('code_metrics', {}).get('python_files', 0):,}
- **Average Complexity**: {metrics.get('code_metrics', {}).get('average_complexity', 0):.1f}

### Test Metrics
- **Test Coverage**: {metrics.get('test_metrics', {}).get('test_coverage', 0):.1f}%
- **Total Tests**: {metrics.get('test_metrics', {}).get('tests_total', 0):,}
- **Tests Passed**: {metrics.get('test_metrics', {}).get('tests_passed', 0):,}
- **Test Duration**: {metrics.get('test_metrics', {}).get('test_duration', 0):.1f}s

### Security Metrics
- **Total Vulnerabilities**: {metrics.get('security_metrics', {}).get('vulnerabilities_total', 0)}
- **High Severity**: {metrics.get('security_metrics', {}).get('vulnerabilities_high', 0)}
- **Dependencies Scanned**: {metrics.get('security_metrics', {}).get('dependencies_scanned', 0):,}

### Git Metrics
- **Total Commits**: {metrics.get('git_metrics', {}).get('total_commits', 0):,}
- **Contributors**: {metrics.get('git_metrics', {}).get('contributors', 0)}
- **Days Since Last Commit**: {metrics.get('git_metrics', {}).get('days_since_last_commit', 0)}
- **Commits (30 days)**: {metrics.get('git_metrics', {}).get('commits_last_30_days', 0)}

### Documentation Metrics
- **Documentation Files**: {metrics.get('documentation_metrics', {}).get('documentation_files', 0)}
- **Documentation Lines**: {metrics.get('documentation_metrics', {}).get('documentation_lines', 0):,}
- **API Doc Coverage**: {metrics.get('documentation_metrics', {}).get('api_documentation_coverage', 0):.1f}%
- **README Quality**: {metrics.get('documentation_metrics', {}).get('readme_quality_score', 0):.1f}%

## 📊 Trends and Recommendations

"""
        
        # Add recommendations based on scores
        if health_scores.get('code_quality', 0) < 70:
            report += "### 🔧 Code Quality Improvements\n"
            report += "- Increase test coverage above 80%\n"
            report += "- Reduce code complexity where possible\n"
            report += "- Fix failing tests\n\n"
        
        if health_scores.get('security', 0) < 80:
            report += "### 🔒 Security Improvements\n"
            report += "- Address high and medium severity vulnerabilities\n"
            report += "- Update dependencies with security patches\n"
            report += "- Run regular security scans\n\n"
        
        if health_scores.get('documentation', 0) < 70:
            report += "### 📚 Documentation Improvements\n"
            report += "- Add more API documentation and docstrings\n"
            report += "- Enhance README with examples and guides\n"
            report += "- Create user and developer documentation\n\n"
        
        report += f"""
---

*Report generated by metrics_collector.py on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC*
"""
        
        return report

    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        print("🚀 Starting comprehensive metrics collection...")
        
        all_metrics = {
            "project": self.metrics["project"],
            "collection_timestamp": self.metrics["collection_timestamp"],
            "version": self.metrics["version"],
            "commit_hash": self.metrics["commit_hash"],
        }
        
        try:
            all_metrics["code_metrics"] = self.collect_code_metrics()
        except Exception as e:
            print(f"❌ Error collecting code metrics: {e}")
            all_metrics["code_metrics"] = {}
        
        try:
            all_metrics["test_metrics"] = self.collect_test_metrics()
        except Exception as e:
            print(f"❌ Error collecting test metrics: {e}")
            all_metrics["test_metrics"] = {}
        
        try:
            all_metrics["security_metrics"] = self.collect_security_metrics()
        except Exception as e:
            print(f"❌ Error collecting security metrics: {e}")
            all_metrics["security_metrics"] = {}
        
        try:
            all_metrics["git_metrics"] = self.collect_git_metrics()
        except Exception as e:
            print(f"❌ Error collecting git metrics: {e}")
            all_metrics["git_metrics"] = {}
        
        try:
            all_metrics["performance_metrics"] = self.collect_performance_metrics()
        except Exception as e:
            print(f"❌ Error collecting performance metrics: {e}")
            all_metrics["performance_metrics"] = {}
        
        try:
            all_metrics["documentation_metrics"] = self.collect_documentation_metrics()
        except Exception as e:
            print(f"❌ Error collecting documentation metrics: {e}")
            all_metrics["documentation_metrics"] = {}
        
        # Calculate health scores
        try:
            all_metrics["health_scores"] = self.calculate_health_scores(all_metrics)
        except Exception as e:
            print(f"❌ Error calculating health scores: {e}")
            all_metrics["health_scores"] = {}
        
        print("✅ Metrics collection completed!")
        return all_metrics

    def save_metrics(self, metrics: Dict[str, Any], output_file: str = ".github/project-metrics.json"):
        """Save metrics to JSON file."""
        output_path = self.repo_path / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"💾 Metrics saved to {output_path}")

    def cleanup_temp_files(self):
        """Clean up temporary files created during metrics collection."""
        temp_files = [
            "test-report.json",
            "coverage.json",
            "bandit-report.json",
            "safety-report.json",
            "benchmark-results.json",
        ]
        
        for temp_file in temp_files:
            file_path = self.repo_path / temp_file
            if file_path.exists():
                file_path.unlink()


def main():
    """Main entry point for metrics collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect comprehensive project metrics")
    parser.add_argument("--repo-path", default=".", help="Path to repository root")
    parser.add_argument("--output", default=".github/project-metrics.json", help="Output file path")
    parser.add_argument("--report", action="store_true", help="Generate human-readable report")
    parser.add_argument("--report-file", default="metrics-report.md", help="Report output file")
    
    args = parser.parse_args()
    
    try:
        collector = MetricsCollector(args.repo_path)
        metrics = collector.collect_all_metrics()
        
        # Save metrics
        collector.save_metrics(metrics, args.output)
        
        # Generate report if requested
        if args.report:
            report = collector.generate_report(metrics)
            with open(args.report_file, "w") as f:
                f.write(report)
            print(f"📊 Report saved to {args.report_file}")
        
        # Clean up
        collector.cleanup_temp_files()
        
        # Print summary
        health_scores = metrics.get("health_scores", {})
        overall_score = health_scores.get("overall", 0)
        
        print(f"\n🎯 Overall Health Score: {overall_score:.1f}%")
        
        if overall_score >= 80:
            print("🟢 Project health is excellent!")
        elif overall_score >= 60:
            print("🟡 Project health is good, some improvements possible.")
        else:
            print("🔴 Project health needs improvement.")
            sys.exit(1)
    
    except Exception as e:
        print(f"❌ Metrics collection failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()