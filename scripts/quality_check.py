#!/usr/bin/env python3
"""
Comprehensive Quality Check Script
Runs all quality checks and provides detailed reporting for the Causal Eval Bench project.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json
import argparse


class QualityChecker:
    """Comprehensive quality checking for the project."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
        self.start_time = time.time()
        
    def run_command(self, command: List[str], description: str) -> Tuple[bool, str, str]:
        """Run a command and capture its output."""
        print(f"🔍 {description}...")
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300  # 5 minute timeout
            )
            success = result.returncode == 0
            return success, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out after 5 minutes"
        except Exception as e:
            return False, "", str(e)
    
    def check_formatting(self) -> Dict:
        """Check code formatting with Black."""
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "black", "--check", "."],
            "Checking code formatting with Black"
        )
        
        return {
            "name": "Code Formatting (Black)",
            "passed": success,
            "output": stdout,
            "error": stderr,
            "recommendation": "Run 'make format' to fix formatting issues" if not success else None
        }
    
    def check_import_sorting(self) -> Dict:
        """Check import sorting with isort."""
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "isort", "--check-only", "."],
            "Checking import sorting with isort"
        )
        
        return {
            "name": "Import Sorting (isort)",
            "passed": success,
            "output": stdout,
            "error": stderr,
            "recommendation": "Run 'make format' to fix import sorting" if not success else None
        }
    
    def check_linting(self) -> Dict:
        """Check linting with Ruff."""
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "ruff", "check", "."],
            "Running linting checks with Ruff"
        )
        
        return {
            "name": "Linting (Ruff)",
            "passed": success,
            "output": stdout,
            "error": stderr,
            "recommendation": "Run 'make lint-fix' to auto-fix issues" if not success else None
        }
    
    def check_type_checking(self) -> Dict:
        """Check type annotations with MyPy."""
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "mypy", "causal_eval/" if (self.project_root / "causal_eval").exists() else "scripts/"],
            "Checking type annotations with MyPy"
        )
        
        return {
            "name": "Type Checking (MyPy)",
            "passed": success,
            "output": stdout,
            "error": stderr,
            "recommendation": "Fix type annotation issues reported above" if not success else None
        }
    
    def check_security(self) -> Dict:
        """Check security issues with Bandit."""
        target = "causal_eval/" if (self.project_root / "causal_eval").exists() else "scripts/"
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "bandit", "-r", target, "-f", "json"],
            "Checking security issues with Bandit"
        )
        
        # Parse Bandit JSON output for better reporting
        issues = []
        if stdout:
            try:
                bandit_data = json.loads(stdout)
                issues = bandit_data.get("results", [])
            except json.JSONDecodeError:
                pass
        
        high_issues = [i for i in issues if i.get("issue_severity") == "HIGH"]
        medium_issues = [i for i in issues if i.get("issue_severity") == "MEDIUM"]
        
        return {
            "name": "Security Check (Bandit)",
            "passed": len(high_issues) == 0,
            "output": f"High: {len(high_issues)}, Medium: {len(medium_issues)}, Total: {len(issues)}",
            "error": stderr,
            "issues": issues,
            "recommendation": "Review and fix high severity security issues" if high_issues else None
        }
    
    def check_dependencies(self) -> Dict:
        """Check for vulnerable dependencies with Safety."""
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "safety", "check", "--json"],
            "Checking dependencies for vulnerabilities with Safety"
        )
        
        vulnerabilities = []
        if stdout:
            try:
                safety_data = json.loads(stdout)
                vulnerabilities = safety_data
            except json.JSONDecodeError:
                pass
        
        return {
            "name": "Dependency Security (Safety)",
            "passed": success and len(vulnerabilities) == 0,
            "output": f"Found {len(vulnerabilities)} vulnerabilities" if vulnerabilities else "No vulnerabilities found",
            "error": stderr,
            "vulnerabilities": vulnerabilities,
            "recommendation": "Update vulnerable dependencies" if vulnerabilities else None
        }
    
    def check_tests(self) -> Dict:
        """Run the test suite."""
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "pytest", "tests/", "--tb=short", "-q"],
            "Running test suite"
        )
        
        # Extract test statistics from output
        lines = stdout.split('\n')
        test_line = next((line for line in lines if 'passed' in line or 'failed' in line), "")
        
        return {
            "name": "Test Suite",
            "passed": success,
            "output": test_line or stdout,
            "error": stderr,
            "recommendation": "Fix failing tests before proceeding" if not success else None
        }
    
    def check_documentation(self) -> Dict:
        """Check documentation build."""
        success, stdout, stderr = self.run_command(
            ["poetry", "run", "mkdocs", "build", "--strict"],
            "Building documentation"
        )
        
        return {
            "name": "Documentation Build",
            "passed": success,
            "output": stdout,
            "error": stderr,
            "recommendation": "Fix documentation errors" if not success else None
        }
    
    def run_all_checks(self, skip_tests: bool = False, skip_docs: bool = False) -> Dict:
        """Run all quality checks."""
        print("🚀 Starting comprehensive quality checks...")
        print("=" * 60)
        
        checks = [
            self.check_formatting,
            self.check_import_sorting,
            self.check_linting,
            self.check_type_checking,
            self.check_security,
            self.check_dependencies,
        ]
        
        if not skip_tests:
            checks.append(self.check_tests)
        
        if not skip_docs:
            checks.append(self.check_documentation)
        
        results = []
        passed_count = 0
        
        for check in checks:
            result = check()
            results.append(result)
            
            # Print immediate feedback
            status = "✅" if result["passed"] else "❌"
            print(f"{status} {result['name']}")
            
            if result["passed"]:
                passed_count += 1
            elif result.get("recommendation"):
                print(f"   💡 {result['recommendation']}")
        
        # Summary
        total_checks = len(results)
        duration = time.time() - self.start_time
        
        print("=" * 60)
        print(f"📊 Quality Check Summary")
        print(f"   ✅ Passed: {passed_count}/{total_checks}")
        print(f"   ❌ Failed: {total_checks - passed_count}/{total_checks}")
        print(f"   ⏱️  Duration: {duration:.1f}s")
        
        overall_passed = passed_count == total_checks
        if overall_passed:
            print("🎉 All quality checks passed!")
        else:
            print("⚠️  Some quality checks failed. Please review and fix issues.")
        
        return {
            "overall_passed": overall_passed,
            "passed_count": passed_count,
            "total_count": total_checks,
            "duration": duration,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def generate_report(self, results: Dict, output_file: str = None) -> str:
        """Generate a detailed quality report."""
        report = []
        report.append("# Quality Check Report")
        report.append(f"**Generated**: {results['timestamp']}")
        report.append(f"**Duration**: {results['duration']:.1f}s")
        report.append(f"**Status**: {'✅ PASSED' if results['overall_passed'] else '❌ FAILED'}")
        report.append(f"**Score**: {results['passed_count']}/{results['total_count']}")
        report.append("")
        
        # Detailed results
        report.append("## Detailed Results")
        report.append("")
        
        for result in results["results"]:
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            report.append(f"### {result['name']} - {status}")
            report.append("")
            
            if result.get("output"):
                report.append("**Output:**")
                report.append("```")
                report.append(result["output"])
                report.append("```")
                report.append("")
            
            if result.get("error") and result["error"].strip():
                report.append("**Errors:**")
                report.append("```")
                report.append(result["error"])
                report.append("```")
                report.append("")
            
            if result.get("recommendation"):
                report.append(f"**Recommendation:** {result['recommendation']}")
                report.append("")
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"📄 Report saved to: {output_file}")
        
        return report_text


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive quality checks")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-docs", action="store_true", help="Skip documentation build")
    parser.add_argument("--report", help="Save detailed report to file")
    parser.add_argument("--json", help="Save results as JSON to file")
    
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent
    
    # Run quality checks
    checker = QualityChecker(project_root)
    results = checker.run_all_checks(
        skip_tests=args.skip_tests,
        skip_docs=args.skip_docs
    )
    
    # Generate reports
    if args.report:
        checker.generate_report(results, args.report)
    
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 JSON results saved to: {args.json}")
    
    # Exit with error code if checks failed
    sys.exit(0 if results["overall_passed"] else 1)


if __name__ == "__main__":
    main()