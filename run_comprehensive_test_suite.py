#!/usr/bin/env python3
"""Comprehensive test suite runner for the entire causal evaluation system."""

import asyncio
import sys
import time
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import subprocess
import os

# Test result tracking
@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float
    details: str = ""
    errors: List[str] = None

class ComprehensiveTestRunner:
    """Runs all test suites and generates comprehensive report."""
    
    def __init__(self):
        self.test_results: List[TestResult] = []
        self.total_start_time = time.time()
    
    def run_test_file(self, test_file: str, description: str) -> TestResult:
        """Run a single test file and return result."""
        
        print(f"\n=== Running {description} ===")
        start_time = time.time()
        
        try:
            # Run the test file
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start_time
            
            # Parse output for success/failure
            output = result.stdout
            error_output = result.stderr
            
            # Simple heuristics for success detection
            success_indicators = [
                "All tests passed",
                "🎉",
                "All.*tests.*passed",
                "PASS"
            ]
            
            failure_indicators = [
                "FAIL",
                "failed",
                "error",
                "❌"
            ]
            
            passed = (result.returncode == 0 and 
                     any(indicator in output.lower() for indicator in success_indicators))
            
            # Extract error details if failed
            errors = []
            if not passed:
                if error_output:
                    errors.append(f"STDERR: {error_output}")
                if result.returncode != 0:
                    errors.append(f"Exit code: {result.returncode}")
            
            details = f"Duration: {duration:.2f}s\\nOutput: {output[:500]}..."
            if error_output:
                details += f"\\nErrors: {error_output[:200]}..."
            
            return TestResult(
                name=description,
                passed=passed,
                duration=duration,
                details=details,
                errors=errors or []
            )
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return TestResult(
                name=description,
                passed=False,
                duration=duration,
                details="Test timed out after 5 minutes",
                errors=["Timeout"]
            )
        
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                name=description,
                passed=False,
                duration=duration,
                details=f"Exception running test: {str(e)}",
                errors=[str(e)]
            )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all available test suites."""
        
        print("=== CAUSAL EVALUATION BENCH - COMPREHENSIVE TEST SUITE ===")
        print(f"Started at: {time.ctime()}")
        
        # Define test files and their descriptions
        test_files = [
            ("test_minimal_causal_system.py", "Core Causal Logic"),
            ("test_robust_evaluation.py", "Robust Error Handling"),
            ("test_scalable_system.py", "Scalability & Performance"),
            ("test_basic_functionality_simple.py", "Basic Functionality"),
        ]
        
        # Optional test files (run if they exist)
        optional_tests = [
            ("test_comprehensive_suite.py", "Advanced Integration Tests"),
            ("test_generation2_robustness.py", "Generation 2 Robustness"),
            ("test_generation3_scaling.py", "Generation 3 Scaling"),
        ]
        
        # Run core tests
        for test_file, description in test_files:
            if os.path.exists(test_file):
                result = self.run_test_file(test_file, description)
                self.test_results.append(result)
                
                if result.passed:
                    print(f"✅ {description}: PASSED ({result.duration:.2f}s)")
                else:
                    print(f"❌ {description}: FAILED ({result.duration:.2f}s)")
                    for error in result.errors:
                        print(f"   Error: {error}")
            else:
                print(f"⚠️  {description}: Test file not found ({test_file})")
        
        # Run optional tests
        for test_file, description in optional_tests:
            if os.path.exists(test_file):
                result = self.run_test_file(test_file, f"Optional: {description}")
                self.test_results.append(result)
                
                if result.passed:
                    print(f"✅ {description}: PASSED ({result.duration:.2f}s)")
                else:
                    print(f"❌ {description}: FAILED ({result.duration:.2f}s)")
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        total_duration = time.time() - self.total_start_time
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if result.passed)
        failed_tests = total_tests - passed_tests
        
        # Calculate success rate
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Group results
        core_tests = [r for r in self.test_results if not r.name.startswith("Optional:")]
        optional_tests = [r for r in self.test_results if r.name.startswith("Optional:")]
        
        core_passed = sum(1 for r in core_tests if r.passed)
        core_total = len(core_tests)
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate_percent": success_rate,
                "total_duration_seconds": total_duration,
                "core_tests_passed": core_passed,
                "core_tests_total": core_total,
                "core_success_rate": (core_passed / core_total * 100) if core_total > 0 else 0
            },
            "test_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration": r.duration,
                    "errors": r.errors
                }
                for r in self.test_results
            ],
            "failed_tests_details": [
                {
                    "name": r.name,
                    "errors": r.errors,
                    "details": r.details
                }
                for r in self.test_results if not r.passed
            ]
        }
        
        return report
    
    def print_final_report(self, report: Dict[str, Any]):
        """Print final comprehensive report."""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST SUITE REPORT")
        print("="*80)
        
        summary = report["summary"]
        
        print(f"Total Tests Run: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']}")
        print(f"Failed: {summary['failed_tests']}")
        print(f"Success Rate: {summary['success_rate_percent']:.1f}%")
        print(f"Total Duration: {summary['total_duration_seconds']:.2f}s")
        
        print(f"\nCore Tests: {summary['core_tests_passed']}/{summary['core_tests_total']} passed ({summary['core_success_rate']:.1f}%)")
        
        # Test breakdown
        print("\n--- Test Results ---")
        for result_data in report["test_results"]:
            status = "PASS" if result_data["passed"] else "FAIL"
            print(f"{status:4s} | {result_data['name']:35s} | {result_data['duration']:6.2f}s")
        
        # Failed tests details
        if report["failed_tests_details"]:
            print("\n--- Failed Tests Details ---")
            for failed_test in report["failed_tests_details"]:
                print(f"\n❌ {failed_test['name']}:")
                for error in failed_test["errors"]:
                    print(f"   - {error}")
        
        # Overall assessment
        print("\n--- Overall Assessment ---")
        if summary["core_success_rate"] >= 100:
            print("🎉 EXCELLENT: All core tests passed!")
            print("✅ System is ready for production deployment")
        elif summary["core_success_rate"] >= 75:
            print("✅ GOOD: Most core tests passed")
            print("⚠️  Some issues detected but system is largely functional")
        elif summary["core_success_rate"] >= 50:
            print("⚠️  MODERATE: Significant issues detected")
            print("🔧 System requires fixes before production deployment")
        else:
            print("❌ POOR: Major system issues detected")
            print("🚨 System not ready for deployment - requires immediate attention")
        
        print("\n" + "="*80)


def main():
    """Main entry point."""
    
    runner = ComprehensiveTestRunner()
    
    try:
        report = runner.run_all_tests()
        runner.print_final_report(report)
        
        # Exit with error code if core tests failed
        core_success_rate = report["summary"]["core_success_rate"]
        if core_success_rate < 75:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n\nUnexpected error running test suite: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()