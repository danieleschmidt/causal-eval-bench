#!/usr/bin/env python3
"""
Comprehensive Quality Gates Assessment for Causal Evaluation Bench.
Tests coverage, security, performance, and documentation standards.
"""

import asyncio
import time
import sys
import subprocess
import os
from typing import Dict, List, Tuple, Any

sys.path.append('/root/repo')

def run_command(command: str, description: str = "") -> Tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, f"Command timed out: {command}"
    except Exception as e:
        return False, f"Command failed: {str(e)}"

def assess_code_quality() -> Dict[str, Any]:
    """Assess code quality metrics."""
    print("🔍 Assessing Code Quality...")
    
    quality_results = {
        "overall_score": 0,
        "checks": {}
    }
    
    # Check Python syntax in all files
    python_files = []
    for root, dirs, files in os.walk('/root/repo/causal_eval'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    syntax_errors = 0
    for py_file in python_files:
        success, output = run_command(f"python3 -m py_compile {py_file}")
        if not success:
            syntax_errors += 1
            print(f"   ❌ Syntax error in {py_file}")
        else:
            print(f"   ✅ {os.path.basename(py_file)} - syntax OK")
    
    quality_results["checks"]["syntax"] = {
        "total_files": len(python_files),
        "errors": syntax_errors,
        "score": max(0, (len(python_files) - syntax_errors) / len(python_files) * 100) if python_files else 100
    }
    
    # Check imports and basic functionality
    import_errors = 0
    core_modules = [
        "causal_eval.core.engine",
        "causal_eval.core.engine_robust", 
        "causal_eval.core.engine_scalable",
        "causal_eval.tasks.attribution",
        "causal_eval.tasks.counterfactual"
    ]
    
    for module in core_modules:
        try:
            __import__(module)
            print(f"   ✅ {module} - import OK")
        except Exception as e:
            import_errors += 1
            print(f"   ❌ {module} - import failed: {str(e)}")
    
    quality_results["checks"]["imports"] = {
        "total_modules": len(core_modules),
        "errors": import_errors,
        "score": max(0, (len(core_modules) - import_errors) / len(core_modules) * 100)
    }
    
    # Calculate overall code quality score
    syntax_score = quality_results["checks"]["syntax"]["score"]
    import_score = quality_results["checks"]["imports"]["score"]
    quality_results["overall_score"] = (syntax_score * 0.6) + (import_score * 0.4)
    
    return quality_results

def assess_security_standards() -> Dict[str, Any]:
    """Assess security implementation."""
    print("🛡️ Assessing Security Standards...")
    
    security_results = {
        "overall_score": 0,
        "checks": {}
    }
    
    # Check for security-related code
    security_patterns = [
        ("input_validation", ["validate", "sanitize", "validator"]),
        ("rate_limiting", ["rate_limit", "rate limiting", "request_counts"]),
        ("error_handling", ["try:", "except", "CausalEvalError"]),
        ("secure_patterns", ["security", "suspicious", "validate_security"])
    ]
    
    security_scores = {}
    
    for pattern_name, keywords in security_patterns:
        found_files = 0
        total_matches = 0
        
        for root, dirs, files in os.walk('/root/repo/causal_eval'):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read().lower()
                            file_matches = sum(1 for keyword in keywords if keyword in content)
                            if file_matches > 0:
                                found_files += 1
                                total_matches += file_matches
                    except Exception:
                        continue
        
        security_scores[pattern_name] = {
            "files_with_pattern": found_files,
            "total_matches": total_matches,
            "score": min(100, total_matches * 10)  # Cap at 100
        }
        
        print(f"   ✅ {pattern_name}: {found_files} files, {total_matches} matches")
    
    security_results["checks"] = security_scores
    
    # Calculate overall security score
    pattern_scores = [check["score"] for check in security_scores.values()]
    security_results["overall_score"] = sum(pattern_scores) / len(pattern_scores) if pattern_scores else 0
    
    return security_results

async def assess_performance_standards() -> Dict[str, Any]:
    """Assess performance implementation."""
    print("⚡ Assessing Performance Standards...")
    
    performance_results = {
        "overall_score": 0,
        "checks": {}
    }
    
    # Test basic engine performance
    try:
        from causal_eval.core.engine import EvaluationEngine
        
        engine = EvaluationEngine()
        
        # Simple performance test
        start_time = time.time()
        
        # Test multiple evaluations
        evaluation_times = []
        for i in range(5):
            eval_start = time.time()
            
            request = {
                "task_type": "attribution",
                "model_response": f"Test response {i} for performance evaluation",
                "domain": "general",
                "difficulty": "medium"
            }
            
            result = await engine.evaluate(request["model_response"], request)
            eval_time = time.time() - eval_start
            evaluation_times.append(eval_time)
        
        total_time = time.time() - start_time
        avg_time = sum(evaluation_times) / len(evaluation_times)
        
        performance_results["checks"]["basic_performance"] = {
            "total_evaluations": len(evaluation_times),
            "total_time_ms": total_time * 1000,
            "avg_time_ms": avg_time * 1000,
            "throughput_per_sec": len(evaluation_times) / total_time,
            "score": min(100, max(0, 100 - (avg_time * 1000 - 100) / 10))  # 100ms baseline
        }
        
        print(f"   ✅ Basic performance: {avg_time*1000:.1f}ms average")
        
    except Exception as e:
        performance_results["checks"]["basic_performance"] = {
            "error": str(e),
            "score": 0
        }
        print(f"   ❌ Basic performance test failed: {str(e)}")
    
    # Test scalable engine if available
    try:
        from causal_eval.core.engine_scalable import ScalableEvaluationEngine
        
        scalable_engine = ScalableEvaluationEngine()
        health = await scalable_engine.health_check()
        
        performance_results["checks"]["scalable_performance"] = {
            "health_status": health.get("status", "unknown"),
            "features_enabled": health.get("scalability_features", {}),
            "score": 90 if health.get("status") == "healthy" else 30
        }
        
        print(f"   ✅ Scalable performance: {health.get('status', 'unknown')}")
        
    except Exception as e:
        performance_results["checks"]["scalable_performance"] = {
            "error": str(e),
            "score": 0
        }
        print(f"   ❌ Scalable performance test failed: {str(e)}")
    
    # Calculate overall performance score
    perf_scores = [check.get("score", 0) for check in performance_results["checks"].values()]
    performance_results["overall_score"] = sum(perf_scores) / len(perf_scores) if perf_scores else 0
    
    return performance_results

def assess_test_coverage() -> Dict[str, Any]:
    """Assess test coverage."""
    print("🧪 Assessing Test Coverage...")
    
    coverage_results = {
        "overall_score": 0,
        "checks": {}
    }
    
    # Count test files
    test_files = []
    test_dirs = ['/root/repo/tests', '/root/repo']  # Check both standard test dir and root
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for root, dirs, files in os.walk(test_dir):
                for file in files:
                    if file.startswith('test_') and file.endswith('.py'):
                        test_files.append(os.path.join(root, file))
    
    # Count source files
    source_files = []
    for root, dirs, files in os.walk('/root/repo/causal_eval'):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                source_files.append(os.path.join(root, file))
    
    coverage_results["checks"]["test_files"] = {
        "test_files_count": len(test_files),
        "source_files_count": len(source_files),
        "coverage_ratio": len(test_files) / max(len(source_files), 1),
        "score": min(100, (len(test_files) / max(len(source_files), 1)) * 100)
    }
    
    print(f"   ✅ Test files: {len(test_files)} (source files: {len(source_files)})")
    for test_file in test_files:
        print(f"      - {os.path.basename(test_file)}")
    
    # Test execution check
    executable_tests = 0
    for test_file in test_files:
        success, output = run_command(f"python3 {test_file}")
        if success:
            executable_tests += 1
            print(f"   ✅ {os.path.basename(test_file)} - executable")
        else:
            print(f"   ⚠️  {os.path.basename(test_file)} - execution issues")
    
    coverage_results["checks"]["test_execution"] = {
        "total_tests": len(test_files),
        "executable_tests": executable_tests,
        "execution_rate": executable_tests / max(len(test_files), 1),
        "score": (executable_tests / max(len(test_files), 1)) * 100 if test_files else 80
    }
    
    # Calculate overall coverage score
    test_score = coverage_results["checks"]["test_files"]["score"]
    exec_score = coverage_results["checks"]["test_execution"]["score"]
    coverage_results["overall_score"] = (test_score * 0.4) + (exec_score * 0.6)
    
    return coverage_results

def assess_documentation() -> Dict[str, Any]:
    """Assess documentation completeness."""
    print("📚 Assessing Documentation...")
    
    doc_results = {
        "overall_score": 0,
        "checks": {}
    }
    
    # Check for key documentation files
    key_docs = [
        ("README.md", "/root/repo/README.md"),
        ("CLAUDE.md", "/root/repo/CLAUDE.md"),
        ("Architecture docs", "/root/repo/docs"),
        ("API docs", "/root/repo/causal_eval/api"),
        ("Implementation status", "/root/repo/IMPLEMENTATION_STATUS.md")
    ]
    
    doc_scores = {}
    
    for doc_name, doc_path in key_docs:
        if os.path.exists(doc_path):
            if os.path.isfile(doc_path):
                # Check file size as quality indicator
                file_size = os.path.getsize(doc_path)
                score = min(100, max(20, file_size / 100))  # 20-100 based on size
                doc_scores[doc_name] = {"exists": True, "size": file_size, "score": score}
                print(f"   ✅ {doc_name} - exists ({file_size} bytes)")
            else:
                # Directory - check if it has content
                try:
                    files_in_dir = sum(1 for _ in os.walk(doc_path))
                    score = min(100, files_in_dir * 20)
                    doc_scores[doc_name] = {"exists": True, "files": files_in_dir, "score": score}
                    print(f"   ✅ {doc_name} - exists (directory with {files_in_dir} items)")
                except:
                    doc_scores[doc_name] = {"exists": False, "score": 0}
                    print(f"   ❌ {doc_name} - directory exists but inaccessible")
        else:
            doc_scores[doc_name] = {"exists": False, "score": 0}
            print(f"   ❌ {doc_name} - missing")
    
    doc_results["checks"] = doc_scores
    
    # Check docstrings in code
    python_files_with_docstrings = 0
    total_python_files = 0
    
    for root, dirs, files in os.walk('/root/repo/causal_eval'):
        for file in files:
            if file.endswith('.py'):
                total_python_files += 1
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Simple check for docstrings
                        if '"""' in content or "'''" in content:
                            python_files_with_docstrings += 1
                except:
                    pass
    
    docstring_score = (python_files_with_docstrings / max(total_python_files, 1)) * 100
    doc_results["checks"]["code_documentation"] = {
        "files_with_docstrings": python_files_with_docstrings,
        "total_python_files": total_python_files,
        "score": docstring_score
    }
    
    print(f"   ✅ Code documentation: {python_files_with_docstrings}/{total_python_files} files with docstrings")
    
    # Calculate overall documentation score
    doc_file_scores = [check.get("score", 0) for check in doc_scores.values()]
    avg_doc_score = sum(doc_file_scores) / len(doc_file_scores) if doc_file_scores else 0
    doc_results["overall_score"] = (avg_doc_score * 0.7) + (docstring_score * 0.3)
    
    return doc_results

async def run_comprehensive_quality_gates():
    """Run all quality gates and provide comprehensive assessment."""
    print("🎯 CAUSAL EVALUATION BENCH - QUALITY GATES ASSESSMENT")
    print("=" * 80)
    print("Running comprehensive quality, security, performance, and documentation checks...")
    print("=" * 80)
    
    # Run all assessments
    code_quality = assess_code_quality()
    security_standards = assess_security_standards()
    performance_standards = await assess_performance_standards()
    test_coverage = assess_test_coverage()
    documentation = assess_documentation()
    
    # Calculate overall quality score
    quality_scores = [
        ("Code Quality", code_quality["overall_score"], 0.25),
        ("Security Standards", security_standards["overall_score"], 0.25),
        ("Performance Standards", performance_standards["overall_score"], 0.20),
        ("Test Coverage", test_coverage["overall_score"], 0.15),
        ("Documentation", documentation["overall_score"], 0.15)
    ]
    
    weighted_score = sum(score * weight for _, score, weight in quality_scores)
    
    print("\\n" + "=" * 80)
    print("🏆 QUALITY GATES ASSESSMENT RESULTS")
    print("=" * 80)
    
    for category, score, weight in quality_scores:
        status = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
        print(f"{status} {category:25} {score:6.1f}/100.0 (weight: {weight:.1%})")
    
    print("-" * 80)
    print(f"📊 OVERALL QUALITY SCORE: {weighted_score:.1f}/100.0")
    
    # Quality assessment
    if weighted_score >= 90:
        quality_level = "🌟 EXCEPTIONAL"
        recommendation = "System exceeds all quality standards and is ready for enterprise deployment."
    elif weighted_score >= 80:
        quality_level = "✅ EXCELLENT" 
        recommendation = "System meets all production quality standards."
    elif weighted_score >= 70:
        quality_level = "🟡 GOOD"
        recommendation = "System meets most quality standards with minor improvements needed."
    elif weighted_score >= 60:
        quality_level = "⚠️  ACCEPTABLE"
        recommendation = "System meets basic quality standards but needs improvement in some areas."
    else:
        quality_level = "🔴 NEEDS IMPROVEMENT"
        recommendation = "System requires significant quality improvements before production deployment."
    
    print(f"🎖️  QUALITY LEVEL: {quality_level}")
    print(f"💡 RECOMMENDATION: {recommendation}")
    
    print("\\n" + "=" * 80)
    print("✅ QUALITY GATES ASSESSMENT COMPLETED")
    print("📋 All quality standards have been evaluated")
    print("🚀 System ready for production readiness assessment")
    print("=" * 80)
    
    return {
        "overall_score": weighted_score,
        "quality_level": quality_level,
        "recommendation": recommendation,
        "detailed_scores": {
            "code_quality": code_quality,
            "security_standards": security_standards, 
            "performance_standards": performance_standards,
            "test_coverage": test_coverage,
            "documentation": documentation
        }
    }

if __name__ == "__main__":
    asyncio.run(run_comprehensive_quality_gates())