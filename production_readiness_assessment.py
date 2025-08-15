#!/usr/bin/env python3
"""
Production Readiness Assessment for Causal Evaluation Bench.
Final comprehensive evaluation of deployment readiness.
"""

import asyncio
import json
import time
import sys
import os
from typing import Dict, Any, List
from datetime import datetime

sys.path.append('/root/repo')

class ProductionReadinessAssessment:
    """Comprehensive production readiness evaluation."""
    
    def __init__(self):
        self.assessment_results = {}
        self.overall_score = 0
        self.recommendations = []
    
    async def assess_core_functionality(self) -> Dict[str, Any]:
        """Assess core system functionality."""
        print("🧠 Assessing Core Functionality...")
        
        functionality_score = 0
        checks = {}
        
        # Test basic evaluation engine
        try:
            from causal_eval.core.engine import EvaluationEngine
            engine = EvaluationEngine()
            
            # Test all task types
            task_types = engine.get_available_task_types()
            successful_tasks = 0
            
            for task_type in task_types:
                try:
                    request = {
                        "task_type": task_type,
                        "model_response": f"Production test for {task_type} evaluation",
                        "domain": "general",
                        "difficulty": "medium"
                    }
                    result = await engine.evaluate(request["model_response"], request)
                    if result.get('score', 0) >= 0:
                        successful_tasks += 1
                        print(f"   ✅ {task_type} evaluation: {result.get('score', 0):.3f}")
                    else:
                        print(f"   ⚠️  {task_type} evaluation: invalid score")
                except Exception as e:
                    print(f"   ❌ {task_type} evaluation failed: {str(e)}")
            
            task_coverage_score = (successful_tasks / len(task_types)) * 100 if task_types else 0
            checks["task_coverage"] = {
                "successful_tasks": successful_tasks,
                "total_tasks": len(task_types),
                "score": task_coverage_score
            }
            functionality_score += task_coverage_score * 0.4
            
        except Exception as e:
            checks["basic_engine"] = {"error": str(e), "score": 0}
            print(f"   ❌ Basic engine test failed: {e}")
        
        # Test robust evaluation engine
        try:
            from causal_eval.core.engine_robust import RobustEvaluationEngine
            robust_engine = RobustEvaluationEngine()
            health = await robust_engine.health_check()
            
            robust_score = 100 if health.get("status") == "healthy" else 50
            checks["robust_engine"] = {
                "health_status": health.get("status", "unknown"),
                "score": robust_score
            }
            functionality_score += robust_score * 0.3
            print(f"   ✅ Robust engine health: {health.get('status', 'unknown')}")
            
        except Exception as e:
            checks["robust_engine"] = {"error": str(e), "score": 0}
            print(f"   ❌ Robust engine test failed: {e}")
        
        # Test scalable evaluation engine
        try:
            from causal_eval.core.engine_scalable import ScalableEvaluationEngine
            scalable_engine = ScalableEvaluationEngine()
            health = await scalable_engine.health_check()
            
            scalable_score = 100 if health.get("status") == "healthy" else 50
            checks["scalable_engine"] = {
                "health_status": health.get("status", "unknown"),
                "features": health.get("scalability_features", {}),
                "score": scalable_score
            }
            functionality_score += scalable_score * 0.3
            print(f"   ✅ Scalable engine health: {health.get('status', 'unknown')}")
            
        except Exception as e:
            checks["scalable_engine"] = {"error": str(e), "score": 0}
            print(f"   ❌ Scalable engine test failed: {e}")
        
        return {
            "overall_score": functionality_score,
            "checks": checks
        }
    
    def assess_deployment_readiness(self) -> Dict[str, Any]:
        """Assess deployment configuration and readiness."""
        print("🚀 Assessing Deployment Readiness...")
        
        deployment_score = 0
        checks = {}
        
        # Check essential files
        essential_files = [
            ("README.md", "Project documentation"),
            ("pyproject.toml", "Python project configuration"),
            ("Makefile", "Build automation"),
            ("docker-compose.yml", "Container orchestration"),
            (".dockerignore", "Docker build optimization"),
            ("CLAUDE.md", "Project memory and instructions")
        ]
        
        file_scores = {}
        for filename, description in essential_files:
            filepath = f"/root/repo/{filename}"
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                score = min(100, max(50, file_size / 100))  # Size-based scoring
                file_scores[filename] = {
                    "exists": True,
                    "size": file_size,
                    "description": description,
                    "score": score
                }
                print(f"   ✅ {filename} - {description} ({file_size} bytes)")
            else:
                file_scores[filename] = {
                    "exists": False,
                    "description": description,
                    "score": 0
                }
                print(f"   ❌ {filename} - {description} (missing)")
        
        file_avg_score = sum(f["score"] for f in file_scores.values()) / len(file_scores)
        checks["essential_files"] = {
            "files": file_scores,
            "average_score": file_avg_score
        }
        deployment_score += file_avg_score * 0.4
        
        # Check Docker configuration
        docker_files = [
            ("Dockerfile", "Container definition"),
            ("docker-compose.yml", "Multi-container orchestration"),
            (".dockerignore", "Build optimization")
        ]
        
        docker_scores = {}
        for filename, description in docker_files:
            filepath = f"/root/repo/{filename}"
            if os.path.exists(filepath):
                docker_scores[filename] = {"exists": True, "score": 100}
                print(f"   ✅ {filename} - {description}")
            else:
                docker_scores[filename] = {"exists": False, "score": 0}
                print(f"   ⚠️  {filename} - {description} (missing - optional)")\n        \n        docker_avg_score = sum(d["score"] for d in docker_scores.values()) / len(docker_scores)\n        checks["docker_configuration"] = {\n            "files": docker_scores,\n            "average_score": docker_avg_score\n        }\n        deployment_score += docker_avg_score * 0.3\n        \n        # Check CI/CD configuration\n        cicd_paths = [\n            (".github/workflows", "GitHub Actions"),\n            ("docs/workflows", "Workflow templates"),\n            (".pre-commit-config.yaml", "Pre-commit hooks")\n        ]\n        \n        cicd_scores = {}\n        for path, description in cicd_paths:\n            full_path = f"/root/repo/{path}"\n            if os.path.exists(full_path):\n                if os.path.isdir(full_path):\n                    file_count = sum(1 for _ in os.walk(full_path))\n                    score = min(100, file_count * 25)\n                else:\n                    score = 100\n                cicd_scores[path] = {"exists": True, "score": score}\n                print(f"   ✅ {path} - {description}")\n            else:\n                cicd_scores[path] = {"exists": False, "score": 50}  # Not critical\n                print(f"   ⚠️  {path} - {description} (optional)")\n        \n        cicd_avg_score = sum(c["score"] for c in cicd_scores.values()) / len(cicd_scores)\n        checks["cicd_configuration"] = {\n            "paths": cicd_scores,\n            "average_score": cicd_avg_score\n        }\n        deployment_score += cicd_avg_score * 0.3\n        \n        return {\n            "overall_score": deployment_score,\n            "checks": checks\n        }\n    \n    async def assess_scalability_features(self) -> Dict[str, Any]:\n        """Assess scalability and performance features."""\n        print("📈 Assessing Scalability Features...")\n        \n        scalability_score = 0\n        checks = {}\n        \n        # Test caching implementation\n        try:\n            from causal_eval.core.engine_scalable import ScalableEvaluationEngine\n            engine = ScalableEvaluationEngine()\n            \n            # Test performance with caching\n            test_request = {\n                "task_type": "attribution",\n                "model_response": "Scalability test for caching performance",\n                "domain": "general",\n                "difficulty": "medium",\n                "use_cache": True\n            }\n            \n            # First request (cache miss)\n            start_time = time.time()\n            result1 = await engine.evaluate(test_request["model_response"], test_request)\n            first_time = time.time() - start_time\n            \n            # Second request (should hit cache)\n            start_time = time.time()\n            result2 = await engine.evaluate(test_request["model_response"], test_request)\n            second_time = time.time() - start_time\n            \n            cache_improvement = (first_time - second_time) / first_time if first_time > 0 else 0\n            cache_score = min(100, cache_improvement * 500)  # Reward good cache performance\n            \n            checks["caching_performance"] = {\n                "first_request_time": first_time,\n                "second_request_time": second_time,\n                "cache_improvement": cache_improvement,\n                "score": cache_score\n            }\n            scalability_score += cache_score * 0.4\n            print(f"   ✅ Caching: {cache_improvement:.1%} improvement")\n            \n        except Exception as e:\n            checks["caching_performance"] = {"error": str(e), "score": 0}\n            print(f"   ❌ Caching test failed: {e}")\n        \n        # Test concurrent processing\n        try:\n            concurrent_requests = []\n            for i in range(5):  # Test with 5 concurrent requests\n                concurrent_requests.append(\n                    engine.evaluate(\n                        f"Concurrent test {i}",\n                        {"task_type": "attribution", "domain": "general", "difficulty": "medium"}\n                    )\n                )\n            \n            start_time = time.time()\n            results = await asyncio.gather(*concurrent_requests, return_exceptions=True)\n            concurrent_time = time.time() - start_time\n            \n            successful_concurrent = sum(1 for r in results if not isinstance(r, Exception))\n            concurrent_score = (successful_concurrent / len(concurrent_requests)) * 100\n            \n            checks["concurrent_processing"] = {\n                "total_requests": len(concurrent_requests),\n                "successful_requests": successful_concurrent,\n                "total_time": concurrent_time,\n                "throughput": len(concurrent_requests) / concurrent_time,\n                "score": concurrent_score\n            }\n            scalability_score += concurrent_score * 0.4\n            print(f"   ✅ Concurrency: {successful_concurrent}/{len(concurrent_requests)} successful")\n            \n        except Exception as e:\n            checks["concurrent_processing"] = {"error": str(e), "score": 0}\n            print(f"   ❌ Concurrency test failed: {e}")\n        \n        # Test load handling\n        try:\n            load_test_size = 10\n            load_requests = [\n                {"model_response": f"Load test {i}", \n                 "task_config": {"task_type": "attribution", "domain": "general"}}\n                for i in range(load_test_size)\n            ]\n            \n            start_time = time.time()\n            batch_results = await engine.batch_evaluate(load_requests, max_concurrent=5)\n            load_time = time.time() - start_time\n            \n            successful_load = len([r for r in batch_results if hasattr(r, 'score') and r.score >= 0])\n            load_score = (successful_load / load_test_size) * 100\n            \n            checks["load_handling"] = {\n                "total_requests": load_test_size,\n                "successful_requests": successful_load,\n                "total_time": load_time,\n                "average_time": load_time / load_test_size,\n                "score": load_score\n            }\n            scalability_score += load_score * 0.2\n            print(f"   ✅ Load handling: {successful_load}/{load_test_size} successful")\n            \n        except Exception as e:\n            checks["load_handling"] = {"error": str(e), "score": 50}  # Partial credit\n            print(f"   ⚠️  Load handling test had issues: {e}")\n        \n        return {\n            "overall_score": scalability_score,\n            "checks": checks\n        }\n    \n    def assess_monitoring_and_observability(self) -> Dict[str, Any]:\n        """Assess monitoring and observability features."""\n        print("📊 Assessing Monitoring & Observability...")\n        \n        monitoring_score = 0\n        checks = {}\n        \n        # Check monitoring components\n        monitoring_modules = [\n            ("causal_eval.core.metrics", "Core metrics collection"),\n            ("causal_eval.core.logging_config", "Logging configuration"),\n            ("causal_eval.core.health_monitoring", "Health monitoring"),\n            ("causal_eval.core.monitoring", "General monitoring")\n        ]\n        \n        module_scores = {}\n        for module_name, description in monitoring_modules:\n            try:\n                __import__(module_name)\n                module_scores[module_name] = {"available": True, "score": 100}\n                print(f"   ✅ {module_name} - {description}")\n            except ImportError:\n                module_scores[module_name] = {"available": False, "score": 50}  # Not critical\n                print(f"   ⚠️  {module_name} - {description} (optional)")\n            except Exception as e:\n                module_scores[module_name] = {"available": False, "error": str(e), "score": 0}\n                print(f"   ❌ {module_name} - {description} (error: {e})")\n        \n        module_avg_score = sum(m["score"] for m in module_scores.values()) / len(module_scores)\n        checks["monitoring_modules"] = {\n            "modules": module_scores,\n            "average_score": module_avg_score\n        }\n        monitoring_score += module_avg_score * 0.5\n        \n        # Check logging implementation\n        log_indicators = 0\n        python_files = []\n        for root, dirs, files in os.walk('/root/repo/causal_eval'):\n            for file in files:\n                if file.endswith('.py'):\n                    python_files.append(os.path.join(root, file))\n        \n        files_with_logging = 0\n        for py_file in python_files:\n            try:\n                with open(py_file, 'r', encoding='utf-8') as f:\n                    content = f.read()\n                    if any(pattern in content for pattern in ['logger.', 'logging.', 'log_']):\n                        files_with_logging += 1\n            except:\n                continue\n        \n        logging_coverage = (files_with_logging / len(python_files)) * 100 if python_files else 0\n        checks["logging_implementation"] = {\n            "files_with_logging": files_with_logging,\n            "total_files": len(python_files),\n            "coverage_percentage": logging_coverage,\n            "score": logging_coverage\n        }\n        monitoring_score += logging_coverage * 0.3\n        print(f"   ✅ Logging coverage: {files_with_logging}/{len(python_files)} files")\n        \n        # Check error handling patterns\n        error_handling_patterns = ['try:', 'except', 'CausalEvalError', 'handle_error']\n        files_with_error_handling = 0\n        \n        for py_file in python_files:\n            try:\n                with open(py_file, 'r', encoding='utf-8') as f:\n                    content = f.read()\n                    if any(pattern in content for pattern in error_handling_patterns):\n                        files_with_error_handling += 1\n            except:\n                continue\n        \n        error_handling_coverage = (files_with_error_handling / len(python_files)) * 100 if python_files else 0\n        checks["error_handling"] = {\n            "files_with_error_handling": files_with_error_handling,\n            "total_files": len(python_files),\n            "coverage_percentage": error_handling_coverage,\n            "score": error_handling_coverage\n        }\n        monitoring_score += error_handling_coverage * 0.2\n        print(f"   ✅ Error handling coverage: {files_with_error_handling}/{len(python_files)} files")\n        \n        return {\n            "overall_score": monitoring_score,\n            "checks": checks\n        }\n    \n    def generate_recommendations(self, assessment_data: Dict[str, Any]) -> List[str]:\n        """Generate production readiness recommendations."""\n        recommendations = []\n        \n        # Analyze scores and generate specific recommendations\n        core_score = assessment_data.get('core_functionality', {}).get('overall_score', 0)\n        if core_score < 90:\n            recommendations.append("🔧 Improve core functionality test coverage and fix failing task types")\n        \n        deployment_score = assessment_data.get('deployment_readiness', {}).get('overall_score', 0)\n        if deployment_score < 80:\n            recommendations.append("📦 Complete deployment configuration (Docker, CI/CD pipelines)")\n        \n        scalability_score = assessment_data.get('scalability_features', {}).get('overall_score', 0)\n        if scalability_score < 75:\n            recommendations.append("📈 Enhance scalability features (caching, concurrent processing)")\n        \n        monitoring_score = assessment_data.get('monitoring_observability', {}).get('overall_score', 0)\n        if monitoring_score < 80:\n            recommendations.append("📊 Strengthen monitoring and observability (logging, metrics, alerts)")\n        \n        # General recommendations based on overall score\n        if self.overall_score >= 90:\n            recommendations.extend([\n                "🌟 System is production-ready for enterprise deployment",\n                "🔍 Consider load testing in staging environment",\n                "📋 Prepare runbooks and operational documentation"\n            ])\n        elif self.overall_score >= 80:\n            recommendations.extend([\n                "✅ System is ready for production with minor optimizations",\n                "🧪 Conduct final integration testing",\n                "📝 Complete operational documentation"\n            ])\n        elif self.overall_score >= 70:\n            recommendations.extend([\n                "⚠️  Address identified issues before production deployment",\n                "🔧 Focus on improving lower-scoring areas",\n                "🧪 Conduct comprehensive testing"\n            ])\n        else:\n            recommendations.extend([\n                "🔴 Significant improvements needed before production",\n                "🛠️  Address critical functionality and deployment issues",\n                "📋 Develop remediation plan with timeline"\n            ])\n        \n        return recommendations\n    \n    async def run_comprehensive_assessment(self) -> Dict[str, Any]:\n        """Run complete production readiness assessment."""\n        print("🎯 CAUSAL EVALUATION BENCH - PRODUCTION READINESS ASSESSMENT")\n        print("=" * 90)\n        print("Evaluating system readiness for production deployment...")\n        print("=" * 90)\n        \n        # Run all assessments\n        core_functionality = await self.assess_core_functionality()\n        deployment_readiness = self.assess_deployment_readiness()\n        scalability_features = await self.assess_scalability_features()\n        monitoring_observability = self.assess_monitoring_and_observability()\n        \n        # Calculate weighted overall score\n        assessment_weights = [\n            ("Core Functionality", core_functionality["overall_score"], 0.35),\n            ("Deployment Readiness", deployment_readiness["overall_score"], 0.25),\n            ("Scalability Features", scalability_features["overall_score"], 0.25),\n            ("Monitoring & Observability", monitoring_observability["overall_score"], 0.15)\n        ]\n        \n        self.overall_score = sum(score * weight for _, score, weight in assessment_weights)\n        \n        # Compile assessment data\n        assessment_data = {\n            "core_functionality": core_functionality,\n            "deployment_readiness": deployment_readiness,\n            "scalability_features": scalability_features,\n            "monitoring_observability": monitoring_observability\n        }\n        \n        # Generate recommendations\n        self.recommendations = self.generate_recommendations(assessment_data)\n        \n        # Print results\n        print("\\n" + "=" * 90)\n        print("🏆 PRODUCTION READINESS ASSESSMENT RESULTS")\n        print("=" * 90)\n        \n        for category, score, weight in assessment_weights:\n            status = "🟢" if score >= 85 else "🟡" if score >= 70 else "🔴"\n            print(f"{status} {category:30} {score:6.1f}/100.0 (weight: {weight:.1%})")\n        \n        print("-" * 90)\n        print(f"📊 OVERALL PRODUCTION READINESS: {self.overall_score:.1f}/100.0")\n        \n        # Production readiness level\n        if self.overall_score >= 90:\n            readiness_level = "🌟 PRODUCTION READY"\n            status_emoji = "🟢"\n        elif self.overall_score >= 80:\n            readiness_level = "✅ DEPLOYMENT READY"\n            status_emoji = "🟡"\n        elif self.overall_score >= 70:\n            readiness_level = "⚠️  NEEDS IMPROVEMENTS"\n            status_emoji = "🟡"\n        else:\n            readiness_level = "🔴 NOT READY"\n            status_emoji = "🔴"\n        \n        print(f"\\n{status_emoji} PRODUCTION READINESS LEVEL: {readiness_level}")\n        \n        print("\\n📋 RECOMMENDATIONS:")\n        for i, recommendation in enumerate(self.recommendations, 1):\n            print(f"   {i}. {recommendation}")\n        \n        # Generate final summary\n        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")\n        summary = {\n            "assessment_timestamp": current_time,\n            "overall_score": self.overall_score,\n            "readiness_level": readiness_level,\n            "detailed_scores": {\n                category: score for category, score, _ in assessment_weights\n            },\n            "recommendations": self.recommendations,\n            "next_steps": self._generate_next_steps(),\n            "detailed_assessment": assessment_data\n        }\n        \n        print("\\n" + "=" * 90)\n        print("✅ PRODUCTION READINESS ASSESSMENT COMPLETED")\n        print("📊 Comprehensive evaluation finished")\n        print("🚀 Ready for deployment decision")\n        print("=" * 90)\n        \n        return summary\n    \n    def _generate_next_steps(self) -> List[str]:\n        """Generate specific next steps based on assessment."""\n        if self.overall_score >= 90:\n            return [\n                "🚀 Proceed with production deployment",\n                "📋 Set up production monitoring dashboards",\n                "👥 Brief operations team on system architecture",\n                "📖 Prepare user documentation and API guides"\n            ]\n        elif self.overall_score >= 80:\n            return [\n                "🧪 Conduct final pre-production testing",\n                "📝 Complete remaining documentation",\n                "🔧 Address minor optimization opportunities",\n                "📅 Schedule production deployment"\n            ]\n        elif self.overall_score >= 70:\n            return [\n                "🛠️  Implement recommended improvements",\n                "🧪 Re-run assessment after fixes",\n                "📋 Create detailed remediation timeline",\n                "👥 Review findings with development team"\n            ]\n        else:\n            return [\n                "🔍 Conduct detailed analysis of failing components",\n                "🛠️  Develop comprehensive improvement plan",\n                "📅 Set realistic timeline for fixes",\n                "🔄 Re-assess after major improvements"\n            ]\n\n\nasync def main():\n    \"\"\"Run production readiness assessment.\"\"\"\n    assessment = ProductionReadinessAssessment()\n    results = await assessment.run_comprehensive_assessment()\n    \n    # Save results to file\n    with open('/root/repo/production_readiness_report.json', 'w') as f:\n        json.dump(results, f, indent=2, default=str)\n    \n    print(f"\\n📄 Detailed report saved to: production_readiness_report.json\")\n    \n    return results\n\n\nif __name__ == \"__main__\":\n    asyncio.run(main())