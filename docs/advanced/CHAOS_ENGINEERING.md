# Chaos Engineering for Causal Eval Bench

This guide covers implementing chaos engineering practices to ensure the resilience and reliability of your causal evaluation infrastructure in production environments.

## 🌪️ Introduction to Chaos Engineering

Chaos engineering is the discipline of experimenting on a system to build confidence in its capability to withstand turbulent conditions in production. For evaluation systems like Causal Eval Bench, this means testing how your system behaves under various failure scenarios.

## 🎯 Chaos Engineering Principles

### Core Principles for Evaluation Systems

1. **Evaluation Continuity**: Ensure evaluations can continue even under partial system failures
2. **Data Integrity**: Maintain evaluation result accuracy during disruptions  
3. **Graceful Degradation**: Reduce functionality gracefully rather than complete failure
4. **Rapid Recovery**: Minimize time to restore full evaluation capabilities

### Hypothesis-Driven Experiments

Structure chaos experiments as testable hypotheses:

```python
from causal_eval.chaos import ChaosExperiment

class EvaluationResilienceTest(ChaosExperiment):
    def hypothesis(self):
        return """
        Given that our evaluation system is designed for high availability,
        when we introduce a database connection failure,
        then evaluation requests should continue to be processed
        using cached results and alternative storage,
        and system should recover within 60 seconds
        without data loss.
        """
    
    def steady_state(self):
        """Define what normal system behavior looks like."""
        return {
            "evaluation_success_rate": "> 99%",
            "average_response_time": "< 2s",
            "active_evaluations": "> 0",
            "data_consistency": "100%"
        }
    
    def turbulence(self):
        """Define the chaos to introduce."""
        return [
            self.inject_database_failure(),
            self.inject_network_latency(delay_ms=500),
            self.inject_memory_pressure(percentage=80)
        ]
```

## 🧪 Chaos Experiments for Evaluation Systems

### Database Resilience

Test database failure scenarios:

```python
from causal_eval.chaos import DatabaseChaos

class DatabaseFailureChaos(DatabaseChaos):
    def setup_experiment(self):
        """Setup database chaos experiment."""
        self.target_databases = ["primary", "cache"]
        self.failure_types = [
            "connection_timeout",
            "high_latency", 
            "partial_unavailability",
            "data_corruption_simulation"
        ]
    
    def connection_timeout_experiment(self):
        """Test behavior when database connections time out."""
        with self.inject_connection_timeout(duration=30):
            # Continue evaluation requests
            results = self.run_evaluation_batch(size=10)
            
            # Verify fallback mechanisms
            assert results.success_rate > 0.8, "Fallback mechanisms failed"
            assert results.used_cache, "Cache fallback not triggered"
            assert results.data_integrity, "Data integrity compromised"
    
    def high_latency_experiment(self):
        """Test system behavior under high database latency."""
        with self.inject_latency(min_delay=1000, max_delay=5000):
            start_time = time.time()
            
            # Run evaluation with latency
            results = self.run_evaluation_request()
            
            elapsed_time = time.time() - start_time
            
            # Verify timeout handling
            assert elapsed_time < 10, "Request timeout not working"
            assert results.status in ["success", "partial"], "No graceful degradation"
    
    def data_corruption_simulation(self):
        """Test handling of corrupted evaluation data."""
        with self.inject_data_corruption(corruption_rate=0.1):
            # Attempt to process potentially corrupted data
            results = self.run_evaluation_with_validation()
            
            # Verify data validation catches corruption
            assert results.validation_errors > 0, "Data validation not working"
            assert results.clean_data_percentage > 0.9, "Too much data lost"
```

### API Resilience

Test API failure scenarios:

```python
from causal_eval.chaos import APIChaos

class APIResilienceChaos(APIChaos):
    def setup_experiment(self):
        """Setup API chaos experiments."""
        self.target_endpoints = [
            "/evaluate/single",
            "/evaluate/batch", 
            "/results/{id}",
            "/models/list"
        ]
    
    def high_traffic_experiment(self):
        """Test system under high concurrent load."""
        concurrent_requests = 100
        request_duration = 60  # seconds
        
        with self.generate_high_traffic(
            concurrent_users=concurrent_requests,
            duration=request_duration
        ):
            # Monitor system metrics
            metrics = self.monitor_system_metrics()
            
            # Verify system handles load gracefully
            assert metrics.success_rate > 0.95, "Success rate too low under load"
            assert metrics.average_response_time < 5.0, "Response time too high"
            assert metrics.error_rate < 0.05, "Error rate too high"
    
    def partial_service_failure(self):
        """Test behavior when some API services fail."""
        # Disable model service
        with self.disable_service("model_service"):
            # Try to run evaluation
            response = self.client.post("/evaluate/single", json={
                "model": "gpt-4",
                "question": "Test question",
                "task_type": "causal_attribution"
            })
            
            # Should get informative error, not crash
            assert response.status_code in [503, 424], "Wrong error code"
            assert "model_service" in response.json()["error"], "Not informative"
    
    def cascade_failure_prevention(self):
        """Test prevention of cascade failures."""
        with self.inject_service_delays({
            "database": 5000,  # 5s delay
            "cache": 3000,     # 3s delay
            "model_api": 10000 # 10s delay
        }):
            # System should use circuit breakers
            results = self.run_evaluation_requests(count=5)
            
            # Verify circuit breakers activated
            assert results.circuit_breaker_trips > 0, "Circuit breakers not working"
            assert results.fast_failures > 0, "Not failing fast"
```

### Model Service Resilience

Test external model API failure scenarios:

```python
from causal_eval.chaos import ModelServiceChaos

class ModelServiceResilienceChaos(ModelServiceChaos):
    def setup_experiment(self):
        """Setup model service chaos experiments."""
        self.model_services = [
            "openai_api",
            "anthropic_api", 
            "local_model_service"
        ]
    
    def api_rate_limiting_experiment(self):
        """Test behavior when model APIs rate limit requests."""
        with self.inject_rate_limiting("openai_api", limit=5):
            # Run batch evaluation that exceeds rate limit
            results = self.run_large_evaluation_batch(size=20)
            
            # Verify rate limiting is handled gracefully
            assert results.completed_evaluations > 0, "No evaluations completed"
            assert results.retry_attempts > 0, "No retry logic triggered"
            assert results.backoff_used, "No exponential backoff"
    
    def model_api_timeout_experiment(self):
        """Test handling of model API timeouts."""
        with self.inject_api_timeouts("anthropic_api", timeout_rate=0.3):
            # Run evaluations with timeout probability
            results = self.run_evaluation_with_retries()
            
            # Verify timeout handling
            assert results.timeout_count > 0, "Timeouts not detected"
            assert results.fallback_model_used, "No fallback model"
            assert results.partial_results_saved, "Partial results lost"
    
    def model_degraded_performance(self):
        """Test handling of degraded model performance."""
        with self.inject_model_degradation(
            accuracy_reduction=0.2,
            latency_increase=3.0
        ):
            # Run evaluation with degraded model
            results = self.run_performance_monitoring()
            
            # Verify degradation detection
            assert results.performance_alerts > 0, "Performance degradation not detected"
            assert results.quality_checks_failed > 0, "Quality degradation not caught"
```

### Infrastructure Resilience

Test infrastructure failure scenarios:

```python
from causal_eval.chaos import InfrastructureChaos

class InfrastructureResilienceChaos(InfrastructureChaos):
    def setup_experiment(self):
        """Setup infrastructure chaos experiments."""
        self.target_components = [
            "web_servers",
            "worker_processes",
            "load_balancer",
            "container_runtime"
        ]
    
    def container_restart_experiment(self):
        """Test system behavior during container restarts."""
        # Kill random containers
        with self.kill_random_containers(kill_rate=0.2):
            # Continue processing during restarts
            results = self.run_continuous_evaluation(duration=120)
            
            # Verify graceful handling
            assert results.availability > 0.95, "Availability too low"
            assert results.request_success_rate > 0.9, "Too many failed requests"
            assert results.auto_recovery_time < 30, "Recovery too slow"
    
    def network_partition_experiment(self):
        """Test handling of network partitions."""
        with self.create_network_partition([
            "web_tier", "application_tier", "data_tier"
        ]):
            # Test cross-tier communication
            results = self.test_tier_communication()
            
            # Verify partition tolerance
            assert results.tier_isolation_working, "Tiers not properly isolated"
            assert results.data_consistency_maintained, "Data consistency lost"
    
    def resource_exhaustion_experiment(self):
        """Test behavior under resource exhaustion."""
        scenarios = [
            self.exhaust_memory(percentage=90),
            self.exhaust_cpu(percentage=95),
            self.exhaust_disk_space(percentage=85),
            self.exhaust_file_descriptors(percentage=90)
        ]
        
        for scenario in scenarios:
            with scenario:
                results = self.monitor_resource_handling()
                
                # Verify graceful degradation
                assert results.oom_kills == 0, "Out of memory kills occurred"
                assert results.response_time_increase < 10, "Response time too high"
                assert results.error_rate < 0.1, "Error rate too high"
```

## 🛠️ Chaos Testing Framework

### Core Chaos Framework

Build a framework for systematic chaos testing:

```python
from causal_eval.chaos.framework import ChaosTestFramework
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ChaosExperimentResult:
    experiment_name: str
    hypothesis_validated: bool
    steady_state_maintained: bool
    recovery_time_seconds: float
    observations: Dict[str, Any]
    recommendations: List[str]

class CausalEvalChaosFramework(ChaosTestFramework):
    def __init__(self, environment: str = "staging"):
        super().__init__(environment)
        self.evaluation_client = EvaluationClient()
        self.monitoring = MonitoringClient()
        self.safety_checks = SafetyValidator()
    
    def run_chaos_experiment(self, experiment_class: type) -> ChaosExperimentResult:
        """Run a chaos experiment with full safety checks."""
        
        # Pre-experiment safety validation
        if not self.safety_checks.validate_experiment_safety(experiment_class):
            raise UnsafeExperimentError("Experiment failed safety validation")
        
        experiment = experiment_class()
        
        # Establish steady state baseline
        baseline_metrics = self.monitoring.capture_baseline(duration=60)
        
        try:
            # Start experiment monitoring
            with self.monitoring.experiment_context(experiment.name):
                # Inject chaos
                with experiment.inject_chaos():
                    # Monitor system behavior
                    chaos_metrics = self.monitoring.capture_metrics(
                        duration=experiment.duration
                    )
                
                # Wait for recovery
                recovery_metrics = self.monitoring.wait_for_recovery(
                    baseline=baseline_metrics,
                    timeout=experiment.recovery_timeout
                )
            
            # Analyze results
            result = self.analyze_experiment_results(
                experiment=experiment,
                baseline=baseline_metrics,
                chaos=chaos_metrics,
                recovery=recovery_metrics
            )
            
            return result
            
        except Exception as e:
            # Emergency recovery procedures
            self.emergency_recovery()
            raise ChaosExperimentError(f"Experiment failed: {e}")
    
    def analyze_experiment_results(self, **kwargs) -> ChaosExperimentResult:
        """Analyze chaos experiment results."""
        experiment = kwargs['experiment']
        baseline = kwargs['baseline']
        chaos = kwargs['chaos']
        recovery = kwargs['recovery']
        
        # Validate hypothesis
        hypothesis_validated = self.validate_hypothesis(
            experiment.hypothesis(),
            chaos
        )
        
        # Check steady state maintenance
        steady_state_maintained = self.check_steady_state(
            experiment.steady_state(),
            chaos
        )
        
        # Calculate recovery time
        recovery_time = self.calculate_recovery_time(
            baseline, recovery
        )
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            experiment, chaos, recovery
        )
        
        return ChaosExperimentResult(
            experiment_name=experiment.name,
            hypothesis_validated=hypothesis_validated,
            steady_state_maintained=steady_state_maintained,
            recovery_time_seconds=recovery_time,
            observations=self.extract_observations(chaos),
            recommendations=recommendations
        )
```

### Automated Chaos Schedules

Schedule regular chaos experiments:

```python
from causal_eval.chaos.scheduler import ChaosScheduler
from datetime import datetime, timedelta

class AutomatedChaosScheduler(ChaosScheduler):
    def __init__(self):
        super().__init__()
        self.setup_experiment_schedule()
    
    def setup_experiment_schedule(self):
        """Setup automated chaos experiment schedule."""
        
        # Daily experiments (low impact)
        self.schedule_daily([
            DatabaseConnectionPoolExhaustion,
            APIRateLimitingTest,
            CacheEvictionTest
        ], time="02:00")
        
        # Weekly experiments (medium impact)
        self.schedule_weekly([
            DatabaseFailoverTest,
            ServiceRestartTest,
            NetworkLatencyTest
        ], day="sunday", time="01:00")
        
        # Monthly experiments (high impact)
        self.schedule_monthly([
            FullDataCenterFailure,
            MajorDatabaseMigration,
            CompleteServiceRestart
        ], day=1, time="03:00")
    
    def run_scheduled_experiments(self):
        """Execute scheduled chaos experiments."""
        due_experiments = self.get_due_experiments()
        
        for experiment in due_experiments:
            try:
                # Run experiment with safety checks
                result = self.chaos_framework.run_chaos_experiment(experiment)
                
                # Store results
                self.store_experiment_result(result)
                
                # Generate alerts if needed
                if not result.hypothesis_validated:
                    self.alert_engineering_team(result)
                
                # Update system resilience metrics
                self.update_resilience_metrics(result)
                
            except Exception as e:
                self.handle_experiment_failure(experiment, e)
```

## 📊 Chaos Metrics and Monitoring

### Resilience Metrics

Track system resilience over time:

```python
from causal_eval.chaos.metrics import ResilienceMetrics

class ChaosMetricsCollector:
    def __init__(self):
        self.metrics = ResilienceMetrics()
    
    def track_experiment_metrics(self, experiment_result: ChaosExperimentResult):
        """Track metrics from chaos experiments."""
        
        # Update resilience score
        self.metrics.update_resilience_score(
            experiment_name=experiment_result.experiment_name,
            success=experiment_result.hypothesis_validated,
            recovery_time=experiment_result.recovery_time_seconds
        )
        
        # Track failure modes
        self.metrics.track_failure_modes(
            experiment_result.observations.get("failure_modes", [])
        )
        
        # Update MTTR (Mean Time To Recovery)
        self.metrics.update_mttr(
            component=experiment_result.experiment_name.split("_")[0],
            recovery_time=experiment_result.recovery_time_seconds
        )
        
        # Track system adaptations
        self.metrics.track_adaptations(
            experiment_result.recommendations
        )
    
    def generate_resilience_report(self) -> Dict[str, Any]:
        """Generate comprehensive resilience report."""
        return {
            "overall_resilience_score": self.metrics.overall_score(),
            "mttr_by_component": self.metrics.mttr_by_component(),
            "failure_mode_frequency": self.metrics.failure_mode_stats(),
            "improvement_trends": self.metrics.improvement_trends(),
            "recommendations": self.metrics.get_recommendations()
        }
```

### Chaos Dashboards

Create monitoring dashboards for chaos experiments:

```python
from causal_eval.chaos.dashboard import ChaosDashboard

class ChaosEngineeringDashboard(ChaosDashboard):
    def __init__(self):
        super().__init__()
        self.setup_dashboard_panels()
    
    def setup_dashboard_panels(self):
        """Setup chaos engineering dashboard panels."""
        
        # Experiment execution panel
        self.add_panel(
            title="Chaos Experiment Status",
            type="stat",
            metrics=[
                "chaos_experiments_total",
                "chaos_experiments_successful", 
                "chaos_experiments_failed",
                "avg_recovery_time"
            ]
        )
        
        # System resilience panel
        self.add_panel(
            title="System Resilience Score",
            type="gauge",
            metrics=["resilience_score"],
            thresholds=[
                {"value": 0.7, "color": "red"},
                {"value": 0.85, "color": "yellow"},
                {"value": 0.95, "color": "green"}
            ]
        )
        
        # Failure mode analysis panel
        self.add_panel(
            title="Common Failure Modes",
            type="bar_chart",
            metrics=["failure_mode_frequency"],
            time_range="30d"
        )
        
        # Recovery time trends panel
        self.add_panel(
            title="Recovery Time Trends",
            type="time_series",
            metrics=["mttr_by_component"],
            time_range="90d"
        )
```

## 🎮 Game Days and Disaster Recovery

### Disaster Recovery Testing

Implement comprehensive disaster recovery testing:

```python
from causal_eval.chaos.gameday import GameDayScenario

class EvaluationSystemGameDay(GameDayScenario):
    def __init__(self):
        super().__init__(name="evaluation_system_dr_test")
        self.scenario_duration = timedelta(hours=4)
    
    def setup_disaster_scenario(self):
        """Setup a comprehensive disaster scenario."""
        
        # Simulate major cloud provider outage
        self.add_failure_injection(
            name="primary_region_failure",
            description="Simulate complete primary region outage",
            actions=[
                self.disable_region("us-east-1"),
                self.redirect_traffic_to_secondary("us-west-2"),
                self.failover_database("primary", "secondary")
            ]
        )
        
        # Add additional complications
        self.add_failure_injection(
            name="partial_secondary_degradation",
            description="Secondary region running at reduced capacity",
            actions=[
                self.reduce_capacity("us-west-2", percentage=60),
                self.inject_intermittent_failures(rate=0.05)
            ]
        )
    
    def define_success_criteria(self):
        """Define what constitutes successful disaster recovery."""
        return {
            "rto_target": timedelta(minutes=15),  # Recovery Time Objective
            "rpo_target": timedelta(minutes=5),   # Recovery Point Objective
            "minimum_functionality": 0.8,         # 80% of normal capacity
            "data_loss_tolerance": 0.001          # 0.1% data loss maximum
        }
    
    def run_game_day(self):
        """Execute the game day scenario."""
        
        # Pre-game day preparation
        self.validate_runbooks()
        self.brief_response_team()
        self.setup_monitoring()
        
        # Execute disaster scenario
        with self.disaster_context():
            # Inject failures according to scenario
            self.execute_failure_injections()
            
            # Monitor response team actions
            response_actions = self.monitor_response_team()
            
            # Track recovery metrics
            recovery_metrics = self.track_recovery_progress()
            
            # Test communication procedures
            communication_test = self.test_communication_channels()
        
        # Post-game day analysis
        results = self.analyze_game_day_results(
            response_actions, recovery_metrics, communication_test
        )
        
        return results
    
    def generate_after_action_report(self, results):
        """Generate comprehensive after-action report."""
        return {
            "scenario_summary": self.get_scenario_summary(),
            "success_criteria_met": results.success_criteria_met,
            "timeline_analysis": results.timeline,
            "team_performance": results.team_metrics,
            "lessons_learned": results.lessons_learned,
            "improvement_actions": results.action_items,
            "runbook_updates": results.runbook_changes
        }
```

### Chaos Engineering Best Practices

Follow these best practices for safe and effective chaos engineering:

```python
from causal_eval.chaos.best_practices import ChaosSafetyGuard

class ChaosEngineeringSafetyGuard:
    def __init__(self):
        self.safety_rules = self.load_safety_rules()
        self.blast_radius_limits = self.load_blast_radius_limits()
    
    def validate_experiment_safety(self, experiment):
        """Validate experiment safety before execution."""
        
        safety_checks = [
            self.check_blast_radius(experiment),
            self.check_safety_mechanisms(experiment),
            self.check_rollback_procedures(experiment),
            self.check_monitoring_coverage(experiment),
            self.check_team_availability(experiment)
        ]
        
        failed_checks = [check for check in safety_checks if not check.passed]
        
        if failed_checks:
            raise UnsafeExperimentError(
                f"Safety validation failed: {[c.reason for c in failed_checks]}"
            )
        
        return True
    
    def load_safety_rules(self):
        """Load chaos engineering safety rules."""
        return [
            "Never run chaos experiments in production without approval",
            "Always have rollback procedures ready",
            "Monitor blast radius continuously",
            "Stop experiment if unexpected behavior occurs",
            "Have team available during experiment execution",
            "Document all experiment procedures",
            "Test in staging environment first",
            "Limit experiment duration",
            "Use circuit breakers and safety mechanisms"
        ]
    
    def emergency_stop_experiment(self, experiment_id: str):
        """Emergency stop for chaos experiments."""
        
        # Immediately stop chaos injection
        self.stop_chaos_injection(experiment_id)
        
        # Trigger rollback procedures
        self.trigger_rollback(experiment_id)
        
        # Alert engineering team
        self.alert_team("EMERGENCY_STOP", experiment_id)
        
        # Begin recovery procedures
        self.start_recovery_procedures(experiment_id)
        
        # Document incident
        self.create_incident_report(experiment_id)
```

This chaos engineering framework provides comprehensive tools for testing the resilience of your Causal Eval Bench deployment. Regular chaos experiments help identify weaknesses before they cause production outages and build confidence in your system's ability to handle unexpected failures.