# Performance Optimization Guide

This guide covers advanced performance optimization techniques for Causal Eval Bench, tailored for high-throughput evaluation scenarios and production deployments.

## 📊 Performance Monitoring

### Built-in Metrics

Causal Eval Bench includes comprehensive performance monitoring:

```python
from causal_eval.monitoring import PerformanceProfiler

# Enable performance profiling
profiler = PerformanceProfiler(
    enable_cpu_profiling=True,
    enable_memory_profiling=True,
    enable_io_profiling=True,
    sampling_interval=0.1
)

# Profile evaluation performance
with profiler:
    results = benchmark.evaluate(model, tasks=["all"])

# Analyze performance bottlenecks
analysis = profiler.analyze()
print(f"CPU bottlenecks: {analysis.cpu_hotspots}")
print(f"Memory peaks: {analysis.memory_peaks}")
print(f"I/O wait times: {analysis.io_bottlenecks}")
```

### Custom Performance Metrics

Create domain-specific performance metrics:

```python
from causal_eval.metrics import CustomPerformanceMetric

class CausalReasoningLatency(CustomPerformanceMetric):
    def __init__(self):
        super().__init__(name="causal_reasoning_latency")
    
    def measure(self, task_type: str, complexity: str) -> float:
        start_time = time.perf_counter()
        # Your evaluation logic here
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        self.record_measurement(
            value=latency,
            tags={"task_type": task_type, "complexity": complexity}
        )
        return latency
```

## 🚀 Optimization Strategies

### 1. Evaluation Parallelization

Optimize evaluation throughput with smart parallelization:

```python
from causal_eval.optimization import ParallelEvaluator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Thread-based parallelization for I/O-bound tasks
thread_evaluator = ParallelEvaluator(
    executor=ThreadPoolExecutor(max_workers=8),
    batch_size=32,
    chunk_strategy="dynamic"
)

# Process-based parallelization for CPU-bound tasks
process_evaluator = ParallelEvaluator(
    executor=ProcessPoolExecutor(max_workers=4),
    batch_size=16,
    chunk_strategy="balanced"
)

# Adaptive parallelization
adaptive_evaluator = ParallelEvaluator.create_adaptive(
    cpu_intensive_threshold=0.7,
    io_intensive_threshold=0.3,
    memory_limit_gb=16
)

# Run optimized evaluation
results = adaptive_evaluator.evaluate_batch(
    models=["gpt-4", "claude-3", "llama-3"],
    tasks=["causal_attribution", "counterfactual"],
    optimize_for="throughput"  # or "latency", "memory"
)
```

### 2. Intelligent Caching

Implement multi-layer caching for maximum performance:

```python
from causal_eval.caching import SmartCache, CacheLayer

# Configure multi-layer cache
cache = SmartCache([
    CacheLayer.memory(max_size_mb=512),
    CacheLayer.redis(host="localhost", port=6379),
    CacheLayer.disk(path="/tmp/causal_eval_cache", max_size_gb=10)
])

# Cache evaluation results with smart invalidation
@cache.memoize(ttl=3600, invalidate_on=["model_update", "task_change"])
def evaluate_with_cache(model_name: str, task_config: dict):
    return benchmark.evaluate(model_name, **task_config)

# Preload common evaluations
cache.preload_patterns([
    {"model": "gpt-4", "task": "causal_attribution"},
    {"model": "claude-3", "task": "counterfactual"},
    {"model": "*", "task": "intervention"}  # wildcard patterns
])
```

### 3. Memory Optimization

Optimize memory usage for large-scale evaluations:

```python
from causal_eval.optimization import MemoryOptimizer

# Configure memory-efficient evaluation
optimizer = MemoryOptimizer(
    max_memory_gb=8,
    enable_gradient_checkpointing=True,
    use_memory_mapping=True,
    batch_size_strategy="adaptive"
)

# Memory-optimized evaluation
with optimizer:
    # Automatic batch size adjustment based on available memory
    results = benchmark.evaluate_large_dataset(
        dataset_size=100000,
        memory_efficient=True,
        streaming=True
    )
```

### 4. GPU Acceleration

Leverage GPU acceleration for compatible models:

```python
from causal_eval.gpu import GPUAccelerator

# Configure GPU acceleration
gpu_accel = GPUAccelerator(
    device="cuda:0",
    memory_fraction=0.8,
    enable_mixed_precision=True
)

# GPU-accelerated evaluation
with gpu_accel:
    results = benchmark.evaluate(
        model="local_transformer",
        accelerate=True,
        precision="fp16"  # Mixed precision for speed
    )
```

## 📈 Advanced Profiling

### CPU Profiling

Identify CPU bottlenecks in your evaluation pipeline:

```python
import cProfile
import pstats
from causal_eval.profiling import CausalEvalProfiler

# Custom profiler for causal reasoning tasks
profiler = CausalEvalProfiler()

# Profile specific evaluation components
@profiler.profile_function
def profile_causal_attribution():
    task = CausalAttribution()
    for i in range(100):
        question = task.generate_question()
        response = model.generate(question.prompt)
        score = task.evaluate_response(response, question.ground_truth)

# Generate detailed profiling report
report = profiler.generate_report(
    sort_by="cumulative",
    top_n=20,
    include_callgraph=True
)
```

### Memory Profiling

Track memory usage patterns:

```python
from causal_eval.profiling import MemoryProfiler
import tracemalloc

# Enable detailed memory tracking
memory_profiler = MemoryProfiler(
    snapshot_interval=1.0,
    track_lineno=True
)

with memory_profiler:
    # Your evaluation code here
    results = benchmark.evaluate(model, tasks=["all"])

# Analyze memory usage
analysis = memory_profiler.analyze()
print(f"Peak memory usage: {analysis.peak_memory_mb:.1f} MB")
print(f"Memory growth rate: {analysis.growth_rate_mb_per_sec:.2f} MB/s")

# Identify memory leaks
leaks = memory_profiler.detect_leaks()
for leak in leaks:
    print(f"Potential leak: {leak.location} ({leak.size_mb:.1f} MB)")
```

### I/O Profiling

Optimize I/O operations:

```python
from causal_eval.profiling import IOProfiler

io_profiler = IOProfiler()

with io_profiler:
    # Profile file I/O during evaluation
    test_set = benchmark.load_test_set("large_causal_dataset.json")
    results = benchmark.evaluate(model, test_set=test_set)

# Analyze I/O patterns
io_stats = io_profiler.get_stats()
print(f"Total I/O operations: {io_stats.total_operations}")
print(f"Average I/O latency: {io_stats.avg_latency_ms:.2f} ms")
print(f"I/O throughput: {io_stats.throughput_mb_per_sec:.1f} MB/s")
```

## 🎯 Performance Benchmarking

### Built-in Benchmarks

Run comprehensive performance benchmarks:

```python
from causal_eval.benchmarks import PerformanceBenchmark

# Standard performance benchmark
perf_benchmark = PerformanceBenchmark()

# Run full benchmark suite
benchmark_results = perf_benchmark.run_full_suite(
    models=["gpt-4", "claude-3"],
    include_stress_tests=True,
    include_memory_tests=True,
    include_scalability_tests=True
)

# Generate performance report
report = perf_benchmark.generate_report(
    benchmark_results,
    format="html",
    include_visualizations=True
)
```

### Custom Benchmarks

Create domain-specific performance benchmarks:

```python
from causal_eval.benchmarks import CustomBenchmark

class CausalReasoningBenchmark(CustomBenchmark):
    def __init__(self):
        super().__init__(name="causal_reasoning_perf")
    
    def run_latency_test(self, model, n_samples=1000):
        """Test single-question latency."""
        latencies = []
        for _ in range(n_samples):
            start = time.perf_counter()
            result = model.evaluate_single_question(self.generate_question())
            end = time.perf_counter()
            latencies.append(end - start)
        
        return {
            "mean_latency": np.mean(latencies),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99)
        }
    
    def run_throughput_test(self, model, duration_seconds=60):
        """Test sustained throughput."""
        start_time = time.time()
        questions_processed = 0
        
        while time.time() - start_time < duration_seconds:
            batch = self.generate_batch(size=10)
            model.evaluate_batch(batch)
            questions_processed += len(batch)
        
        actual_duration = time.time() - start_time
        return {
            "throughput_qps": questions_processed / actual_duration,
            "total_questions": questions_processed
        }
```

## 🔧 Production Optimizations

### Database Optimization

Optimize database performance for evaluation storage:

```python
from causal_eval.database import OptimizedDatabase

# Configure high-performance database settings
db = OptimizedDatabase(
    connection_pool_size=20,
    max_overflow=30,
    pool_timeout=30,
    pool_recycle=3600,
    enable_query_optimization=True,
    enable_connection_pooling=True
)

# Optimize queries for evaluation results
@db.optimize_query
def store_evaluation_results(results):
    # Batch insert for better performance
    db.bulk_insert("evaluation_results", results, batch_size=1000)

# Use read replicas for analysis queries
@db.read_replica
def analyze_historical_performance():
    return db.query("""
        SELECT model_name, AVG(score), COUNT(*)
        FROM evaluation_results
        WHERE created_at >= NOW() - INTERVAL '30 days'
        GROUP BY model_name
    """)
```

### API Optimization

Optimize API performance for high-throughput scenarios:

```python
from fastapi import FastAPI
from causal_eval.api.optimization import OptimizedAPIRouter

app = FastAPI()

# Optimized router with built-in performance enhancements
router = OptimizedAPIRouter(
    enable_response_compression=True,
    enable_request_caching=True,
    rate_limit_per_minute=1000,
    enable_connection_pooling=True
)

@router.post("/evaluate/batch")
async def evaluate_batch_optimized(requests: List[EvaluationRequest]):
    # Automatic request batching and parallel processing
    return await router.process_batch(
        requests,
        batch_size=32,
        max_workers=8,
        timeout_seconds=300
    )
```

### Container Optimization

Optimize Docker containers for performance:

```dockerfile
# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main --no-dev

FROM python:3.11-slim as production

# Copy optimized Python environment
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Performance optimizations
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONHASHSEED=random
ENV PYTHONIOENCODING=utf-8

# Optimize Python startup
ENV PYTHONINSPECT=0
ENV PYTHONOPTIMIZE=2

# Memory optimizations
ENV MALLOC_ARENA_MAX=2
ENV MALLOC_MMAP_THRESHOLD_=131072
ENV MALLOC_TRIM_THRESHOLD_=131072
ENV MALLOC_TOP_PAD_=131072
ENV MALLOC_MMAP_MAX_=65536

# Application code
COPY . /app
WORKDIR /app

# Use high-performance WSGI server
CMD ["uvicorn", "causal_eval.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker"]
```

## 📊 Monitoring and Alerting

### Performance Monitoring Dashboard

Set up comprehensive performance monitoring:

```python
from causal_eval.monitoring import PerformanceDashboard
from prometheus_client import Counter, Histogram, Gauge

# Define custom metrics
evaluation_duration = Histogram(
    'causal_eval_duration_seconds',
    'Time spent on evaluation',
    ['model_name', 'task_type']
)

evaluation_errors = Counter(
    'causal_eval_errors_total',
    'Total evaluation errors',
    ['model_name', 'error_type']
)

memory_usage = Gauge(
    'causal_eval_memory_usage_bytes',
    'Current memory usage'
)

# Create monitoring dashboard
dashboard = PerformanceDashboard(
    metrics=[evaluation_duration, evaluation_errors, memory_usage],
    refresh_interval=5,
    alert_thresholds={
        'error_rate': 0.05,
        'latency_p95': 10.0,
        'memory_usage_gb': 8.0
    }
)

# Start monitoring
dashboard.start()
```

### Automated Performance Alerts

Configure intelligent alerting:

```python
from causal_eval.alerting import PerformanceAlerting

alerting = PerformanceAlerting(
    channels=["slack", "email", "pagerduty"],
    escalation_policy="follow-the-sun"
)

# Define alert conditions
alerting.add_alert(
    name="high_evaluation_latency",
    condition="p95_latency > 5.0",
    severity="warning",
    cooldown_minutes=10
)

alerting.add_alert(
    name="evaluation_error_spike",
    condition="error_rate > 0.1",
    severity="critical",
    cooldown_minutes=5
)

alerting.add_alert(
    name="memory_exhaustion",
    condition="memory_usage_percent > 90",
    severity="critical",
    cooldown_minutes=2
)
```

## 🎛️ Configuration Tuning

### Environment-Specific Optimization

Optimize for different deployment environments:

```yaml
# config/performance.yaml
production:
  evaluation:
    batch_size: 64
    max_workers: 16
    cache_ttl: 3600
    enable_gpu: true
  
  api:
    worker_count: 8
    worker_class: "uvicorn.workers.UvicornWorker"
    max_requests: 10000
    max_requests_jitter: 1000
  
  database:
    pool_size: 20
    max_overflow: 30
    statement_timeout: 300

development:
  evaluation:
    batch_size: 8
    max_workers: 2
    cache_ttl: 60
    enable_gpu: false
  
  api:
    worker_count: 1
    reload: true
    debug: true
  
  database:
    pool_size: 5
    max_overflow: 10
    statement_timeout: 30
```

### Dynamic Configuration

Implement runtime configuration tuning:

```python
from causal_eval.config import DynamicConfig

config = DynamicConfig()

# Auto-tune based on system resources
config.auto_tune(
    target_cpu_utilization=0.8,
    target_memory_utilization=0.7,
    target_latency_p95=2.0
)

# Monitor and adjust in real-time
@config.adaptive_tuning(interval_seconds=60)
def tune_batch_size():
    current_latency = get_current_p95_latency()
    current_throughput = get_current_throughput()
    
    if current_latency > 5.0:
        config.batch_size = max(1, config.batch_size // 2)
    elif current_latency < 1.0 and current_throughput < target_throughput:
        config.batch_size = min(128, config.batch_size * 2)
```

## 📝 Performance Testing

### Load Testing

Implement comprehensive load testing:

```python
from locust import HttpUser, task, between
from causal_eval.testing import LoadTest

class CausalEvalLoadTest(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        # Authentication and setup
        self.client.post("/auth/login", json={
            "username": "test_user",
            "password": "test_password"
        })
    
    @task(3)
    def evaluate_single_question(self):
        """Test single question evaluation."""
        self.client.post("/evaluate/single", json={
            "model": "gpt-4",
            "question": "Does smoking cause lung cancer?",
            "task_type": "causal_attribution"
        })
    
    @task(1)
    def evaluate_batch(self):
        """Test batch evaluation."""
        self.client.post("/evaluate/batch", json={
            "model": "gpt-4",
            "questions": [
                {"question": "Question 1", "task_type": "causal_attribution"},
                {"question": "Question 2", "task_type": "counterfactual"},
                {"question": "Question 3", "task_type": "intervention"}
            ]
        })
    
    @task(1)
    def get_results(self):
        """Test results retrieval."""
        self.client.get("/results/recent?limit=10")

# Run load test
load_test = LoadTest(
    users=100,
    spawn_rate=10,
    duration="10m",
    host="http://localhost:8000"
)

results = load_test.run()
```

### Regression Testing

Automate performance regression detection:

```python
from causal_eval.testing import PerformanceRegressionTest

regression_test = PerformanceRegressionTest(
    baseline_branch="main",
    threshold_percent=10  # Alert if performance degrades by >10%
)

# Run performance comparison
comparison = regression_test.compare_performance(
    test_suite="standard_benchmark",
    current_branch="feature/optimization"
)

if comparison.has_regression:
    print(f"Performance regression detected:")
    for metric, degradation in comparison.regressions.items():
        print(f"  {metric}: {degradation:.1f}% slower")
```

This performance optimization guide provides comprehensive strategies for maximizing the performance of Causal Eval Bench in production environments. Regular monitoring and profiling will help you identify bottlenecks and optimize your specific use cases.