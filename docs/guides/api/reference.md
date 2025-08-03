# API Reference Guide

## Overview

Causal Eval Bench provides both a Python API and a REST API for running causal reasoning evaluations.

## Python API

### Core Classes

#### CausalBenchmark

The main entry point for running evaluations.

```python
from causal_eval import CausalBenchmark

benchmark = CausalBenchmark()
```

**Methods:**

- `evaluate(evaluator, tasks=None, **kwargs) -> EvaluationResults`
- `generate_report(results, format='json') -> str`
- `list_available_tasks() -> List[str]`

#### ModelEvaluator

Wrapper for different model APIs.

```python
from causal_eval import ModelEvaluator

# OpenAI GPT models
evaluator = ModelEvaluator(
    model_name="gpt-4",
    api_key="your-openai-key"
)

# Anthropic Claude models  
evaluator = ModelEvaluator(
    model_name="claude-3-opus",
    api_key="your-anthropic-key"
)

# Local Hugging Face models
evaluator = ModelEvaluator(
    model_name="microsoft/DialoGPT-medium",
    model_type="huggingface"
)
```

**Methods:**

- `evaluate_task(task) -> TaskResult`
- `generate_response(prompt) -> str`
- `get_model_info() -> Dict[str, Any]`

#### Task Classes

Base class for all causal reasoning tasks.

```python
from causal_eval.tasks import CausalAttribution

task = CausalAttribution()
question = task.generate_question()
score = task.evaluate_response(response, question.ground_truth)
```

**Available Tasks:**

- `CausalAttribution`: Causation vs correlation
- `CounterfactualReasoning`: "What if" scenarios  
- `CausalIntervention`: Effect of interventions
- `CausalChain`: Multi-step causal chains
- `ConfoundingAnalysis`: Identifying confounders

### Test Generation

#### CausalTestGenerator

Generate new causal reasoning tests.

```python
from causal_eval.generation import CausalTestGenerator

generator = CausalTestGenerator(
    domains=["medical", "economic"],
    difficulty="medium"
)

test_set = generator.generate_test_set(n_questions=100)
```

**Configuration Options:**

- `domains`: List of domains to generate from
- `difficulty`: "easy", "medium", "hard", or "expert"
- `avoid_contamination`: Ensure novel questions
- `balance_types`: Equal distribution across task types

### Analysis Tools

#### ErrorAnalyzer

Analyze failure patterns in evaluation results.

```python
from causal_eval.analysis import ErrorAnalyzer

analyzer = ErrorAnalyzer()
error_analysis = analyzer.analyze_errors(results)

# Get most common error types
top_errors = error_analysis.top_errors(n=5)
```

#### CausalProfiler

Create detailed capability profiles for models.

```python
from causal_eval.analysis import CausalProfiler

profiler = CausalProfiler()
profile = profiler.create_profile(model, results)

# Visualize strengths and weaknesses
profiler.plot_capability_matrix(profile)
```

## REST API

### Base URL

```
http://localhost:8080/api/v1
```

### Authentication

Include API key in request headers:

```
Authorization: Bearer your-api-key
```

### Endpoints

#### Start Evaluation

**POST** `/evaluations`

Start a new evaluation run.

```json
{
  "model_config": {
    "name": "gpt-4",
    "api_key": "your-key"
  },
  "evaluation_config": {
    "tasks": ["causal_attribution", "counterfactual"],
    "num_examples": 50,
    "domains": ["medical", "social"]
  }
}
```

**Response:**
```json
{
  "evaluation_id": "eval_123",
  "status": "running",
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### Get Evaluation Status

**GET** `/evaluations/{evaluation_id}`

```json
{
  "evaluation_id": "eval_123",
  "status": "completed",
  "progress": 100,
  "results": {
    "overall_score": 0.75,
    "task_scores": {
      "causal_attribution": 0.80,
      "counterfactual": 0.70
    }
  }
}
```

#### List Evaluations

**GET** `/evaluations`

Query parameters:
- `limit`: Number of results (default: 50)
- `offset`: Pagination offset (default: 0)
- `status`: Filter by status

#### Generate Test Set

**POST** `/test-generation`

```json
{
  "generator_config": {
    "domains": ["medical"],
    "difficulty": "medium",
    "num_questions": 100
  }
}
```

#### Get Available Tasks

**GET** `/tasks`

```json
{
  "tasks": [
    {
      "name": "causal_attribution",
      "description": "Test ability to distinguish causation from correlation",
      "domains": ["medical", "social", "economic"]
    }
  ]
}
```

#### Health Check

**GET** `/health`

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime": 86400
}
```

### Error Responses

All endpoints return structured error responses:

```json
{
  "error": {
    "code": "INVALID_MODEL",
    "message": "Unsupported model type",
    "details": {
      "supported_models": ["gpt-4", "claude-3"]
    }
  }
}
```

**Error Codes:**

- `INVALID_MODEL`: Unsupported model configuration
- `QUOTA_EXCEEDED`: API rate limit exceeded
- `EVALUATION_NOT_FOUND`: Evaluation ID not found
- `VALIDATION_ERROR`: Invalid request parameters

### Rate Limiting

- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1000 requests/hour  
- **Enterprise**: Custom limits

Rate limit headers included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

### Webhooks

Configure webhooks to receive evaluation completion notifications:

**POST** `/webhooks`

```json
{
  "url": "https://your-app.com/webhook",
  "events": ["evaluation.completed", "evaluation.failed"]
}
```

Webhook payload:
```json
{
  "event": "evaluation.completed",
  "evaluation_id": "eval_123",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "overall_score": 0.75
  }
}
```

## SDK Examples

### Python SDK

```python
import causal_eval

# Configure client
client = causal_eval.Client(api_key="your-key")

# Start evaluation
evaluation = client.evaluations.create(
    model="gpt-4",
    tasks=["causal_attribution"],
    num_examples=50
)

# Wait for completion
results = evaluation.wait_for_completion()
print(f"Score: {results.overall_score}")
```

### JavaScript SDK

```javascript
const CausalEval = require('causal-eval-js');

const client = new CausalEval.Client({
  apiKey: 'your-key'
});

// Start evaluation
const evaluation = await client.evaluations.create({
  model: 'gpt-4',
  tasks: ['causal_attribution'],
  numExamples: 50
});

// Get results
const results = await evaluation.waitForCompletion();
console.log(`Score: ${results.overallScore}`);
```

## Advanced Usage

### Custom Task Implementation

```python
from causal_eval.core import CausalTask, Question

class CustomTask(CausalTask):
    def generate_question(self) -> Question:
        return Question(
            prompt="Your custom prompt",
            ground_truth={"answer": "expected_response"},
            metadata={"domain": "custom"}
        )
    
    def evaluate_response(self, response: str, ground_truth: dict) -> float:
        # Custom evaluation logic
        return 1.0 if response.lower() == ground_truth["answer"] else 0.0

# Register custom task
from causal_eval import register_task
register_task("custom_task", CustomTask)
```

### Batch Evaluation

```python
# Evaluate multiple models in parallel
models = ["gpt-4", "claude-3", "llama-2"]
batch_results = benchmark.batch_evaluate(
    models=models,
    tasks=["causal_attribution"],
    parallel=True
)

# Compare results
comparison = benchmark.compare_models(batch_results)
comparison.plot_comparison()
```

### Streaming Results

```python
# Stream evaluation progress
for update in benchmark.evaluate_stream(evaluator, tasks):
    print(f"Progress: {update.progress}%")
    print(f"Current score: {update.current_score}")
```

## Best Practices

1. **Use appropriate sample sizes** for reliable results
2. **Cache test sets** to ensure reproducible evaluations  
3. **Monitor API rate limits** to avoid disruptions
4. **Validate model responses** before evaluation
5. **Use domain-specific test sets** for targeted evaluation

## Support

- **API Documentation**: Interactive docs at `/docs`
- **SDKs**: Available for Python, JavaScript, and R
- **Rate Limits**: Contact support for increased limits
- **Enterprise**: Custom deployment and SLA options