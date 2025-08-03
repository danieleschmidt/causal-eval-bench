# Getting Started with Causal Eval Bench

## Quick Start Guide

This guide will help you get up and running with Causal Eval Bench in just a few minutes.

## Installation

### Option 1: Using pip (Recommended)

```bash
pip install causal-eval-bench
```

### Option 2: From Source

```bash
git clone https://github.com/your-org/causal-eval-bench
cd causal-eval-bench
pip install -e ".[all]"
```

### Option 3: Using Docker

```bash
docker pull your-org/causal-eval:latest
docker run -it -p 8080:8080 your-org/causal-eval:latest
```

## Your First Evaluation

Let's run a simple causal reasoning evaluation:

```python
from causal_eval import CausalBenchmark, ModelEvaluator

# Initialize the benchmark
benchmark = CausalBenchmark()

# Create an evaluator for your model
evaluator = ModelEvaluator(
    model_name="gpt-4",
    api_key="your-api-key"
)

# Run a quick evaluation
results = benchmark.evaluate(
    evaluator,
    tasks=["causal_attribution"],
    num_examples=10
)

print(f"Causal Attribution Score: {results.causal_attribution:.2%}")
```

## Understanding the Results

Causal Eval Bench provides detailed analysis of model performance across different causal reasoning tasks:

- **Causal Attribution**: Ability to distinguish causation from correlation
- **Counterfactual Reasoning**: Understanding "what if" scenarios
- **Intervention Analysis**: Predicting effects of actions
- **Causal Chain Reasoning**: Following multi-step causal relationships

## Next Steps

1. **Explore different tasks**: Try running evaluations on all available tasks
2. **Customize your evaluation**: Learn about advanced configuration options
3. **Generate custom tests**: Create domain-specific evaluation scenarios
4. **Analyze results**: Use built-in analysis tools to understand model strengths and weaknesses

## Common Use Cases

### Academic Research
- Benchmark your causal reasoning models
- Compare different approaches to causal understanding
- Generate research insights about model capabilities

### Model Development
- Test causal reasoning during model training
- Identify weak areas in causal understanding
- Track improvement over training iterations

### Production Validation
- Validate causal reasoning capabilities before deployment
- Monitor causal understanding in production models
- Ensure robust causal reasoning for safety-critical applications

## Getting Help

- **Documentation**: [Full documentation](https://docs.causal-eval-bench.org)
- **Examples**: Check the `examples/` directory
- **Community**: Join our [Discord](https://discord.gg/your-org)
- **Issues**: Report bugs on [GitHub](https://github.com/your-org/causal-eval-bench/issues)