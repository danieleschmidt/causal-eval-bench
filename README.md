# causal-eval-bench

[![Build Status](https://img.shields.io/github/actions/workflow/status/your-org/causal-eval-bench/ci.yml?branch=main)](https://github.com/your-org/causal-eval-bench/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-NAACL%202025-red.svg)](https://aclanthology.org/2025.naacl-long.622.pdf)

Comprehensive evaluation framework for testing genuine causal reasoning in language models. Goes beyond correlation to test understanding of cause-and-effect, counterfactuals, and causal interventions.

## 🎯 Key Features

- **Causal Reasoning Tests**: Evaluate understanding of causation vs correlation
- **Counterfactual Generation**: Test ability to reason about alternative scenarios
- **Intervention Analysis**: Assess understanding of causal interventions
- **Domain Coverage**: 15+ domains from medicine to economics
- **Automatic Test Generation**: Create novel causal reasoning problems
- **Interactive Leaderboard**: Track model performance with live updates

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Evaluation Tasks](#evaluation-tasks)
- [Test Generation](#test-generation)
- [Running Evaluations](#running-evaluations)
- [Analysis Tools](#analysis-tools)
- [Leaderboard](#leaderboard)
- [Custom Domains](#custom-domains)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🚀 Installation

### From PyPI

```bash
pip install causal-eval-bench
```

### From Source

```bash
git clone https://github.com/your-org/causal-eval-bench
cd causal-eval-bench
pip install -e ".[all]"
```

### Docker Installation

```bash
docker pull your-org/causal-eval:latest
docker run -it -p 8080:8080 your-org/causal-eval:latest
```

## ⚡ Quick Start

### Basic Evaluation

```python
from causal_eval import CausalBenchmark, ModelEvaluator

# Initialize benchmark
benchmark = CausalBenchmark()

# Evaluate a model
evaluator = ModelEvaluator(
    model_name="gpt-4",
    api_key="your-api-key"
)

# Run evaluation
results = benchmark.evaluate(
    evaluator,
    tasks=["causal_attribution", "counterfactual", "intervention"],
    num_examples=100
)

print(f"Overall Score: {results.overall_score:.2%}")
print(f"Causal Attribution: {results.causal_attribution:.2%}")
print(f"Counterfactual Reasoning: {results.counterfactual:.2%}")
```

### Quick Test

```python
from causal_eval import quick_test

# Run a quick causal reasoning test
score = quick_test(
    model="claude-3",
    test_type="causal_vs_correlation",
    n_questions=20
)

print(f"Model correctly identified causation in {score:.1%} of cases")
```

## 🧪 Evaluation Tasks

### 1. Causal Attribution

Test ability to identify true causes vs. mere correlations.

```python
from causal_eval.tasks import CausalAttribution

task = CausalAttribution()

# Example question
question = task.generate_question()
print(question.prompt)
# "Ice cream sales and swimming pool accidents both increase in summer. 
#  Does eating ice cream cause swimming accidents?"

# Evaluate response
response = model.generate(question.prompt)
score = task.evaluate_response(response, question.ground_truth)
```

### 2. Counterfactual Reasoning

Assess understanding of "what if" scenarios.

```python
from causal_eval.tasks import CounterfactualReasoning

task = CounterfactualReasoning()

# Generate counterfactual scenario
scenario = task.create_scenario(
    context="A student studied hard and passed the exam",
    intervention="What if the student had not studied?"
)

# Test model's counterfactual reasoning
response = model.reason_counterfactual(scenario)
evaluation = task.evaluate_counterfactual(response)

print(f"Logical consistency: {evaluation.consistency}")
print(f"Causal understanding: {evaluation.causal_score}")
```

### 3. Causal Intervention

Test understanding of the effects of interventions.

```python
from causal_eval.tasks import CausalIntervention

task = CausalIntervention()

# Create intervention scenario
scenario = {
    "system": "A thermostat controls room temperature",
    "intervention": "Manually set thermostat to 30°C",
    "question": "What happens to room temperature?"
}

# Evaluate intervention understanding
response = model.predict_intervention(scenario)
score = task.evaluate_intervention_prediction(response)
```

### 4. Causal Chain Reasoning

Evaluate understanding of multi-step causal chains.

```python
from causal_eval.tasks import CausalChain

task = CausalChain()

# Create causal chain
chain = task.create_chain(
    steps=[
        "Rain falls",
        "Ground becomes wet",
        "Plants absorb water",
        "Plants grow"
    ]
)

# Test chain reasoning
questions = task.generate_chain_questions(chain)
scores = []
for q in questions:
    response = model.answer(q)
    scores.append(task.evaluate_chain_reasoning(response, q))
```

### 5. Confounding Variables

Test ability to identify and reason about confounders.

```python
from causal_eval.tasks import ConfoundingAnalysis

task = ConfoundingAnalysis()

# Present scenario with potential confounder
scenario = task.create_confounded_scenario(
    observed="Coffee drinkers have higher productivity",
    potential_confounder="Sleep quality",
    domain="workplace"
)

# Test confounder identification
response = model.analyze_confounding(scenario)
evaluation = task.evaluate_confounding_analysis(response)
```

## 🔧 Test Generation

### Automatic Test Creation

```python
from causal_eval.generation import CausalTestGenerator

generator = CausalTestGenerator(
    domains=["medical", "economic", "social"],
    difficulty="medium",
    avoid_contamination=True  # Ensure novel questions
)

# Generate diverse test set
test_set = generator.generate_test_set(
    n_questions=500,
    balance_types=True
)

# Validate test quality
validation = generator.validate_test_set(test_set)
print(f"Question quality: {validation.avg_quality:.2f}/5")
print(f"Difficulty variance: {validation.difficulty_std:.2f}")
```

### Domain-Specific Generation

```python
from causal_eval.generation import DomainSpecificGenerator

# Medical domain tests
medical_gen = DomainSpecificGenerator(domain="medical")
medical_tests = medical_gen.generate(
    n_questions=100,
    include_real_studies=True,
    complexity="expert"
)

# Economic domain tests
econ_gen = DomainSpecificGenerator(domain="economics")
econ_tests = econ_gen.generate(
    focus_areas=["market_dynamics", "policy_effects"],
    time_period="2020-2024"
)
```

### Adversarial Test Generation

```python
from causal_eval.generation import AdversarialGenerator

# Generate tricky causal questions
adv_gen = AdversarialGenerator()

adversarial_tests = adv_gen.generate(
    base_model="gpt-4",
    exploit_weaknesses=True,
    categories=[
        "reverse_causation",
        "spurious_correlation",
        "hidden_confounders"
    ]
)
```

## 📊 Running Evaluations

### Comprehensive Evaluation

```python
from causal_eval import ComprehensiveBenchmark

benchmark = ComprehensiveBenchmark()

# Configure evaluation
config = {
    "tasks": "all",  # or specific list
    "domains": ["medical", "social", "economic", "scientific"],
    "difficulty_levels": ["easy", "medium", "hard"],
    "include_adversarial": True,
    "n_questions_per_category": 50
}

# Run full evaluation
results = benchmark.run_evaluation(
    model=your_model,
    config=config,
    save_responses=True
)

# Generate detailed report
benchmark.generate_report(
    results,
    output_format="pdf",
    include_error_analysis=True
)
```

### Comparative Evaluation

```python
from causal_eval import ModelComparison

comparison = ModelComparison()

# Compare multiple models
models = ["gpt-4", "claude-3", "llama-3", "gemini-pro"]
comparative_results = comparison.compare_models(
    models=models,
    benchmark="causal_eval_v1",
    parallel=True
)

# Visualize comparison
comparison.plot_radar_chart(
    comparative_results,
    dimensions=["attribution", "counterfactual", "intervention"],
    save_to="model_comparison.png"
)
```

### Longitudinal Evaluation

```python
from causal_eval import LongitudinalEvaluator

# Track performance over time
evaluator = LongitudinalEvaluator(
    model="your-model",
    checkpoint_frequency="weekly"
)

# Run periodic evaluations
evaluator.start_tracking(
    duration_days=30,
    test_set="causal_eval_stable_v1"
)

# Analyze trends
trends = evaluator.analyze_trends()
print(f"Performance trend: {trends.direction}")
print(f"Improvement rate: {trends.improvement_rate:.2%}/week")
```

## 📈 Analysis Tools

### Error Analysis

```python
from causal_eval.analysis import ErrorAnalyzer

analyzer = ErrorAnalyzer()

# Analyze failure patterns
error_analysis = analyzer.analyze_errors(
    results=evaluation_results,
    group_by=["task_type", "domain", "difficulty"]
)

# Most common error types
for error_type in error_analysis.top_errors(n=5):
    print(f"{error_type.name}: {error_type.frequency:.1%}")
    print(f"Example: {error_type.example}")
```

### Causal Understanding Profile

```python
from causal_eval.analysis import CausalProfiler

profiler = CausalProfiler()

# Create detailed profile
profile = profiler.create_profile(
    model=model,
    test_results=results
)

# Visualize strengths and weaknesses
profiler.plot_capability_matrix(
    profile,
    save_to="causal_profile.png"
)

# Get recommendations
recommendations = profiler.get_improvement_recommendations(profile)
```

### Statistical Analysis

```python
from causal_eval.analysis import StatisticalAnalyzer

stats = StatisticalAnalyzer()

# Perform statistical tests
significance = stats.test_significance(
    model_a_results=results_a,
    model_b_results=results_b,
    test="mcnemar",
    correction="bonferroni"
)

print(f"P-value: {significance.p_value:.4f}")
print(f"Effect size: {significance.effect_size:.3f}")
```

## 🏆 Leaderboard

### Submit Results

```python
from causal_eval.leaderboard import LeaderboardClient

client = LeaderboardClient(api_key="your-api-key")

# Submit evaluation results
submission = client.submit_results(
    model_name="YourModel-v1",
    results=evaluation_results,
    metadata={
        "model_size": "7B",
        "training_data": "custom",
        "timestamp": "2024-01-15"
    }
)

print(f"Submission ID: {submission.id}")
print(f"Rank: {submission.rank}")
```

### Generate Leaderboard Badge

```python
# Generate badge for README
badge_url = client.get_badge_url(
    model_name="YourModel-v1",
    metric="overall_score"
)

print(f"Add to README: ![CausalEval Score]({badge_url})")
```

### Access Leaderboard Data

```python
# Get current leaderboard
leaderboard = client.get_leaderboard(
    metric="causal_attribution",
    top_k=10
)

for rank, entry in enumerate(leaderboard, 1):
    print(f"{rank}. {entry.model}: {entry.score:.3f}")
```

## 🎯 Custom Domains

### Adding New Domains

```python
from causal_eval.domains import DomainBuilder

# Create custom domain
climate_domain = DomainBuilder.create_domain(
    name="climate_science",
    description="Climate and weather causation",
    example_phenomena=[
        "greenhouse_effect",
        "ocean_currents",
        "weather_patterns"
    ]
)

# Add domain-specific tests
climate_domain.add_test_template(
    name="climate_feedback_loops",
    template="""
    Given: {climate_condition}
    Intervention: {human_action}
    Question: What are the causal effects on {target_variable}?
    """,
    ground_truth_generator=climate_causal_model
)

# Register domain
benchmark.register_domain(climate_domain)
```

### Domain-Specific Metrics

```python
from causal_eval.metrics import DomainMetric

# Create specialized metric
class MedicalCausalMetric(DomainMetric):
    def __init__(self):
        super().__init__(domain="medical")
    
    def score(self, response, ground_truth):
        # Custom scoring for medical causation
        score = self.base_score(response, ground_truth)
        
        # Penalize dangerous causal misunderstandings
        if self.is_dangerous_inference(response):
            score *= 0.5
            
        return score
```

## 📚 API Reference

### Core Classes

```python
class CausalBenchmark:
    def evaluate(self, model, tasks, **kwargs) -> EvaluationResults
    def generate_report(self, results, format) -> None
    
class ModelEvaluator:
    def __init__(self, model_name, **credentials)
    def evaluate_task(self, task) -> TaskResult
    
class CausalTestGenerator:
    def generate_test_set(self, n_questions, **config) -> TestSet
    def validate_test_set(self, test_set) -> ValidationReport
```

### Task Interfaces

```python
class CausalTask(ABC):
    @abstractmethod
    def generate_question(self) -> Question
    
    @abstractmethod
    def evaluate_response(self, response, ground_truth) -> float
```

### Analysis Tools

```python
class ErrorAnalyzer:
    def analyze_errors(self, results) -> ErrorAnalysis
    def get_recommendations(self, analysis) -> List[str]
    
class CausalProfiler:
    def create_profile(self, model, results) -> CausalProfile
    def plot_capability_matrix(self, profile) -> None
```

## 🤝 Contributing

We welcome contributions! Priority areas:
- New causal reasoning tasks
- Domain-specific test sets
- Multilingual causal evaluation
- Improved evaluation metrics

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/causal-eval-bench
cd causal-eval-bench

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Generate new test cases
python scripts/generate_tests.py --domain medical --n 100
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [CausalML](https://github.com/uber/causalml) - Causal inference in ML
- [DoWhy](https://github.com/microsoft/dowhy) - Causal reasoning framework
- [CausalNLP](https://github.com/causal-nlp/causalnlp) - Causality in NLP
- [CLADDER](https://github.com/causal-ladder/cladder) - Causal reasoning dataset

## 📞 Support

- 📧 Email: causal-eval@your-org.com
- 💬 Discord: [Join our community](https://discord.gg/your-org)
- 📖 Documentation: [Full docs](https://docs.your-org.com/causal-eval)
- 🎓 Tutorial: [Causal Reasoning Guide](https://learn.your-org.com/causality)

## 📚 References

- [CausalEval: Better Causal Reasoning](https://aclanthology.org/2025.naacl-long.622.pdf) - Main paper
- [Unveiling Causal Reasoning in LLMs](https://arxiv.org/html/2506.21215) - Analysis
- [Causal Representation Learning](https://arxiv.org/abs/2102.11107) - Theory
- [Benchmarking Causal Understanding](https://arxiv.org/abs/2301.05015) - Related work
