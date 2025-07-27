# Contributing to Causal Eval Bench

We welcome contributions from researchers, developers, and the broader AI community! This guide will help you get started with contributing to Causal Eval Bench.

## 🌟 Ways to Contribute

### 1. Code Contributions
- **New evaluation tasks**: Implement novel causal reasoning tasks
- **Bug fixes**: Fix issues and improve stability
- **Performance improvements**: Optimize evaluation speed and accuracy
- **New features**: Add functionality requested by the community

### 2. Research Contributions
- **Domain expertise**: Add domain-specific test sets
- **Evaluation methods**: Propose improved evaluation metrics
- **Bias analysis**: Identify and address evaluation biases
- **Validation studies**: Verify benchmark quality and relevance

### 3. Documentation
- **User guides**: Improve documentation and tutorials
- **API documentation**: Enhance code documentation
- **Examples**: Create practical examples and case studies
- **Translations**: Help translate documentation

### 4. Community Support
- **Issue triage**: Help categorize and prioritize issues
- **Code reviews**: Review pull requests from other contributors
- **Testing**: Test new features and report bugs
- **Discussions**: Participate in design discussions

## 🚀 Getting Started

### 1. Development Setup

```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/causal-eval-bench.git
cd causal-eval-bench

# Set up the development environment
make dev

# Run tests to ensure everything works
make test
```

### 2. Understanding the Codebase

```
causal-eval-bench/
├── causal_eval/           # Main package
│   ├── tasks/             # Evaluation tasks
│   ├── evaluation/        # Evaluation engine
│   ├── generation/        # Test generation
│   ├── analysis/          # Analysis tools
│   ├── api/              # REST API
│   ├── cli/              # Command line interface
│   └── models/           # Data models
├── tests/                # Test suite
├── docs/                 # Documentation
└── scripts/              # Utility scripts
```

### 3. Development Workflow

1. **Create a feature branch**: `git checkout -b feature/your-feature-name`
2. **Make your changes**: Follow our coding standards
3. **Write tests**: Ensure your changes are well tested
4. **Run the test suite**: `make test`
5. **Update documentation**: If you change APIs or add features
6. **Commit your changes**: Use conventional commit messages
7. **Push to your fork**: `git push origin feature/your-feature-name`
8. **Create a pull request**: Describe your changes clearly

## 📝 Coding Standards

### Code Style
- **Python**: Follow PEP 8, use Black for formatting
- **Line length**: 88 characters maximum
- **Imports**: Use isort for import organization
- **Type hints**: Use type hints for all function signatures
- **Docstrings**: Use Google-style docstrings

### Example Code Style

```python
from typing import List, Optional

def evaluate_causal_reasoning(
    questions: List[str],
    model_name: str,
    timeout: Optional[int] = None
) -> float:
    """Evaluate causal reasoning performance on a set of questions.
    
    Args:
        questions: List of causal reasoning questions to evaluate.
        model_name: Name of the model to evaluate.
        timeout: Optional timeout in seconds for each question.
        
    Returns:
        Overall evaluation score between 0 and 1.
        
    Raises:
        ValueError: If questions list is empty.
        ModelError: If model evaluation fails.
    """
    if not questions:
        raise ValueError("Questions list cannot be empty")
    
    # Implementation here
    return 0.85
```

### Testing Standards
- **Coverage**: Aim for >80% test coverage
- **Test types**: Write unit, integration, and end-to-end tests
- **Test naming**: Use descriptive test names
- **Fixtures**: Use pytest fixtures for common test data
- **Mocking**: Mock external services and APIs

### Documentation Standards
- **API docs**: Document all public functions and classes
- **User guides**: Provide clear examples and use cases
- **Code comments**: Explain complex logic and algorithms
- **Changelog**: Update CHANGELOG.md for user-facing changes

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration
make test-e2e

# Run with coverage
make coverage

# Run performance tests
make test-performance
```

### Writing Tests

```python
import pytest
from causal_eval.tasks import CausalAttribution

class TestCausalAttribution:
    """Test cases for causal attribution task."""
    
    def test_question_generation(self):
        """Test that questions are generated correctly."""
        task = CausalAttribution()
        question = task.generate_question()
        
        assert "prompt" in question
        assert "ground_truth" in question
        assert question["task_type"] == "causal_attribution"
    
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_difficulty_levels(self, difficulty):
        """Test question generation for different difficulty levels."""
        task = CausalAttribution()
        question = task.generate_question(difficulty=difficulty)
        
        assert question["difficulty"] == difficulty
```

## 📋 Pull Request Process

### Before Submitting
1. **Rebase your branch**: `git rebase main`
2. **Run all checks**: `make lint test`
3. **Update documentation**: If you changed APIs
4. **Write clear commit messages**: Use conventional commits

### Pull Request Template
When creating a pull request, include:

- **Description**: Clear description of changes
- **Motivation**: Why is this change needed?
- **Testing**: How did you test your changes?
- **Breaking changes**: Any backwards incompatible changes?
- **Checklist**: Use our PR checklist

### Review Process
1. **Automated checks**: CI/CD pipeline runs automatically
2. **Code review**: At least one maintainer reviews your code
3. **Testing**: We may test your changes manually
4. **Feedback**: Address any feedback from reviewers
5. **Merge**: Once approved, we'll merge your changes

## 🎯 Issue Guidelines

### Reporting Bugs
Use our bug report template and include:
- **Environment**: OS, Python version, package versions
- **Steps to reproduce**: Clear reproduction steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Logs**: Any relevant error messages or logs

### Feature Requests
Use our feature request template and include:
- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives**: Other solutions you considered
- **Implementation**: Willing to implement it yourself?

### Research Proposals
For research-related contributions:
- **Problem statement**: What research question are you addressing?
- **Methodology**: How do you plan to approach it?
- **Validation**: How will you validate your approach?
- **Impact**: How will this benefit the community?

## 🏗️ Adding New Evaluation Tasks

### Task Structure
New evaluation tasks should inherit from `CausalTask`:

```python
from causal_eval.tasks.base import CausalTask
from causal_eval.models import Question, TaskResult

class YourNewTask(CausalTask):
    """Your new causal reasoning task."""
    
    def generate_question(self, **kwargs) -> Question:
        """Generate a test question."""
        # Implementation here
        pass
    
    def evaluate_response(self, response: str, ground_truth: any) -> TaskResult:
        """Evaluate a model's response."""
        # Implementation here
        pass
    
    def get_task_metadata(self) -> dict:
        """Return task metadata."""
        return {
            "name": "your_new_task",
            "description": "Description of your task",
            "domains": ["general"],
            "difficulty_levels": ["easy", "medium", "hard"]
        }
```

### Task Registration
Register your task in `pyproject.toml`:

```toml
[tool.poetry.plugins."causal_eval.tasks"]
your_new_task = "causal_eval.tasks.your_module:YourNewTask"
```

## 🌍 Adding New Domains

### Domain Builder
Use the domain builder to create new domains:

```python
from causal_eval.domains import DomainBuilder

# Create your domain
your_domain = DomainBuilder.create_domain(
    name="your_domain",
    description="Description of your domain",
    example_phenomena=["phenomenon1", "phenomenon2"]
)

# Add domain-specific tests
your_domain.add_test_template(
    name="domain_specific_test",
    template="Test template with {variable}",
    ground_truth_generator=your_ground_truth_function
)
```

### Domain Guidelines
- **Expertise**: Ensure domain accuracy with expert review
- **Diversity**: Include diverse scenarios within the domain
- **Difficulty**: Provide questions at multiple difficulty levels
- **Validation**: Validate questions with domain experts

## 🤝 Community Guidelines

### Code of Conduct
We follow a code of conduct to ensure a welcoming environment:
- **Be respectful**: Treat everyone with respect and kindness
- **Be inclusive**: Welcome contributors from all backgrounds
- **Be constructive**: Provide helpful feedback and suggestions
- **Be patient**: Help newcomers learn and contribute

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General discussions and questions
- **Discord**: Real-time chat and community support
- **Email**: security@causal-eval-bench.org for security issues

### Recognition
We recognize contributors in several ways:
- **Contributors file**: Listed in CONTRIBUTORS.md
- **Release notes**: Mentioned in release announcements
- **Academic papers**: Co-authorship opportunities for significant contributions
- **Conference talks**: Speaking opportunities at conferences

## 📚 Resources

### Documentation
- [Architecture Overview](docs/development/architecture.md)
- [API Reference](docs/api/)
- [User Guide](docs/user-guide/)
- [Research Methodology](docs/research/methodology.md)

### Learning Resources
- [Causal Inference Primer](https://example.com/causal-inference)
- [Evaluation Best Practices](https://example.com/evaluation-best-practices)
- [Python Development Guide](https://example.com/python-development)

### Tools and Setup
- **IDE**: VS Code with Python extension recommended
- **Git**: Use conventional commit messages
- **Testing**: pytest for all testing needs
- **Documentation**: MkDocs for documentation

## ❓ Getting Help

If you need help with contributing:

1. **Read the docs**: Check our comprehensive documentation
2. **Search issues**: Look for similar questions or problems
3. **Ask in discussions**: Use GitHub Discussions for questions
4. **Join Discord**: Get real-time help from the community
5. **Email us**: For sensitive or complex questions

## 🙏 Thank You

Thank you for contributing to Causal Eval Bench! Your contributions help advance the field of causal reasoning in AI and benefit researchers and practitioners worldwide.

---

**Questions?** Reach out to us at contribute@causal-eval-bench.org or join our [Discord community](https://discord.gg/causal-eval).