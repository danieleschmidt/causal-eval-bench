# Developer Contributing Guide

## Development Environment Setup

### Prerequisites

- Python 3.9+ 
- Poetry (for dependency management)
- Git
- Docker & Docker Compose (optional, for containerized development)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/causal-eval-bench
   cd causal-eval-bench
   ```

2. **Install dependencies**
   ```bash
   poetry install --with dev,docs,test
   ```

3. **Activate the virtual environment**
   ```bash
   poetry shell
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Run initial tests**
   ```bash
   make test
   ```

## Development Workflow

### Code Quality Standards

We maintain high code quality through automated tools:

- **Black**: Code formatting
- **isort**: Import sorting  
- **Ruff**: Linting and code analysis
- **MyPy**: Type checking
- **Bandit**: Security scanning

Run all quality checks:
```bash
make lint
```

Auto-fix formatting issues:
```bash
make format
```

### Testing Strategy

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Benchmark critical paths

Run specific test suites:
```bash
make test-unit          # Unit tests only
make test-integration   # Integration tests only  
make test-e2e          # End-to-end tests only
make test              # All tests with coverage
```

### Adding New Features

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write tests first** (TDD approach recommended)
   ```bash
   # Add tests in appropriate test directory
   touch tests/unit/test_your_feature.py
   ```

3. **Implement the feature**
   - Follow existing code patterns
   - Add comprehensive docstrings
   - Include type hints

4. **Update documentation**
   - Update relevant guides
   - Add API documentation if needed
   - Update CHANGELOG.md

5. **Submit a pull request**
   - Use the PR template
   - Ensure all checks pass
   - Request review from maintainers

### Adding New Evaluation Tasks

Causal Eval Bench is designed to be extensible. To add a new causal reasoning task:

1. **Create the task class**
   ```python
   # causal_eval/tasks/your_task.py
   from causal_eval.core.tasks import CausalTask
   
   class YourTask(CausalTask):
       def generate_question(self) -> Question:
           # Implementation
           pass
           
       def evaluate_response(self, response: str, ground_truth: Any) -> float:
           # Implementation  
           pass
   ```

2. **Register the task plugin**
   ```toml
   # pyproject.toml
   [tool.poetry.plugins."causal_eval.tasks"]
   your_task = "causal_eval.tasks.your_task:YourTask"
   ```

3. **Add comprehensive tests**
   ```python
   # tests/unit/tasks/test_your_task.py
   from causal_eval.tasks.your_task import YourTask
   
   def test_your_task_generation():
       # Test question generation
       pass
       
   def test_your_task_evaluation():
       # Test response evaluation
       pass
   ```

4. **Update documentation**
   - Add task description to user guides
   - Include usage examples
   - Document evaluation criteria

### Code Style Guidelines

#### Python Conventions
- Follow PEP 8 style guide
- Use type hints for all function signatures
- Prefer composition over inheritance
- Write descriptive variable and function names

#### Documentation
- Use Google-style docstrings
- Include examples in docstrings when helpful
- Keep docstrings concise but complete

#### Error Handling
- Use specific exception types
- Provide meaningful error messages
- Log errors appropriately

#### Testing
- Test both happy path and edge cases
- Use descriptive test names
- Follow AAA pattern (Arrange, Act, Assert)

### Performance Considerations

- Profile code for performance bottlenecks
- Use async/await for I/O operations
- Cache expensive computations when appropriate
- Consider memory usage for large datasets

### Security Guidelines

- Never commit API keys or secrets
- Validate all user inputs
- Use parameterized queries for database operations
- Follow principle of least privilege

## Project Architecture

### Core Components

- **causal_eval/core/**: Core evaluation engine and base classes
- **causal_eval/tasks/**: Individual causal reasoning tasks
- **causal_eval/api/**: FastAPI web interface
- **causal_eval/generation/**: Test generation utilities
- **causal_eval/analysis/**: Result analysis tools

### Key Design Patterns

- **Plugin Architecture**: Tasks are loaded dynamically via entry points
- **Factory Pattern**: Model evaluators are created via factory
- **Strategy Pattern**: Different evaluation strategies per task type
- **Observer Pattern**: Event-driven result collection

### Database Schema

- **evaluations**: Evaluation run metadata
- **questions**: Generated test questions
- **responses**: Model responses to questions
- **results**: Scored evaluation results

## Getting Help

- **Architecture Questions**: Check `ARCHITECTURE.md`
- **API Reference**: Generated docs at `/docs`
- **Discord**: Join our developer channel
- **Office Hours**: Weekly developer sync (Fridays 3pm UTC)

## Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes
- Annual contributor awards
- Conference acknowledgments

Thank you for contributing to Causal Eval Bench!