# Testing Guide for Causal Eval Bench

This document provides comprehensive information about testing practices, infrastructure, and guidelines for the Causal Eval Bench project.

## Testing Strategy

Our testing strategy is built on multiple layers to ensure comprehensive coverage:

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions and integrations
3. **End-to-End Tests**: Test complete user workflows
4. **Performance Tests**: Benchmark and load testing
5. **API Tests**: Test REST API endpoints and responses

## Test Structure

```
tests/
├── conftest.py           # Shared fixtures and configuration
├── unit/                 # Unit tests for individual components
├── integration/          # Integration tests for component interactions
├── e2e/                  # End-to-end workflow tests
├── performance/          # Performance and load tests
└── load/                 # Load testing with Locust
```

## Running Tests

### Basic Test Execution

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

### Test Categories

Tests are organized using pytest markers for selective execution:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run API tests
pytest -m api

# Skip slow tests
pytest -m "not slow"

# Run evaluation logic tests
pytest -m evaluation
```

### Test Configuration

Test configuration is managed through `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "-ra",
    "--strict-markers",
    "--strict-config",
    "--cov=causal_eval",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
    "--cov-fail-under=80",
]
```

## Test Fixtures

Our test suite includes comprehensive fixtures defined in `conftest.py`:

### Database Fixtures
- `temp_db_file`: Temporary database for testing
- `mock_redis`: Mock Redis client for caching tests

### Model Client Fixtures
- `mock_openai_client`: Mock OpenAI API client
- `mock_anthropic_client`: Mock Anthropic API client

### Test Data Fixtures
- `sample_causal_question`: Example causal reasoning question
- `sample_counterfactual_question`: Example counterfactual question
- `performance_test_data`: Large dataset for performance testing

### Utility Fixtures
- `mock_file_system`: Temporary file system for testing
- `benchmark_config`: Performance benchmark configuration

## Writing Tests

### Unit Test Example

```python
class TestCausalAttribution:
    """Unit tests for causal attribution functionality."""
    
    def test_causal_question_structure(self, sample_causal_question):
        """Test the structure of causal questions."""
        ground_truth = sample_causal_question["ground_truth"]
        
        assert "answer" in ground_truth
        assert "explanation" in ground_truth
        assert "causal_relationship" in ground_truth
        assert isinstance(ground_truth["causal_relationship"], bool)
    
    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
    def test_difficulty_levels(self, difficulty):
        """Test different difficulty levels."""
        assert difficulty in ["easy", "medium", "hard"]
```

### Integration Test Example

```python
@pytest.mark.integration
class TestEvaluationPipeline:
    """Integration tests for the evaluation pipeline."""
    
    async def test_full_evaluation_flow(self, mock_openai_client):
        """Test complete evaluation workflow."""
        # Setup
        evaluator = ModelEvaluator(client=mock_openai_client)
        question = create_test_question()
        
        # Execute
        result = await evaluator.evaluate(question)
        
        # Verify
        assert result.score is not None
        assert 0 <= result.score <= 1
```

### Performance Test Example

```python
@pytest.mark.performance
@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark(group="evaluation")
    def test_single_evaluation_performance(self, benchmark):
        """Benchmark single evaluation performance."""
        
        def run_evaluation():
            # Evaluation logic here
            return evaluate_model_response()
        
        result = benchmark(run_evaluation)
        assert result.score > 0
```

## Test Data Management

### Sample Data
All test data is generated programmatically or loaded from fixtures to ensure:
- **Consistency**: Same test data across all test runs
- **Isolation**: Each test gets fresh, independent data
- **Reproducibility**: Tests produce consistent results

### Mock Strategies
We use comprehensive mocking for external dependencies:

```python
@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Mocked AI response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client
```

## Performance Testing

### Benchmarking
We use `pytest-benchmark` for performance testing:

```python
def test_evaluation_performance(benchmark):
    result = benchmark(expensive_function)
    assert result is not None
```

### Load Testing
Load testing is implemented using Locust:

```python
# tests/load/locustfile.py
from locust import HttpUser, task

class EvaluationUser(HttpUser):
    @task
    def evaluate_model(self):
        self.client.post("/api/v1/evaluate", json=test_payload)
```

### Memory Testing
Memory usage is monitored in performance tests:

```python
def test_memory_usage(self, benchmark_config):
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    # Run memory-intensive operations
    
    peak_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = peak_memory - initial_memory
    
    assert memory_increase < benchmark_config["max_memory_mb"]
```

## Coverage Requirements

### Coverage Targets
- **Overall Coverage**: Minimum 80%
- **Critical Components**: Minimum 90%
- **New Code**: 100% coverage required

### Coverage Reports
Coverage reports are generated in multiple formats:
- **Terminal**: Real-time coverage feedback
- **HTML**: Detailed coverage browser at `htmlcov/index.html`
- **XML**: For CI/CD integration

### Exclusions
Certain code is excluded from coverage requirements:
- Test files themselves
- Migration scripts
- Debug and development utilities
- External integrations (tested separately)

## Continuous Integration

### GitHub Actions Integration
Tests run automatically on:
- **Pull Requests**: Full test suite execution
- **Main Branch**: Extended test suite with performance tests
- **Nightly**: Full regression testing with all combinations

### Test Matrix
We test against multiple configurations:
- **Python Versions**: 3.9, 3.10, 3.11, 3.12
- **Operating Systems**: Ubuntu, macOS, Windows
- **Dependencies**: Minimum and latest versions

### Quality Gates
All tests must pass before code can be merged:
- Unit tests: 100% pass rate
- Integration tests: 100% pass rate
- Coverage: Above threshold
- Performance: Within acceptable limits

## Test Best Practices

### Writing Effective Tests

1. **Descriptive Names**: Test names should clearly describe what is being tested
   ```python
   def test_causal_attribution_correctly_identifies_spurious_correlation():
   ```

2. **Arrange-Act-Assert**: Structure tests clearly
   ```python
   def test_evaluation_scoring():
       # Arrange
       question = create_test_question()
       model = MockModel()
       
       # Act
       result = evaluate(model, question)
       
       # Assert
       assert result.score == expected_score
   ```

3. **Single Responsibility**: Each test should verify one specific behavior

4. **Independent Tests**: Tests should not depend on each other's execution

5. **Comprehensive Assertions**: Verify all relevant aspects of the result

### Fixture Guidelines

1. **Scope Appropriately**: Use session, module, class, or function scope as needed
2. **Name Clearly**: Fixture names should indicate their purpose
3. **Document Dependencies**: Explain complex fixture relationships
4. **Clean Up**: Ensure resources are properly cleaned up

### Mocking Best Practices

1. **Mock External Dependencies**: Always mock API calls, file system, databases
2. **Verify Interactions**: Check that mocks are called with expected parameters
3. **Reset Between Tests**: Ensure mock state doesn't leak between tests
4. **Realistic Responses**: Mock responses should match real API behavior

## Debugging Tests

### Common Issues

1. **Flaky Tests**: Tests that pass/fail intermittently
   - Check for timing dependencies
   - Verify proper test isolation
   - Review async code carefully

2. **Slow Tests**: Tests that take too long to execute
   - Profile test execution
   - Check for unnecessary operations
   - Consider parallelization

3. **Failed Assertions**: Tests that fail unexpectedly
   - Use descriptive assertion messages
   - Print intermediate values for debugging
   - Check test data assumptions

### Debugging Tools

```bash
# Run tests with verbose output
pytest -v

# Run specific test with debugging
pytest tests/unit/test_specific.py::test_function -s

# Run tests with pdb on failure
pytest --pdb

# Generate detailed test report
pytest --tb=long --capture=no
```

## Contributing to Tests

### Before Submitting

1. **Run Full Test Suite**: Ensure all tests pass locally
2. **Check Coverage**: Verify coverage meets requirements
3. **Add Tests for New Features**: Every new feature needs tests
4. **Update Documentation**: Update this guide if needed

### Code Review Checklist

- [ ] Tests cover all new functionality
- [ ] Tests include edge cases and error conditions
- [ ] Test names are descriptive and clear
- [ ] Fixtures are used appropriately
- [ ] Mocks are properly configured
- [ ] Performance impact is considered
- [ ] Tests are properly categorized with markers

## Resources

### Documentation
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- [Locust Documentation](https://docs.locust.io/)

### Tools
- **pytest**: Primary testing framework
- **pytest-cov**: Coverage reporting
- **pytest-benchmark**: Performance testing
- **pytest-asyncio**: Async test support
- **factory-boy**: Test data generation
- **Locust**: Load testing

### Examples
- See `tests/unit/test_example.py` for unit test patterns
- See `tests/integration/test_example.py` for integration test patterns
- See `tests/performance/test_benchmarks.py` for performance test patterns

---

For questions about testing practices or to contribute improvements to our testing infrastructure, please open an issue or submit a pull request.