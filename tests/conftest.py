"""Pytest configuration and shared fixtures for the Causal Eval Bench test suite."""

import os
import tempfile
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool

# Set test environment
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///./test_causal_eval_bench.db"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["DEBUG"] = "false"

from causal_eval.database.models import Base
from causal_eval.database.connection import DatabaseManager
from causal_eval.core.engine import EvaluationEngine
from causal_eval.core.metrics import MetricsCollector


@pytest.fixture(scope="session")
def temp_db_file():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        yield temp_file.name
    # Cleanup
    if os.path.exists(temp_file.name):
        os.unlink(temp_file.name)


@pytest_asyncio.fixture(scope="session")
async def test_db_engine():
    """Create test database engine."""
    # Use in-memory SQLite for tests
    database_url = "sqlite+aiosqlite:///:memory:"
    
    engine = create_async_engine(
        database_url,
        echo=False,
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
        }
    )
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def test_session(test_db_engine):
    """Create test database session."""
    async_session = async_sessionmaker(
        bind=test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture
async def test_db_manager(test_db_engine):
    """Create test database manager."""
    db_manager = DatabaseManager("sqlite+aiosqlite:///:memory:")
    db_manager.async_engine = test_db_engine
    db_manager.async_session_factory = async_sessionmaker(
        bind=test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    db_manager.is_async = True
    
    yield db_manager


@pytest.fixture
def evaluation_engine():
    """Create evaluation engine for testing."""
    return EvaluationEngine()


@pytest.fixture
def metrics_collector():
    """Create metrics collector for testing."""
    return MetricsCollector()


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.exists.return_value = False
    return mock_redis


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Mocked AI response"
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock()]
    mock_response.content[0].text = "Mocked Claude response"
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def sample_causal_question():
    """Sample causal reasoning question for testing."""
    return {
        "id": "test_question_001",
        "prompt": (
            "Ice cream sales and swimming pool accidents both increase in summer. "
            "Does eating ice cream cause swimming accidents?"
        ),
        "ground_truth": {
            "answer": "No",
            "explanation": (
                "This is a classic example of spurious correlation. Both ice cream "
                "sales and swimming accidents increase in summer due to a confounding "
                "variable: hot weather."
            ),
            "causal_relationship": False,
            "confounders": ["temperature", "season", "outdoor_activity"]
        },
        "task_type": "causal_attribution",
        "domain": "general",
        "difficulty": "easy"
    }


@pytest.fixture
def sample_counterfactual_question():
    """Sample counterfactual reasoning question for testing."""
    return {
        "id": "test_question_002",
        "prompt": (
            "A student studied hard for 3 weeks and scored 95% on the exam. "
            "What would likely have happened if the student had not studied at all?"
        ),
        "ground_truth": {
            "answer": "The student would likely have scored much lower",
            "explanation": (
                "Studying is a major causal factor in exam performance. Without "
                "studying, the student would lack the knowledge and preparation "
                "necessary for a high score."
            ),
            "counterfactual_outcome": "low_score",
            "confidence": 0.9
        },
        "task_type": "counterfactual",
        "domain": "education",
        "difficulty": "medium"
    }


@pytest.fixture
def mock_file_system():
    """Mock file system operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def performance_test_data():
    """Generate performance test data."""
    import random
    
    questions = []
    for i in range(100):
        questions.append({
            "id": f"perf_test_{i:03d}",
            "prompt": f"Test question {i} for performance evaluation.",
            "ground_truth": {"answer": random.choice(["Yes", "No"])},
            "task_type": random.choice(["causal_attribution", "counterfactual"]),
            "domain": random.choice(["medical", "social", "economic"]),
            "difficulty": random.choice(["easy", "medium", "hard"])
        })
    
    return questions


# Performance testing fixtures
@pytest.fixture
def benchmark_config():
    """Default benchmark configuration for testing."""
    return {
        "max_execution_time": 30.0,
        "max_memory_mb": 512,
        "target_throughput": 10.0,  # requests per second
        "acceptable_error_rate": 0.05  # 5%
    }


# Parametrized fixtures for different model types
@pytest.fixture(params=["openai", "anthropic", "huggingface"])
def model_type(request):
    """Parametrized fixture for different model types."""
    return request.param


@pytest.fixture(params=["easy", "medium", "hard"])
def difficulty_level(request):
    """Parametrized fixture for different difficulty levels."""
    return request.param


@pytest.fixture(params=["medical", "social", "economic", "scientific"])
def domain_type(request):
    """Parametrized fixture for different domain types."""
    return request.param


# Custom markers for test organization
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "evaluation: mark test as evaluation logic test")
    config.addinivalue_line("markers", "generation: mark test as test generation test")
    config.addinivalue_line("markers", "performance: mark test as performance test")


# Test data cleanup
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test."""
    yield
    
    # Clean up any test files
    test_files = [
        "test_causal_eval_bench.db",
        "test_evaluation_results.json",
        "test_benchmark_data.pkl"
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            os.unlink(file_path)