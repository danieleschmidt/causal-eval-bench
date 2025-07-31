"""
Causal Evaluation Benchmark

A comprehensive evaluation framework for testing genuine causal reasoning in language models.
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"
__email__ = "daniel@terragon-labs.com"

from causal_eval.core.engine import EvaluationEngine
from causal_eval.core.tasks import TaskRegistry
from causal_eval.core.metrics import MetricsCollector

__all__ = [
    "EvaluationEngine",
    "TaskRegistry", 
    "MetricsCollector",
]