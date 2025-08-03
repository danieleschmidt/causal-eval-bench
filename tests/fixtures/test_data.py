"""Test data fixtures and factories for Causal Eval Bench."""

import json
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import uuid4

import factory
from factory import Faker, LazyAttribute, SubFactory
from faker import Faker as FakerInstance

fake = FakerInstance()


class QuestionFactory(factory.Factory):
    """Factory for generating test questions."""
    
    class Meta:
        model = dict
    
    id = LazyAttribute(lambda _: str(uuid4()))
    prompt = Faker('sentence', nb_words=15)
    task_type = Faker('random_element', elements=['causal_attribution', 'counterfactual', 'intervention', 'causal_chain', 'confounding'])
    domain = Faker('random_element', elements=['medical', 'social', 'economic', 'scientific', 'educational'])
    difficulty = Faker('random_element', elements=['easy', 'medium', 'hard', 'expert'])
    created_at = Faker('date_time_between', start_date='-1y', end_date='now')
    
    @LazyAttribute
    def ground_truth(self):
        """Generate ground truth based on task type."""
        base_truth = {
            'answer': fake.sentence(nb_words=10),
            'explanation': fake.paragraph(nb_sentences=3),
            'confidence': round(random.uniform(0.7, 1.0), 2)
        }
        
        if self.task_type == 'causal_attribution':
            base_truth.update({
                'causal_relationship': fake.boolean(),
                'confounders': fake.words(nb=random.randint(1, 4))
            })
        elif self.task_type == 'counterfactual':
            base_truth.update({
                'counterfactual_outcome': fake.word(),
                'alternative_scenarios': [fake.sentence() for _ in range(2)]
            })
        elif self.task_type == 'intervention':
            base_truth.update({
                'intervention_effect': fake.random_element(['positive', 'negative', 'neutral']),
                'effect_magnitude': round(random.uniform(0.1, 1.0), 2)
            })
        
        return base_truth


class EvaluationFactory(factory.Factory):
    """Factory for generating test evaluations."""
    
    class Meta:
        model = dict
    
    id = LazyAttribute(lambda _: str(uuid4()))
    model_name = Faker('word')
    status = Faker('random_element', elements=['pending', 'running', 'completed', 'failed'])
    created_at = Faker('date_time_between', start_date='-30d', end_date='now')
    completed_at = LazyAttribute(lambda obj: obj.created_at + timedelta(minutes=random.randint(1, 60)) if obj.status == 'completed' else None)
    
    @LazyAttribute
    def results(self):
        """Generate evaluation results based on status."""
        if self.status != 'completed':
            return None
        
        return {
            'overall_score': round(random.uniform(0.4, 0.95), 3),
            'task_scores': {
                'causal_attribution': round(random.uniform(0.3, 0.9), 3),
                'counterfactual': round(random.uniform(0.4, 0.85), 3),
                'intervention': round(random.uniform(0.5, 0.9), 3)
            },
            'domain_scores': {
                'medical': round(random.uniform(0.4, 0.9), 3),
                'social': round(random.uniform(0.3, 0.85), 3),
                'economic': round(random.uniform(0.5, 0.88), 3)
            },
            'questions_answered': random.randint(50, 200),
            'average_response_time': round(random.uniform(1.5, 8.0), 2)
        }


class ResponseFactory(factory.Factory):
    """Factory for generating model responses."""
    
    class Meta:
        model = dict
    
    question_id = LazyAttribute(lambda _: str(uuid4()))
    response_text = Faker('paragraph', nb_sentences=3)
    confidence = LazyAttribute(lambda _: round(random.uniform(0.1, 1.0), 2))
    response_time = LazyAttribute(lambda _: round(random.uniform(0.5, 10.0), 2))
    timestamp = Faker('date_time_between', start_date='-7d', end_date='now')
    
    @LazyAttribute
    def metadata(self):
        """Generate response metadata."""
        return {
            'tokens_used': random.randint(50, 500),
            'model_version': f"{fake.word()}-{fake.random_int(1, 10)}",
            'temperature': round(random.uniform(0.1, 1.0), 1),
            'max_tokens': random.randint(100, 1000)
        }


class TestDatasets:
    """Predefined test datasets for various scenarios."""
    
    @staticmethod
    def simple_causal_questions() -> List[Dict[str, Any]]:
        """Simple causal attribution questions for basic testing."""
        return [
            {
                'id': 'simple_001',
                'prompt': 'Ice cream sales and drowning incidents both increase in summer. Does ice cream cause drowning?',
                'ground_truth': {
                    'answer': 'No',
                    'explanation': 'This is spurious correlation. Both increase due to hot weather.',
                    'causal_relationship': False,
                    'confounders': ['temperature', 'season']
                },
                'task_type': 'causal_attribution',
                'domain': 'general',
                'difficulty': 'easy'
            },
            {
                'id': 'simple_002',
                'prompt': 'A person takes an umbrella and it starts raining. Did taking the umbrella cause the rain?',
                'ground_truth': {
                    'answer': 'No',
                    'explanation': 'The person likely took the umbrella because they predicted rain.',
                    'causal_relationship': False,
                    'confounders': ['weather_forecast']
                },
                'task_type': 'causal_attribution',
                'domain': 'general',
                'difficulty': 'easy'
            }
        ]
    
    @staticmethod
    def medical_questions() -> List[Dict[str, Any]]:
        """Medical domain questions for domain-specific testing."""
        return [
            {
                'id': 'med_001',
                'prompt': 'Patients who take vitamin D have lower rates of respiratory infections. Does vitamin D prevent respiratory infections?',
                'ground_truth': {
                    'answer': 'Possibly, but confounders must be considered',
                    'explanation': 'While studies suggest a relationship, factors like overall health, sun exposure, and lifestyle could be confounders.',
                    'causal_relationship': True,
                    'confounders': ['sun_exposure', 'overall_health', 'lifestyle']
                },
                'task_type': 'causal_attribution',
                'domain': 'medical',
                'difficulty': 'medium'
            }
        ]
    
    @staticmethod
    def counterfactual_scenarios() -> List[Dict[str, Any]]:
        """Counterfactual reasoning scenarios."""
        return [
            {
                'id': 'cf_001',
                'prompt': 'A student studied 4 hours daily and got an A. What if they had studied only 1 hour daily?',
                'ground_truth': {
                    'answer': 'They would likely have gotten a lower grade',
                    'explanation': 'Study time is generally positively correlated with academic performance.',
                    'counterfactual_outcome': 'lower_grade',
                    'confidence': 0.8
                },
                'task_type': 'counterfactual',
                'domain': 'educational',
                'difficulty': 'easy'
            }
        ]
    
    @staticmethod
    def challenging_questions() -> List[Dict[str, Any]]:
        """Challenging questions for advanced testing."""
        return [
            {
                'id': 'hard_001',
                'prompt': 'In a randomized controlled trial, patients receiving Drug A had better outcomes than those receiving placebo. However, patients on Drug A also had higher adherence rates. What can we conclude about Drug A\'s causal effect?',
                'ground_truth': {
                    'answer': 'Causal effect is likely but adherence is a mediating factor',
                    'explanation': 'The randomization supports causality, but adherence mediates the effect. We need to consider per-protocol vs intention-to-treat analysis.',
                    'causal_relationship': True,
                    'confounders': [],
                    'mediators': ['adherence']
                },
                'task_type': 'causal_attribution',
                'domain': 'medical',
                'difficulty': 'expert'
            }
        ]
    
    @staticmethod
    def performance_test_dataset(size: int = 1000) -> List[Dict[str, Any]]:
        """Generate large dataset for performance testing."""
        questions = []
        
        for i in range(size):
            question = QuestionFactory.build()
            question['id'] = f'perf_{i:05d}'
            questions.append(question)
        
        return questions
    
    @staticmethod
    def evaluation_results_dataset(num_evaluations: int = 100) -> List[Dict[str, Any]]:
        """Generate evaluation results dataset."""
        evaluations = []
        
        for i in range(num_evaluations):
            evaluation = EvaluationFactory.build()
            evaluation['id'] = f'eval_{i:05d}'
            evaluations.append(evaluation)
        
        return evaluations
    
    @staticmethod
    def adversarial_examples() -> List[Dict[str, Any]]:
        """Examples designed to test edge cases and potential failure modes."""
        return [
            {
                'id': 'adv_001',
                'prompt': 'Every time I wear my lucky socks, my team wins. Do my socks cause the wins?',
                'ground_truth': {
                    'answer': 'No',
                    'explanation': 'This is a classic example of superstitious thinking and post-hoc reasoning.',
                    'causal_relationship': False,
                    'bias_type': 'confirmation_bias'
                },
                'task_type': 'causal_attribution',
                'domain': 'psychology',
                'difficulty': 'medium'
            },
            {
                'id': 'adv_002',
                'prompt': 'Countries with higher chocolate consumption have more Nobel laureates per capita. Does chocolate consumption cause Nobel prizes?',
                'ground_truth': {
                    'answer': 'No',
                    'explanation': 'This is spurious correlation likely explained by economic development confounders.',
                    'causal_relationship': False,
                    'confounders': ['economic_development', 'education_investment', 'research_funding']
                },
                'task_type': 'causal_attribution',
                'domain': 'social',
                'difficulty': 'hard'
            }
        ]


class TestDataLoader:
    """Utility class for loading and managing test data."""
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self._cache = {}
    
    def load_dataset(self, dataset_name: str, **kwargs) -> List[Dict[str, Any]]:
        """Load a specific dataset by name."""
        if dataset_name in self._cache:
            return self._cache[dataset_name]
        
        dataset_methods = {
            'simple_causal': TestDatasets.simple_causal_questions,
            'medical': TestDatasets.medical_questions,
            'counterfactual': TestDatasets.counterfactual_scenarios,
            'challenging': TestDatasets.challenging_questions,
            'adversarial': TestDatasets.adversarial_examples,
            'performance': lambda: TestDatasets.performance_test_dataset(kwargs.get('size', 1000)),
            'evaluations': lambda: TestDatasets.evaluation_results_dataset(kwargs.get('num_evaluations', 100))
        }
        
        if dataset_name not in dataset_methods:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset = dataset_methods[dataset_name]()
        self._cache[dataset_name] = dataset
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """Save dataset to file for reproducible testing."""
        if self.data_path:
            filepath = f"{self.data_path}/{filename}"
            with open(filepath, 'w') as f:
                json.dump(dataset, f, indent=2, default=str)
    
    def create_mixed_dataset(self, datasets: List[str], proportions: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """Create a mixed dataset from multiple sources."""
        if proportions and len(proportions) != len(datasets):
            raise ValueError("Proportions must match number of datasets")
        
        if not proportions:
            proportions = [1.0 / len(datasets)] * len(datasets)
        
        mixed_data = []
        total_size = sum(len(self.load_dataset(ds)) for ds in datasets)
        
        for dataset_name, proportion in zip(datasets, proportions):
            dataset = self.load_dataset(dataset_name)
            sample_size = int(len(dataset) * proportion)
            mixed_data.extend(random.sample(dataset, min(sample_size, len(dataset))))
        
        random.shuffle(mixed_data)
        return mixed_data


# Utility functions for test data generation
def generate_question_batch(task_type: str, domain: str, count: int = 10) -> List[Dict[str, Any]]:
    """Generate a batch of questions for specific task type and domain."""
    questions = []
    for i in range(count):
        question = QuestionFactory.build(task_type=task_type, domain=domain)
        question['id'] = f"{task_type}_{domain}_{i:03d}"
        questions.append(question)
    return questions


def generate_evaluation_scenario(model_names: List[str], question_count: int = 50) -> Dict[str, Any]:
    """Generate a complete evaluation scenario with multiple models."""
    questions = TestDatasets.performance_test_dataset(question_count)
    
    scenario = {
        'questions': questions,
        'models': [],
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'total_questions': len(questions),
            'domains': list(set(q['domain'] for q in questions)),
            'task_types': list(set(q['task_type'] for q in questions))
        }
    }
    
    for model_name in model_names:
        model_data = {
            'name': model_name,
            'responses': [],
            'evaluation': EvaluationFactory.build(model_name=model_name, status='completed')
        }
        
        for question in questions:
            response = ResponseFactory.build(question_id=question['id'])
            model_data['responses'].append(response)
        
        scenario['models'].append(model_data)
    
    return scenario