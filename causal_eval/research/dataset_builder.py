"""
Dataset builder for creating standardized causal reasoning evaluation datasets.
"""

import json
import logging
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    """Metadata for causal reasoning datasets."""
    
    name: str
    version: str
    description: str
    creation_date: str
    authors: List[str]
    domains: List[str]
    task_types: List[str]
    total_samples: int
    difficulty_distribution: Dict[str, int]
    domain_distribution: Dict[str, int]
    license: str = "Apache-2.0"
    citation: Optional[str] = None
    baseline_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    data_collection_method: str = "generated"
    quality_assurance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSample:
    """Individual sample in causal reasoning dataset."""
    
    sample_id: str
    task_type: str
    domain: str
    difficulty: str
    prompt: str
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_criteria: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetSample':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CausalDataset:
    """Complete causal reasoning evaluation dataset."""
    
    metadata: DatasetMetadata
    samples: List[DatasetSample] = field(default_factory=list)
    splits: Dict[str, List[str]] = field(default_factory=dict)  # split_name -> sample_ids
    
    @property
    def train_samples(self) -> List[DatasetSample]:
        """Get training samples."""
        if "train" not in self.splits:
            return []
        return [s for s in self.samples if s.sample_id in self.splits["train"]]
    
    @property
    def validation_samples(self) -> List[DatasetSample]:
        """Get validation samples."""
        if "validation" not in self.splits:
            return []
        return [s for s in self.samples if s.sample_id in self.splits["validation"]]
    
    @property
    def test_samples(self) -> List[DatasetSample]:
        """Get test samples."""
        if "test" not in self.splits:
            return []
        return [s for s in self.samples if s.sample_id in self.splits["test"]]
    
    def get_samples_by_criteria(
        self,
        task_type: Optional[str] = None,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None
    ) -> List[DatasetSample]:
        """Get samples matching specified criteria."""
        filtered_samples = self.samples
        
        if task_type:
            filtered_samples = [s for s in filtered_samples if s.task_type == task_type]
        if domain:
            filtered_samples = [s for s in filtered_samples if s.domain == domain]
        if difficulty:
            filtered_samples = [s for s in filtered_samples if s.difficulty == difficulty]
        
        return filtered_samples
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        if not self.samples:
            return {}
        
        stats = {
            "total_samples": len(self.samples),
            "task_type_distribution": {},
            "domain_distribution": {},
            "difficulty_distribution": {},
            "average_prompt_length": 0,
            "splits": {split: len(ids) for split, ids in self.splits.items()}
        }
        
        prompt_lengths = []
        
        for sample in self.samples:
            # Task type distribution
            stats["task_type_distribution"][sample.task_type] = \
                stats["task_type_distribution"].get(sample.task_type, 0) + 1
            
            # Domain distribution
            stats["domain_distribution"][sample.domain] = \
                stats["domain_distribution"].get(sample.domain, 0) + 1
            
            # Difficulty distribution
            stats["difficulty_distribution"][sample.difficulty] = \
                stats["difficulty_distribution"].get(sample.difficulty, 0) + 1
            
            # Prompt length
            prompt_lengths.append(len(sample.prompt))
        
        stats["average_prompt_length"] = np.mean(prompt_lengths)
        stats["prompt_length_stats"] = {
            "min": np.min(prompt_lengths),
            "max": np.max(prompt_lengths),
            "std": np.std(prompt_lengths),
            "median": np.median(prompt_lengths)
        }
        
        return stats


class DatasetBuilder:
    """Builder for creating standardized causal reasoning datasets."""
    
    def __init__(self, scenario_generator=None, task_registry=None):
        """Initialize the dataset builder."""
        self.scenario_generator = scenario_generator
        self.task_registry = task_registry
        self.built_datasets: List[CausalDataset] = []
        logger.info("Dataset builder initialized")
    
    def build_dataset(
        self,
        name: str,
        version: str,
        description: str,
        authors: List[str],
        config: Dict[str, Any]
    ) -> CausalDataset:
        """Build a complete causal reasoning dataset."""
        
        logger.info(f"Building dataset: {name} v{version}")
        
        # Extract configuration
        domains = config.get("domains", ["general"])
        task_types = config.get("task_types", ["attribution", "counterfactual", "intervention"])
        difficulty_levels = config.get("difficulty_levels", ["easy", "medium", "hard"])
        samples_per_combination = config.get("samples_per_combination", 10)
        train_ratio = config.get("train_ratio", 0.7)
        val_ratio = config.get("val_ratio", 0.15)
        test_ratio = config.get("test_ratio", 0.15)
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Generate samples
        samples = self._generate_samples(
            domains, task_types, difficulty_levels, samples_per_combination
        )
        
        # Create splits
        splits = self._create_splits(samples, train_ratio, val_ratio, test_ratio)
        
        # Calculate distributions
        difficulty_dist = {}
        domain_dist = {}
        
        for sample in samples:
            difficulty_dist[sample.difficulty] = difficulty_dist.get(sample.difficulty, 0) + 1
            domain_dist[sample.domain] = domain_dist.get(sample.domain, 0) + 1
        
        # Create metadata
        metadata = DatasetMetadata(
            name=name,
            version=version,
            description=description,
            creation_date=datetime.now().isoformat(),
            authors=authors,
            domains=domains,
            task_types=task_types,
            total_samples=len(samples),
            difficulty_distribution=difficulty_dist,
            domain_distribution=domain_dist,
            data_collection_method="automated_generation"
        )
        
        # Create dataset
        dataset = CausalDataset(
            metadata=metadata,
            samples=samples,
            splits=splits
        )
        
        # Run quality assurance
        qa_results = self._run_quality_assurance(dataset)
        dataset.metadata.quality_assurance = qa_results
        
        self.built_datasets.append(dataset)
        logger.info(f"Dataset built with {len(samples)} samples")
        
        return dataset
    
    def _generate_samples(
        self,
        domains: List[str],
        task_types: List[str],
        difficulty_levels: List[str],
        samples_per_combination: int
    ) -> List[DatasetSample]:
        """Generate samples for all combinations of parameters."""
        
        samples = []
        sample_counter = 0
        
        for domain in domains:
            for task_type in task_types:
                for difficulty in difficulty_levels:
                    logger.info(f"Generating {samples_per_combination} samples for {domain}-{task_type}-{difficulty}")
                    
                    for i in range(samples_per_combination):
                        sample = self._generate_single_sample(
                            domain, task_type, difficulty, sample_counter
                        )
                        samples.append(sample)
                        sample_counter += 1
        
        return samples
    
    def _generate_single_sample(
        self,
        domain: str,
        task_type: str,
        difficulty: str,
        sample_id: int
    ) -> DatasetSample:
        """Generate a single dataset sample."""
        
        # Generate unique sample ID
        sample_id_str = f"{domain}_{task_type}_{difficulty}_{sample_id:06d}"
        
        # Generate scenario if scenario generator is available
        if self.scenario_generator:
            from causal_eval.generation.scenario_generator import ScenarioComplexity
            
            complexity_map = {
                "easy": ScenarioComplexity.SIMPLE,
                "medium": ScenarioComplexity.MODERATE,
                "hard": ScenarioComplexity.COMPLEX
            }
            
            complexity = complexity_map.get(difficulty, ScenarioComplexity.MODERATE)
            scenario = self.scenario_generator.generate_scenario(domain, complexity)
            
            # Generate task-specific prompt
            prompt = self._generate_task_prompt(task_type, scenario)
            
            # Extract ground truth
            ground_truth = self._extract_ground_truth(task_type, scenario)
            
            # Create evaluation criteria
            evaluation_criteria = self._create_evaluation_criteria(task_type, scenario)
            
        else:
            # Fallback to template-based generation
            prompt = self._generate_template_prompt(domain, task_type, difficulty)
            ground_truth = {"type": task_type, "domain": domain}
            evaluation_criteria = {"scoring": "manual"}
        
        return DatasetSample(
            sample_id=sample_id_str,
            task_type=task_type,
            domain=domain,
            difficulty=difficulty,
            prompt=prompt,
            ground_truth=ground_truth,
            evaluation_criteria=evaluation_criteria,
            metadata={
                "generation_timestamp": time.time(),
                "generation_method": "scenario_based" if self.scenario_generator else "template_based"
            }
        )
    
    def _generate_task_prompt(self, task_type: str, scenario) -> str:
        """Generate task-specific prompt from scenario."""
        
        prompt_templates = {
            "attribution": f"""
Analyze the following scenario and determine the nature of the causal relationship.

Scenario: {scenario.description}

Primary Variables:
- Variable A: {scenario.get_primary_relationship().cause if scenario.get_primary_relationship() else 'Unknown'}
- Variable B: {scenario.get_primary_relationship().effect if scenario.get_primary_relationship() else 'Unknown'}

Question: What is the causal relationship between Variable A and Variable B?

Please analyze:
1. Whether there is a genuine causal relationship
2. The direction of causation (if any)
3. Potential confounding variables
4. Your confidence in the assessment

Provide detailed reasoning for your analysis.""",
            
            "counterfactual": f"""
Consider the following scenario and analyze a counterfactual situation.

Original Scenario: {scenario.description}

Counterfactual Question: {self._generate_counterfactual_question(scenario)}

Please provide:
1. Your predicted outcome for the counterfactual scenario
2. The reasoning behind your prediction
3. Key assumptions you're making
4. Factors that might affect the outcome

Explain the causal mechanisms that support your reasoning.""",
            
            "intervention": f"""
Analyze the following system and predict the effects of an intervention.

System Description: {scenario.description}

Proposed Intervention: {self._generate_intervention_description(scenario)}

Please predict:
1. The direct effects of this intervention
2. Potential indirect effects and side effects
3. The time frame for these effects
4. Factors that might moderate the intervention's impact

Provide detailed causal reasoning for your predictions.""",
            
            "chain": f"""
Examine the following scenario and trace the causal chain.

Initial Condition: {self._extract_initial_condition(scenario)}
Final Outcome: {self._extract_final_outcome(scenario)}

Question: What are the intermediate causal steps that connect the initial condition to the final outcome?

Please provide:
1. A complete causal chain with intermediate steps
2. The mechanisms underlying each step
3. Points where the chain might be interrupted
4. Alternative causal pathways

Explain the reasoning behind each link in the chain.""",
            
            "confounding": f"""
Analyze the following observational data for potential confounding variables.

Observed Relationship: {scenario.description}

Question: Is this relationship likely to be causal, or might it be explained by confounding variables?

Please assess:
1. Whether the relationship is genuinely causal
2. Potential confounding variables
3. How these confounders might affect the relationship
4. How you would design a study to control for confounding

Provide detailed analysis of the causal structure."""
        }
        
        return prompt_templates.get(task_type, scenario.description).strip()
    
    def _generate_counterfactual_question(self, scenario) -> str:
        """Generate counterfactual question from scenario."""
        primary_rel = scenario.get_primary_relationship()
        if primary_rel:
            return f"What if {primary_rel.cause.replace('_', ' ')} had not occurred or been different?"
        return "What if the key variable had been different?"
    
    def _generate_intervention_description(self, scenario) -> str:
        """Generate intervention description from scenario."""
        primary_rel = scenario.get_primary_relationship()
        if primary_rel:
            return f"Directly manipulating {primary_rel.cause.replace('_', ' ')} to increase its level"
        return "A targeted intervention on the primary variable"
    
    def _extract_initial_condition(self, scenario) -> str:
        """Extract initial condition for causal chain."""
        primary_rel = scenario.get_primary_relationship()
        if primary_rel:
            return f"Changes in {primary_rel.cause.replace('_', ' ')}"
        return "Initial system state"
    
    def _extract_final_outcome(self, scenario) -> str:
        """Extract final outcome for causal chain."""
        primary_rel = scenario.get_primary_relationship()
        if primary_rel:
            return f"Observed changes in {primary_rel.effect.replace('_', ' ')}"
        return "Final system outcome"
    
    def _extract_ground_truth(self, task_type: str, scenario) -> Dict[str, Any]:
        """Extract ground truth information from scenario."""
        
        base_truth = {
            "task_type": task_type,
            "scenario_id": scenario.scenario_id,
            "domain": scenario.domain,
            "complexity": scenario.complexity.value,
            "primary_relationship": None,
            "confounders": list(scenario.get_confounders()),
            "learning_objectives": scenario.learning_objectives,
        }
        
        primary_rel = scenario.get_primary_relationship()
        if primary_rel:
            base_truth["primary_relationship"] = {
                "cause": primary_rel.cause,
                "effect": primary_rel.effect,
                "type": primary_rel.relationship_type.value,
                "strength": primary_rel.strength,
                "mechanism": primary_rel.mechanism
            }
        
        # Task-specific ground truth
        task_specific = {
            "attribution": {
                "is_causal": primary_rel.relationship_type.value == "direct_causal" if primary_rel else False,
                "relationship_type": primary_rel.relationship_type.value if primary_rel else "unknown"
            },
            "counterfactual": {
                "outcome_direction": "decrease" if primary_rel and primary_rel.relationship_type.value == "direct_causal" else "unclear",
                "key_mechanisms": [primary_rel.mechanism] if primary_rel else []
            },
            "intervention": {
                "primary_effect": primary_rel.effect if primary_rel else "unknown",
                "effect_magnitude": primary_rel.strength if primary_rel else 0.0,
                "side_effects": [c for c in scenario.get_confounders()]
            },
            "chain": {
                "chain_length": len(scenario.relationships),
                "key_steps": [rel.cause + " -> " + rel.effect for rel in scenario.relationships]
            },
            "confounding": {
                "is_confounded": len(scenario.get_confounders()) > 0,
                "confounders": list(scenario.get_confounders())
            }
        }
        
        base_truth.update(task_specific.get(task_type, {}))
        return base_truth
    
    def _create_evaluation_criteria(self, task_type: str, scenario) -> Dict[str, Any]:
        """Create evaluation criteria for the sample."""
        
        base_criteria = {
            "scoring_method": "weighted_average",
            "max_score": 1.0,
            "partial_credit": True
        }
        
        task_criteria = {
            "attribution": {
                "weights": {
                    "causal_assessment": 0.4,
                    "relationship_identification": 0.3,
                    "confounder_awareness": 0.2,
                    "reasoning_quality": 0.1
                },
                "key_concepts": ["causation", "correlation", "confounding"]
            },
            "counterfactual": {
                "weights": {
                    "outcome_prediction": 0.4,
                    "mechanism_explanation": 0.3,
                    "assumption_identification": 0.2,
                    "reasoning_coherence": 0.1
                },
                "key_concepts": ["counterfactual", "mechanism", "assumption"]
            },
            "intervention": {
                "weights": {
                    "effect_prediction": 0.35,
                    "side_effect_identification": 0.25,
                    "time_frame_assessment": 0.2,
                    "mechanism_understanding": 0.2
                },
                "key_concepts": ["intervention", "effect", "mechanism"]
            },
            "chain": {
                "weights": {
                    "chain_completeness": 0.4,
                    "step_accuracy": 0.3,
                    "mechanism_explanation": 0.2,
                    "reasoning_coherence": 0.1
                },
                "key_concepts": ["chain", "sequence", "mechanism"]
            },
            "confounding": {
                "weights": {
                    "confounder_identification": 0.4,
                    "causal_assessment": 0.3,
                    "study_design": 0.2,
                    "reasoning_quality": 0.1
                },
                "key_concepts": ["confounding", "causal", "design"]
            }
        }
        
        base_criteria.update(task_criteria.get(task_type, {}))
        return base_criteria
    
    def _generate_template_prompt(self, domain: str, task_type: str, difficulty: str) -> str:
        """Generate basic template prompt when scenario generator is not available."""
        
        template = f"""
Analyze the following {task_type} scenario in the {domain} domain.

[This is a {difficulty} difficulty causal reasoning task]

Question: Please provide your analysis of the causal relationships involved.

Consider:
- The strength and direction of causal relationships
- Potential confounding variables
- Alternative explanations
- The quality of evidence

Provide detailed reasoning for your conclusions.
"""
        return template.strip()
    
    def _create_splits(
        self,
        samples: List[DatasetSample],
        train_ratio: float,
        val_ratio: float,
        test_ratio: float
    ) -> Dict[str, List[str]]:
        """Create train/validation/test splits."""
        
        # Shuffle samples for random assignment
        sample_ids = [s.sample_id for s in samples]
        np.random.shuffle(sample_ids)
        
        n_total = len(sample_ids)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            "train": sample_ids[:n_train],
            "validation": sample_ids[n_train:n_train + n_val],
            "test": sample_ids[n_train + n_val:]
        }
        
        logger.info(f"Created splits - Train: {len(splits['train'])}, Val: {len(splits['validation'])}, Test: {len(splits['test'])}")
        
        return splits
    
    def _run_quality_assurance(self, dataset: CausalDataset) -> Dict[str, Any]:
        """Run quality assurance checks on the dataset."""
        
        qa_results = {
            "checks_passed": [],
            "checks_failed": [],
            "warnings": [],
            "overall_quality": "unknown"
        }
        
        # Check 1: Minimum sample size
        min_samples = 50
        if len(dataset.samples) >= min_samples:
            qa_results["checks_passed"].append("minimum_sample_size")
        else:
            qa_results["checks_failed"].append("minimum_sample_size")
        
        # Check 2: Balanced splits
        split_sizes = [len(ids) for ids in dataset.splits.values()]
        if len(split_sizes) > 1 and max(split_sizes) / min(split_sizes) < 10:  # No split is 10x larger than others
            qa_results["checks_passed"].append("balanced_splits")
        else:
            qa_results["warnings"].append("unbalanced_splits")
        
        # Check 3: Domain coverage
        domains = set(s.domain for s in dataset.samples)
        if len(domains) >= 2:
            qa_results["checks_passed"].append("domain_diversity")
        else:
            qa_results["warnings"].append("limited_domain_coverage")
        
        # Check 4: Task type coverage
        task_types = set(s.task_type for s in dataset.samples)
        if len(task_types) >= 3:
            qa_results["checks_passed"].append("task_type_diversity")
        else:
            qa_results["warnings"].append("limited_task_type_coverage")
        
        # Check 5: Difficulty distribution
        difficulties = [s.difficulty for s in dataset.samples]
        difficulty_counts = {d: difficulties.count(d) for d in set(difficulties)}
        if len(difficulty_counts) >= 2 and max(difficulty_counts.values()) / min(difficulty_counts.values()) < 5:
            qa_results["checks_passed"].append("difficulty_distribution")
        else:
            qa_results["warnings"].append("skewed_difficulty_distribution")
        
        # Check 6: Prompt quality
        avg_prompt_length = np.mean([len(s.prompt) for s in dataset.samples])
        if 100 <= avg_prompt_length <= 2000:  # Reasonable prompt length
            qa_results["checks_passed"].append("prompt_length")
        else:
            qa_results["warnings"].append("unusual_prompt_lengths")
        
        # Check 7: Ground truth completeness
        complete_ground_truth = sum(1 for s in dataset.samples if s.ground_truth and len(s.ground_truth) > 2)
        if complete_ground_truth / len(dataset.samples) > 0.9:
            qa_results["checks_passed"].append("ground_truth_completeness")
        else:
            qa_results["checks_failed"].append("incomplete_ground_truth")
        
        # Overall quality assessment
        passed_count = len(qa_results["checks_passed"])
        failed_count = len(qa_results["checks_failed"])
        warning_count = len(qa_results["warnings"])
        
        if failed_count == 0 and warning_count <= 2:
            qa_results["overall_quality"] = "excellent"
        elif failed_count <= 1 and warning_count <= 4:
            qa_results["overall_quality"] = "good"
        elif failed_count <= 2:
            qa_results["overall_quality"] = "acceptable"
        else:
            qa_results["overall_quality"] = "poor"
        
        qa_results["summary"] = {
            "checks_passed": passed_count,
            "checks_failed": failed_count,
            "warnings": warning_count,
            "total_samples": len(dataset.samples)
        }
        
        logger.info(f"QA completed - Quality: {qa_results['overall_quality']}, Passed: {passed_count}, Failed: {failed_count}")
        
        return qa_results
    
    def save_dataset(self, dataset: CausalDataset, output_path: str) -> str:
        """Save dataset to disk in standard format."""
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate dataset hash for integrity
        dataset_content = json.dumps([s.to_dict() for s in dataset.samples], sort_keys=True)
        dataset_hash = hashlib.sha256(dataset_content.encode()).hexdigest()[:16]
        
        # Save metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            metadata_dict = asdict(dataset.metadata)
            metadata_dict["dataset_hash"] = dataset_hash
            json.dump(metadata_dict, f, indent=2)
        
        # Save samples
        samples_file = output_dir / "samples.jsonl"
        with open(samples_file, 'w') as f:
            for sample in dataset.samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
        
        # Save splits
        splits_file = output_dir / "splits.json"
        with open(splits_file, 'w') as f:
            json.dump(dataset.splits, f, indent=2)
        
        # Save statistics
        stats_file = output_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            stats = dataset.calculate_statistics()
            stats["dataset_hash"] = dataset_hash
            json.dump(stats, f, indent=2)
        
        # Generate README
        readme_file = output_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(self._generate_dataset_readme(dataset))
        
        logger.info(f"Dataset saved to {output_dir}")
        return str(output_dir)
    
    def load_dataset(self, dataset_path: str) -> CausalDataset:
        """Load dataset from disk."""
        
        dataset_dir = Path(dataset_path)
        
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
        # Load metadata
        metadata_file = dataset_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata_dict = json.load(f)
            dataset_hash = metadata_dict.pop("dataset_hash", None)
            metadata = DatasetMetadata(**metadata_dict)
        
        # Load samples
        samples = []
        samples_file = dataset_dir / "samples.jsonl"
        with open(samples_file, 'r') as f:
            for line in f:
                sample_dict = json.loads(line.strip())
                samples.append(DatasetSample.from_dict(sample_dict))
        
        # Load splits
        splits_file = dataset_dir / "splits.json"
        if splits_file.exists():
            with open(splits_file, 'r') as f:
                splits = json.load(f)
        else:
            splits = {}
        
        # Verify integrity if hash is available
        if dataset_hash:
            current_content = json.dumps([s.to_dict() for s in samples], sort_keys=True)
            current_hash = hashlib.sha256(current_content.encode()).hexdigest()[:16]
            if current_hash != dataset_hash:
                logger.warning(f"Dataset integrity check failed. Expected hash: {dataset_hash}, Got: {current_hash}")
        
        dataset = CausalDataset(
            metadata=metadata,
            samples=samples,
            splits=splits
        )
        
        logger.info(f"Loaded dataset with {len(samples)} samples from {dataset_path}")
        return dataset
    
    def _generate_dataset_readme(self, dataset: CausalDataset) -> str:
        """Generate README file for the dataset."""
        
        stats = dataset.calculate_statistics()
        
        readme = f"""# {dataset.metadata.name} v{dataset.metadata.version}

## Description

{dataset.metadata.description}

## Dataset Information

- **Total Samples**: {stats['total_samples']}
- **Domains**: {', '.join(dataset.metadata.domains)}
- **Task Types**: {', '.join(dataset.metadata.task_types)}
- **Creation Date**: {dataset.metadata.creation_date}
- **Authors**: {', '.join(dataset.metadata.authors)}

## Dataset Splits

"""
        
        for split_name, sample_ids in dataset.splits.items():
            percentage = len(sample_ids) / len(dataset.samples) * 100
            readme += f"- **{split_name.title()}**: {len(sample_ids)} samples ({percentage:.1f}%)\n"
        
        readme += f"""
## Distribution Analysis

### Task Types
"""
        for task_type, count in stats["task_type_distribution"].items():
            percentage = count / stats["total_samples"] * 100
            readme += f"- **{task_type}**: {count} samples ({percentage:.1f}%)\n"
        
        readme += f"""
### Domains
"""
        for domain, count in stats["domain_distribution"].items():
            percentage = count / stats["total_samples"] * 100
            readme += f"- **{domain}**: {count} samples ({percentage:.1f}%)\n"
        
        readme += f"""
### Difficulty Levels
"""
        for difficulty, count in stats["difficulty_distribution"].items():
            percentage = count / stats["total_samples"] * 100
            readme += f"- **{difficulty}**: {count} samples ({percentage:.1f}%)\n"
        
        readme += f"""
## Prompt Statistics

- **Average Length**: {stats['average_prompt_length']:.0f} characters
- **Min Length**: {stats['prompt_length_stats']['min']} characters
- **Max Length**: {stats['prompt_length_stats']['max']} characters
- **Standard Deviation**: {stats['prompt_length_stats']['std']:.0f} characters

## Quality Assurance

Quality Level: **{dataset.metadata.quality_assurance.get('overall_quality', 'Unknown')}**

- Checks Passed: {len(dataset.metadata.quality_assurance.get('checks_passed', []))}
- Checks Failed: {len(dataset.metadata.quality_assurance.get('checks_failed', []))}
- Warnings: {len(dataset.metadata.quality_assurance.get('warnings', []))}

## Usage

This dataset is designed for evaluating causal reasoning capabilities in language models. Each sample contains:

- A structured prompt testing specific causal reasoning skills
- Ground truth information for automated evaluation
- Evaluation criteria with scoring weights
- Metadata for analysis and filtering

## Citation

```bibtex
@dataset{{causal_eval_dataset,
    title = {{{dataset.metadata.name}}},
    version = {{{dataset.metadata.version}}},
    authors = {{{', '.join(dataset.metadata.authors)}}},
    year = {{{datetime.now().year}}},
    url = {{https://github.com/your-org/causal-eval-bench}}
}}
```

## License

{dataset.metadata.license}
"""
        
        return readme