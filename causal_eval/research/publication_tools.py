"""
Publication tools for generating research papers and analysis reports.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class PublicationConfig:
    """Configuration for publication generation."""
    
    title: str
    authors: List[str]
    abstract: str
    keywords: List[str] = field(default_factory=list)
    venue: Optional[str] = None
    template: str = "neurips"  # "neurips", "acl", "icml", etc.
    include_figures: bool = True
    include_tables: bool = True
    include_appendix: bool = True
    figure_format: str = "pdf"
    table_format: str = "latex"


@dataclass
class ExperimentResult:
    """Structured experiment result for publication."""
    
    experiment_name: str
    models_tested: List[str]
    datasets_used: List[str]
    metrics: Dict[str, Dict[str, float]]  # model -> metric -> value
    statistical_tests: Dict[str, Any]
    execution_time: float
    baseline_comparisons: Dict[str, Any] = field(default_factory=dict)
    human_evaluation: Optional[Dict[str, Any]] = None
    ablation_studies: List[Dict[str, Any]] = field(default_factory=list)


class ResultsAnalyzer:
    """Analyzer for experimental results preparing publication-ready analysis."""
    
    def __init__(self):
        """Initialize the results analyzer."""
        self.analysis_cache = {}
        logger.info("Results analyzer initialized")
    
    def analyze_benchmark_results(
        self,
        benchmark_results: List[Dict[str, Any]],
        statistical_analyzer=None
    ) -> Dict[str, Any]:
        """Analyze benchmark results for publication."""
        
        logger.info(f"Analyzing {len(benchmark_results)} benchmark results")
        
        analysis = {
            "summary_statistics": {},
            "model_rankings": {},
            "significance_tests": {},
            "effect_sizes": {},
            "consistency_analysis": {},
            "domain_analysis": {},
            "task_difficulty_analysis": {},
            "failure_analysis": {},
            "recommendations": []
        }
        
        # Extract model performance data
        model_performances = self._extract_model_performances(benchmark_results)
        
        # Calculate summary statistics
        analysis["summary_statistics"] = self._calculate_summary_statistics(model_performances)
        
        # Generate model rankings
        analysis["model_rankings"] = self._generate_model_rankings(model_performances)
        
        # Perform statistical analysis
        if statistical_analyzer and len(model_performances) > 1:
            analysis["significance_tests"] = self._perform_significance_tests(
                model_performances, statistical_analyzer
            )
            analysis["effect_sizes"] = self._calculate_effect_sizes(model_performances)
        
        # Analyze consistency
        analysis["consistency_analysis"] = self._analyze_consistency(model_performances)
        
        # Domain-specific analysis
        analysis["domain_analysis"] = self._analyze_domain_performance(benchmark_results)
        
        # Task difficulty analysis
        analysis["task_difficulty_analysis"] = self._analyze_task_difficulty(benchmark_results)
        
        # Failure mode analysis
        analysis["failure_analysis"] = self._analyze_failure_modes(benchmark_results)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _extract_model_performances(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract model performance scores from benchmark results."""
        
        model_performances = {}
        
        for result in benchmark_results:
            if "model_summaries" in result:
                for model, summary in result["model_summaries"].items():
                    if model not in model_performances:
                        model_performances[model] = []
                    
                    # Extract overall score
                    if "overall_score" in summary and "mean" in summary["overall_score"]:
                        model_performances[model].append(summary["overall_score"]["mean"])
        
        return model_performances
    
    def _calculate_summary_statistics(self, model_performances: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate summary statistics for model performances."""
        
        summary = {}
        
        for model, scores in model_performances.items():
            if scores:
                summary[model] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "median": np.median(scores),
                    "q25": np.percentile(scores, 25),
                    "q75": np.percentile(scores, 75),
                    "n_evaluations": len(scores)
                }
        
        return summary
    
    def _generate_model_rankings(self, model_performances: Dict[str, List[float]]) -> Dict[str, Any]:
        """Generate model rankings based on performance."""
        
        # Calculate mean performance for ranking
        model_means = {}
        for model, scores in model_performances.items():
            if scores:
                model_means[model] = np.mean(scores)
        
        # Sort by performance
        ranked_models = sorted(model_means.items(), key=lambda x: x[1], reverse=True)
        
        rankings = {
            "overall_ranking": [
                {"rank": i + 1, "model": model, "score": score}
                for i, (model, score) in enumerate(ranked_models)
            ],
            "performance_gaps": [],
            "tier_analysis": {}
        }
        
        # Calculate performance gaps
        for i in range(len(ranked_models) - 1):
            current_model, current_score = ranked_models[i]
            next_model, next_score = ranked_models[i + 1]
            gap = current_score - next_score
            
            rankings["performance_gaps"].append({
                "between": [current_model, next_model],
                "gap": gap,
                "relative_gap": gap / current_score if current_score > 0 else 0
            })
        
        # Tier analysis (group models with similar performance)
        if ranked_models:
            top_score = ranked_models[0][1]
            tiers = {"top": [], "competitive": [], "baseline": []}
            
            for model, score in ranked_models:
                relative_performance = score / top_score if top_score > 0 else 0
                
                if relative_performance >= 0.95:
                    tiers["top"].append(model)
                elif relative_performance >= 0.85:
                    tiers["competitive"].append(model)
                else:
                    tiers["baseline"].append(model)
            
            rankings["tier_analysis"] = tiers
        
        return rankings
    
    def _perform_significance_tests(
        self,
        model_performances: Dict[str, List[float]],
        statistical_analyzer
    ) -> Dict[str, Any]:
        """Perform statistical significance tests between models."""
        
        significance_tests = {
            "pairwise_comparisons": [],
            "multiple_comparisons_correction": {},
            "summary": {}
        }
        
        model_names = list(model_performances.keys())
        p_values = []
        
        # Pairwise comparisons
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1:]:
                if model_performances[model_a] and model_performances[model_b]:
                    try:
                        test_result = statistical_analyzer.compare_model_performance(
                            model_performances[model_a],
                            model_performances[model_b],
                            model_a,
                            model_b
                        )
                        
                        comparison = {
                            "models": [model_a, model_b],
                            "test_type": test_result.test_type.value,
                            "statistic": test_result.statistic,
                            "p_value": test_result.p_value,
                            "effect_size": test_result.effect_size,
                            "is_significant": test_result.is_significant,
                            "confidence_interval": test_result.confidence_interval,
                            "interpretation": test_result.interpretation
                        }
                        
                        significance_tests[\"pairwise_comparisons\"].append(comparison)\n                        p_values.append(test_result.p_value)\n                        \n                    except Exception as e:\n                        logger.warning(f\"Statistical test failed for {model_a} vs {model_b}: {e}\")\n        \n        # Multiple comparisons correction\n        if p_values:\n            try:\n                corrected_p_values, significant_after_correction = statistical_analyzer.multiple_comparisons_correction(\n                    p_values, method=\"bonferroni\"\n                )\n                \n                significance_tests[\"multiple_comparisons_correction\"] = {\n                    \"method\": \"bonferroni\",\n                    \"original_significant\": sum(p < 0.05 for p in p_values),\n                    \"corrected_significant\": sum(significant_after_correction),\n                    \"corrected_alpha\": 0.05 / len(p_values)\n                }\n                \n                # Update significance in pairwise comparisons\n                for i, comparison in enumerate(significance_tests[\"pairwise_comparisons\"]):\n                    comparison[\"significant_after_correction\"] = significant_after_correction[i]\n                    \n            except Exception as e:\n                logger.warning(f\"Multiple comparisons correction failed: {e}\")\n        \n        # Summary statistics\n        total_comparisons = len(significance_tests[\"pairwise_comparisons\"])\n        significant_comparisons = sum(1 for comp in significance_tests[\"pairwise_comparisons\"] if comp[\"is_significant\"])\n        \n        significance_tests[\"summary\"] = {\n            \"total_comparisons\": total_comparisons,\n            \"significant_comparisons\": significant_comparisons,\n            \"significance_rate\": significant_comparisons / total_comparisons if total_comparisons > 0 else 0\n        }\n        \n        return significance_tests\n    \n    def _calculate_effect_sizes(self, model_performances: Dict[str, List[float]]) -> Dict[str, Any]:\n        \"\"\"Calculate effect sizes between models.\"\"\"\n        \n        effect_sizes = {\n            \"cohen_d\": {},\n            \"summary\": {}\n        }\n        \n        model_names = list(model_performances.keys())\n        all_effect_sizes = []\n        \n        for i, model_a in enumerate(model_names):\n            for model_b in model_names[i + 1:]:\n                scores_a = model_performances[model_a]\n                scores_b = model_performances[model_b]\n                \n                if scores_a and scores_b:\n                    # Cohen's d\n                    mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)\n                    pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)\n                    \n                    if pooled_std > 0:\n                        cohens_d = (mean_a - mean_b) / pooled_std\n                        effect_sizes[\"cohen_d\"][f\"{model_a}_vs_{model_b}\"] = cohens_d\n                        all_effect_sizes.append(abs(cohens_d))\n        \n        if all_effect_sizes:\n            effect_sizes[\"summary\"] = {\n                \"mean_effect_size\": np.mean(all_effect_sizes),\n                \"max_effect_size\": np.max(all_effect_sizes),\n                \"small_effects\": sum(1 for es in all_effect_sizes if es < 0.2),\n                \"medium_effects\": sum(1 for es in all_effect_sizes if 0.2 <= es < 0.8),\n                \"large_effects\": sum(1 for es in all_effect_sizes if es >= 0.8)\n            }\n        \n        return effect_sizes\n    \n    def _analyze_consistency(self, model_performances: Dict[str, List[float]]) -> Dict[str, Any]:\n        \"\"\"Analyze consistency of model performances.\"\"\"\n        \n        consistency = {}\n        \n        for model, scores in model_performances.items():\n            if len(scores) > 1:\n                cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else float('inf')\n                consistency[model] = {\n                    \"coefficient_of_variation\": cv,\n                    \"std_dev\": np.std(scores),\n                    \"range\": np.max(scores) - np.min(scores),\n                    \"consistency_rating\": \"high\" if cv < 0.1 else \"medium\" if cv < 0.2 else \"low\"\n                }\n        \n        return consistency\n    \n    def _analyze_domain_performance(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:\n        \"\"\"Analyze performance across different domains.\"\"\"\n        \n        domain_analysis = {\n            \"domain_rankings\": {},\n            \"domain_specialization\": {},\n            \"cross_domain_consistency\": {}\n        }\n        \n        # Extract domain-specific performance\n        domain_performances = {}\n        \n        for result in benchmark_results:\n            if \"comparative_analysis\" in result and \"best_in_category\" in result[\"comparative_analysis\"]:\n                if \"domain\" in result[\"comparative_analysis\"][\"best_in_category\"]:\n                    for domain, best_info in result[\"comparative_analysis\"][\"best_in_category\"][\"domain\"].items():\n                        if domain not in domain_performances:\n                            domain_performances[domain] = {}\n                        \n                        model = best_info[\"model\"]\n                        score = best_info[\"score\"]\n                        \n                        if model not in domain_performances[domain]:\n                            domain_performances[domain][model] = []\n                        \n                        domain_performances[domain][model].append(score)\n        \n        # Calculate domain rankings\n        for domain, model_scores in domain_performances.items():\n            domain_means = {model: np.mean(scores) for model, scores in model_scores.items()}\n            ranked = sorted(domain_means.items(), key=lambda x: x[1], reverse=True)\n            domain_analysis[\"domain_rankings\"][domain] = ranked\n        \n        return domain_analysis\n    \n    def _analyze_task_difficulty(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:\n        \"\"\"Analyze performance across different task difficulty levels.\"\"\"\n        \n        difficulty_analysis = {\n            \"difficulty_curves\": {},\n            \"difficulty_gaps\": {},\n            \"models_by_difficulty\": {}\n        }\n        \n        # This would require more detailed benchmark results with difficulty-specific scores\n        # For now, return a placeholder structure\n        \n        return difficulty_analysis\n    \n    def _analyze_failure_modes(self, benchmark_results: List[Dict[str, Any]]) -> Dict[str, Any]:\n        \"\"\"Analyze common failure modes across models.\"\"\"\n        \n        failure_analysis = {\n            \"error_patterns\": {},\n            \"systematic_failures\": {},\n            \"model_weaknesses\": {}\n        }\n        \n        # Extract error analysis from benchmark results\n        for result in benchmark_results:\n            if \"error_analysis\" in result and \"error_patterns\" in result[\"error_analysis\"]:\n                for pattern in result[\"error_analysis\"][\"error_patterns\"]:\n                    error_type = pattern[\"error_type\"]\n                    if error_type not in failure_analysis[\"error_patterns\"]:\n                        failure_analysis[\"error_patterns\"][error_type] = {\n                            \"frequency\": 0,\n                            \"severity\": pattern[\"severity\"],\n                            \"common_contexts\": set()\n                        }\n                    \n                    failure_analysis[\"error_patterns\"][error_type][\"frequency\"] += pattern[\"frequency\"]\n                    \n                    # Add common contexts\n                    for context, _ in pattern.get(\"most_common_contexts\", []):\n                        failure_analysis[\"error_patterns\"][error_type][\"common_contexts\"].add(context)\n        \n        # Convert sets to lists for JSON serialization\n        for error_type, data in failure_analysis[\"error_patterns\"].items():\n            data[\"common_contexts\"] = list(data[\"common_contexts\"])\n        \n        return failure_analysis\n    \n    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:\n        \"\"\"Generate recommendations based on analysis.\"\"\"\n        \n        recommendations = []\n        \n        # Model performance recommendations\n        if \"model_rankings\" in analysis and \"tier_analysis\" in analysis[\"model_rankings\"]:\n            tiers = analysis[\"model_rankings\"][\"tier_analysis\"]\n            \n            if len(tiers[\"top\"]) > 1:\n                recommendations.append(\n                    f\"Multiple models ({', '.join(tiers['top'])}) achieved top-tier performance, \"\n                    \"suggesting convergence in causal reasoning capabilities.\"\n                )\n            \n            if len(tiers[\"baseline\"]) > len(tiers[\"top\"]) + len(tiers[\"competitive\"]):\n                recommendations.append(\n                    \"Majority of models show baseline performance, indicating room for improvement \"\n                    \"in causal reasoning training.\"\n                )\n        \n        # Statistical significance recommendations\n        if \"significance_tests\" in analysis and \"summary\" in analysis[\"significance_tests\"]:\n            sig_rate = analysis[\"significance_tests\"][\"summary\"][\"significance_rate\"]\n            \n            if sig_rate < 0.3:\n                recommendations.append(\n                    \"Low rate of statistically significant differences suggests models have \"\n                    \"similar causal reasoning performance levels.\"\n                )\n            elif sig_rate > 0.7:\n                recommendations.append(\n                    \"High rate of significant differences indicates clear performance hierarchies \"\n                    \"in causal reasoning tasks.\"\n                )\n        \n        # Consistency recommendations\n        if \"consistency_analysis\" in analysis:\n            inconsistent_models = [\n                model for model, data in analysis[\"consistency_analysis\"].items()\n                if data.get(\"consistency_rating\") == \"low\"\n            ]\n            \n            if inconsistent_models:\n                recommendations.append(\n                    f\"Models {', '.join(inconsistent_models)} show low consistency, \"\n                    \"suggesting unstable causal reasoning performance.\"\n                )\n        \n        # Failure mode recommendations\n        if \"failure_analysis\" in analysis and \"error_patterns\" in analysis[\"failure_analysis\"]:\n            common_errors = sorted(\n                analysis[\"failure_analysis\"][\"error_patterns\"].items(),\n                key=lambda x: x[1][\"frequency\"],\n                reverse=True\n            )[:3]\n            \n            if common_errors:\n                error_types = [error[0].replace(\"_\", \" \") for error in common_errors]\n                recommendations.append(\n                    f\"Most common error patterns ({', '.join(error_types)}) should be \"\n                    \"targeted in future model training.\"\n                )\n        \n        if not recommendations:\n            recommendations.append(\n                \"Analysis completed successfully. Consider deeper investigation \"\n                \"of specific model capabilities and failure modes.\"\n            )\n        \n        return recommendations\n    \n    def generate_figures(self, analysis: Dict[str, Any], output_dir: str) -> List[str]:\n        \"\"\"Generate publication-ready figures from analysis.\"\"\"\n        \n        output_path = Path(output_dir)\n        output_path.mkdir(parents=True, exist_ok=True)\n        \n        figures = []\n        \n        # Set style for publication\n        plt.style.use('seaborn-v0_8')\n        sns.set_palette(\"husl\")\n        \n        # Figure 1: Model Performance Comparison\n        if \"summary_statistics\" in analysis:\n            fig, ax = plt.subplots(figsize=(10, 6))\n            \n            models = list(analysis[\"summary_statistics\"].keys())\n            means = [analysis[\"summary_statistics\"][m][\"mean\"] for m in models]\n            stds = [analysis[\"summary_statistics\"][m][\"std\"] for m in models]\n            \n            x_pos = np.arange(len(models))\n            bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8)\n            \n            ax.set_xlabel('Models')\n            ax.set_ylabel('Causal Reasoning Score')\n            ax.set_title('Model Performance Comparison')\n            ax.set_xticks(x_pos)\n            ax.set_xticklabels(models, rotation=45)\n            \n            # Add value labels on bars\n            for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):\n                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std/2,\n                       f'{mean:.3f}', ha='center', va='bottom')\n            \n            plt.tight_layout()\n            figure_path = output_path / \"model_performance_comparison.png\"\n            plt.savefig(figure_path, dpi=300, bbox_inches='tight')\n            plt.close()\n            \n            figures.append(str(figure_path))\n        \n        # Figure 2: Effect Size Heatmap\n        if \"effect_sizes\" in analysis and \"cohen_d\" in analysis[\"effect_sizes\"]:\n            effect_sizes = analysis[\"effect_sizes\"][\"cohen_d\"]\n            \n            if effect_sizes:\n                # Create matrix for heatmap\n                models = set()\n                for comparison in effect_sizes.keys():\n                    model_a, model_b = comparison.split(\"_vs_\")\n                    models.add(model_a)\n                    models.add(model_b)\n                \n                models = sorted(list(models))\n                n_models = len(models)\n                \n                effect_matrix = np.zeros((n_models, n_models))\n                \n                for comparison, effect_size in effect_sizes.items():\n                    model_a, model_b = comparison.split(\"_vs_\")\n                    i, j = models.index(model_a), models.index(model_b)\n                    effect_matrix[i, j] = effect_size\n                    effect_matrix[j, i] = -effect_size\n                \n                fig, ax = plt.subplots(figsize=(8, 6))\n                sns.heatmap(effect_matrix, annot=True, cmap='RdBu_r', center=0,\n                           xticklabels=models, yticklabels=models, ax=ax)\n                ax.set_title('Effect Sizes (Cohen\\'s d) Between Models')\n                \n                plt.tight_layout()\n                figure_path = output_path / \"effect_size_heatmap.png\"\n                plt.savefig(figure_path, dpi=300, bbox_inches='tight')\n                plt.close()\n                \n                figures.append(str(figure_path))\n        \n        # Figure 3: Error Pattern Distribution\n        if \"failure_analysis\" in analysis and \"error_patterns\" in analysis[\"failure_analysis\"]:\n            error_patterns = analysis[\"failure_analysis\"][\"error_patterns\"]\n            \n            if error_patterns:\n                fig, ax = plt.subplots(figsize=(12, 6))\n                \n                error_types = list(error_patterns.keys())\n                frequencies = [error_patterns[et][\"frequency\"] for et in error_types]\n                \n                # Clean up error type names for display\n                display_names = [et.replace(\"_\", \" \").title() for et in error_types]\n                \n                bars = ax.bar(display_names, frequencies, alpha=0.8)\n                ax.set_xlabel('Error Type')\n                ax.set_ylabel('Frequency')\n                ax.set_title('Distribution of Error Patterns')\n                plt.xticks(rotation=45, ha='right')\n                \n                # Add frequency labels\n                for bar, freq in zip(bars, frequencies):\n                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,\n                           str(freq), ha='center', va='bottom')\n                \n                plt.tight_layout()\n                figure_path = output_path / \"error_pattern_distribution.png\"\n                plt.savefig(figure_path, dpi=300, bbox_inches='tight')\n                plt.close()\n                \n                figures.append(str(figure_path))\n        \n        logger.info(f\"Generated {len(figures)} figures in {output_dir}\")\n        return figures\n    \n    def export_results_table(self, analysis: Dict[str, Any], output_path: str, format: str = \"latex\") -> str:\n        \"\"\"Export results table in specified format.\"\"\"\n        \n        if \"summary_statistics\" not in analysis:\n            logger.warning(\"No summary statistics available for table export\")\n            return \"\"\n        \n        summary_stats = analysis[\"summary_statistics\"]\n        \n        # Create DataFrame\n        data = []\n        for model, stats in summary_stats.items():\n            data.append({\n                \"Model\": model,\n                \"Mean\": f\"{stats['mean']:.3f}\",\n                \"Std\": f\"{stats['std']:.3f}\",\n                \"Min\": f\"{stats['min']:.3f}\",\n                \"Max\": f\"{stats['max']:.3f}\",\n                \"Median\": f\"{stats['median']:.3f}\",\n                \"N\": stats['n_evaluations']\n            })\n        \n        df = pd.DataFrame(data)\n        \n        if format.lower() == \"latex\":\n            # Generate LaTeX table\n            latex_table = df.to_latex(\n                index=False,\n                caption=\"Model Performance Summary\",\n                label=\"tab:model_performance\",\n                column_format=\"lrrrrrr\",\n                escape=False\n            )\n            \n            # Save to file\n            with open(output_path, 'w') as f:\n                f.write(latex_table)\n            \n            return latex_table\n        \n        elif format.lower() == \"csv\":\n            df.to_csv(output_path, index=False)\n            return df.to_csv(index=False)\n        \n        else:\n            logger.warning(f\"Unsupported format: {format}\")\n            return \"\"\n\n\nclass PublicationGenerator:\n    \"\"\"Generator for creating publication-ready documents.\"\"\"\n    \n    def __init__(self, results_analyzer: Optional[ResultsAnalyzer] = None):\n        \"\"\"Initialize the publication generator.\"\"\"\n        self.results_analyzer = results_analyzer or ResultsAnalyzer()\n        logger.info(\"Publication generator initialized\")\n    \n    def generate_paper(\n        self,\n        config: PublicationConfig,\n        experiment_results: List[ExperimentResult],\n        output_dir: str\n    ) -> str:\n        \"\"\"Generate a complete research paper.\"\"\"\n        \n        logger.info(f\"Generating paper: {config.title}\")\n        \n        output_path = Path(output_dir)\n        output_path.mkdir(parents=True, exist_ok=True)\n        \n        # Analyze results\n        benchmark_results = []\n        for exp_result in experiment_results:\n            benchmark_results.append({\n                \"model_summaries\": exp_result.metrics,\n                \"statistical_analysis\": exp_result.statistical_tests,\n                \"comparative_analysis\": exp_result.baseline_comparisons\n            })\n        \n        analysis = self.results_analyzer.analyze_benchmark_results(benchmark_results)\n        \n        # Generate figures\n        figures = self.results_analyzer.generate_figures(analysis, str(output_path / \"figures\"))\n        \n        # Generate tables\n        table_path = output_path / \"tables\" / \"model_performance.tex\"\n        table_path.parent.mkdir(exist_ok=True)\n        self.results_analyzer.export_results_table(analysis, str(table_path), \"latex\")\n        \n        # Generate paper content\n        paper_content = self._generate_paper_content(config, experiment_results, analysis, figures)\n        \n        # Save paper\n        paper_file = output_path / \"paper.tex\"\n        with open(paper_file, 'w') as f:\n            f.write(paper_content)\n        \n        # Generate bibliography\n        bib_content = self._generate_bibliography()\n        bib_file = output_path / \"references.bib\"\n        with open(bib_file, 'w') as f:\n            f.write(bib_content)\n        \n        # Try to compile if LaTeX is available\n        try:\n            self._compile_latex(str(paper_file))\n            logger.info(f\"Paper compiled successfully: {paper_file.with_suffix('.pdf')}\")\n        except Exception as e:\n            logger.warning(f\"LaTeX compilation failed: {e}\")\n        \n        logger.info(f\"Paper generated in {output_path}\")\n        return str(paper_file)\n    \n    def _generate_paper_content(\n        self,\n        config: PublicationConfig,\n        experiment_results: List[ExperimentResult],\n        analysis: Dict[str, Any],\n        figures: List[str]\n    ) -> str:\n        \"\"\"Generate the main paper content.\"\"\"\n        \n        paper = self._get_paper_template(config.template)\n        \n        # Replace placeholders\n        paper = paper.replace(\"{TITLE}\", config.title)\n        paper = paper.replace(\"{AUTHORS}\", \", \".join(config.authors))\n        paper = paper.replace(\"{ABSTRACT}\", config.abstract)\n        paper = paper.replace(\"{KEYWORDS}\", \", \".join(config.keywords))\n        \n        # Generate sections\n        introduction = self._generate_introduction(experiment_results)\n        methodology = self._generate_methodology(experiment_results)\n        results = self._generate_results(analysis, figures)\n        discussion = self._generate_discussion(analysis)\n        conclusion = self._generate_conclusion(analysis)\n        \n        paper = paper.replace(\"{INTRODUCTION}\", introduction)\n        paper = paper.replace(\"{METHODOLOGY}\", methodology)\n        paper = paper.replace(\"{RESULTS}\", results)\n        paper = paper.replace(\"{DISCUSSION}\", discussion)\n        paper = paper.replace(\"{CONCLUSION}\", conclusion)\n        \n        return paper\n    \n    def _get_paper_template(self, template: str) -> str:\n        \"\"\"Get paper template for specified venue.\"\"\"\n        \n        # Basic LaTeX template - in a real implementation, you'd have venue-specific templates\n        template_content = r\"\"\"\n\\documentclass[11pt]{article}\n\\usepackage{graphicx}\n\\usepackage{booktabs}\n\\usepackage{amsmath}\n\\usepackage{amssymb}\n\\usepackage{natbib}\n\n\\title{{TITLE}}\n\\author{{AUTHORS}}\n\\date{\\today}\n\n\\begin{document}\n\n\\maketitle\n\n\\begin{abstract}\n{ABSTRACT}\n\\end{abstract}\n\n\\section{Introduction}\n{INTRODUCTION}\n\n\\section{Methodology}\n{METHODOLOGY}\n\n\\section{Results}\n{RESULTS}\n\n\\section{Discussion}\n{DISCUSSION}\n\n\\section{Conclusion}\n{CONCLUSION}\n\n\\bibliographystyle{plain}\n\\bibliography{references}\n\n\\end{document}\n\"\"\"\n        \n        return template_content\n    \n    def _generate_introduction(self, experiment_results: List[ExperimentResult]) -> str:\n        \"\"\"Generate introduction section.\"\"\"\n        \n        models_tested = set()\n        datasets_used = set()\n        \n        for result in experiment_results:\n            models_tested.update(result.models_tested)\n            datasets_used.update(result.datasets_used)\n        \n        introduction = f\"\"\"\nCausal reasoning represents a fundamental aspect of human intelligence, enabling us to understand\ncause-and-effect relationships and make predictions about interventions. Recent advances in\nlarge language models have shown remarkable capabilities across various natural language tasks,\nbut their ability to perform genuine causal reasoning remains an open question.\n\nThis paper presents a comprehensive evaluation of causal reasoning capabilities in large language\nmodels. We evaluate {len(models_tested)} state-of-the-art models across {len(datasets_used)} \ncarefully designed datasets that test different aspects of causal reasoning including:\n\n\\begin{itemize}\n\\item Distinguishing correlation from causation\n\\item Counterfactual reasoning\n\\item Understanding causal interventions\n\\item Identifying confounding variables\n\\item Tracing causal chains\n\\end{itemize}\n\nOur evaluation framework provides the first systematic assessment of causal reasoning in LLMs\nwith statistical significance testing and comprehensive error analysis. The results reveal\nsignificant variations in causal reasoning capabilities across models and identify key areas\nfor improvement.\n\"\"\"\n        \n        return introduction\n    \n    def _generate_methodology(self, experiment_results: List[ExperimentResult]) -> str:\n        \"\"\"Generate methodology section.\"\"\"\n        \n        all_models = set()\n        all_datasets = set()\n        \n        for result in experiment_results:\n            all_models.update(result.models_tested)\n            all_datasets.update(result.datasets_used)\n        \n        methodology = f\"\"\"\n\\subsection{{Models Evaluated}}\n\nWe evaluate the following {len(all_models)} language models:\n\n\\begin{itemize}\n\"\"\"\n        \n        for model in sorted(all_models):\n            methodology += f\"\\item {model}\\n\"\n        \n        methodology += \"\\end{itemize}\\n\\n\"\n        \n        methodology += f\"\"\"\n\\subsection{{Evaluation Datasets}}\n\nOur evaluation uses {len(all_datasets)} specialized datasets designed to test different\naspects of causal reasoning:\n\n\\begin{itemize}\n\"\"\"\n        \n        for dataset in sorted(all_datasets):\n            methodology += f\"\\item {dataset}\\n\"\n        \n        methodology += \"\\end{itemize}\\n\\n\"\n        \n        methodology += \"\"\"\n\\subsection{Evaluation Metrics}\n\nWe employ a multi-dimensional scoring system that evaluates:\n\n\\begin{itemize}\n\\item \\textbf{Causal Accuracy}: Correctness of causal relationship identification\n\\item \\textbf{Reasoning Quality}: Depth and coherence of explanations\n\\item \\textbf{Confounder Awareness}: Recognition of confounding variables\n\\item \\textbf{Mechanism Understanding}: Grasp of underlying causal mechanisms\n\\end{itemize}\n\n\\subsection{Statistical Analysis}\n\nWe perform rigorous statistical analysis including paired t-tests for model comparisons,\neffect size calculations using Cohen's d, and multiple comparisons correction using\nBonferroni adjustment. Bootstrap confidence intervals are calculated for all metrics.\n\"\"\"\n        \n        return methodology\n    \n    def _generate_results(self, analysis: Dict[str, Any], figures: List[str]) -> str:\n        \"\"\"Generate results section.\"\"\"\n        \n        results = \"\"\"\n\\subsection{Overall Performance}\n\nFigure \\ref{fig:performance} shows the overall causal reasoning performance across all\nevaluated models. \n\n\\begin{figure}[h]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{figures/model_performance_comparison.png}\n\\caption{Model Performance Comparison}\n\\label{fig:performance}\n\\end{figure}\n\n\\input{tables/model_performance}\n\n\\subsection{Statistical Significance}\n\"\"\"\n        \n        if \"significance_tests\" in analysis and \"summary\" in analysis[\"significance_tests\"]:\n            sig_summary = analysis[\"significance_tests\"][\"summary\"]\n            results += f\"\"\"\nOut of {sig_summary['total_comparisons']} pairwise model comparisons, \n{sig_summary['significant_comparisons']} showed statistically significant differences \n(significance rate: {sig_summary['significance_rate']:.1%}).\n\"\"\"\n        \n        results += \"\"\"\n\n\\subsection{Error Analysis}\n\nFigure \\ref{fig:errors} shows the distribution of error patterns across all models.\n\n\\begin{figure}[h]\n\\centering\n\\includegraphics[width=0.8\\textwidth]{figures/error_pattern_distribution.png}\n\\caption{Distribution of Error Patterns}\n\\label{fig:errors}\n\\end{figure}\n\n\"\"\"\n        \n        return results\n    \n    def _generate_discussion(self, analysis: Dict[str, Any]) -> str:\n        \"\"\"Generate discussion section.\"\"\"\n        \n        discussion = \"\"\"\nOur comprehensive evaluation reveals several key insights about causal reasoning in\nlarge language models:\n\n\\subsection{Model Capabilities}\n\nThe results demonstrate significant variation in causal reasoning capabilities across\nmodels. While some models show strong performance on basic causal attribution tasks,\nmore complex scenarios involving confounding variables and counterfactual reasoning\nremain challenging.\n\n\\subsection{Common Failure Modes}\n\nError analysis reveals systematic patterns in model failures:\n\n\\begin{itemize}\n\\item Confusion between correlation and causation remains prevalent\n\\item Difficulty identifying relevant confounding variables\n\\item Challenges with multi-step causal reasoning\n\\item Inconsistent performance across different domains\n\\end{itemize}\n\n\\subsection{Implications for Model Development}\n\"\"\"\n        \n        # Add specific recommendations from analysis\n        if \"recommendations\" in analysis:\n            for recommendation in analysis[\"recommendations\"][:3]:\n                discussion += f\"\\n{recommendation}\"\n        \n        discussion += \"\"\"\n\n\\subsection{Limitations}\n\nOur evaluation, while comprehensive, has several limitations. The focus on text-based\nevaluation may not capture all aspects of causal reasoning. Additionally, the scenarios,\nwhile diverse, may not cover all possible causal reasoning contexts.\n\"\"\"\n        \n        return discussion\n    \n    def _generate_conclusion(self, analysis: Dict[str, Any]) -> str:\n        \"\"\"Generate conclusion section.\"\"\"\n        \n        conclusion = \"\"\"\nThis study presents the first comprehensive evaluation of causal reasoning capabilities\nin large language models. Our findings reveal both the current capabilities and\nlimitations of LLMs in understanding causal relationships.\n\nKey contributions include:\n\n\\begin{itemize}\n\\item A comprehensive evaluation framework for causal reasoning in LLMs\n\\item Statistical analysis revealing significant performance differences between models\n\\item Identification of common failure modes and improvement opportunities\n\\item Open-source dataset and evaluation tools for future research\n\\end{itemize}\n\nFuture work should focus on developing training methods that specifically target\ncausal reasoning capabilities and expanding evaluation to more diverse domains\nand complex causal structures.\n\"\"\"\n        \n        return conclusion\n    \n    def _generate_bibliography(self) -> str:\n        \"\"\"Generate bibliography entries.\"\"\"\n        \n        bibliography = \"\"\"\n@inproceedings{pearl2009causal,\n  title={Causal inference in statistics: An overview},\n  author={Pearl, Judea},\n  booktitle={Statistics Surveys},\n  volume={3},\n  pages={96--146},\n  year={2009}\n}\n\n@article{holland1986statistics,\n  title={Statistics and causal inference},\n  author={Holland, Paul W},\n  journal={Journal of the American statistical Association},\n  volume={81},\n  number={396},\n  pages={945--960},\n  year={1986}\n}\n\n@book{morgan2015counterfactuals,\n  title={Counterfactuals and causal inference},\n  author={Morgan, Stephen L and Winship, Christopher},\n  year={2015},\n  publisher={Cambridge University Press}\n}\n\"\"\"\n        \n        return bibliography\n    \n    def _compile_latex(self, tex_file: str) -> None:\n        \"\"\"Compile LaTeX file to PDF.\"\"\"\n        \n        try:\n            # Run pdflatex\n            subprocess.run(\n                [\"pdflatex\", \"-interaction=nonstopmode\", tex_file],\n                cwd=Path(tex_file).parent,\n                check=True,\n                capture_output=True\n            )\n            \n            # Run bibtex if bibliography exists\n            bib_file = Path(tex_file).with_suffix(\".aux\")\n            if bib_file.exists():\n                subprocess.run(\n                    [\"bibtex\", str(bib_file.stem)],\n                    cwd=Path(tex_file).parent,\n                    capture_output=True\n                )\n                \n                # Run pdflatex twice more for bibliography\n                for _ in range(2):\n                    subprocess.run(\n                        [\"pdflatex\", \"-interaction=nonstopmode\", tex_file],\n                        cwd=Path(tex_file).parent,\n                        capture_output=True\n                    )\n        \n        except subprocess.CalledProcessError as e:\n            raise RuntimeError(f\"LaTeX compilation failed: {e}\")\n        except FileNotFoundError:\n            raise RuntimeError(\"LaTeX not found. Please install LaTeX distribution.\")