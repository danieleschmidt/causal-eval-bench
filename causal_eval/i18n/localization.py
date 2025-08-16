"""
Internationalization and Localization Support

This module provides comprehensive i18n support for the causal evaluation framework,
enabling global deployment with multi-language and multi-cultural support.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for the framework."""
    
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HINDI = "hi"


@dataclass
class CulturalContext:
    """Cultural context for causal reasoning evaluation."""
    
    language: SupportedLanguage
    region: str
    cultural_values: Dict[str, Any]
    causal_reasoning_patterns: Dict[str, Any]
    communication_style: str  # "direct", "indirect", "high_context", "low_context"
    temporal_orientation: str  # "linear", "cyclical", "event_based"
    authority_distance: str   # "high", "medium", "low"


class LocalizationManager:
    """Manages localization and cultural adaptation."""
    
    def __init__(self, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.default_language = default_language
        self.current_language = default_language
        self.translations = {}
        self.cultural_contexts = {}
        
        # Load translations and cultural contexts
        self._load_translations()
        self._load_cultural_contexts()
        
        logger.info(f"Localization manager initialized with default language: {default_language.value}")
    
    def set_language(self, language: SupportedLanguage) -> None:
        """Set the current language."""
        self.current_language = language
        logger.info(f"Language set to: {language.value}")
    
    def get_text(self, key: str, **kwargs) -> str:
        """Get localized text for a given key."""
        lang_code = self.current_language.value
        
        # Try to get translation for current language
        if lang_code in self.translations and key in self.translations[lang_code]:
            text = self.translations[lang_code][key]
        # Fallback to default language
        elif self.default_language.value in self.translations and key in self.translations[self.default_language.value]:
            text = self.translations[self.default_language.value][key]
        # Fallback to key itself
        else:
            text = key
            logger.warning(f"No translation found for key: {key} in language: {lang_code}")
        
        # Format with provided arguments
        try:
            return text.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing format argument for key {key}: {e}")
            return text
    
    def get_cultural_context(self, language: Optional[SupportedLanguage] = None) -> CulturalContext:
        """Get cultural context for a language."""
        lang = language or self.current_language
        return self.cultural_contexts.get(lang, self._get_default_cultural_context())
    
    def adapt_causal_scenario(self, scenario: Dict[str, Any], target_language: Optional[SupportedLanguage] = None) -> Dict[str, Any]:
        """Adapt a causal scenario for cultural context."""
        lang = target_language or self.current_language
        cultural_context = self.get_cultural_context(lang)
        
        adapted_scenario = scenario.copy()
        
        # Adapt based on cultural reasoning patterns
        if cultural_context.causal_reasoning_patterns.get("prefer_concrete_examples", False):
            adapted_scenario = self._add_concrete_examples(adapted_scenario)
        
        if cultural_context.communication_style in ["indirect", "high_context"]:
            adapted_scenario = self._adapt_for_indirect_communication(adapted_scenario)
        
        # Adapt temporal references
        if cultural_context.temporal_orientation == "cyclical":
            adapted_scenario = self._adapt_temporal_references(adapted_scenario)
        
        return adapted_scenario
    
    def _load_translations(self) -> None:
        """Load translation files."""
        
        # Base translations for core framework messages
        self.translations = {
            "en": {
                "evaluation_complete": "Evaluation completed successfully",
                "model_performance": "Model Performance",
                "causal_attribution": "Causal Attribution",
                "counterfactual_reasoning": "Counterfactual Reasoning",
                "causal_intervention": "Causal Intervention",
                "statistical_significance": "Statistical Significance",
                "confidence_interval": "Confidence Interval",
                "effect_size": "Effect Size",
                "baseline_comparison": "Baseline Comparison",
                "error_analysis": "Error Analysis",
                "research_recommendations": "Research Recommendations",
                "data_privacy_notice": "This evaluation respects data privacy and follows {compliance_standard} guidelines",
                "cultural_adaptation_notice": "This evaluation has been adapted for {culture} cultural context",
                "scenario_introduction": "Please analyze the following scenario and determine the causal relationships:",
                "explanation_request": "Please provide your reasoning and explanation:",
                "confidence_request": "How confident are you in your answer? (0-100%)",
                "invalid_response": "Invalid response format. Please follow the instructions.",
                "processing_error": "An error occurred during processing. Please try again.",
                "results_summary": "Summary of Results",
                "performance_metrics": "Performance Metrics",
                "comparative_analysis": "Comparative Analysis"
            },
            "es": {
                "evaluation_complete": "Evaluación completada exitosamente",
                "model_performance": "Rendimiento del Modelo",
                "causal_attribution": "Atribución Causal",
                "counterfactual_reasoning": "Razonamiento Contrafáctico",
                "causal_intervention": "Intervención Causal",
                "statistical_significance": "Significancia Estadística",
                "confidence_interval": "Intervalo de Confianza",
                "effect_size": "Tamaño del Efecto",
                "baseline_comparison": "Comparación de Referencia",
                "error_analysis": "Análisis de Errores",
                "research_recommendations": "Recomendaciones de Investigación",
                "data_privacy_notice": "Esta evaluación respeta la privacidad de datos y sigue las pautas de {compliance_standard}",
                "cultural_adaptation_notice": "Esta evaluación ha sido adaptada para el contexto cultural {culture}",
                "scenario_introduction": "Por favor analice el siguiente escenario y determine las relaciones causales:",
                "explanation_request": "Por favor proporcione su razonamiento y explicación:",
                "confidence_request": "¿Qué tan seguro está de su respuesta? (0-100%)",
                "invalid_response": "Formato de respuesta inválido. Por favor siga las instrucciones.",
                "processing_error": "Ocurrió un error durante el procesamiento. Por favor intente de nuevo.",
                "results_summary": "Resumen de Resultados",
                "performance_metrics": "Métricas de Rendimiento",
                "comparative_analysis": "Análisis Comparativo"
            },
            "fr": {
                "evaluation_complete": "Évaluation terminée avec succès",
                "model_performance": "Performance du Modèle",
                "causal_attribution": "Attribution Causale",
                "counterfactual_reasoning": "Raisonnement Contrefactuel",
                "causal_intervention": "Intervention Causale",
                "statistical_significance": "Signification Statistique",
                "confidence_interval": "Intervalle de Confiance",
                "effect_size": "Taille de l'Effet",
                "baseline_comparison": "Comparaison de Référence",
                "error_analysis": "Analyse des Erreurs",
                "research_recommendations": "Recommandations de Recherche",
                "data_privacy_notice": "Cette évaluation respecte la confidentialité des données et suit les directives {compliance_standard}",
                "cultural_adaptation_notice": "Cette évaluation a été adaptée au contexte culturel {culture}",
                "scenario_introduction": "Veuillez analyser le scénario suivant et déterminer les relations causales:",
                "explanation_request": "Veuillez fournir votre raisonnement et explication:",
                "confidence_request": "À quel point êtes-vous confiant dans votre réponse? (0-100%)",
                "invalid_response": "Format de réponse invalide. Veuillez suivre les instructions.",
                "processing_error": "Une erreur s'est produite pendant le traitement. Veuillez réessayer.",
                "results_summary": "Résumé des Résultats",
                "performance_metrics": "Métriques de Performance",
                "comparative_analysis": "Analyse Comparative"
            },
            "de": {
                "evaluation_complete": "Bewertung erfolgreich abgeschlossen",
                "model_performance": "Modellleistung",
                "causal_attribution": "Kausale Zuordnung",
                "counterfactual_reasoning": "Kontrafaktisches Denken",
                "causal_intervention": "Kausale Intervention",
                "statistical_significance": "Statistische Signifikanz",
                "confidence_interval": "Konfidenzintervall",
                "effect_size": "Effektgröße",
                "baseline_comparison": "Baseline-Vergleich",
                "error_analysis": "Fehleranalyse",
                "research_recommendations": "Forschungsempfehlungen",
                "data_privacy_notice": "Diese Bewertung respektiert den Datenschutz und folgt den {compliance_standard} Richtlinien",
                "cultural_adaptation_notice": "Diese Bewertung wurde für den kulturellen Kontext {culture} angepasst",
                "scenario_introduction": "Bitte analysieren Sie das folgende Szenario und bestimmen Sie die kausalen Beziehungen:",
                "explanation_request": "Bitte geben Sie Ihre Begründung und Erklärung an:",
                "confidence_request": "Wie sicher sind Sie sich bei Ihrer Antwort? (0-100%)",
                "invalid_response": "Ungültiges Antwortformat. Bitte befolgen Sie die Anweisungen.",
                "processing_error": "Ein Fehler ist während der Verarbeitung aufgetreten. Bitte versuchen Sie es erneut.",
                "results_summary": "Zusammenfassung der Ergebnisse",
                "performance_metrics": "Leistungsmetriken",
                "comparative_analysis": "Vergleichsanalyse"
            },
            "ja": {
                "evaluation_complete": "評価が正常に完了しました",
                "model_performance": "モデルパフォーマンス",
                "causal_attribution": "因果帰属",
                "counterfactual_reasoning": "反実仮想推論",
                "causal_intervention": "因果介入",
                "statistical_significance": "統計的有意性",
                "confidence_interval": "信頼区間",
                "effect_size": "効果量",
                "baseline_comparison": "ベースライン比較",
                "error_analysis": "エラー分析",
                "research_recommendations": "研究推奨事項",
                "data_privacy_notice": "この評価はデータプライバシーを尊重し、{compliance_standard}ガイドラインに従います",
                "cultural_adaptation_notice": "この評価は{culture}の文化的文脈に適応されています",
                "scenario_introduction": "以下のシナリオを分析し、因果関係を判断してください：",
                "explanation_request": "推論と説明を提供してください：",
                "confidence_request": "あなたの答えにどの程度確信がありますか？（0-100%）",
                "invalid_response": "無効な応答形式です。指示に従ってください。",
                "processing_error": "処理中にエラーが発生しました。もう一度お試しください。",
                "results_summary": "結果の要約",
                "performance_metrics": "パフォーマンス指標",
                "comparative_analysis": "比較分析"
            },
            "zh-CN": {
                "evaluation_complete": "评估成功完成",
                "model_performance": "模型性能",
                "causal_attribution": "因果归因",
                "counterfactual_reasoning": "反事实推理",
                "causal_intervention": "因果干预",
                "statistical_significance": "统计显著性",
                "confidence_interval": "置信区间",
                "effect_size": "效应量",
                "baseline_comparison": "基线比较",
                "error_analysis": "错误分析",
                "research_recommendations": "研究建议",
                "data_privacy_notice": "此评估尊重数据隐私并遵循{compliance_standard}准则",
                "cultural_adaptation_notice": "此评估已适应{culture}文化背景",
                "scenario_introduction": "请分析以下情景并确定因果关系：",
                "explanation_request": "请提供您的推理和解释：",
                "confidence_request": "您对答案有多确信？（0-100%）",
                "invalid_response": "无效的响应格式。请遵循说明。",
                "processing_error": "处理过程中发生错误。请重试。",
                "results_summary": "结果摘要",
                "performance_metrics": "性能指标",
                "comparative_analysis": "比较分析"
            }
        }
    
    def _load_cultural_contexts(self) -> None:
        """Load cultural context information."""
        
        self.cultural_contexts = {
            SupportedLanguage.ENGLISH: CulturalContext(
                language=SupportedLanguage.ENGLISH,
                region="Western",
                cultural_values={
                    "individualism_score": 0.8,
                    "uncertainty_avoidance": 0.4,
                    "long_term_orientation": 0.6
                },
                causal_reasoning_patterns={
                    "prefer_linear_causation": True,
                    "analytical_approach": True,
                    "direct_questioning": True,
                    "prefer_concrete_examples": False
                },
                communication_style="direct",
                temporal_orientation="linear",
                authority_distance="medium"
            ),
            SupportedLanguage.SPANISH: CulturalContext(
                language=SupportedLanguage.SPANISH,
                region="Latin",
                cultural_values={
                    "individualism_score": 0.4,
                    "uncertainty_avoidance": 0.7,
                    "long_term_orientation": 0.4
                },
                causal_reasoning_patterns={
                    "prefer_linear_causation": True,
                    "analytical_approach": True,
                    "direct_questioning": True,
                    "prefer_concrete_examples": True
                },
                communication_style="direct",
                temporal_orientation="linear",
                authority_distance="high"
            ),
            SupportedLanguage.JAPANESE: CulturalContext(
                language=SupportedLanguage.JAPANESE,
                region="East Asian",
                cultural_values={
                    "individualism_score": 0.2,
                    "uncertainty_avoidance": 0.8,
                    "long_term_orientation": 0.9
                },
                causal_reasoning_patterns={
                    "prefer_linear_causation": False,
                    "analytical_approach": False,
                    "direct_questioning": False,
                    "prefer_concrete_examples": True,
                    "holistic_thinking": True,
                    "context_dependent": True
                },
                communication_style="indirect",
                temporal_orientation="cyclical",
                authority_distance="high"
            ),
            SupportedLanguage.CHINESE_SIMPLIFIED: CulturalContext(
                language=SupportedLanguage.CHINESE_SIMPLIFIED,
                region="East Asian",
                cultural_values={
                    "individualism_score": 0.2,
                    "uncertainty_avoidance": 0.6,
                    "long_term_orientation": 0.9
                },
                causal_reasoning_patterns={
                    "prefer_linear_causation": False,
                    "analytical_approach": False,
                    "direct_questioning": False,
                    "prefer_concrete_examples": True,
                    "holistic_thinking": True,
                    "dialectical_thinking": True
                },
                communication_style="high_context",
                temporal_orientation="cyclical",
                authority_distance="high"
            ),
            SupportedLanguage.GERMAN: CulturalContext(
                language=SupportedLanguage.GERMAN,
                region="Germanic",
                cultural_values={
                    "individualism_score": 0.7,
                    "uncertainty_avoidance": 0.8,
                    "long_term_orientation": 0.7
                },
                causal_reasoning_patterns={
                    "prefer_linear_causation": True,
                    "analytical_approach": True,
                    "direct_questioning": True,
                    "systematic_approach": True,
                    "detailed_analysis": True
                },
                communication_style="direct",
                temporal_orientation="linear",
                authority_distance="low"
            ),
            SupportedLanguage.FRENCH: CulturalContext(
                language=SupportedLanguage.FRENCH,
                region="Romance",
                cultural_values={
                    "individualism_score": 0.7,
                    "uncertainty_avoidance": 0.7,
                    "long_term_orientation": 0.6
                },
                causal_reasoning_patterns={
                    "prefer_linear_causation": True,
                    "analytical_approach": True,
                    "direct_questioning": True,
                    "intellectual_approach": True
                },
                communication_style="direct",
                temporal_orientation="linear",
                authority_distance="high"
            ),
            SupportedLanguage.ARABIC: CulturalContext(
                language=SupportedLanguage.ARABIC,
                region="Middle Eastern",
                cultural_values={
                    "individualism_score": 0.3,
                    "uncertainty_avoidance": 0.7,
                    "long_term_orientation": 0.4
                },
                causal_reasoning_patterns={
                    "prefer_linear_causation": True,
                    "analytical_approach": True,
                    "direct_questioning": False,
                    "context_dependent": True,
                    "authority_based": True
                },
                communication_style="high_context",
                temporal_orientation="event_based",
                authority_distance="high"
            )
        }
    
    def _get_default_cultural_context(self) -> CulturalContext:
        """Get default cultural context."""
        return self.cultural_contexts[SupportedLanguage.ENGLISH]
    
    def _add_concrete_examples(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Add concrete examples to abstract scenarios."""
        adapted = scenario.copy()
        
        if "description" in adapted:
            # Add specific examples to make abstract concepts more concrete
            description = adapted["description"]
            if "correlation" in description.lower() and "example" not in description.lower():
                adapted["description"] += "\n\nFor example, consider how both ice cream sales and swimming accidents increase in summer - this shows correlation but not causation."
        
        return adapted
    
    def _adapt_for_indirect_communication(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt scenario for indirect communication style."""
        adapted = scenario.copy()
        
        # Soften direct questions and add context
        if "question" in adapted:
            question = adapted["question"]
            if question.endswith("?"):
                adapted["question"] = f"Please consider: {question}"
        
        # Add polite framing
        if "instructions" in adapted:
            adapted["instructions"] = "If you would kindly consider the following scenario: " + adapted["instructions"]
        
        return adapted
    
    def _adapt_temporal_references(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt temporal references for cyclical time orientation."""
        adapted = scenario.copy()
        
        # Replace linear time references with cyclical or contextual ones
        temporal_replacements = {
            "before": "in the context that precedes",
            "after": "following the natural progression",
            "sequence": "pattern of relationships",
            "timeline": "flow of events"
        }
        
        for field in ["description", "question", "instructions"]:
            if field in adapted:
                text = adapted[field]
                for linear_term, cyclical_term in temporal_replacements.items():
                    text = text.replace(linear_term, cyclical_term)
                adapted[field] = text
        
        return adapted


class ComplianceManager:
    """Manages regulatory compliance across different regions."""
    
    def __init__(self):
        self.compliance_frameworks = {
            "GDPR": {
                "regions": ["EU", "EEA"],
                "data_processing_principles": [
                    "lawfulness_fairness_transparency",
                    "purpose_limitation",
                    "data_minimization",
                    "accuracy",
                    "storage_limitation",
                    "integrity_confidentiality"
                ],
                "user_rights": [
                    "right_to_information",
                    "right_of_access",
                    "right_to_rectification",
                    "right_to_erasure",
                    "right_to_restrict_processing",
                    "right_to_data_portability",
                    "right_to_object"
                ]
            },
            "CCPA": {
                "regions": ["California", "US"],
                "consumer_rights": [
                    "right_to_know",
                    "right_to_delete",
                    "right_to_opt_out",
                    "right_to_non_discrimination"
                ]
            },
            "PIPEDA": {
                "regions": ["Canada"],
                "principles": [
                    "accountability",
                    "identifying_purposes",
                    "consent",
                    "limiting_collection",
                    "limiting_use_disclosure_retention",
                    "accuracy",
                    "safeguards",
                    "openness",
                    "individual_access",
                    "challenging_compliance"
                ]
            },
            "LGPD": {
                "regions": ["Brazil"],
                "principles": [
                    "purpose",
                    "adequacy",
                    "necessity",
                    "free_access",
                    "data_quality",
                    "transparency",
                    "security",
                    "prevention",
                    "non_discrimination",
                    "accountability"
                ]
            }
        }
    
    def get_applicable_frameworks(self, region: str) -> List[str]:
        """Get applicable compliance frameworks for a region."""
        applicable = []
        for framework, details in self.compliance_frameworks.items():
            if region in details.get("regions", []):
                applicable.append(framework)
        return applicable
    
    def generate_privacy_notice(self, frameworks: List[str], language: SupportedLanguage) -> str:
        """Generate appropriate privacy notice."""
        localization_manager = LocalizationManager(language)
        
        if "GDPR" in frameworks:
            return localization_manager.get_text(
                "data_privacy_notice",
                compliance_standard="GDPR"
            )
        elif "CCPA" in frameworks:
            return localization_manager.get_text(
                "data_privacy_notice", 
                compliance_standard="CCPA"
            )
        else:
            return localization_manager.get_text(
                "data_privacy_notice",
                compliance_standard="international data protection"
            )
    
    def ensure_compliance(self, data_processing: Dict[str, Any], region: str) -> Dict[str, Any]:
        """Ensure data processing complies with regional requirements."""
        frameworks = self.get_applicable_frameworks(region)
        
        compliant_processing = data_processing.copy()
        
        for framework in frameworks:
            if framework == "GDPR":
                compliant_processing = self._apply_gdpr_compliance(compliant_processing)
            elif framework == "CCPA":
                compliant_processing = self._apply_ccpa_compliance(compliant_processing)
            elif framework == "PIPEDA":
                compliant_processing = self._apply_pipeda_compliance(compliant_processing)
            elif framework == "LGPD":
                compliant_processing = self._apply_lgpd_compliance(compliant_processing)
        
        return compliant_processing
    
    def _apply_gdpr_compliance(self, processing: Dict[str, Any]) -> Dict[str, Any]:
        """Apply GDPR compliance measures."""
        gdpr_compliant = processing.copy()
        
        # Ensure data minimization
        if "personal_data" in gdpr_compliant:
            gdpr_compliant["personal_data"] = {
                k: v for k, v in gdpr_compliant["personal_data"].items()
                if self._is_necessary_for_purpose(k, gdpr_compliant.get("purpose", ""))
            }
        
        # Add consent tracking
        gdpr_compliant["consent_obtained"] = True
        gdpr_compliant["lawful_basis"] = "consent"
        gdpr_compliant["retention_period"] = "evaluation_completion_plus_30_days"
        
        return gdpr_compliant
    
    def _apply_ccpa_compliance(self, processing: Dict[str, Any]) -> Dict[str, Any]:
        """Apply CCPA compliance measures."""
        ccpa_compliant = processing.copy()
        
        # Add CCPA-specific tracking
        ccpa_compliant["sale_opt_out_available"] = True
        ccpa_compliant["deletion_request_honored"] = True
        ccpa_compliant["data_categories_disclosed"] = True
        
        return ccpa_compliant
    
    def _apply_pipeda_compliance(self, processing: Dict[str, Any]) -> Dict[str, Any]:
        """Apply PIPEDA compliance measures."""
        pipeda_compliant = processing.copy()
        
        # PIPEDA-specific measures
        pipeda_compliant["purpose_identified"] = True
        pipeda_compliant["consent_meaningful"] = True
        pipeda_compliant["collection_limited"] = True
        
        return pipeda_compliant
    
    def _apply_lgpd_compliance(self, processing: Dict[str, Any]) -> Dict[str, Any]:
        """Apply LGPD compliance measures."""
        lgpd_compliant = processing.copy()
        
        # LGPD-specific measures
        lgpd_compliant["legal_basis"] = "consent"
        lgpd_compliant["purpose_specific"] = True
        lgpd_compliant["data_subject_rights_respected"] = True
        
        return lgpd_compliant
    
    def _is_necessary_for_purpose(self, data_field: str, purpose: str) -> bool:
        """Check if data field is necessary for stated purpose."""
        # Simplified necessity check
        necessary_fields = {
            "evaluation": ["responses", "timestamps", "session_id"],
            "research": ["anonymized_responses", "performance_metrics"],
            "analytics": ["aggregated_statistics"]
        }
        
        return data_field in necessary_fields.get(purpose, [])


class GlobalizationManager:
    """Manages global deployment considerations."""
    
    def __init__(self):
        self.localization_manager = LocalizationManager()
        self.compliance_manager = ComplianceManager()
        
        # Regional configurations
        self.regional_configs = {
            "EU": {
                "default_language": SupportedLanguage.ENGLISH,
                "compliance_frameworks": ["GDPR"],
                "cultural_considerations": ["high_privacy_expectations", "detailed_explanations_preferred"],
                "timezone": "CET",
                "number_format": "european",
                "date_format": "DD/MM/YYYY"
            },
            "US": {
                "default_language": SupportedLanguage.ENGLISH,
                "compliance_frameworks": ["CCPA"],
                "cultural_considerations": ["direct_communication", "efficiency_valued"],
                "timezone": "EST",
                "number_format": "american",
                "date_format": "MM/DD/YYYY"
            },
            "Asia": {
                "default_language": SupportedLanguage.ENGLISH,
                "compliance_frameworks": ["PDPA"],
                "cultural_considerations": ["indirect_communication", "context_important", "hierarchy_respected"],
                "timezone": "JST",
                "number_format": "asian",
                "date_format": "YYYY/MM/DD"
            },
            "LatAm": {
                "default_language": SupportedLanguage.SPANISH,
                "compliance_frameworks": ["LGPD"],
                "cultural_considerations": ["relationship_focused", "concrete_examples_helpful"],
                "timezone": "BRT",
                "number_format": "latin",
                "date_format": "DD/MM/YYYY"
            }
        }
    
    def configure_for_region(self, region: str, language: Optional[SupportedLanguage] = None) -> Dict[str, Any]:
        """Configure framework for specific region."""
        
        if region not in self.regional_configs:
            logger.warning(f"Unknown region: {region}. Using default configuration.")
            region = "US"  # Default fallback
        
        config = self.regional_configs[region].copy()
        
        # Set language
        target_language = language or config["default_language"]
        self.localization_manager.set_language(target_language)
        
        # Get cultural context
        cultural_context = self.localization_manager.get_cultural_context(target_language)
        
        # Configure compliance
        compliance_config = {}
        for framework in config["compliance_frameworks"]:
            compliance_config[framework] = self.compliance_manager.compliance_frameworks.get(framework, {})
        
        return {
            "region": region,
            "language": target_language.value,
            "cultural_context": cultural_context,
            "compliance_config": compliance_config,
            "regional_settings": config,
            "privacy_notice": self.compliance_manager.generate_privacy_notice(
                config["compliance_frameworks"], 
                target_language
            )
        }
    
    def adapt_evaluation_for_global_use(
        self, 
        evaluation_config: Dict[str, Any], 
        target_regions: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Adapt evaluation configuration for multiple regions."""
        
        adapted_configs = {}
        
        for region in target_regions:
            regional_config = self.configure_for_region(region)
            
            # Adapt evaluation scenarios
            adapted_evaluation = evaluation_config.copy()
            
            # Apply cultural adaptations
            if "scenarios" in adapted_evaluation:
                adapted_scenarios = []
                for scenario in adapted_evaluation["scenarios"]:
                    adapted_scenario = self.localization_manager.adapt_causal_scenario(
                        scenario, 
                        regional_config["cultural_context"].language
                    )
                    adapted_scenarios.append(adapted_scenario)
                adapted_evaluation["scenarios"] = adapted_scenarios
            
            # Apply compliance measures
            if "data_processing" in adapted_evaluation:
                adapted_evaluation["data_processing"] = self.compliance_manager.ensure_compliance(
                    adapted_evaluation["data_processing"],
                    region
                )
            
            # Add regional metadata
            adapted_evaluation["regional_config"] = regional_config
            
            adapted_configs[region] = adapted_evaluation
        
        return adapted_configs
    
    def generate_global_deployment_guide(self) -> Dict[str, Any]:
        """Generate comprehensive global deployment guide."""
        
        return {
            "supported_regions": list(self.regional_configs.keys()),
            "supported_languages": [lang.value for lang in SupportedLanguage],
            "compliance_frameworks": list(self.compliance_manager.compliance_frameworks.keys()),
            "deployment_checklist": [
                "Configure target regions",
                "Set default languages",
                "Validate compliance requirements",
                "Test cultural adaptations",
                "Review privacy notices",
                "Validate localized content",
                "Test multi-language scenarios",
                "Verify regional formatting",
                "Confirm timezone handling",
                "Validate accessibility standards"
            ],
            "best_practices": [
                "Always provide fallback to English",
                "Test with native speakers",
                "Consider right-to-left languages",
                "Validate cultural sensitivity",
                "Monitor compliance updates",
                "Regular localization updates",
                "Cultural competency training",
                "Regional user feedback loops"
            ],
            "technical_requirements": [
                "UTF-8 encoding support",
                "Bidirectional text support",
                "Cultural calendar systems",
                "Regional number formatting",
                "Timezone conversion utilities",
                "Character set validation",
                "Font support verification",
                "Input method compatibility"
            ]
        }


# Export the main classes for use
__all__ = [
    "SupportedLanguage",
    "CulturalContext", 
    "LocalizationManager",
    "ComplianceManager",
    "GlobalizationManager"
]