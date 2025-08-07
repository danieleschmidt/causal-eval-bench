"""
Advanced scenario generator for creating novel causal reasoning test cases.
"""

import random
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import itertools
import numpy as np

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """Types of causal relationships."""
    DIRECT_CAUSAL = "direct_causal"
    REVERSE_CAUSAL = "reverse_causal"
    BIDIRECTIONAL = "bidirectional"
    CONFOUNDED = "confounded"
    SPURIOUS_CORRELATION = "spurious_correlation" 
    MEDIATED = "mediated"
    NO_RELATIONSHIP = "no_relationship"


class ScenarioComplexity(Enum):
    """Complexity levels for scenarios."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class CausalVariable:
    """Represents a variable in a causal scenario."""
    
    name: str
    description: str
    domain: str  # medical, education, business, etc.
    variable_type: str  # continuous, categorical, binary, count
    typical_range: Optional[Tuple] = None
    possible_values: Optional[List[str]] = None
    unit: Optional[str] = None


@dataclass
class CausalRelationship:
    """Represents a causal relationship between variables."""
    
    cause: str
    effect: str
    relationship_type: CausalRelationType
    strength: float  # 0.0 to 1.0
    mechanism: str
    confounders: List[str] = field(default_factory=list)


@dataclass
class CausalScenario:
    """A complete causal reasoning scenario."""
    
    scenario_id: str
    title: str
    description: str
    domain: str
    complexity: ScenarioComplexity
    variables: Dict[str, CausalVariable]
    relationships: List[CausalRelationship] 
    ground_truth: Dict[str, Any]
    difficulty_level: str
    learning_objectives: List[str]
    common_misconceptions: List[str]


class ScenarioGenerator:
    """Generates novel causal reasoning scenarios."""
    
    def __init__(self):
        """Initialize the scenario generator."""
        self.generated_scenarios: List[CausalScenario] = []
        self.variable_library = self._build_variable_library()
        self.mechanism_templates = self._build_mechanism_templates()
        logger.info("Scenario generator initialized")
    
    def _build_variable_library(self) -> Dict[str, Dict[str, List[CausalVariable]]]:
        """Build library of variables organized by domain and type."""
        
        library = {
            "medical": {
                "treatments": [
                    CausalVariable("medication_adherence", "Patient's adherence to prescribed medication", "medical", "continuous", 
                                 typical_range=(0.0, 1.0), unit="proportion"),
                    CausalVariable("exercise_frequency", "Number of exercise sessions per week", "medical", "count",
                                 typical_range=(0, 14), unit="sessions/week"),
                    CausalVariable("dietary_modification", "Adherence to dietary recommendations", "medical", "binary",
                                 possible_values=["yes", "no"]),
                    CausalVariable("sleep_duration", "Average hours of sleep per night", "medical", "continuous",
                                 typical_range=(3.0, 12.0), unit="hours")
                ],
                "outcomes": [
                    CausalVariable("blood_pressure", "Systolic blood pressure", "medical", "continuous",
                                 typical_range=(80, 200), unit="mmHg"),
                    CausalVariable("symptom_severity", "Patient-reported symptom severity", "medical", "continuous",
                                 typical_range=(0, 10), unit="scale 0-10"),
                    CausalVariable("hospitalization", "Hospital readmission within 30 days", "medical", "binary",
                                 possible_values=["yes", "no"]),
                    CausalVariable("quality_of_life", "Health-related quality of life score", "medical", "continuous",
                                 typical_range=(0, 100), unit="HRQL score")
                ],
                "confounders": [
                    CausalVariable("age", "Patient age", "medical", "continuous",
                                 typical_range=(18, 95), unit="years"),
                    CausalVariable("disease_severity", "Baseline disease severity", "medical", "categorical",
                                 possible_values=["mild", "moderate", "severe"]),
                    CausalVariable("socioeconomic_status", "Socioeconomic status", "medical", "categorical",
                                 possible_values=["low", "middle", "high"]),
                    CausalVariable("comorbidities", "Number of comorbid conditions", "medical", "count",
                                 typical_range=(0, 10))
                ]
            },
            "education": {
                "interventions": [
                    CausalVariable("study_time", "Hours spent studying per week", "education", "continuous",
                                 typical_range=(0, 80), unit="hours/week"),
                    CausalVariable("active_learning", "Use of active learning techniques", "education", "binary",
                                 possible_values=["yes", "no"]),
                    CausalVariable("tutoring", "Participation in tutoring program", "education", "binary",
                                 possible_values=["yes", "no"]),
                    CausalVariable("technology_use", "Educational technology usage", "education", "categorical",
                                 possible_values=["none", "basic", "advanced"])
                ],
                "outcomes": [
                    CausalVariable("test_scores", "Standardized test performance", "education", "continuous",
                                 typical_range=(0, 100), unit="percentage"),
                    CausalVariable("course_completion", "Course completion rate", "education", "binary",
                                 possible_values=["completed", "dropped"]),
                    CausalVariable("skill_retention", "Knowledge retention after 6 months", "education", "continuous",
                                 typical_range=(0, 100), unit="percentage"),
                    CausalVariable("engagement", "Student engagement level", "education", "continuous",
                                 typical_range=(1, 5), unit="likert scale")
                ],
                "confounders": [
                    CausalVariable("prior_knowledge", "Baseline knowledge level", "education", "continuous",
                                 typical_range=(0, 100), unit="percentage"),
                    CausalVariable("motivation", "Student motivation level", "education", "continuous",
                                 typical_range=(1, 5), unit="likert scale"),
                    CausalVariable("learning_style", "Preferred learning style", "education", "categorical",
                                 possible_values=["visual", "auditory", "kinesthetic", "mixed"]),
                    CausalVariable("socioeconomic_background", "Family socioeconomic status", "education", "categorical",
                                 possible_values=["low", "middle", "high"])
                ]
            },
            "business": {
                "strategies": [
                    CausalVariable("advertising_spend", "Monthly advertising expenditure", "business", "continuous",
                                 typical_range=(1000, 100000), unit="dollars"),
                    CausalVariable("price_discount", "Percentage price discount offered", "business", "continuous",
                                 typical_range=(0, 50), unit="percentage"),
                    CausalVariable("product_quality", "Investment in product quality improvements", "business", "binary",
                                 possible_values=["yes", "no"]),
                    CausalVariable("customer_service", "Customer service enhancement program", "business", "binary",
                                 possible_values=["implemented", "not_implemented"])
                ],
                "outcomes": [
                    CausalVariable("sales_revenue", "Monthly sales revenue", "business", "continuous",
                                 typical_range=(10000, 1000000), unit="dollars"),
                    CausalVariable("customer_satisfaction", "Customer satisfaction score", "business", "continuous",
                                 typical_range=(1, 5), unit="likert scale"),
                    CausalVariable("market_share", "Percentage market share", "business", "continuous",
                                 typical_range=(0, 100), unit="percentage"),
                    CausalVariable("customer_retention", "Customer retention rate", "business", "continuous",
                                 typical_range=(0, 1), unit="proportion")
                ],
                "confounders": [
                    CausalVariable("market_conditions", "Overall market conditions", "business", "categorical",
                                 possible_values=["recession", "stable", "growth"]),
                    CausalVariable("competitor_actions", "Competitor pricing and promotion activity", "business", "categorical",
                                 possible_values=["low", "moderate", "high"]),
                    CausalVariable("seasonal_effects", "Seasonal demand patterns", "business", "categorical",
                                 possible_values=["low_season", "peak_season", "holiday_season"]),
                    CausalVariable("brand_reputation", "Company brand reputation score", "business", "continuous",
                                 typical_range=(1, 10), unit="reputation score")
                ]
            },
            "environmental": {
                "factors": [
                    CausalVariable("pollution_levels", "Air pollution concentration", "environmental", "continuous",
                                 typical_range=(0, 500), unit="AQI"),
                    CausalVariable("deforestation_rate", "Annual deforestation rate", "environmental", "continuous",
                                 typical_range=(0, 10), unit="percent/year"),
                    CausalVariable("industrial_activity", "Level of industrial activity", "environmental", "continuous",
                                 typical_range=(0, 100), unit="index"),
                    CausalVariable("renewable_energy", "Percentage of renewable energy usage", "environmental", "continuous",
                                 typical_range=(0, 100), unit="percentage")
                ],
                "outcomes": [
                    CausalVariable("biodiversity_index", "Local biodiversity index", "environmental", "continuous",
                                 typical_range=(0, 1), unit="index"),
                    CausalVariable("water_quality", "Water quality measurements", "environmental", "continuous",
                                 typical_range=(0, 100), unit="quality score"),
                    CausalVariable("health_outcomes", "Population health indicators", "environmental", "continuous",
                                 typical_range=(0, 100), unit="health index"),
                    CausalVariable("climate_impact", "Local climate change indicators", "environmental", "continuous",
                                 typical_range=(0, 10), unit="severity scale")
                ],
                "confounders": [
                    CausalVariable("geographic_location", "Geographic region characteristics", "environmental", "categorical",
                                 possible_values=["urban", "suburban", "rural", "coastal"]),
                    CausalVariable("population_density", "Population density of area", "environmental", "continuous",
                                 typical_range=(1, 10000), unit="people/km²"),
                    CausalVariable("economic_development", "Level of economic development", "environmental", "categorical",
                                 possible_values=["developing", "developed", "highly_developed"]),
                    CausalVariable("natural_disasters", "Frequency of natural disasters", "environmental", "count",
                                 typical_range=(0, 5), unit="events/year")
                ]
            },
            "technology": {
                "interventions": [
                    CausalVariable("algorithm_complexity", "Machine learning algorithm complexity", "technology", "categorical",
                                 possible_values=["simple", "moderate", "complex"]),
                    CausalVariable("data_quality", "Training data quality score", "technology", "continuous",
                                 typical_range=(0, 1), unit="quality score"),
                    CausalVariable("processing_power", "Computational resources allocated", "technology", "continuous",
                                 typical_range=(1, 1000), unit="GPU hours"),
                    CausalVariable("feature_engineering", "Advanced feature engineering applied", "technology", "binary",
                                 possible_values=["yes", "no"])
                ],
                "outcomes": [
                    CausalVariable("model_accuracy", "Model prediction accuracy", "technology", "continuous",
                                 typical_range=(0, 1), unit="accuracy score"),
                    CausalVariable("inference_speed", "Model inference speed", "technology", "continuous",
                                 typical_range=(1, 1000), unit="predictions/second"),
                    CausalVariable("user_adoption", "System user adoption rate", "technology", "continuous",
                                 typical_range=(0, 1), unit="adoption rate"),
                    CausalVariable("system_reliability", "System uptime percentage", "technology", "continuous",
                                 typical_range=(0.9, 1.0), unit="uptime percentage")
                ],
                "confounders": [
                    CausalVariable("team_expertise", "Development team experience level", "technology", "categorical",
                                 possible_values=["junior", "mid", "senior", "expert"]),
                    CausalVariable("project_timeline", "Available development time", "technology", "continuous",
                                 typical_range=(1, 24), unit="months"),
                    CausalVariable("budget_constraints", "Project budget limitations", "technology", "categorical",
                                 possible_values=["low", "medium", "high", "unlimited"]),
                    CausalVariable("technology_maturity", "Maturity of underlying technology", "technology", "categorical",
                                 possible_values=["experimental", "emerging", "mature", "legacy"])
                ]
            }
        }
        
        return library
    
    def _build_mechanism_templates(self) -> Dict[str, List[str]]:
        """Build templates for causal mechanisms."""
        
        templates = {
            "direct_physiological": [
                "{cause} directly affects {effect} through {mechanism} pathways",
                "{cause} modulates {effect} via {mechanism} processes",
                "{cause} influences {effect} by altering {mechanism} function"
            ],
            "behavioral_change": [
                "{cause} motivates individuals to change their behavior regarding {effect}",
                "{cause} creates incentives that lead to modifications in {effect}",
                "{cause} influences decision-making processes affecting {effect}"
            ],
            "resource_allocation": [
                "{cause} determines the allocation of resources that impact {effect}",
                "{cause} affects the availability of resources needed for {effect}",
                "{cause} influences resource distribution patterns affecting {effect}"
            ],
            "information_processing": [
                "{cause} affects cognitive processing mechanisms that influence {effect}",
                "{cause} modifies information flow patterns impacting {effect}",
                "{cause} changes learning and adaptation processes affecting {effect}"
            ],
            "environmental_mediation": [
                "{cause} alters environmental conditions that subsequently affect {effect}",
                "{cause} modifies the context in which {effect} occurs",
                "{cause} creates environmental pressures that influence {effect}"
            ],
            "network_effects": [
                "{cause} propagates through social/professional networks to influence {effect}",
                "{cause} creates cascading effects through interconnected systems affecting {effect}",
                "{cause} influences network dynamics that subsequently impact {effect}"
            ]
        }
        
        return templates
    
    def generate_scenario(
        self,
        domain: str,
        complexity: ScenarioComplexity = ScenarioComplexity.MODERATE,
        target_relationship: Optional[CausalRelationType] = None,
        custom_variables: Optional[List[CausalVariable]] = None
    ) -> CausalScenario:
        """Generate a novel causal scenario."""
        
        if domain not in self.variable_library:
            raise ValueError(f"Domain '{domain}' not supported. Available: {list(self.variable_library.keys())}")
        
        # Generate scenario ID
        scenario_id = f"{domain}_{complexity.value}_{len(self.generated_scenarios):04d}"
        
        # Select variables based on complexity
        variables = self._select_variables(domain, complexity, custom_variables)
        
        # Generate relationships
        relationships = self._generate_relationships(variables, complexity, target_relationship)
        
        # Create ground truth
        ground_truth = self._create_ground_truth(relationships)
        
        # Generate scenario description
        title, description = self._generate_scenario_description(domain, variables, relationships)
        
        # Determine difficulty and learning objectives
        difficulty = self._determine_difficulty(complexity, relationships)
        learning_objectives = self._generate_learning_objectives(relationships)
        misconceptions = self._generate_common_misconceptions(relationships)
        
        scenario = CausalScenario(
            scenario_id=scenario_id,
            title=title,
            description=description,
            domain=domain,
            complexity=complexity,
            variables=variables,
            relationships=relationships,
            ground_truth=ground_truth,
            difficulty_level=difficulty,
            learning_objectives=learning_objectives,
            common_misconceptions=misconceptions
        )
        
        self.generated_scenarios.append(scenario)
        logger.info(f"Generated scenario {scenario_id}: {title}")
        
        return scenario

    def _select_variables(
        self,
        domain: str,
        complexity: ScenarioComplexity,
        custom_variables: Optional[List[CausalVariable]] = None
    ) -> Dict[str, CausalVariable]:
        """Select appropriate variables for the scenario."""
        
        if custom_variables:
            return {var.name: var for var in custom_variables}
        
        domain_vars = self.variable_library[domain]
        selected_vars = {}
        
        # Number of variables based on complexity
        var_counts = {
            ScenarioComplexity.SIMPLE: {"primary": 2, "confounders": 0, "additional": 0},
            ScenarioComplexity.MODERATE: {"primary": 2, "confounders": 2, "additional": 1},
            ScenarioComplexity.COMPLEX: {"primary": 3, "confounders": 3, "additional": 2},
            ScenarioComplexity.EXPERT: {"primary": 4, "confounders": 4, "additional": 3}
        }
        
        counts = var_counts[complexity]
        
        # Select primary variables (cause and effect)
        cause_categories = [cat for cat in domain_vars.keys() if cat not in ["confounders"]]
        effect_categories = [cat for cat in domain_vars.keys() if cat not in ["confounders"]]
        
        # Ensure we have different categories for cause and effect
        if len(cause_categories) > 1:
            cause_cat = random.choice(cause_categories)
            effect_cat = random.choice([cat for cat in effect_categories if cat != cause_cat])
        else:
            cause_cat = effect_cat = cause_categories[0]
        
        # Select cause variables
        cause_vars = random.sample(domain_vars[cause_cat], min(counts["primary"] // 2 + 1, len(domain_vars[cause_cat])))
        for var in cause_vars:
            selected_vars[var.name] = var
        
        # Select effect variables
        effect_vars = random.sample(domain_vars[effect_cat], min(counts["primary"] - len(cause_vars), len(domain_vars[effect_cat])))
        for var in effect_vars:
            if var.name not in selected_vars:  # Avoid duplicates
                selected_vars[var.name] = var
        
        # Add confounders
        if "confounders" in domain_vars and counts["confounders"] > 0:
            confounders = random.sample(domain_vars["confounders"], min(counts["confounders"], len(domain_vars["confounders"])))
            for var in confounders:
                selected_vars[var.name] = var
        
        # Add additional variables from other categories
        if counts["additional"] > 0:
            remaining_vars = []
            for cat, vars_list in domain_vars.items():
                if cat != "confounders":
                    remaining_vars.extend([v for v in vars_list if v.name not in selected_vars])
            
            if remaining_vars:
                additional = random.sample(remaining_vars, min(counts["additional"], len(remaining_vars)))
                for var in additional:
                    selected_vars[var.name] = var
        
        return selected_vars

    def _generate_relationships(
        self,
        variables: Dict[str, CausalVariable],
        complexity: ScenarioComplexity,
        target_relationship: Optional[CausalRelationType] = None
    ) -> List[CausalRelationship]:
        """Generate causal relationships between variables."""
        
        relationships = []
        var_names = list(variables.keys())
        
        # Identify potential confounders
        confounders = [name for name, var in variables.items() 
                      if "confounder" in var.description.lower() or 
                         any(keyword in name.lower() for keyword in ["age", "socioeconomic", "baseline", "prior"])]
        
        # Generate primary relationship
        primary_relationship = self._create_primary_relationship(
            var_names, confounders, target_relationship, variables
        )
        relationships.append(primary_relationship)
        
        # Add additional relationships based on complexity
        if complexity in [ScenarioComplexity.COMPLEX, ScenarioComplexity.EXPERT]:
            # Add secondary relationships
            num_additional = 2 if complexity == ScenarioComplexity.COMPLEX else 4
            
            for _ in range(num_additional):
                # Create relationships between remaining variables
                remaining_pairs = [(a, b) for a in var_names for b in var_names 
                                 if a != b and not any(r.cause == a and r.effect == b for r in relationships)]
                
                if remaining_pairs:
                    cause, effect = random.choice(remaining_pairs)
                    rel_type = random.choice([CausalRelationType.DIRECT_CAUSAL, 
                                           CausalRelationType.CONFOUNDED,
                                           CausalRelationType.SPURIOUS_CORRELATION])
                    
                    relationship = self._create_relationship(
                        cause, effect, rel_type, confounders, variables
                    )
                    relationships.append(relationship)
        
        return relationships

    def _create_primary_relationship(
        self,
        var_names: List[str],
        confounders: List[str],
        target_relationship: Optional[CausalRelationType],
        variables: Dict[str, CausalVariable]
    ) -> CausalRelationship:
        """Create the primary causal relationship."""
        
        # Select cause and effect
        non_confounders = [name for name in var_names if name not in confounders]
        
        if len(non_confounders) >= 2:
            cause, effect = random.sample(non_confounders, 2)
        else:
            cause, effect = random.sample(var_names, 2)
        
        # Determine relationship type
        if target_relationship:
            rel_type = target_relationship
        else:
            # Bias toward meaningful relationships
            rel_type = random.choice([
                CausalRelationType.DIRECT_CAUSAL,
                CausalRelationType.DIRECT_CAUSAL,  # Higher weight
                CausalRelationType.CONFOUNDED,
                CausalRelationType.SPURIOUS_CORRELATION,
                CausalRelationType.REVERSE_CAUSAL
            ])
        
        return self._create_relationship(cause, effect, rel_type, confounders, variables)

    def _create_relationship(
        self,
        cause: str,
        effect: str,
        rel_type: CausalRelationType,
        all_confounders: List[str],
        variables: Dict[str, CausalVariable]
    ) -> CausalRelationship:
        """Create a causal relationship with specified parameters."""
        
        # Determine strength based on relationship type
        strength_ranges = {
            CausalRelationType.DIRECT_CAUSAL: (0.6, 0.9),
            CausalRelationType.REVERSE_CAUSAL: (0.5, 0.8),
            CausalRelationType.CONFOUNDED: (0.3, 0.7),
            CausalRelationType.SPURIOUS_CORRELATION: (0.1, 0.4),
            CausalRelationType.MEDIATED: (0.4, 0.7),
            CausalRelationType.NO_RELATIONSHIP: (0.0, 0.1)
        }
        
        min_strength, max_strength = strength_ranges[rel_type]
        strength = random.uniform(min_strength, max_strength)
        
        # Select confounders for this relationship
        relationship_confounders = []
        if rel_type in [CausalRelationType.CONFOUNDED, CausalRelationType.SPURIOUS_CORRELATION]:
            available_confounders = [c for c in all_confounders if c not in [cause, effect]]
            if available_confounders:
                num_confounders = random.randint(1, min(3, len(available_confounders)))
                relationship_confounders = random.sample(available_confounders, num_confounders)
        
        # Generate mechanism description
        mechanism = self._generate_mechanism_description(
            cause, effect, rel_type, variables[cause], variables[effect]
        )
        
        return CausalRelationship(
            cause=cause,
            effect=effect,
            relationship_type=rel_type,
            strength=strength,
            mechanism=mechanism,
            confounders=relationship_confounders
        )

    def _generate_mechanism_description(
        self,
        cause: str,
        effect: str,
        rel_type: CausalRelationType,
        cause_var: CausalVariable,
        effect_var: CausalVariable
    ) -> str:
        """Generate a plausible mechanism description."""
        
        domain = cause_var.domain
        
        # Select appropriate mechanism template based on domain and relationship type
        mechanism_type = self._select_mechanism_type(domain, rel_type)
        templates = self.mechanism_templates.get(mechanism_type, self.mechanism_templates["direct_physiological"])
        
        template = random.choice(templates)
        
        # Generate mechanism details based on domain
        mechanism_details = self._generate_mechanism_details(domain, cause_var, effect_var)
        
        return template.format(
            cause=cause.replace("_", " "),
            effect=effect.replace("_", " "),
            mechanism=mechanism_details
        )

    def _select_mechanism_type(self, domain: str, rel_type: CausalRelationType) -> str:
        """Select appropriate mechanism type based on domain and relationship."""
        
        domain_mechanisms = {
            "medical": ["direct_physiological", "behavioral_change"],
            "education": ["information_processing", "behavioral_change"],
            "business": ["resource_allocation", "behavioral_change", "network_effects"],
            "environmental": ["environmental_mediation", "network_effects"],
            "technology": ["information_processing", "resource_allocation"]
        }
        
        available_mechanisms = domain_mechanisms.get(domain, ["direct_physiological"])
        return random.choice(available_mechanisms)

    def _generate_mechanism_details(self, domain: str, cause_var: CausalVariable, effect_var: CausalVariable) -> str:
        """Generate domain-specific mechanism details."""
        
        mechanism_details = {
            "medical": ["physiological", "metabolic", "neurological", "cardiovascular", "immunological"],
            "education": ["cognitive", "motivational", "social learning", "memory consolidation", "skill transfer"],
            "business": ["market dynamics", "consumer behavior", "operational efficiency", "brand perception", "competitive positioning"],
            "environmental": ["ecological", "atmospheric", "hydrological", "biological", "geochemical"],
            "technology": ["algorithmic", "computational", "data processing", "system optimization", "user interface"]
        }
        
        details = mechanism_details.get(domain, ["systematic"])
        return random.choice(details)

    def _create_ground_truth(self, relationships: List[CausalRelationship]) -> Dict[str, Any]:
        """Create ground truth information for the scenario."""
        
        primary_rel = relationships[0] if relationships else None
        
        ground_truth = {
            "primary_relationship": {
                "cause": primary_rel.cause if primary_rel else None,
                "effect": primary_rel.effect if primary_rel else None,
                "type": primary_rel.relationship_type.value if primary_rel else None,
                "strength": primary_rel.strength if primary_rel else None
            },
            "all_relationships": [
                {
                    "cause": rel.cause,
                    "effect": rel.effect,
                    "type": rel.relationship_type.value,
                    "strength": rel.strength,
                    "confounders": rel.confounders
                }
                for rel in relationships
            ],
            "confounders": list(set([
                conf for rel in relationships 
                for conf in rel.confounders
            ]))
        }
        
        return ground_truth

    def _generate_scenario_description(
        self,
        domain: str,
        variables: Dict[str, CausalVariable],
        relationships: List[CausalRelationship]
    ) -> Tuple[str, str]:
        """Generate title and description for the scenario."""
        
        primary_rel = relationships[0] if relationships else None
        
        if primary_rel:
            cause_name = primary_rel.cause.replace("_", " ").title()
            effect_name = primary_rel.effect.replace("_", " ").title()
            
            title = f"{cause_name} and {effect_name} in {domain.title()}"
            
            description = f"""
A researcher is investigating the relationship between {cause_name.lower()} and {effect_name.lower()} 
in the {domain} domain. The study involves {len(variables)} key variables and seeks to understand 
whether changes in {cause_name.lower()} actually cause changes in {effect_name.lower()}, 
or if the relationship might be explained by other factors.

Key Variables:
{chr(10).join([f"- {var.name.replace('_', ' ').title()}: {var.description}" for var in variables.values()])}

The research question is: Does {cause_name.lower()} have a causal effect on {effect_name.lower()}?
"""
        else:
            title = f"Variable Relationships in {domain.title()}"
            description = f"A study examining relationships between variables in the {domain} domain."
        
        return title.strip(), description.strip()

    def _determine_difficulty(self, complexity: ScenarioComplexity, relationships: List[CausalRelationship]) -> str:
        """Determine difficulty level based on complexity and relationships."""
        
        base_difficulty = {
            ScenarioComplexity.SIMPLE: "beginner",
            ScenarioComplexity.MODERATE: "intermediate", 
            ScenarioComplexity.COMPLEX: "advanced",
            ScenarioComplexity.EXPERT: "expert"
        }
        
        difficulty = base_difficulty[complexity]
        
        # Adjust based on relationship types
        complex_relationships = [
            CausalRelationType.CONFOUNDED,
            CausalRelationType.SPURIOUS_CORRELATION,
            CausalRelationType.MEDIATED,
            CausalRelationType.BIDIRECTIONAL
        ]
        
        if any(rel.relationship_type in complex_relationships for rel in relationships):
            if difficulty == "beginner":
                difficulty = "intermediate"
            elif difficulty == "intermediate":
                difficulty = "advanced"
        
        return difficulty

    def _generate_learning_objectives(self, relationships: List[CausalRelationship]) -> List[str]:
        """Generate learning objectives based on relationships."""
        
        objectives = []
        
        # Base objectives
        objectives.append("Identify potential causal relationships between variables")
        objectives.append("Distinguish between correlation and causation")
        
        # Relationship-specific objectives
        relationship_types = [rel.relationship_type for rel in relationships]
        
        if CausalRelationType.CONFOUNDED in relationship_types:
            objectives.append("Recognize confounding variables and their effects")
        
        if CausalRelationType.SPURIOUS_CORRELATION in relationship_types:
            objectives.append("Identify spurious correlations caused by third variables")
        
        if CausalRelationType.REVERSE_CAUSAL in relationship_types:
            objectives.append("Consider alternative causal directions")
        
        if CausalRelationType.MEDIATED in relationship_types:
            objectives.append("Understand mediated causal pathways")
        
        # Add domain-specific objectives
        objectives.append("Apply causal reasoning principles to real-world scenarios")
        objectives.append("Evaluate evidence for causal claims")
        
        return objectives

    def _generate_common_misconceptions(self, relationships: List[CausalRelationship]) -> List[str]:
        """Generate common misconceptions based on relationships."""
        
        misconceptions = []
        
        # General misconceptions
        misconceptions.append("Assuming correlation implies causation")
        misconceptions.append("Ignoring potential confounding variables")
        misconceptions.append("Failing to consider alternative explanations")
        
        # Relationship-specific misconceptions  
        relationship_types = [rel.relationship_type for rel in relationships]
        
        if CausalRelationType.REVERSE_CAUSAL in relationship_types:
            misconceptions.append("Assuming the wrong causal direction")
        
        if CausalRelationType.SPURIOUS_CORRELATION in relationship_types:
            misconceptions.append("Missing the true common cause of both variables")
        
        if CausalRelationType.CONFOUNDED in relationship_types:
            misconceptions.append("Attributing causation without controlling for confounders")
        
        misconceptions.append("Oversimplifying complex causal networks")
        misconceptions.append("Generalizing from limited or biased samples")
        
        return misconceptions

    def generate_batch(
        self,
        domains: List[str],
        n_scenarios_per_domain: int = 5,
        complexity_distribution: Optional[Dict[ScenarioComplexity, float]] = None
    ) -> List[CausalScenario]:
        """Generate a batch of scenarios across multiple domains."""
        
        if complexity_distribution is None:
            complexity_distribution = {
                ScenarioComplexity.SIMPLE: 0.2,
                ScenarioComplexity.MODERATE: 0.4,
                ScenarioComplexity.COMPLEX: 0.3,
                ScenarioComplexity.EXPERT: 0.1
            }
        
        scenarios = []
        
        for domain in domains:
            for _ in range(n_scenarios_per_domain):
                # Sample complexity based on distribution
                complexity = np.random.choice(
                    list(complexity_distribution.keys()),
                    p=list(complexity_distribution.values())
                )
                
                scenario = self.generate_scenario(domain, complexity)
                scenarios.append(scenario)
        
        logger.info(f"Generated batch of {len(scenarios)} scenarios across {len(domains)} domains")
        return scenarios

    def export_scenarios(self, scenarios: List[CausalScenario], output_path: str, format: str = "json") -> None:
        """Export scenarios to file."""
        
        import json
        from pathlib import Path
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            # Convert scenarios to dict format for JSON serialization
            scenario_dicts = []
            for scenario in scenarios:
                scenario_dict = {
                    "scenario_id": scenario.scenario_id,
                    "title": scenario.title,
                    "description": scenario.description,
                    "domain": scenario.domain,
                    "complexity": scenario.complexity.value,
                    "difficulty_level": scenario.difficulty_level,
                    "learning_objectives": scenario.learning_objectives,
                    "common_misconceptions": scenario.common_misconceptions,
                    "variables": {
                        name: {
                            "name": var.name,
                            "description": var.description,
                            "domain": var.domain,
                            "variable_type": var.variable_type,
                            "typical_range": var.typical_range,
                            "possible_values": var.possible_values,
                            "unit": var.unit
                        }
                        for name, var in scenario.variables.items()
                    },
                    "relationships": [
                        {
                            "cause": rel.cause,
                            "effect": rel.effect,
                            "relationship_type": rel.relationship_type.value,
                            "strength": rel.strength,
                            "mechanism": rel.mechanism,
                            "confounders": rel.confounders
                        }
                        for rel in scenario.relationships
                    ],
                    "ground_truth": scenario.ground_truth
                }
                scenario_dicts.append(scenario_dict)
            
            with open(output_file, 'w') as f:
                json.dump(scenario_dicts, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported {len(scenarios)} scenarios to {output_path}")

    def get_scenario_stats(self) -> Dict[str, Any]:
        """Get statistics about generated scenarios."""
        
        if not self.generated_scenarios:
            return {"total_scenarios": 0}
        
        stats = {
            "total_scenarios": len(self.generated_scenarios),
            "by_domain": {},
            "by_complexity": {},
            "by_difficulty": {},
            "relationship_types": {}
        }
        
        for scenario in self.generated_scenarios:
            # Domain stats
            domain = scenario.domain
            if domain not in stats["by_domain"]:
                stats["by_domain"][domain] = 0
            stats["by_domain"][domain] += 1
            
            # Complexity stats
            complexity = scenario.complexity.value
            if complexity not in stats["by_complexity"]:
                stats["by_complexity"][complexity] = 0
            stats["by_complexity"][complexity] += 1
            
            # Difficulty stats
            difficulty = scenario.difficulty_level
            if difficulty not in stats["by_difficulty"]:
                stats["by_difficulty"][difficulty] = 0
            stats["by_difficulty"][difficulty] += 1
            
            # Relationship type stats
            for rel in scenario.relationships:
                rel_type = rel.relationship_type.value
                if rel_type not in stats["relationship_types"]:
                    stats["relationship_types"][rel_type] = 0
                stats["relationship_types"][rel_type] += 1
        
        return stats