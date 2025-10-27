# app/services/knowledge_guide.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class KnowledgeGuide:
    def __init__(self, knowledge_corpus_path: str):
        self.knowledge_corpus = None
        self.domain_map = None
        self.domain_descriptions = None
        self._load_knowledge_corpus(knowledge_corpus_path)
    
    def _load_knowledge_corpus(self, corpus_path: str):
        """Load and process the knowledge corpus"""
        try:
            if Path(corpus_path).exists():
                self.knowledge_corpus = pd.read_csv(corpus_path)
                logger.info(f"Loaded knowledge corpus with {len(self.knowledge_corpus)} entries")
            else:
                # Create default knowledge corpus structure
                self.knowledge_corpus = self._create_default_corpus()
                logger.info("Created default knowledge corpus")
            
            self._build_domain_mappings()
            
        except Exception as e:
            logger.error(f"Failed to load knowledge corpus: {str(e)}")
            raise
    
    def _create_default_corpus(self) -> pd.DataFrame:
        """Create default knowledge corpus based on autism diagnostic domains"""
        default_data = {
            'Linked_Model': ['TSN', 'TSN', 'TSN', 'SGCN', 'SGCN', 'SGCN', 'ST-GCN', 'ST-GCN', 'ST-GCN'],
            'Category': [
                'Social Communication & Interaction',
                'Communication',
                'Joint Attention',
                'Social Communication & Interaction', 
                'Early Social Attention',
                'Facial Affect Recognition',
                'Object & Repetitive Behaviors',
                'Motor Development & Atypical Movements',
                'Stereotyped Motor Movements'
            ],
            'Description': [
                'Difficulties in social-emotional reciprocity',
                'Impairments in verbal and non-verbal communication',
                'Challenges in sharing focus of interest with others',
                'Atypical social approach and response',
                'Reduced attention to social stimuli',
                'Difficulties recognizing and responding to facial expressions',
                'Presence of restricted, repetitive patterns of behavior',
                'Atypical motor patterns and coordination difficulties',
                'Repetitive, seemingly driven motor behaviors'
            ],
            'Severity_Indicator': [
                'Higher scores indicate greater social communication challenges',
                'Elevated scores suggest communication impairments',
                'Increased scores reflect joint attention difficulties',
                'Higher values indicate social interaction challenges',
                'Elevated scores show reduced social attention',
                'Increased scores suggest facial affect recognition issues',
                'Higher scores indicate more pronounced repetitive behaviors',
                'Elevated scores reflect motor coordination challenges',
                'Increased scores suggest presence of motor stereotypies'
            ],
            'Clinical_References': [
                'ADOS-2: Social Affect domain',
                'ADOS-2: Communication subscale',
                'Early Start Denver Model indicators',
                'ADOS-2: Social Response measures',
                'Autism Diagnostic Observation',
                'Facial Emotion Recognition tasks',
                'ADOS-2: Restricted and Repetitive Behaviors',
                'Motor coordination assessment protocols',
                'Stereotypy Severity Measures'
            ]
        }
        return pd.DataFrame(default_data)
    
    def _build_domain_mappings(self):
        """Build mappings between models and diagnostic domains"""
        try:
            # Create domain map
            self.domain_map = (
                self.knowledge_corpus.groupby("Linked_Model")["Category"]
                .unique()
                .to_dict()
            )
            
            # Create domain descriptions
            self.domain_descriptions = (
                self.knowledge_corpus.groupby("Category")
                .agg({
                    'Description': 'first',
                    'Severity_Indicator': 'first',
                    'Clinical_References': 'first'
                })
                .to_dict(orient='index')
            )
            
            logger.info("Built domain mappings successfully")
            
        except Exception as e:
            logger.error(f"Failed to build domain mappings: {str(e)}")
            raise
    
    def identify_dominant_models(self, tsn_score: float, sgcn_score: float, stgcn_score: float, 
                               threshold: float = 0.9) -> List[str]:
        """
        Identify which models are driving the prediction
        
        Args:
            tsn_score, sgcn_score, stgcn_score: Individual model predictions
            threshold: Consider models within this percentage of max as dominant
            
        Returns:
            List of dominant model names
        """
        try:
            model_scores = {
                'TSN': tsn_score,
                'SGCN': sgcn_score, 
                'ST-GCN': stgcn_score
            }
            
            max_score = max(model_scores.values())
            threshold_value = threshold * max_score
            
            dominant_models = [
                model for model, score in model_scores.items() 
                if score >= threshold_value
            ]
            
            return dominant_models
            
        except Exception as e:
            logger.error(f"Failed to identify dominant models: {str(e)}")
            return []
    
    def map_to_diagnostic_domains(self, dominant_models: List[str]) -> Dict:
        """
        Map dominant models to diagnostic domains
        
        Returns:
            Dictionary with domains and their descriptions
        """
        try:
            domains = {}
            
            for model in dominant_models:
                if model in self.domain_map:
                    for domain in self.domain_map[model]:
                        if domain not in domains:
                            domains[domain] = self.domain_descriptions.get(domain, {})
            
            return domains
            
        except Exception as e:
            logger.error(f"Failed to map to diagnostic domains: {str(e)}")
            return {}
    
    def generate_explanation(self, final_score: float, component_scores: Dict, 
                           dominant_models: List[str], domains: Dict) -> Dict:
        """
        Generate comprehensive explanation for the prediction
        
        Returns:
            Dictionary with structured explanation
        """
        try:
            # Base explanation
            model_names = ", ".join(dominant_models)
            domain_names = ", ".join(domains.keys())
            
            base_explanation = (
                f"Score {final_score:.2f} — driven mainly by {model_names} "
                f"models (domains: {domain_names})."
            )
            
            # Detailed domain explanations
            domain_details = []
            for domain, info in domains.items():
                domain_detail = {
                    'domain': domain,
                    'description': info.get('Description', ''),
                    'severity_interpretation': info.get('Severity_Indicator', ''),
                    'clinical_reference': info.get('Clinical_References', '')
                }
                domain_details.append(domain_detail)
            
            # Risk level interpretation
            risk_level = self._interpret_risk_level(final_score)
            
            # Clinical recommendations
            recommendations = self._generate_recommendations(dominant_models, domains, final_score)
            
            explanation = {
                'base_explanation': base_explanation,
                'risk_level': risk_level,
                'dominant_models': dominant_models,
                'domains': domain_details,
                'clinical_recommendations': recommendations,
                'component_breakdown': {
                    'optical_flow_score': component_scores['optical_flow'],
                    'skeleton_2d_score': component_scores['skeleton_2d'],
                    'skeleton_3d_score': component_scores['skeleton_3d']
                }
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {str(e)}")
            return {
                'base_explanation': f"Score {final_score:.2f} — technical explanation unavailable.",
                'risk_level': 'Unknown',
                'dominant_models': [],
                'domains': [],
                'clinical_recommendations': ['Consult with healthcare provider for detailed assessment'],
                'component_breakdown': component_scores
            }
    
    def _interpret_risk_level(self, score: float) -> str:
        """Interpret the final score into risk levels"""
        if score <= 3.0:
            return "Low Risk"
        elif score <= 5.0:
            return "Medium Risk" 
        elif score <= 7.0:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def _generate_recommendations(self, dominant_models: List[str], domains: Dict, score: float) -> List[str]:
        """Generate clinical recommendations based on findings"""
        recommendations = []
        
        # Base recommendations based on risk level
        if score <= 3.0:
            recommendations.append("Routine developmental monitoring recommended")
        elif score <= 5.0:
            recommendations.append("Consider comprehensive developmental screening")
            recommendations.append("Monitor social communication milestones")
        elif score <= 7.0:
            recommendations.append("Recommend comprehensive diagnostic assessment")
            recommendations.append("Early intervention services may be beneficial")
        else:
            recommendations.append("Urgent comprehensive evaluation needed")
            recommendations.append("Immediate referral to autism specialist recommended")
        
        # Domain-specific recommendations
        domain_names = list(domains.keys())
        
        if any('Social' in domain for domain in domain_names):
            recommendations.append("Focus on social communication skills assessment")
            
        if any('Communication' in domain for domain in domain_names):
            recommendations.append("Evaluate verbal and non-verbal communication abilities")
            
        if any('Motor' in domain for domain in domain_names):
            recommendations.append("Assess motor coordination and movement patterns")
            
        if any('Repetitive' in domain for domain in domain_names):
            recommendations.append("Evaluate restricted and repetitive behaviors")
        
        return recommendations
    
    def create_comprehensive_report(self, prediction_result: Dict) -> Dict:
        """
        Create comprehensive report with knowledge-guided explanations
        
        Args:
            prediction_result: Output from the main predictor
            
        Returns:
            Enhanced result with explanations
        """
        try:
            component_scores = prediction_result['component_analysis']
            
            # Identify dominant models
            dominant_models = self.identify_dominant_models(
                component_scores['optical_flow'],
                component_scores['skeleton_2d'], 
                component_scores['skeleton_3d']
            )
            
            # Map to diagnostic domains
            domains = self.map_to_diagnostic_domains(dominant_models)
            
            # Generate explanation
            explanation = self.generate_explanation(
                prediction_result['prediction']['final_score'],
                component_scores,
                dominant_models,
                domains
            )
            
            # Enhance the original result
            enhanced_result = prediction_result.copy()
            enhanced_result['knowledge_guided_explanation'] = explanation
            
            logger.info("Generated knowledge-guided explanation successfully")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Failed to create comprehensive report: {str(e)}")
            # Return original result if explanation fails
            return prediction_result