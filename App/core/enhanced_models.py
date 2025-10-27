# app/core/enhanced_models.py
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class DomainDetail(BaseModel):
    domain: str = Field(..., description="Diagnostic domain name")
    description: str = Field(..., description="Domain description")
    severity_interpretation: str = Field(..., description="How to interpret scores in this domain")
    clinical_reference: str = Field(..., description="Clinical assessment reference")

class KnowledgeExplanation(BaseModel):
    base_explanation: str = Field(..., description="Base explanation text")
    risk_level: str = Field(..., description="Overall risk level")
    dominant_models: List[str] = Field(..., description="Models driving the prediction")
    domains: List[DomainDetail] = Field(..., description="Detailed domain information")
    clinical_recommendations: List[str] = Field(..., description="Clinical recommendations")
    component_breakdown: Dict[str, float] = Field(..., description="Detailed component scores")

class EnhancedScreeningResponse(BaseModel):
    prediction: Dict[str, Any] = Field(..., description="Prediction results")
    component_analysis: Dict[str, Any] = Field(..., description="Component scores analysis")
    interpretation: Dict[str, Any] = Field(..., description="Score interpretation")
    knowledge_guided_explanation: KnowledgeExplanation = Field(..., description="Knowledge-guided explanation")
    processing_metadata: Dict[str, Any] = Field(..., description="Processing metadata")