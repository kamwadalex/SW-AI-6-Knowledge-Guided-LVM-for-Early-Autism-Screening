# app/models/enhanced_predictor.py
import torch
import numpy as np
import logging
from typing import Dict
import time

from app.models.predictor import AutismScreeningPredictor
from app.services.knowledge_guide import KnowledgeGuide
from app.core.config import settings

logger = logging.getLogger(__name__)

class EnhancedAutismScreeningPredictor(AutismScreeningPredictor):
    def __init__(self):
        super().__init__()
        self.knowledge_guide = KnowledgeGuide("knowledge_corpus.csv")
        logger.info("Enhanced predictor with knowledge guidance initialized")
    
    def predict_with_explanation(self, model_inputs: Dict[str, torch.Tensor]) -> Dict:
        """
        Enhanced prediction with knowledge-guided explanations
        
        Returns:
            Dictionary with original predictions plus explanations
        """
        try:
            start_time = time.time()
            
            # Get base prediction
            base_result = super().predict(model_inputs)
            
            # Add knowledge-guided explanations
            enhanced_result = self.knowledge_guide.create_comprehensive_report(base_result)
            
            # Add processing metadata
            total_time = time.time() - start_time
            enhanced_result['processing_metadata']['total_processing_time'] = total_time
            enhanced_result['processing_metadata']['knowledge_guidance'] = True
            
            logger.info(f"Enhanced prediction completed in {total_time:.3f}s")
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced prediction failed: {str(e)}")
            # Fall back to base prediction
            return super().predict(model_inputs)

# Singleton instance
enhanced_predictor = EnhancedAutismScreeningPredictor()