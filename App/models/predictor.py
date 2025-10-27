# app/models/predictor.py
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import time
from app.core.config import settings
from app.models.model_loader import model_loader

logger = logging.getLogger(__name__)

class AutismScreeningPredictor:
    def __init__(self):
        self.models = model_loader.get_models()
        self.device = settings.DEVICE
        self._validate_models()
    
    def _validate_models(self):
        """Validate that all required models are loaded"""
        required_models = ['tsn', 'sgcn', 'stgcn', 'fusion']
        for model_name in required_models:
            if self.models[model_name] is None:
                raise ValueError(f"Model {model_name} is not loaded")
        logger.info("All models validated and ready for inference")
    
    def predict_tsn(self, tsn_input: torch.Tensor) -> float:
        """Run inference with TSN model on optical flow data"""
        try:
            with torch.no_grad():
                start_time = time.time()
                prediction = self.models['tsn'](tsn_input)
                inference_time = time.time() - start_time
                
                # Ensure scalar output
                if prediction.dim() > 0:
                    prediction = prediction.mean()
                
                score = prediction.item()
                logger.info(f"TSN inference completed in {inference_time:.3f}s - Score: {score:.3f}")
                return score
                
        except Exception as e:
            logger.error(f"TSN inference failed: {str(e)}")
            raise
    
    def predict_sgcn(self, sgcn_input: torch.Tensor) -> float:
        """Run inference with SGCN model on 2D skeleton data"""
        try:
            with torch.no_grad():
                start_time = time.time()
                prediction = self.models['sgcn'](sgcn_input)
                inference_time = time.time() - start_time
                
                # Ensure scalar output
                if prediction.dim() > 0:
                    prediction = prediction.mean()
                
                score = prediction.item()
                logger.info(f"SGCN inference completed in {inference_time:.3f}s - Score: {score:.3f}")
                return score
                
        except Exception as e:
            logger.error(f"SGCN inference failed: {str(e)}")
            raise
    
    def predict_stgcn(self, stgcn_input: torch.Tensor) -> float:
        """Run inference with STGCN model on 3D skeleton data"""
        try:
            with torch.no_grad():
                start_time = time.time()
                prediction = self.models['stgcn'](stgcn_input)
                inference_time = time.time() - start_time
                
                # Ensure scalar output
                if prediction.dim() > 0:
                    prediction = prediction.mean()
                
                score = prediction.item()
                logger.info(f"STGCN inference completed in {inference_time:.3f}s - Score: {score:.3f}")
                return score
                
        except Exception as e:
            logger.error(f"STGCN inference failed: {str(e)}")
            raise
    
    def fuse_predictions(self, tsn_score: float, sgcn_score: float, stgcn_score: float) -> Dict:
        """Fuse individual model predictions using the fusion model"""
        try:
            start_time = time.time()
            
            # Prepare input for fusion model
            fusion_input = np.array([[tsn_score, sgcn_score, stgcn_score]])
            
            # Get fusion prediction
            final_score = self.models['fusion'].predict(fusion_input)[0]
            
            # Get confidence scores or probabilities if available
            confidence = self._calculate_confidence(tsn_score, sgcn_score, stgcn_score, final_score)
            
            fusion_time = time.time() - start_time
            logger.info(f"Fusion completed in {fusion_time:.3f}s - Final score: {final_score:.3f}")
            
            return {
                'final_score': float(final_score),
                'confidence': float(confidence),
                'component_scores': {
                    'optical_flow': float(tsn_score),
                    'skeleton_2d': float(sgcn_score),
                    'skeleton_3d': float(stgcn_score)
                }
            }
            
        except Exception as e:
            logger.error(f"Fusion failed: {str(e)}")
            raise
    
    def _calculate_confidence(self, tsn_score: float, sgcn_score: float, stgcn_score: float, final_score: float) -> float:
        """Calculate confidence score based on prediction consistency"""
        try:
            # Calculate variance between component scores
            scores = np.array([tsn_score, sgcn_score, stgcn_score])
            variance = np.var(scores)
            
            # Normalize variance to confidence (lower variance = higher confidence)
            max_expected_variance = 5.0  # Based on ADOS score range 1-10
            variance_confidence = max(0, 1 - (variance / max_expected_variance))
            
            # Additional confidence based on how close scores are to final prediction
            deviations = np.abs(scores - final_score)
            deviation_confidence = 1 - (np.mean(deviations) / max_expected_variance)
            
            # Combined confidence
            confidence = (variance_confidence + deviation_confidence) / 2
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {str(e)}")
            return 0.5  # Default medium confidence
    
    def interpret_score(self, final_score: float) -> Dict:
        """Interpret the final ADOS score into severity categories"""
        try:
            # ADOS Comparison Score interpretation (1-10 scale)
            if final_score <= 3:
                severity = "Minimal"
                description = "Minimal evidence of autism symptoms"
                recommendation = "Routine developmental monitoring recommended"
            elif final_score <= 5:
                severity = "Mild"
                description = "Mild autism symptoms present"
                recommendation = "Consider comprehensive evaluation"
            elif final_score <= 7:
                severity = "Moderate"
                description = "Moderate autism symptoms"
                recommendation = "Recommend comprehensive diagnostic assessment"
            else:  # 8-10
                severity = "Severe"
                description = "Significant autism symptoms"
                recommendation = "Urgent comprehensive evaluation needed"
            
            return {
                'severity': severity,
                'description': description,
                'recommendation': recommendation,
                'score_range': '1-10',
                'interpretation_notes': 'Based on ADOS Comparison Score standards'
            }
            
        except Exception as e:
            logger.error(f"Score interpretation failed: {str(e)}")
            return {
                'severity': 'Unknown',
                'description': 'Interpretation unavailable',
                'recommendation': 'Consult with healthcare provider',
                'score_range': '1-10',
                'interpretation_notes': 'Interpretation error occurred'
            }
    
    def predict(self, model_inputs: Dict[str, torch.Tensor]) -> Dict:
        """
        Main prediction method that orchestrates all models and fusion
        
        Args:
            model_inputs: Dictionary containing preprocessed inputs for each model
                - 'tsn': Optical flow tensor [1, 10, 3, 224, 224]
                - 'sgcn': 2D skeleton tensor [1, 4, 24, 2] 
                - 'stgcn': 3D skeleton tensor [1, 3, 32, 24]
        
        Returns:
            Dictionary containing complete prediction results
        """
        try:
            total_start_time = time.time()
            logger.info("Starting autism screening prediction pipeline")
            
            # Validate inputs
            self._validate_inputs(model_inputs)
            
            # Run individual model predictions
            tsn_score = self.predict_tsn(model_inputs['tsn'])
            sgcn_score = self.predict_sgcn(model_inputs['sgcn'])
            stgcn_score = self.predict_stgcn(model_inputs['stgcn'])
            
            # Fuse predictions
            fusion_result = self.fuse_predictions(tsn_score, sgcn_score, stgcn_score)
            
            # Interpret the final score
            interpretation = self.interpret_score(fusion_result['final_score'])
            
            # Calculate total processing time
            total_time = time.time() - total_start_time
            
            # Compile final results
            result = {
                'prediction': {
                    'final_score': fusion_result['final_score'],
                    'severity': interpretation['severity'],
                    'confidence': fusion_result['confidence']
                },
                'component_analysis': fusion_result['component_scores'],
                'interpretation': interpretation,
                'processing_metadata': {
                    'total_processing_time': total_time,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'model_versions': self._get_model_versions()
                }
            }
            
            logger.info(f"Prediction pipeline completed in {total_time:.3f}s - Final score: {fusion_result['final_score']:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction pipeline failed: {str(e)}")
            raise
    
    def _validate_inputs(self, model_inputs: Dict):
        """Validate that all required inputs are present and have correct shapes"""
        required_inputs = ['tsn', 'sgcn', 'stgcn']
        
        for input_name in required_inputs:
            if input_name not in model_inputs:
                raise ValueError(f"Missing required input: {input_name}")
            
            tensor = model_inputs[input_name]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Input {input_name} must be a torch.Tensor")
            
            # Basic shape validation
            if input_name == 'tsn' and tensor.dim() != 5:
                raise ValueError(f"TSN input must be 5D tensor, got {tensor.dim()}D")
            elif input_name == 'sgcn' and tensor.dim() != 4:
                raise ValueError(f"SGCN input must be 4D tensor, got {tensor.dim()}D")
            elif input_name == 'stgcn' and tensor.dim() != 4:
                raise ValueError(f"STGCN input must be 4D tensor, got {tensor.dim()}D")
    
    def _get_model_versions(self) -> Dict:
        """Get model version information"""
        return {
            'tsn': 'ResNet18-based TSN for optical flow',
            'sgcn': 'Spatial GCN for 2D skeletons',
            'stgcn': 'Spatial-Temporal GCN for 3D skeletons',
            'fusion': 'GradientBoostingRegressor',
            'pipeline_version': '1.0.0'
        }
    
    def batch_predict(self, batch_inputs: List[Dict]) -> List[Dict]:
        """Run predictions on multiple inputs (batch processing)"""
        try:
            results = []
            for i, model_inputs in enumerate(batch_inputs):
                logger.info(f"Processing batch item {i+1}/{len(batch_inputs)}")
                try:
                    result = self.predict(model_inputs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch item {i+1} failed: {str(e)}")
                    # Add error result for failed prediction
                    results.append({
                        'error': str(e),
                        'success': False
                    })
            return results
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise

# Singleton instance
predictor = AutismScreeningPredictor()