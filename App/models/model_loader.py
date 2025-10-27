# app/models/model_loader.py
import torch
import joblib
import logging
from pathlib import Path
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.tsn_model = None
        self.sgcn_model = None
        self.stgcn_model = None 
        self.fusion_model = None
        self._load_models()
    
    def _load_models(self):
        """Load all trained models with proper error handling"""
        try:
            # Load TSN model
            if Path(settings.TSN_MODEL_PATH).exists():
                self.tsn_model = torch.load(settings.TSN_MODEL_PATH, map_location='cpu')
                self.tsn_model.eval()
                logger.info("TSN model loaded successfully")
            
            # Load SGCN model  
            if Path(settings.SGCN_MODEL_PATH).exists():
                self.sgcn_model = torch.load(settings.SGCN_MODEL_PATH, map_location='cpu')
                self.sgcn_model.eval()
                logger.info("SGCN model loaded successfully")
                
            # Load STGCN model
            if Path(settings.STGCN_MODEL_PATH).exists():
                checkpoint = torch.load(settings.STGCN_MODEL_PATH, map_location='cpu')
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    from app.models.stgcn import STGCNRegression
                    # Need to recreate STGCN architecture
                    self.stgcn_model = STGCNRegression(adj_tensor=torch.eye(24))  # placeholder adj
                    self.stgcn_model.load_state_dict(checkpoint["model_state"])
                else:
                    self.stgcn_model = checkpoint
                self.stgcn_model.eval()
                logger.info("STGCN model loaded successfully")
                
            # Load fusion model
            if Path(settings.FUSION_MODEL_PATH).exists():
                self.fusion_model = joblib.load(settings.FUSION_MODEL_PATH)
                logger.info("Fusion model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def get_models(self):
        return {
            'tsn': self.tsn_model,
            'sgcn': self.sgcn_model,
            'stgcn': self.stgcn_model,
            'fusion': self.fusion_model
        }

# Singleton instance
model_loader = ModelLoader()