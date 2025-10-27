# app/models/model_loader.py
import torch
import joblib
import logging
from pathlib import Path
from app.core.config import settings
from app.models.stgcn import STGCNRegression
from app.models.graph_utils import SMPL_ADJ_MATRIX

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, device="cuda"):
        self.device = device
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
                self.tsn_model = torch.load(settings.TSN_MODEL_PATH, map_location=self.device)
                self.tsn_model.eval()
                logger.info("TSN model loaded successfully")
            
            # Load SGCN model  
            if Path(settings.SGCN_MODEL_PATH).exists():
                self.sgcn_model = torch.load(settings.SGCN_MODEL_PATH, map_location=self.device)
                self.sgcn_model.eval()
                logger.info("SGCN model loaded successfully")
                
            # Load STGCN model
            if Path(settings.STGCN_MODEL_PATH).exists():
                checkpoint = torch.load(settings.STGCN_MODEL_PATH, map_location=self.device)
                
                # Initialize STGCN with correct architecture
                self.stgcn_model = STGCNRegression(
                    A_tensor=SMPL_ADJ_MATRIX.to(self.device),
                    in_channels=3,
                    hidden_channels=128,
                    num_blocks=3
                )
                
                # Load weights
                if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                    self.stgcn_model.load_state_dict(checkpoint["model_state"])
                else:
                    self.stgcn_model.load_state_dict(checkpoint)
                    
                self.stgcn_model.to(self.device)
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
model_loader = ModelLoader(device="cuda")