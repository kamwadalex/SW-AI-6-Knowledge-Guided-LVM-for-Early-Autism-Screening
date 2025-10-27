# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
import os

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "Autism Screening API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Multimodal AI system for autism screening using computer vision"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False  # Set to True for development
    
    # Model paths
    TSN_MODEL_PATH: str = "model_weights/tsn_optical_flow.pth"
    SGCN_MODEL_PATH: str = "model_weights/sgcn_2d.pth"
    STGCN_MODEL_PATH: str = "model_weights/stgcn_3d.pth"
    FUSION_MODEL_PATH: str = "model_weights/fusion.pkl"
    
    # Video processing
    FRAME_SIZE: tuple = (224, 224)
    MAX_VIDEO_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    
    # Model-specific parameters
    TSN_NUM_FRAMES: int = 10
    SGCN_NUM_FRAMES: int = 4
    STGCN_NUM_FRAMES: int = 32
    
    # Inference settings
    DEVICE: str = "cuda"
    BATCH_SIZE: int = 1
    
    # Security
    API_KEYS: List[str] = []  # Add your API keys here
    
    # CORS
    CORS_ORIGINS: List[str] = ["*"]
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()