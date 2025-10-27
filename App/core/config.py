# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List, Optional
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Action Recognition API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Model paths
    TSN_MODEL_PATH: str = "model_weights/tsn_optical_flow.pth"
    SGCN_MODEL_PATH: str = "model_weights/sgcn_2d.pth" 
    STGCN_MODEL_PATH: str = "model_weights/stgcn_3d.pth"
    FUSION_MODEL_PATH: str = "model_weights/fusion.pkl"
    
    # Inference settings
    BATCH_SIZE: int = 1
    FRAME_COUNT: int = 64  # Adjust based on your model requirements
    IMAGE_SIZE: tuple = (224, 224)
    
    # API settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()