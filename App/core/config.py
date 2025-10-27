# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
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
    
    # Hardware
    DEVICE: str = "cuda"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()