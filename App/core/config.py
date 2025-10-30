import os
from functools import lru_cache


class Settings:
	app_name: str = os.getenv("APP_NAME", "SW-AI-6 API")
	environment: str = os.getenv("ENVIRONMENT", "development")
	log_level: str = os.getenv("LOG_LEVEL", "INFO")
	allow_origins: str = os.getenv("ALLOW_ORIGINS", "*")
	port: int = int(os.getenv("PORT", "8000"))
	host: str = os.getenv("HOST", "0.0.0.0")

	# Model/device config
	device: str = os.getenv("DEVICE", "auto")  # auto|cpu|cuda
	model_tsn_path: str = os.getenv("MODEL_TSN_PATH", "model_weights/tsn_optical_flow.pth")
	model_sgcn_path: str = os.getenv("MODEL_SGCN_PATH", "model_weights/sgcn_2d.pth")
	model_stgcn_path: str = os.getenv("MODEL_STGCN_PATH", "model_weights/stgcn_3d.pth")
	model_fusion_path: str = os.getenv("MODEL_FUSION_PATH", "model_weights/fusion.pkl")

	# Knowledge corpus (optional CSV)
	knowledge_corpus_path: str = os.getenv("KNOWLEDGE_CORPUS_PATH", "knowledge_corpus.csv")

	# Severity bands mapping (thresholds inclusive ranges)
	severity_bands: str = os.getenv("SEVERITY_BANDS", "0-2:Minimal,3-4:Low,5-7:Moderate,8-10:High")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
	return Settings()


