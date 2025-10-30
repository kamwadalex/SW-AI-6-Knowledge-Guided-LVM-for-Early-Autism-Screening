from pathlib import Path
from typing import Optional


class FusionService:
	def __init__(self, model_path: Optional[Path] = None) -> None:
		self.model_path = model_path
		self.model = None
		if model_path and Path(model_path).exists():
			try:
				import joblib
				self.model = joblib.load(model_path)
			except Exception:
				self.model = None

	def predict(self, tsn: float, sgcn: float, stgcn: float) -> float:
		if self.model is None:
			# fallback: simple average
			return float((tsn + sgcn + stgcn) / 3.0)
		return float(self.model.predict([[tsn, sgcn, stgcn]])[0])


