from pathlib import Path
from typing import Dict, Tuple
import math

import numpy as np

from App.core.config import get_settings
from App.services.feature_extraction import FeatureExtractionService
from App.services.models.tsn import TSNRegressorInference
from App.services.models.sgcn import SGCNInference
from App.services.models.stgcn import STGCNInference
from App.services.fusion import FusionService
from App.services.knowledge import KnowledgeService


class InferencePipeline:
	def __init__(self) -> None:
		settings = get_settings()
		self.device = settings.device
		self.features = FeatureExtractionService()
		self.tsn = TSNRegressorInference(checkpoint_path=Path(settings.model_tsn_path))
		self.sgcn = SGCNInference(checkpoint_path=Path(settings.model_sgcn_path))
		self.stgcn = STGCNInference(checkpoint_path=Path(settings.model_stgcn_path))
		self.fusion = FusionService(model_path=Path(settings.model_fusion_path))
		self.knowledge = KnowledgeService()

	def run_on_video(self, video_path: Path) -> Tuple[Dict[str, float], float]:
		# 1) Frames
		frames = self.features.extract_frames(video_path)
		# 2) Optical flow → TSN
		flow_rgb = self.features.compute_optical_flow(frames)
		tsn_score = self.tsn.predict_from_flow_rgb_stack(flow_rgb, device=self.device)
		# Map TSN raw output to 0–10 using sigmoid for user-facing consistency
		try:
			tsn_score = 10.0 / (1.0 + math.exp(-float(tsn_score)))
		except Exception:
			pass
		# 3) 2D skeleton → SGCN
		seq2d = np.array(self.features.extract_2d_pose(frames), dtype=np.float32)  # (T,24,2)
		seq2d = self.features.normalize_2d(seq2d)
		seq2d = self.features.pad_or_sample(seq2d, target_len=4)
		sgcn_score = self.sgcn.predict_from_2d_sequence(seq2d, device=self.device)
		# 4) 3D skeleton → ST-GCN
		seq3d = np.array(self.features.extract_3d_pose(frames), dtype=np.float32)  # (T,24,3)
		seq3d = self.features.normalize_3d(seq3d)
		seq3d = self.features.pad_or_sample(seq3d, target_len=32)
		seq3d_ch_first = np.transpose(seq3d, (2,0,1))  # (3,T,24)
		stgcn_score = self.stgcn.predict_from_3d_sequence(seq3d_ch_first, device=self.device)
		# 5) Fusion
		scores = {"tsn": tsn_score, "sgcn": sgcn_score, "stgcn": stgcn_score}
		fused = self.fusion.predict(scores["tsn"], scores["sgcn"], scores["stgcn"])
		return scores, fused


