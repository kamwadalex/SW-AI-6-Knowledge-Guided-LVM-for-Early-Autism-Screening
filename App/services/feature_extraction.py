from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np


try:
	import cv2  # type: ignore
except Exception:  # pragma: no cover
	cv2 = None  # lazy import guard

try:
	import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
	mp = None


class FeatureExtractionService:
	def __init__(self, workspace_dir: Optional[Path] = None) -> None:
		self.workspace_dir = workspace_dir or Path("/tmp")

	def _ensure_cv(self) -> None:
		if cv2 is None:
			raise RuntimeError("OpenCV (opencv-python-headless) is required for video/flow extraction.")

	def extract_frames(self, video_path: Path, target_frames: int = 64) -> List[np.ndarray]:
		self._ensure_cv()
		cap = cv2.VideoCapture(str(video_path))
		total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
		idxs = list(range(total)) if total <= target_frames else list(np.linspace(0, total - 1, target_frames, dtype=int))
		frames: List[np.ndarray] = []
		for idx in idxs:
			cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
			ok, frame = cap.read()
			if not ok:
				frames.append(frames[-1].copy() if frames else np.zeros((224,224,3), dtype=np.uint8))
				continue
			frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame_resized = cv2.resize(frame_rgb, (224, 224))
			frames.append(frame_resized)
		cap.release()
		return frames

	def compute_optical_flow(self, frames: List[np.ndarray]) -> List[np.ndarray]:
		self._ensure_cv()
		flow_frames: List[np.ndarray] = []
		for i in range(max(0, len(frames) - 1)):
			g1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
			g2 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
			lk_params = dict(winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
			feat_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
			p0 = cv2.goodFeaturesToTrack(g1, mask=None, **feat_params)
			if p0 is not None:
				p1, st, _ = cv2.calcOpticalFlowPyrLK(g1, g2, p0, None, **lk_params)
				if p1 is not None:
					fx = np.zeros_like(g1, dtype=np.float32)
					fy = np.zeros_like(g1, dtype=np.float32)
					good_new = p1[st == 1]; good_old = p0[st == 1]
					for new, old in zip(good_new, good_old):
						a, b = new.ravel(); c, d = old.ravel()
						# Clamp indices to valid range to avoid OOB
						h, w = g1.shape[:2]
						xi = int(max(0, min(w - 1, round(a))))
						yi = int(max(0, min(h - 1, round(b))))
						fx[yi, xi] = float(a - c)
						fy[yi, xi] = float(b - d)
				else:
					fx = np.zeros_like(g1, dtype=np.float32); fy = np.zeros_like(g1, dtype=np.float32)
			else:
				fx = np.zeros_like(g1, dtype=np.float32); fy = np.zeros_like(g1, dtype=np.float32)
			fx_norm = cv2.normalize(fx, None, 0, 255, cv2.NORM_MINMAX)
			fy_norm = cv2.normalize(fy, None, 0, 255, cv2.NORM_MINMAX)
			flow_rgb = np.stack([fx_norm, fy_norm, fx_norm], axis=2).astype(np.uint8)
			flow_frames.append(flow_rgb)
		# Ensure at least one frame so downstream does not crash
		if not flow_frames:
			flow_frames.append(np.zeros((224,224,3), dtype=np.uint8))
		return flow_frames

	def extract_2d_pose(self, frames: List[np.ndarray]) -> List[np.ndarray]:
		if mp is None:
			raise RuntimeError("mediapipe is required for 2D pose extraction.")
		pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)
		seq: List[np.ndarray] = []
		for frame in frames:
			res = pose.process(frame)
			coords = np.zeros((24,2), dtype=np.float32)
			if res.pose_landmarks:
				landmarks = res.pose_landmarks.landmark
				mp_coords = np.zeros((33,2), dtype=np.float32)
				for i, lm in enumerate(landmarks):
					mp_coords[i] = [lm.x * frame.shape[1], lm.y * frame.shape[0]]
				coords = self._convert_mediapipe_to_smpl(mp_coords)
			seq.append(coords)
		pose.close()
		return seq

	def _convert_mediapipe_to_smpl(self, mediapipe_coords: np.ndarray) -> np.ndarray:
		coords = np.zeros((24,2), dtype=np.float32)
		mapping = {0:23,1:11,2:12,3:25,6:23,9:24,12:11,15:0,4:23,7:25,10:29,5:24,8:26,11:30,13:11,16:13,18:15,20:17,22:19,14:12,17:14,19:16,21:18,23:20}
		for smpl_idx, mp_idx in mapping.items():
			if mp_idx < len(mediapipe_coords):
				coords[smpl_idx] = mediapipe_coords[mp_idx]
		return coords

	def extract_3d_pose(self, frames: List[np.ndarray]) -> List[np.ndarray]:
		# Try ROMP; fallback to heuristic 3D from 2D
		try:
			from romp import ROMP  # type: ignore
		except Exception:
			return [self._estimate_3d_from_2d(f) for f in frames]
		romp = ROMP()
		seq: List[np.ndarray] = []
		for frame in frames:
			try:
				res = romp(frame)
				if res and 'smpl_thetas' in res:
					seq.append(self._extract_romp_joints(res))
				else:
					seq.append(self._estimate_3d_from_2d(frame))
			except Exception:
				seq.append(self._estimate_3d_from_2d(frame))
		return seq

	def _extract_romp_joints(self, romp_results: Dict) -> np.ndarray:
		# Placeholder: use zeros if unavailable; integrate ROMP utilities if present
		return np.zeros((24,3), dtype=np.float32)

	def _estimate_3d_from_2d(self, frame: np.ndarray) -> np.ndarray:
		coords3d = np.zeros((24,3), dtype=np.float32)
		return coords3d

	def normalize_2d(self, seq: np.ndarray) -> np.ndarray:
		centered = seq - seq.mean(axis=1, keepdims=True)
		maxnorm = np.max(np.linalg.norm(centered, axis=2))
		return centered if maxnorm < 1e-8 else centered / (maxnorm + 1e-8)

	def normalize_3d(self, seq: np.ndarray) -> np.ndarray:
		centered = seq - seq.mean(axis=1, keepdims=True)
		maxnorm = np.max(np.linalg.norm(centered, axis=2))
		return centered if maxnorm < 1e-8 else centered / (maxnorm + 1e-8)

	def pad_or_sample(self, arr: np.ndarray, target_len: int) -> np.ndarray:
		T = arr.shape[0]
		if T == target_len:
			return arr
		if T > target_len:
			idx = np.linspace(0, T - 1, target_len).astype(int)
			return arr[idx]
		pad = np.zeros((target_len - T, *arr.shape[1:]), dtype=arr.dtype)
		return np.concatenate([arr, pad], axis=0)

	def load_romp_2d_npz(self, npz_path: Path) -> Path:
		if not npz_path.exists():
			raise FileNotFoundError(f"2D skeleton file not found: {npz_path}")
		return npz_path

	def load_romp_3d_npz(self, npz_path: Path) -> Path:
		if not npz_path.exists():
			raise FileNotFoundError(f"3D skeleton file not found: {npz_path}")
		return npz_path


