from pathlib import Path
from typing import Optional

import numpy as np


class TSNRegressorInference:
	def __init__(self, checkpoint_path: Optional[Path] = None) -> None:
		self.checkpoint_path = checkpoint_path
		self.available = False
		try:
			import torch  # noqa: F401
			import torchvision  # noqa: F401
			self.available = checkpoint_path is not None and Path(checkpoint_path).exists()
		except Exception:
			self.available = False

	def _build_model(self):
		import torch
		import torch.nn as nn
		from torchvision import models
		class TSNRegressor(nn.Module):
			def __init__(self):
				super().__init__()
				base = models.resnet18(weights=None)
				base.fc = nn.Identity()
				self.base = base
				self.reg_head = nn.Sequential(
					nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
				)
			def forward(self, x):
				B,T,C,H,W = x.shape
				x = x.view(B*T, C, H, W)
				with torch.no_grad():
					feats = self.base(x)
				feats = feats.view(B, T, -1).mean(1)
				out = self.reg_head(feats)
				return out.squeeze(1)
		return TSNRegressor()

	def predict_from_flow_rgb_stack(self, flow_rgb_frames: np.ndarray, device: str = "auto") -> float:
		if not self.available:
			raise RuntimeError("TSN inference unavailable. Install torch/torchvision and provide checkpoint.")
		import torch
		from torchvision import transforms
		model = self._build_model()
		state = torch.load(str(self.checkpoint_path), map_location="cpu")
		try:
			model.load_state_dict(state, strict=False)
		except Exception:
			if isinstance(state, dict) and "model_state" in state:
				model.load_state_dict(state["model_state"], strict=False)
		device_sel = torch.device("cuda" if (device == "cuda" or (device=="auto" and torch.cuda.is_available())) else "cpu")
		model.to(device_sel); model.eval()
		tfm = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
		])
		if len(flow_rgb_frames) <= 10:
			idxs = list(range(len(flow_rgb_frames)))
		else:
			idxs = list(np.linspace(0, len(flow_rgb_frames)-1, 10, dtype=int))
		frames = []
		from PIL import Image
		for i in idxs:
			arr = flow_rgb_frames[i]
			if isinstance(arr, np.ndarray):
				img = Image.fromarray(arr.astype(np.uint8))
				frames.append(tfm(img))
			else:
				frames.append(tfm(arr))
		import torch as _t
		tensor = _t.stack(frames, dim=0).unsqueeze(0).to(device_sel)
		with torch.no_grad():
			pred = model(tensor)
		return float(pred.detach().cpu().item())


