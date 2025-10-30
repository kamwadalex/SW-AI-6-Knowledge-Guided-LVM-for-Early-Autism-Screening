from pathlib import Path
from typing import Optional

import numpy as np


class SGCNInference:
	def __init__(self, checkpoint_path: Optional[Path] = None) -> None:
		self.checkpoint_path = checkpoint_path
		self.available = False
		try:
			import torch  # noqa: F401
			self.available = checkpoint_path is not None and Path(checkpoint_path).exists()
		except Exception:
			self.available = False

	def _build_model(self, A_tensor):
		import torch
		import torch.nn as nn
		import torch.nn.functional as F
		class SpatialGCNLayer(nn.Module):
			def __init__(self, in_channels, out_channels, bias=True):
				super().__init__(); self.fc = nn.Linear(in_channels, out_channels, bias=bias)
			def forward(self, x, A):
				B,T,J,C = x.shape
				x = self.fc(x)
				x = x.view(B*T, J, -1)
				x = torch.matmul(A, x)
				x = x.view(B, T, J, -1)
				return F.relu(x)
		class SGCNRegression(nn.Module):
			def __init__(self, adj, in_channels=2, hidden=128, layers=3, dropout=0.2):
				super().__init__(); self.adj = adj
				self.layers = nn.ModuleList([SpatialGCNLayer(in_channels if i==0 else hidden, hidden) for i in range(layers)])
				self.dropout = nn.Dropout(dropout)
				self.regressor = nn.Sequential(nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden//2, 1))
			def forward(self, x):
				A = self.adj
				for lyr in self.layers:
					x = lyr(x, A)
				x = x.mean(dim=2).mean(dim=1)
				x = self.dropout(x)
				return self.regressor(x).squeeze(1)
		return SGCNRegression(A_tensor)

	def predict_from_2d_sequence(self, seq_2d: np.ndarray, device: str = "auto") -> float:
		if not self.available:
			raise RuntimeError("SGCN inference unavailable. Install torch and provide checkpoint.")
		import torch
		# Build adjacency for 24 joints
		edges = [(0,1),(0,2),(0,3),(1,4),(4,7),(7,10),(2,5),(5,8),(8,11),(3,6),(6,9),(9,12),(12,13),(13,16),(16,18),(18,20),(20,22),(12,14),(14,17),(17,19),(19,21),(21,23),(12,15)]
		A = np.zeros((24,24), dtype=np.float32)
		for i,j in edges:
			if i<24 and j<24:
				A[i,j]=1; A[j,i]=1
		np.fill_diagonal(A,1)
		D = np.sum(A, axis=1); D_inv = np.diag(1.0/ (np.sqrt(D)+1e-8)); A_norm = D_inv @ A @ D_inv
		A_tensor = torch.tensor(A_norm, dtype=torch.float32)
		model = self._build_model(A_tensor)
		state = torch.load(str(self.checkpoint_path), map_location="cpu")
		try:
			model.load_state_dict(state if not isinstance(state, dict) else state.get("model_state", state), strict=False)
		except Exception:
			pass
		device_sel = torch.device("cuda" if (device=="cuda" or (device=="auto" and torch.cuda.is_available())) else "cpu")
		model.to(device_sel); model.eval()
		tensor = torch.tensor(seq_2d, dtype=torch.float32).unsqueeze(0).to(device_sel)  # (1,T,24,2)
		with torch.no_grad():
			pred = model(tensor)
		return float(pred.detach().cpu().item())


