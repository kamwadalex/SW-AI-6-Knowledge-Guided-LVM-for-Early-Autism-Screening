from pathlib import Path
from typing import Optional

import numpy as np


class STGCNInference:
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
		class SpatialConv(nn.Module):
			def __init__(self, in_channels, out_channels):
				super().__init__(); self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=1)
			def forward(self, x, A):
				B,C,T,V = x.shape
				x_t = x.permute(0,2,3,1).contiguous().view(B*T, V, C)
				x_t = torch.matmul(A, x_t)
				x_t = x_t.view(B, T, V, C).permute(0,3,1,2)
				return self.fc(x_t)
		class STGCNBlock(nn.Module):
			def __init__(self, in_ch, out_ch, A, tks=9, stride=1, drop=0.2):
				super().__init__(); self.A=A; self.spatial=SpatialConv(in_ch,out_ch)
				pad=(tks-1)//2; self.temporal=nn.Conv2d(out_ch,out_ch,kernel_size=(tks,1),padding=(pad,0),stride=(stride,1))
				self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(); self.drop=nn.Dropout(drop)
			def forward(self,x):
				o = self.spatial(x,self.A); o=self.temporal(o); o=self.bn(o); o=self.relu(o); o=self.drop(o); return o
		class STGCNRegression(nn.Module):
			def __init__(self, A, in_ch=3, hidden=128, blocks=3):
				super().__init__(); self.A=A; self.input_proj=nn.Conv2d(in_ch,hidden,kernel_size=1)
				self.blocks=nn.ModuleList([STGCNBlock(hidden,hidden,A) for _ in range(blocks)])
				self.reg=nn.Sequential(nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(hidden//2,1))
			def forward(self,x):
				x=self.input_proj(x)
				for b in self.blocks: x=b(x)
				e=x.mean(dim=[2,3])
				return self.reg(e).squeeze(1)
		return STGCNRegression(A_tensor)

	def predict_from_3d_sequence(self, seq_3d_ch_first: np.ndarray, device: str = "auto") -> float:
		if not self.available:
			raise RuntimeError("ST-GCN inference unavailable. Install torch and provide checkpoint.")
		import torch
		# Build adjacency
		edges=[(0,1),(0,2),(0,3),(1,4),(4,7),(7,10),(2,5),(5,8),(8,11),(3,6),(6,9),(9,12),(12,13),(13,16),(16,18),(18,20),(20,22),(12,14),(14,17),(17,19),(19,21),(21,23),(12,15)]
		A=np.zeros((24,24),dtype=np.float32)
		for i,j in edges:
			if i<24 and j<24:
				A[i,j]=1;A[j,i]=1
		np.fill_diagonal(A,1)
		D=np.sum(A,axis=1); D_inv=np.diag(1.0/(np.sqrt(D)+1e-8)); A_norm=D_inv@A@D_inv
		A_tensor=torch.tensor(A_norm,dtype=torch.float32)
		model=self._build_model(A_tensor)
		state=torch.load(str(self.checkpoint_path), map_location="cpu")
		try:
			model.load_state_dict(state if not isinstance(state, dict) else state.get("model_state", state), strict=False)
		except Exception:
			pass
		device_sel=torch.device("cuda" if (device=="cuda" or (device=="auto" and torch.cuda.is_available())) else "cpu")
		model.to(device_sel); model.eval()
		tensor=torch.tensor(seq_3d_ch_first,dtype=torch.float32).unsqueeze(0).to(device_sel) # (1,3,T,24)
		with torch.no_grad():
			pred=model(tensor)
		return float(pred.detach().cpu().item())


