# app/models/stgcn.py
import torch
import torch.nn as nn
import numpy as np

class SpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x, A):
        # x shape: [B, C, T, V] where V is number of joints
        B, C, T, V = x.shape
        # Reshape for spatial convolution: [B*T, V, C]
        xt = x.permute(0, 2, 3, 1).contiguous().view(B * T, V, C)
        # Apply adjacency matrix: [B*T, V, C]
        xt = torch.matmul(A, xt)
        # Reshape back: [B, C, T, V]
        xt = xt.view(B, T, V, C).permute(0, 3, 1, 2)
        return self.fc(xt)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A_tensor, temporal_kernel_size=9, stride=1, dropout=0.2):
        super().__init__()
        self.A = A_tensor
        self.spatial = SpatialConv(in_channels, out_channels)
        padding = (temporal_kernel_size - 1) // 2
        self.temporal = nn.Conv2d(
            out_channels, out_channels, 
            kernel_size=(temporal_kernel_size, 1), 
            padding=(padding, 0), 
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.spatial(x, self.A)
        out = self.temporal(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class STGCNRegression(nn.Module):
    def __init__(self, A_tensor, in_channels=3, hidden_channels=128, num_blocks=3):
        super().__init__()
        self.A = A_tensor
        self.input_proj = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            STGCNBlock(hidden_channels, hidden_channels, self.A)
            for _ in range(num_blocks)
        ])
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, x):
        # x shape: [B, C, T, V] = [B, 3, 32, 24]
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        # Global average pooling over temporal and spatial dimensions
        emb = x.mean(dim=[2, 3])  # [B, hidden_channels]
        return self.regressor(emb).squeeze(1)