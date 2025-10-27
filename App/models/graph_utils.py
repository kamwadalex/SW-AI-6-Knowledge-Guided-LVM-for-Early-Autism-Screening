# app/models/graph_utils.py
import torch
import numpy as np

def build_smpl_adjacency_matrix():
    """Build SMPL 24-joint adjacency matrix with normalization"""
    SMPL_EDGES_24 = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (4, 7), (7, 10),
        (2, 5), (5, 8), (8, 11), 
        (3, 6), (6, 9), (9, 12),
        (12, 13), (13, 16), (16, 18), (18, 20), (20, 22),
        (12, 14), (14, 17), (17, 19), (19, 21), (21, 23),
        (12, 15)
    ]
    
    num_nodes = 24
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # Build adjacency matrix
    for i, j in SMPL_EDGES_24:
        if i < num_nodes and j < num_nodes:
            A[i, j] = 1.0
            A[j, i] = 1.0
    
    # Add self-loops
    np.fill_diagonal(A, 1.0)
    
    # Symmetric normalization: D^{-1/2} A D^{-1/2}
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(D) + 1e-8))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    
    return torch.tensor(A_norm, dtype=torch.float32)

# Precompute adjacency matrix
SMPL_ADJ_MATRIX = build_smpl_adjacency_matrix()