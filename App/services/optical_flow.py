# app/services/optical_flow.py
import cv2
import numpy as np
import torch
from PIL import Image
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class OpticalFlowProcessor:
    def __init__(self, method='lucas_kanade'):
        self.method = method
        self.prev_gray = None
        
    def compute_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute optical flow between two consecutive frames"""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        if self.method == 'lucas_kanade':
            return self._lucas_kanade_flow(gray1, gray2)
        else:
            raise ValueError(f"Unsupported optical flow method: {self.method}")
    
    def _lucas_kanade_flow(self, gray1: np.ndarray, gray2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Lucas-Kanade optical flow"""
        # Parameters for Lucas-Kanade
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection
        feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Detect features to track
        p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
        
        if p0 is not None:
            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
            
            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]
                
                # Create dense flow field
                flow_x = np.zeros_like(gray1, dtype=np.float32)
                flow_y = np.zeros_like(gray1, dtype=np.float32)
                
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    flow_x[int(b), int(a)] = a - c
                    flow_y[int(b), int(a)] = b - d
                
                return flow_x, flow_y
        
        # Fallback: return zero flow if no features detected
        return np.zeros_like(gray1, dtype=np.float32), np.zeros_like(gray1, dtype=np.float32)
    
    def flow_to_rgb(self, flow_x: np.ndarray, flow_y: np.ndarray) -> Image.Image:
        """Convert optical flow components to RGB image"""
        # Normalize flow components to [0, 255]
        flow_x_norm = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        flow_y_norm = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create RGB image with x and y components
        flow_rgb = np.stack([flow_x_norm, flow_y_norm, flow_x_norm], axis=2)  # [H, W, 3]
        return Image.fromarray(flow_rgb)