# app/services/video_processor.py
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import logging
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import tempfile
import mediapipe as mp
from app.services.optical_flow import OpticalFlowProcessor

logger = logging.getLogger(__name__)

class VideoPreprocessor:
    def __init__(self, target_size=(224, 224), device="cuda"):
        self.target_size = target_size
        self.device = device
        self.optical_flow_processor = OpticalFlowProcessor()
        self._initialize_pose_estimators()
        
    def _initialize_pose_estimators(self):
        """Initialize MediaPipe for 2D pose and load ROMP for 3D pose"""
        try:
            # Initialize MediaPipe Pose for 2D skeleton
            self.mp_pose = mp.solutions.pose
            self.pose_2d = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            logger.info("MediaPipe Pose initialized for 2D skeleton estimation")
            
            # Initialize ROMP for 3D pose estimation
            try:
                from romp import ROMP
                self.romp = ROMP()
                logger.info("ROMP initialized for 3D skeleton estimation")
            except ImportError:
                logger.warning("ROMP not available, 3D pose estimation will use MediaPipe")
                self.romp = None
                
        except Exception as e:
            logger.error(f"Error initializing pose estimators: {str(e)}")
            raise
    
    def extract_frames(self, video_path: str, target_frames: int = 64) -> List[np.ndarray]:
        """Extract frames from video with uniform sampling"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                raise ValueError("Video has no frames")
            
            logger.info(f"Video: {total_frames} frames, {fps:.2f} FPS")
            
            # Calculate frame indices to sample
            if total_frames <= target_frames:
                frame_indices = list(range(total_frames))
            else:
                frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            
            frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize frame to target size
                    frame_resized = cv2.resize(frame_rgb, self.target_size)
                    frames.append(frame_resized)
                else:
                    if frames:
                        frames.append(frames[-1].copy())
                    else:
                        frames.append(np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8))
            
            cap.release()
            logger.info(f"Extracted {len(frames)} frames from {Path(video_path).name}")
            return frames
            
        except Exception as e:
            logger.error(f"Error extracting frames from {video_path}: {str(e)}")
            raise
    
    def preprocess_frames_for_tsn(self, frames: List[np.ndarray], num_frames: int = 10) -> torch.Tensor:
        """Preprocess frames for TSN model (optical flow input)"""
        try:
            if len(frames) < 2:
                raise ValueError("Need at least 2 frames for optical flow computation")
            
            # Sample frames for optical flow
            if len(frames) <= num_frames:
                sampled_frames = frames
            else:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                sampled_frames = [frames[i] for i in indices]
            
            # Compute optical flow between consecutive frames
            flow_frames = []
            for i in range(len(sampled_frames) - 1):
                flow_x, flow_y = self.optical_flow_processor.compute_flow(
                    sampled_frames[i], sampled_frames[i + 1]
                )
                flow_rgb = self.optical_flow_processor.flow_to_rgb(flow_x, flow_y)
                flow_frames.append(flow_rgb)
            
            # Handle edge cases
            if not flow_frames:
                # Create zero flow frames
                zero_flow = Image.new('RGB', self.target_size, (0, 0, 0))
                flow_frames = [zero_flow] * num_frames
            elif len(flow_frames) < num_frames:
                # Duplicate last frame to reach target length
                last_frame = flow_frames[-1]
                while len(flow_frames) < num_frames:
                    flow_frames.append(last_frame)
            
            # Apply transformations
            transform = self._get_optical_flow_transform()
            processed_frames = []
            
            for flow_frame in flow_frames[:num_frames]:
                tensor_frame = transform(flow_frame)
                processed_frames.append(tensor_frame)
            
            # Stack frames: [num_frames, channels, height, width]
            frames_tensor = torch.stack(processed_frames)  # [T, C, H, W]
            
            return frames_tensor.unsqueeze(0)  # [1, T, C, H, W]
            
        except Exception as e:
            logger.error(f"Error preprocessing frames for TSN: {str(e)}")
            raise
    
    def extract_skeleton_2d(self, frames: List[np.ndarray], num_frames: int = 4) -> torch.Tensor:
        """Extract 2D skeleton coordinates using MediaPipe Pose"""
        try:
            if len(frames) < num_frames:
                sampled_frames = frames
                # Pad with last frame if needed
                while len(sampled_frames) < num_frames:
                    sampled_frames.append(sampled_frames[-1])
            else:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                sampled_frames = [frames[i] for i in indices]
            
            skeleton_2d_list = []
            
            for frame in sampled_frames:
                # Convert to MediaPipe format
                frame_mp = frame.astype(np.uint8)
                results = self.pose_2d.process(frame_mp)
                
                if results.pose_landmarks:
                    # Extract 2D coordinates (33 landmarks from MediaPipe)
                    landmarks = results.pose_landmarks.landmark
                    frame_coords = np.zeros((33, 2), dtype=np.float32)
                    
                    for i, landmark in enumerate(landmarks):
                        frame_coords[i] = [landmark.x * self.target_size[0], 
                                         landmark.y * self.target_size[1]]
                    
                    # Convert MediaPipe 33 joints to SMPL 24 joints format
                    skeleton_2d = self._convert_mediapipe_to_smpl(frame_coords)
                else:
                    # No pose detected, use zeros
                    skeleton_2d = np.zeros((24, 2), dtype=np.float32)
                
                skeleton_2d_list.append(skeleton_2d)
            
            skeleton_2d_array = np.array(skeleton_2d_list)  # [T, 24, 2]
            
            # Normalize coordinates
            skeleton_2d_normalized = self._normalize_skeleton_2d(skeleton_2d_array)
            
            return torch.tensor(skeleton_2d_normalized, dtype=torch.float32).unsqueeze(0)  # [1, T, J, 2]
            
        except Exception as e:
            logger.error(f"Error extracting 2D skeleton: {str(e)}")
            raise
    
    def extract_skeleton_3d(self, frames: List[np.ndarray], num_frames: int = 32) -> torch.Tensor:
        """Extract 3D skeleton coordinates using ROMP"""
        try:
            if len(frames) < num_frames:
                sampled_frames = frames
                while len(sampled_frames) < num_frames:
                    sampled_frames.append(sampled_frames[-1])
            else:
                indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
                sampled_frames = [frames[i] for i in indices]
            
            skeleton_3d_list = []
            
            if self.romp:
                # Use ROMP for 3D pose estimation
                for frame in sampled_frames:
                    try:
                        results = self.romp(frame)
                        if results and 'smpl_thetas' in results:
                            # ROMP returns SMPL parameters, extract 3D joints
                            # This is simplified - in production you'd extract the 24 joints
                            joints_3d = self._extract_joints_from_romp(results)
                            skeleton_3d_list.append(joints_3d)
                        else:
                            # Fallback to 2D-to-3D conversion
                            joints_3d = self._estimate_3d_from_2d(frame)
                            skeleton_3d_list.append(joints_3d)
                    except Exception as e:
                        logger.warning(f"ROMP failed on frame, using fallback: {str(e)}")
                        joints_3d = self._estimate_3d_from_2d(frame)
                        skeleton_3d_list.append(joints_3d)
            else:
                # Fallback: estimate 3D from 2D using simple method
                for frame in sampled_frames:
                    joints_3d = self._estimate_3d_from_2d(frame)
                    skeleton_3d_list.append(joints_3d)
            
            skeleton_3d_array = np.array(skeleton_3d_list)  # [T, 24, 3]
            
            # Normalize coordinates
            skeleton_3d_normalized = self._normalize_skeleton_3d(skeleton_3d_array)
            
            # Convert to channels-first format: [C, T, J] = [3, 32, 24]
            skeleton_3d_transposed = np.transpose(skeleton_3d_normalized, (2, 0, 1))
            
            return torch.tensor(skeleton_3d_transposed, dtype=torch.float32).unsqueeze(0)  # [1, 3, T, J]
            
        except Exception as e:
            logger.error(f"Error extracting 3D skeleton: {str(e)}")
            raise
    
    def _convert_mediapipe_to_smpl(self, mediapipe_coords: np.ndarray) -> np.ndarray:
        """Convert MediaPipe 33 joints to SMPL 24 joints format"""
        # MediaPipe to SMPL joint mapping (simplified)
        # This mapping needs to be adjusted based on your exact SMPL joint definitions
        smpl_coords = np.zeros((24, 2), dtype=np.float32)
        
        # Map key joints (this is a simplified mapping)
        mapping = {
            # Body core
            0: 23,  # Pelvis (approximate)
            11: 1,   # Left hip
            12: 2,   # Right hip
            13: 4,   # Left knee
            14: 5,   # Right knee
            15: 7,   # Left ankle
            16: 8,   # Right ankle
            23: 3,   # Spine (approximate)
            24: 6,   # Neck (approximate)
            25: 9,   # Head (approximate)
            # Arms
            11: 13,  # Left shoulder
            12: 14,  # Right shoulder
            13: 16,  # Left elbow
            14: 17,  # Right elbow
            15: 18,  # Left wrist
            16: 19,  # Right wrist
        }
        
        for smpl_idx, mp_idx in mapping.items():
            if mp_idx < len(mediapipe_coords):
                smpl_coords[smpl_idx] = mediapipe_coords[mp_idx]
        
        return smpl_coords
    
    def _extract_joints_from_romp(self, romp_results: Dict) -> np.ndarray:
        """Extract 3D joints from ROMP results"""
        # ROMP returns SMPL parameters, we need to extract the 24 joints
        # This is a placeholder - you'd need to implement proper joint extraction
        # from SMPL parameters
        try:
            # Simplified: return random joints (replace with actual extraction)
            joints_3d = np.random.randn(24, 3).astype(np.float32)
            return joints_3d
        except:
            # Fallback
            return np.zeros((24, 3), dtype=np.float32)
    
    def _estimate_3d_from_2d(self, frame: np.ndarray) -> np.ndarray:
        """Estimate 3D pose from 2D using MediaPipe and simple depth estimation"""
        try:
            # Get 2D pose first
            frame_mp = frame.astype(np.uint8)
            results = self.pose_2d.process(frame_mp)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                joints_3d = np.zeros((24, 3), dtype=np.float32)
                
                for i in range(24):
                    if i < len(landmarks):
                        landmark = landmarks[i]
                        # Simple depth estimation based on joint relationships
                        depth = self._estimate_joint_depth(i, landmarks)
                        joints_3d[i] = [
                            landmark.x * self.target_size[0],
                            landmark.y * self.target_size[1],
                            depth
                        ]
                
                return joints_3d
            else:
                return np.zeros((24, 3), dtype=np.float32)
                
        except Exception as e:
            logger.warning(f"3D estimation from 2D failed: {str(e)}")
            return np.zeros((24, 3), dtype=np.float32)
    
    def _estimate_joint_depth(self, joint_idx: int, landmarks: List) -> float:
        """Simple depth estimation based on joint relationships"""
        # Basic depth estimation logic
        if joint_idx in [23, 24, 25]:  # Head/neck area
            return 0.0
        elif joint_idx in [11, 12]:  # Shoulders
            return -0.2
        elif joint_idx in [13, 14, 15, 16]:  # Arms
            return -0.4
        else:  # Lower body
            return -0.6
    
    def _normalize_skeleton_2d(self, skeleton: np.ndarray) -> np.ndarray:
        """Normalize 2D skeleton coordinates per sequence"""
        skeleton = skeleton.copy().astype(np.float32)
        skeleton_centered = skeleton - skeleton.mean(axis=1, keepdims=True)
        max_norm = np.max(np.linalg.norm(skeleton_centered, axis=2))
        return skeleton_centered / (max_norm + 1e-8) if max_norm > 1e-8 else skeleton_centered
    
    def _normalize_skeleton_3d(self, skeleton: np.ndarray) -> np.ndarray:
        """Normalize 3D skeleton coordinates per sequence"""
        skeleton = skeleton.copy().astype(np.float32)
        skeleton_centered = skeleton - skeleton.mean(axis=1, keepdims=True)
        max_norm = np.max(np.linalg.norm(skeleton_centered, axis=2))
        return skeleton_centered / (max_norm + 1e-8) if max_norm > 1e-8 else skeleton_centered
    
    def _get_optical_flow_transform(self):
        """Get transformations for optical flow frames"""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
    
    def process_video(self, video_path: str) -> Dict[str, torch.Tensor]:
        """Main method to process video and extract all required inputs"""
        try:
            logger.info(f"Processing video: {Path(video_path).name}")
            
            # Extract frames
            frames = self.extract_frames(video_path, target_frames=64)
            
            # Process for each model
            tsn_input = self.preprocess_frames_for_tsn(frames, num_frames=10)
            sgcn_input = self.extract_skeleton_2d(frames, num_frames=4)
            stgcn_input = self.extract_skeleton_3d(frames, num_frames=32)
            
            logger.info(f"Processing complete - TSN: {tsn_input.shape}, SGCN: {sgcn_input.shape}, STGCN: {stgcn_input.shape}")
            
            return {
                'tsn': tsn_input.to(self.device),
                'sgcn': sgcn_input.to(self.device),
                'stgcn': stgcn_input.to(self.device)
            }
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'pose_2d'):
            self.pose_2d.close()
        if hasattr(self, 'romp'):
            del self.romp