#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm

# ---------------------------
# Environment Configuration
# ---------------------------
def get_config():
    """Detect environment and set paths"""
    config = {
        'use_mediapipe': True,  # Always use MediaPipe for portability
        'is_kaggle': os.path.exists('/kaggle/working'),
    }
    return config

CONFIG = get_config()

# ---------------------------
# Utilities
# ---------------------------

def ensure_dir(p: Path):
    """Create directory if it doesn't exist"""
    p.mkdir(parents=True, exist_ok=True)

def save_metadata(path: Path, data: dict):
    """Save JSON metadata"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

# ---------------------------
# Frame Extraction
# ---------------------------

def extract_frames(video_path: str, out_dir: Path, fps: int = 25) -> Tuple[List[Path], int, int]:
    """
    Extract frames at target fps.
    Returns: (frame_paths, height, width)
    """
    ensure_dir(out_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(orig_fps / float(fps))))

    saved_paths = []
    idx = 0
    saved_idx = 0
    
    pbar = tqdm(total=frame_count, desc="Extracting frames", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            fname = out_dir / f"frame_{saved_idx:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved_paths.append(fname)
            saved_idx += 1
        idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    if not saved_paths:
        raise RuntimeError("No frames extracted")
    
    img = cv2.imread(str(saved_paths[0]))
    H, W = img.shape[:2]
    return saved_paths, H, W

# ---------------------------
# 2D Pose Extraction (MediaPipe)
# ---------------------------

def extract_2d_poses(frame_paths: List[Path]) -> List[Optional[Dict]]:
    """
    Extract 2D poses using MediaPipe.
    Returns list of dicts with keys: 'landmarks' (33x4 list), 'bbox' (optional)
    """
    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError("Install MediaPipe: pip install mediapipe")
    
    mp_pose = mp.solutions.pose
    results = []
    
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose:
        for frame_path in tqdm(frame_paths, desc="2D Pose (MediaPipe)"):
            img = cv2.imread(str(frame_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = pose.process(img_rgb)
            
            if result.pose_landmarks:
                landmarks = []
                for lm in result.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                
                results.append({
                    'landmarks': landmarks,  # Shape: (33, 4)
                    'frame': frame_path.stem
                })
            else:
                results.append(None)
    
    return results

# ---------------------------
# 3D Pose Extraction (Placeholder/MediaPipe)
# ---------------------------

def extract_3d_poses(frame_paths: List[Path], pose2d_results: List) -> np.ndarray:
    """
    Extract 3D poses. Currently uses MediaPipe's z-coordinate.
    
    For better 3D: implement ROMP, MotionBERT, or other 3D pose estimator.
    Returns: (T, 33, 3) array
    """
    T = len(frame_paths)
    poses_3d = np.zeros((T, 33, 3), dtype=np.float32)
    
    for i, result in enumerate(pose2d_results):
        if result is not None:
            landmarks = np.array(result['landmarks'])  # (33, 4)
            # Use x, y, z from MediaPipe (z is depth estimate)
            poses_3d[i] = landmarks[:, :3]
    
    return poses_3d

# ---------------------------
# Optical Flow Extraction
# ---------------------------

def extract_optical_flow(frame_paths: List[Path]) -> np.ndarray:
    """
    Extract dense optical flow between consecutive frames.
    Returns: (T-1, H, W, 2) array where [:,:,:,0]=flow_x, [:,:,:,1]=flow_y
    """
    if len(frame_paths) < 2:
        print("Not enough frames for optical flow")
        return np.array([])
    
    first_frame = cv2.imread(str(frame_paths[0]))
    H, W = first_frame.shape[:2]
    T = len(frame_paths)
    
    flow_array = np.zeros((T-1, H, W, 2), dtype=np.float32)
    
    prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    for i in tqdm(range(1, T), desc="Optical Flow"):
        cur = cv2.imread(str(frame_paths[i]))
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        flow_array[i-1] = flow
        prev = cur_gray
    
    return flow_array

# ---------------------------
# Segmentation
# ---------------------------

def create_segments(
    pose2d: List,
    pose3d: np.ndarray,
    optical_flow: np.ndarray,
    segment_length: int,
    output_dir: Path
) -> List[Dict]:
    """
    Create overlapping segments for temporal modeling.
    Returns list of segment metadata.
    """
    T = len(pose2d)
    segments_info = []
    
    for start in range(0, T, segment_length):
        end = min(start + segment_length, T)
        
        seg_dir = output_dir / f"seg_{start:06d}"
        ensure_dir(seg_dir)
        
        # Save segmented data
        seg_pose2d = pose2d[start:end]
        with open(seg_dir / "pose2d.json", 'w') as f:
            json.dump(seg_pose2d, f)
        
        np.savez_compressed(
            seg_dir / "pose3d.npz",
            pose3d=pose3d[start:end]
        )
        
        # Flow has T-1 frames, align indices
        flow_start = max(0, start - 1)
        flow_end = min(end - 1, len(optical_flow))
        if flow_end > flow_start:
            np.savez_compressed(
                seg_dir / "optical_flow.npz",
                flow=optical_flow[flow_start:flow_end]
            )
        
        segments_info.append({
            'segment_id': f"seg_{start:06d}",
            'start_frame': start,
            'end_frame': end,
            'num_frames': end - start,
            'path': str(seg_dir)
        })
    
    return segments_info

# ---------------------------
# Main Processing Pipeline
# ---------------------------

def process_video(
    video_path: str,
    output_dir: str,
    fps: int = 25,
    segment_length: Optional[int] = 64,
    keep_frames: bool = False
) -> Dict:
    """
    Main preprocessing pipeline.
    
    Args:
        video_path: Path to input video
        output_dir: Where to save features (creates video_name subfolder)
        fps: Target frames per second
        segment_length: Frames per segment (None = no segmentation)
        keep_frames: Keep extracted frame images
    
    Returns:
        Metadata dict with all output paths
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Create output structure
    video_name = video_path.stem
    out_root = Path(output_dir) / video_name
    ensure_dir(out_root)
    
    print(f"\n{'='*60}")
    print(f"Processing: {video_path.name}")
    print(f"Output: {out_root.absolute()}")
    print(f"{'='*60}\n")
    
    # Temporary directory for frames
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"autism_prep_{video_name}_"))
    frames_dir = tmp_dir / "frames"
    
    try:
        # Step 1: Extract frames
        print("[1/4] Extracting frames...")
        frame_paths, H, W = extract_frames(str(video_path), frames_dir, fps)
        T = len(frame_paths)
        print(f"   Extracted {T} frames at {fps} fps ({H}x{W})")
        
        # Step 2: 2D poses
        print("\n[2/4] Extracting 2D poses...")
        pose2d_results = extract_2d_poses(frame_paths)
        detected = sum(1 for x in pose2d_results if x is not None)
        print(f"   Detected poses in {detected}/{T} frames")
        
        # Save 2D poses
        pose2d_path = out_root / "pose2d.json"
        with open(pose2d_path, 'w') as f:
            json.dump(pose2d_results, f)
        
        # Step 3: 3D poses
        print("\n[3/4] Extracting 3D poses...")
        pose3d_array = extract_3d_poses(frame_paths, pose2d_results)
        pose3d_path = out_root / "pose3d.npz"
        np.savez_compressed(pose3d_path, pose3d=pose3d_array)
        print(f"   Saved 3D poses: {pose3d_array.shape}")
        
        # Step 4: Optical flow
        print("\n[4/4] Extracting optical flow...")
        flow_array = extract_optical_flow(frame_paths)
        flow_path = out_root / "optical_flow.npz"
        np.savez_compressed(flow_path, flow=flow_array)
        print(f"   Saved optical flow: {flow_array.shape}")
        
        # Create segments if requested
        segments_info = []
        if segment_length and segment_length > 0:
            print(f"\n[5/4] Creating {segment_length}-frame segments...")
            segments_dir = out_root / "segments"
            ensure_dir(segments_dir)
            segments_info = create_segments(
                pose2d_results, pose3d_array, flow_array,
                segment_length, segments_dir
            )
            print(f"   Created {len(segments_info)} segments")
        
        # Optionally keep frames
        if keep_frames:
            frames_out = out_root / "frames"
            shutil.copytree(frames_dir, frames_out)
            print(f"   Saved frames to {frames_out}")
        
        # Create metadata for fusion model
        metadata = {
            'video_name': video_name,
            'video_path': str(video_path.absolute()),
            'output_dir': str(out_root.absolute()),
            'fps': fps,
            'num_frames': T,
            'frame_size': {'height': H, 'width': W},
            'features': {
                'pose2d': str(pose2d_path.absolute()),
                'pose3d': str(pose3d_path.absolute()),
                'optical_flow': str(flow_path.absolute())
            },
            'segments': segments_info,
            'preprocessing_config': {
                'pose_model': 'MediaPipe',
                'flow_algorithm': 'Farneback',
                'segment_length': segment_length
            }
        }
        
        # Save metadata
        metadata_path = out_root / "metadata.json"
        save_metadata(metadata_path, metadata)
        
        print(f"\n{'='*60}")
        print(f" Processing complete!")
        print(f" Features saved to: {out_root}")
        print(f" Metadata: {metadata_path}")
        print(f"{'='*60}\n")
        
        return metadata
        
    finally:
        # Cleanup temporary files
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

# Fusion Model Data Loader Helper
# ---------------------------

def load_features_for_fusion(features_dir: str) -> Dict:
    """
    Helper function for fusion model to load preprocessed features.
    
    Usage in fusion model:
        features = load_features_for_fusion('./features/video_name')
        pose2d = features['pose2d']
        pose3d = features['pose3d']
        flow = features['optical_flow']
    """
    features_dir = Path(features_dir)
    
    # Load metadata
    with open(features_dir / "metadata.json") as f:
        metadata = json.load(f)
    
    # Load features
    with open(features_dir / "pose2d.json") as f:
        pose2d = json.load(f)
    
    pose3d_data = np.load(features_dir / "pose3d.npz")
    pose3d = pose3d_data['pose3d']
    
    flow_data = np.load(features_dir / "optical_flow.npz")
    optical_flow = flow_data['flow']
    
    return {
        'metadata': metadata,
        'pose2d': pose2d,
        'pose3d': pose3d,
        'optical_flow': optical_flow,
        'segments': metadata.get('segments', [])
    }

# ---------------------------
# CLI
# ---------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess videos for autism screening fusion model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python feature_extraction_final.py --video child.mp4 --output ./features
  
  # Kaggle
  python feature_extraction_final.py \\
      --video /kaggle/input/videos/sample.mp4 \\
      --output /kaggle/working/features \\
      --fps 25 --segment 64
  
  # Batch processing
  for video in /path/to/videos/*.mp4; do
      python feature_extraction_final.py --video "$video" --output ./features
  done
        """
    )
    p.add_argument("--video", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--fps", type=int, default=25, help="Target FPS (default: 25)")
    p.add_argument("--segment", type=int, default=64, help="Segment length in frames (default: 64)")
    p.add_argument("--no-segment", action="store_true", help="Disable segmentation")
    p.add_argument("--keep-frames", action="store_true", help="Keep extracted frame images")
    return p.parse_args()

def main():
    args = parse_args()
    
    segment_length = None if args.no_segment else args.segment
    
    try:
        metadata = process_video(
            video_path=args.video,
            output_dir=args.output,
            fps=args.fps,
            segment_length=segment_length,
            keep_frames=args.keep_frames
        )
        
        print(" Success! Use load_features_for_fusion() to load in fusion model.")
        
    except Exception as e:
        print(f"\n Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()