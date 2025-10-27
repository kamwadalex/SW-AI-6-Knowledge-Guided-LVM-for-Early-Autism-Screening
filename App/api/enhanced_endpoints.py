# app/api/enhanced_endpoints.py
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
import tempfile
import os
import uuid
from pathlib import Path

from app.core.models import ScreeningResponse
from app.services.video_processor import VideoPreprocessor
from app.models.enhanced_predictor import enhanced_predictor

router = APIRouter()
video_processor = VideoPreprocessor()

@router.post(
    "/screen-with-explanation",
    response_model=ScreeningResponse,
    summary="Screen video with detailed explanations",
    description="Upload a video file for autism screening with knowledge-guided explanations"
)
async def screen_video_with_explanation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file for analysis with explanations")
):
    """
    Analyze video with enhanced explanations linking predictions to clinical domains.
    
    - **file**: Video file (MP4, AVI, MOV, MKV) up to 100MB
    """
    try:
        # Validate file (reuse existing validation)
        from app.api.endpoints import _validate_video_file
        _validate_video_file(file)
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_filename = f"enhanced_screen_{uuid.uuid4().hex}{Path(file.filename).suffix}"
        temp_filepath = os.path.join(temp_dir, temp_filename)
        
        try:
            # Save uploaded file temporarily
            with open(temp_filepath, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"Processing video with explanations: {file.filename}")
            
            # Process video and extract features
            model_inputs = video_processor.process_video(temp_filepath)
            
            # Run enhanced prediction pipeline
            result = enhanced_predictor.predict_with_explanation(model_inputs)
            
            logger.info(f"Successfully processed {file.filename} with explanations")
            
            return result
            
        finally:
            # Cleanup temporary file
            background_tasks.add_task(_cleanup_temp_file, temp_filepath)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced video screening failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Enhanced video processing failed: {str(e)}"
        )

def _cleanup_temp_file(filepath: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up temporary file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {filepath}: {str(e)}")