# app/api/endpoints.py
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import tempfile
import os
import uuid
import logging
from pathlib import Path
from typing import List, Optional

from app.core.models import ScreeningResponse, ErrorResponse
from app.services.video_processor import VideoPreprocessor
from app.models.predictor import predictor
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()
video_processor = VideoPreprocessor()

def _validate_video_file(file: UploadFile) -> None:
    """Validate uploaded video file"""
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
        )
    
    # Check file size (rough estimate)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > settings.MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_VIDEO_SIZE // (1024*1024)}MB"
        )

@router.post(
    "/screen",
    response_model=ScreeningResponse,
    summary="Screen video for autism indicators",
    description="Upload a video file for autism screening analysis using multimodal AI models"
)
async def screen_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Video file for analysis")
):
    """
    Analyze video for autism screening using optical flow, 2D, and 3D skeleton analysis.
    
    - **file**: Video file (MP4, AVI, MOV, MKV) up to 100MB
    """
    try:
        # Validate file
        _validate_video_file(file)
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_filename = f"autism_screen_{uuid.uuid4().hex}{Path(file.filename).suffix}"
        temp_filepath = os.path.join(temp_dir, temp_filename)
        
        try:
            # Save uploaded file temporarily
            with open(temp_filepath, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            logger.info(f"Processing video: {file.filename} (size: {len(content)} bytes)")
            
            # Process video and extract features
            model_inputs = video_processor.process_video(temp_filepath)
            
            # Run prediction pipeline
            result = predictor.predict(model_inputs)
            
            # Format response
            response = ScreeningResponse(
                prediction=result['prediction'],
                component_analysis=result['component_analysis'],
                interpretation=result['interpretation'],
                processing_metadata=result['processing_metadata']
            )
            
            logger.info(f"Successfully processed {file.filename} - Score: {result['prediction']['final_score']:.3f}")
            
            return response
            
        finally:
            # Cleanup temporary file
            background_tasks.add_task(_cleanup_temp_file, temp_filepath)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video screening failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(e)}"
        )

@router.post(
    "/batch-screen",
    summary="Batch screen multiple videos",
    description="Upload multiple video files for batch autism screening analysis"
)
async def batch_screen_videos(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple video files for batch analysis")
):
    """
    Batch analyze multiple videos for autism screening.
    
    - **files**: List of video files (MP4, AVI, MOV, MKV) up to 100MB each
    """
    try:
        if len(files) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 files allowed per batch request"
            )
        
        results = []
        temp_files = []
        
        try:
            for file in files:
                # Validate each file
                _validate_video_file(file)
                
                # Create temporary file
                temp_dir = tempfile.gettempdir()
                temp_filename = f"batch_{uuid.uuid4().hex}{Path(file.filename).suffix}"
                temp_filepath = os.path.join(temp_dir, temp_filename)
                temp_files.append(temp_filepath)
                
                # Save uploaded file
                with open(temp_filepath, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                
                # Process video
                try:
                    model_inputs = video_processor.process_video(temp_filepath)
                    result = predictor.predict(model_inputs)
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "result": result
                    })
                    logger.info(f"Batch processed {file.filename} successfully")
                    
                except Exception as e:
                    logger.error(f"Batch processing failed for {file.filename}: {str(e)}")
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "error": str(e)
                    })
            
            return {
                "batch_id": uuid.uuid4().hex,
                "total_files": len(files),
                "successful": len([r for r in results if r["success"]]),
                "failed": len([r for r in results if not r["success"]]),
                "results": results
            }
            
        finally:
            # Cleanup all temporary files
            for temp_file in temp_files:
                background_tasks.add_task(_cleanup_temp_file, temp_file)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch screening failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )

@router.get(
    "/model-info",
    summary="Get model information",
    description="Retrieve information about loaded AI models and their configurations"
)
async def get_model_info():
    """Get information about the loaded models and their versions"""
    try:
        model_info = predictor._get_model_versions()
        
        return {
            "models": model_info,
            "config": {
                "device": settings.DEVICE,
                "frame_size": settings.FRAME_SIZE,
                "allowed_extensions": settings.ALLOWED_EXTENSIONS,
                "max_video_size_mb": settings.MAX_VIDEO_SIZE // (1024 * 1024)
            }
        }
        
    except Exception as e:
        logger.error(f"Model info retrieval failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve model information: {str(e)}"
        )

@router.get(
    "/interpret-score",
    summary="Interpret ADOS score",
    description="Interpret a given ADOS score into severity categories and recommendations"
)
async def interpret_score(
    score: float = Query(..., ge=1.0, le=10.0, description="ADOS score between 1.0 and 10.0")
):
    """
    Interpret an ADOS score without video analysis.
    
    - **score**: ADOS comparison score (1.0 - 10.0)
    """
    try:
        interpretation = predictor.interpret_score(score)
        
        return {
            "score": score,
            "interpretation": interpretation
        }
        
    except Exception as e:
        logger.error(f"Score interpretation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Score interpretation failed: {str(e)}"
        )

def _cleanup_temp_file(filepath: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logger.info(f"Cleaned up temporary file: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary file {filepath}: {str(e)}")

# Exception handlers
@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail
        ).dict()
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            details={"type": type(exc).__name__}
        ).dict()
    )