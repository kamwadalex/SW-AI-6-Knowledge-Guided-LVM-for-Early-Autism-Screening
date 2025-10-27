# app/core/models.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ScreeningResponse(BaseModel):
    """Response model for autism screening results"""
    prediction: Dict[str, Any] = Field(..., description="Final prediction results")
    component_analysis: Dict[str, Any] = Field(..., description="Component scores analysis")
    interpretation: Dict[str, Any] = Field(..., description="Score interpretation")
    processing_metadata: Dict[str, Any] = Field(..., description="Processing metadata")

class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = Field(False, description="Whether the operation was successful")
    error: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

