# app/api/dependencies.py
from fastapi import Header, HTTPException
from typing import Optional

async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key for authenticated endpoints"""
    # In production, implement proper API key validation
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )
    
    # Example validation - replace with your logic
    valid_keys = ["your-api-key-here"]  # Store in environment variables
    if x_api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return x_api_key