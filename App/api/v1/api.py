from fastapi import APIRouter

from App.api.v1.endpoints.health import router as health_router
from App.api.v1.endpoints.example import router as example_router
from App.api.v1.endpoints.inference import router as inference_router
from App.api.v1.endpoints.report import router as report_router
from App.api.v1.endpoints.ui import router as ui_router


api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])  # /health, /ready
api_router.include_router(example_router, prefix="/example", tags=["example"])  # /example/echo
api_router.include_router(inference_router, tags=["inference"])  # /infer
api_router.include_router(report_router, tags=["report"])  # /report
api_router.include_router(ui_router, tags=["ui"])  # /ui


