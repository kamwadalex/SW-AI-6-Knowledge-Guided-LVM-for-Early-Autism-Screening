from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from App.core.config import get_settings
from App.core.logging import configure_logging
from App.middleware.limits import UploadSizeLimitMiddleware


def create_app() -> FastAPI:
	app = FastAPI(
		title="SW-AI-6 API",
		description="Knowledge-Guided LVM for Early Autism Screening - API",
		version="0.1.0",
	)

	settings = get_settings()
	configure_logging(settings.log_level)

	# CORS - restrict via env ALLOW_ORIGINS
	app.add_middleware(
		CORSMiddleware,
		allow_origins=[origin.strip() for origin in settings.allow_origins.split(",")],
		allow_credentials=True,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	# Enforce 100 MB upload limit
	app.add_middleware(UploadSizeLimitMiddleware, max_bytes=100 * 1024 * 1024)

	from App.api.v1.api import api_router as api_v1_router  # noqa: WPS433

	@app.get("/")
	def root() -> dict:
		return {"status": "ok", "service": "SW-AI-6 API"}

	app.include_router(api_v1_router, prefix="/api/v1")
	return app


app = create_app()


