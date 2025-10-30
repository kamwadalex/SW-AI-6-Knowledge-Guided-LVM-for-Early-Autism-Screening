from fastapi import APIRouter


router = APIRouter()


@router.get("/health")
def health() -> dict:
	return {"status": "healthy"}


@router.get("/ready")
def readiness() -> dict:
	# Extend with checks (DB, external services) if needed
	return {"status": "ready"}


