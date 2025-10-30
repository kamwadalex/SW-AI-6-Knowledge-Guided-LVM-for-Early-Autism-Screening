from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from app.services.pipeline import InferencePipeline
from app.services.report import ReportService
from app.services.knowledge_guidance import KnowledgeCorpus, KnowledgeGuidanceService
from app.core.config import get_settings


router = APIRouter()


@router.post("/infer")
async def infer(video: UploadFile = File(...), use_mock: bool = Form(default=False)):
	reports = ReportService(out_dir=Path("reports"))
	settings = get_settings()
	try:
		if use_mock:
			scores = {"tsn": 6.3, "sgcn": 5.8, "stgcn": 7.1}
			fused = (scores["tsn"] + scores["sgcn"] + scores["stgcn"]) / 3.0
		else:
			# Save upload to temp
			uploads = Path("uploads"); uploads.mkdir(parents=True, exist_ok=True)
			vid_path = uploads / video.filename
			with open(vid_path, "wb") as f:
				f.write(await video.read())
			pipeline = InferencePipeline()
			scores, fused = pipeline.run_on_video(vid_path)

		severity = _categorize_severity(fused, settings.severity_bands)
		# Knowledge guidance
		corpus = KnowledgeCorpus(Path(settings.knowledge_corpus_path) if settings.knowledge_corpus_path else None)
		kg = KnowledgeGuidanceService(corpus)
		guidance = kg.build_guidance(scores["tsn"], scores["sgcn"], scores["stgcn"], fused, severity)
		notes = [guidance.get("base_explanation", ""), f"severity: {severity}", f"confidence: {guidance['confidence']['label']}"]
		summary = reports.build_summary({**scores, "severity": severity}, fused, notes)
		summary["knowledge_guidance"] = guidance
		return JSONResponse({"summary": summary})
	except HTTPException:
		raise
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc))


def _categorize_severity(score: float, bands: str) -> str:
	for part in bands.split(","):
		range_part, label = part.split(":")
		lo, hi = range_part.split("-")
		try:
			if float(lo) <= score <= float(hi):
				return label
		except Exception:
			continue
	return "Unknown"


