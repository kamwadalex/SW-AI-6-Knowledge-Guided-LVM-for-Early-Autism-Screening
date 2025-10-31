from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from App.services.report import ReportService


router = APIRouter()


@router.post("/report/{report_id}")
def generate_report(report_id: str, payload: Dict[str, object]) -> Dict[str, str]:
	reports = ReportService(out_dir=Path("reports"))
	try:
		pdf_path = reports.write_pdf(report_id, payload)
		return {"report_id": report_id, "pdf_path": str(pdf_path)}
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc))


@router.get("/report/{report_id}")
def download_report(report_id: str):
	pdf_path = Path("reports") / f"{report_id}.pdf"
	if not pdf_path.exists():
		raise HTTPException(status_code=404, detail="Report not found")
	return FileResponse(str(pdf_path), media_type="application/pdf", filename=f"{report_id}.pdf")


