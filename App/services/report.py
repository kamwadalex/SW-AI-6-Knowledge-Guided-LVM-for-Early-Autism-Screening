from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any


@dataclass
class ReportEntry:
	label: str
	value: float


class ReportService:
	def __init__(self, out_dir: Path) -> None:
		self.out_dir = out_dir
		self.out_dir.mkdir(parents=True, exist_ok=True)

	def build_summary(self, scores: Dict[str, float], fused: float, notes: List[str]) -> Dict[str, object]:
		return {
			"scores": scores,
			"fused_score": fused,
			"notes": notes,
		}

	def write_pdf(self, report_id: str, summary: Dict[str, object]) -> Path:
		try:
			from reportlab.lib.pagesizes import A4
			from reportlab.pdfgen import canvas
		except Exception as exc:
			raise RuntimeError("PDF generation requires reportlab to be installed.") from exc

		pdf_path = self.out_dir / f"{report_id}.pdf"
		c = canvas.Canvas(str(pdf_path), pagesize=A4)
		width, height = A4
		y = height - 50
		c.setFont("Helvetica-Bold", 16)
		c.drawString(50, y, "Autism Screening Report")
		y -= 30
		c.setFont("Helvetica", 11)

		def fmt(value: Any, decimals: int = 3) -> str:
			try:
				f = float(value)
				# Avoid scientific notation and force fixed decimals
				return f"{f:.{decimals}f}"
			except Exception:
				return str(value)

		# Severity and scores
		severity = (summary.get("scores", {}) or {}).get("severity") or summary.get("severity")
		if severity is not None:
			c.drawString(50, y, f"Severity: {severity}")
			y -= 18
		scores = summary.get("scores", {}) or {}
		for key in ("tsn", "sgcn", "stgcn"):
			if key in scores:
				c.drawString(50, y, f"{key.upper()} score: {fmt(scores.get(key))}")
				y -= 16
		c.drawString(50, y, f"Fused score: {fmt(summary.get('fused_score', 'N/A'))}")
		y -= 24
		kg = summary.get("knowledge_guidance", {}) or {}
		# Confidence is kept in JSON but not rendered in PDF
		if kg:
			c.setFont("Helvetica-Bold", 12)
			c.drawString(50, y, "Knowledge Guidance")
			y -= 18
			c.setFont("Helvetica", 10)
			base = kg.get("base_explanation", "")
			if base:
				c.drawString(50, y, base[:1000])
				y -= 16
			domains = kg.get("domains", {})
			for dname, dinfo in list(domains.items())[:6]:
				c.drawString(50, y, f"- {dname}: {dinfo.get('description', '')}")
				y -= 14
				if y < 60:
					c.showPage(); y = height - 50; c.setFont("Helvetica", 10)
		c.setFont("Helvetica-Bold", 12)
		c.drawString(50, y, "Notes")
		y -= 18
		c.setFont("Helvetica", 10)
		for note in summary.get("notes", []):
			c.drawString(60, y, f"- {note}")
			y -= 14
			if y < 60:
				c.showPage(); y = height - 50
		c.showPage()
		c.save()
		return pdf_path


