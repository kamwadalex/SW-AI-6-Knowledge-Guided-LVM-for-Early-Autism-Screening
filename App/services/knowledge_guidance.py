from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import csv
import numpy as np


@dataclass
class DomainInfo:
	description: str
	severity_interpretation: str
	clinical_reference: str


DEFAULT_MODEL_DOMAIN_MAPPINGS: Dict[str, List[str]] = {
	"TSN": [
		"Social Communication & Interaction",
		"Communication",
		"Joint Attention",
	],
	"SGCN": [
		"Social Communication & Interaction",
		"Early Social Attention",
		"Facial Affect Recognition",
	],
	"ST-GCN": [
		"Object & Repetitive Behaviors",
		"Motor Development & Atypical Movements",
		"Stereotyped Motor Movements",
	],
}


DEFAULT_DOMAIN_DETAILS: Dict[str, DomainInfo] = {
	"Social Communication & Interaction": DomainInfo(
		description="Difficulties in social-emotional reciprocity, social approach, and response",
		severity_interpretation="Higher scores may indicate challenges in social reciprocity",
		clinical_reference="ADOS-2 Social Affect domain",
	),
	"Object & Repetitive Behaviors": DomainInfo(
		description="Restricted, repetitive patterns of behavior, interests, or activities",
		severity_interpretation="Elevated scores suggest repetitive behaviors",
		clinical_reference="ADOS-2 Restricted/Repetitive Behaviors",
	),
	"Motor Development & Atypical Movements": DomainInfo(
		description="Atypical motor patterns, coordination difficulties, and movement quality issues",
		severity_interpretation="Elevated scores suggest atypical motor patterns",
		clinical_reference="Motor coordination assessments",
	),
	"Early Social Attention": DomainInfo(
		description="Attention to social stimuli and social bids",
		severity_interpretation="Elevated scores suggest altered social attention",
		clinical_reference="Developmental social attention measures",
	),
}


class KnowledgeCorpus:
	def __init__(self, csv_path: Optional[Path]) -> None:
		self.csv_path = csv_path
		self.records: List[Dict[str, str]] = []
		if csv_path and csv_path.exists():
			with open(csv_path, newline="", encoding="utf-8") as f:
				reader = csv.DictReader(f)
				self.records = [row for row in reader]

	def get_domains_for_model(self, model_name: str) -> List[str]:
		if not self.records:
			return DEFAULT_MODEL_DOMAIN_MAPPINGS.get(model_name, [])
		return sorted({r.get("Category", "") for r in self.records if r.get("Linked_Model") == model_name and r.get("Category")})

	def get_domain_details(self, domain: str) -> DomainInfo:
		if not self.records:
			return DEFAULT_DOMAIN_DETAILS.get(domain, DomainInfo("", "", ""))
		for r in self.records:
			if r.get("Category") == domain:
				return DomainInfo(
					description=r.get("Description", ""),
					severity_interpretation=r.get("Severity_Indicator", ""),
					clinical_reference=r.get("Clinical_References", ""),
				)
		return DomainInfo("", "", "")


class KnowledgeGuidanceService:
	def __init__(self, corpus: KnowledgeCorpus) -> None:
		self.corpus = corpus

	def identify_dominant_models(self, tsn: float, sgcn: float, stgcn: float, threshold: float = 0.9) -> List[str]:
		scores = {"TSN": tsn, "SGCN": sgcn, "ST-GCN": stgcn}
		max_score = max(scores.values())
		cut = threshold * max_score
		return [m for m, s in scores.items() if s >= cut]

	def map_to_domains(self, dominant_models: List[str]) -> Dict[str, DomainInfo]:
		domains: Dict[str, DomainInfo] = {}
		for model in dominant_models:
			for domain in self.corpus.get_domains_for_model(model):
				if domain not in domains:
					domains[domain] = self.corpus.get_domain_details(domain)
		return domains

	def generate_base_explanation(self, final_score: float, dominant_models: List[str], domains: Dict[str, DomainInfo]) -> str:
		model_names = ", ".join(dominant_models) if dominant_models else "no dominant model"
		domain_names = ", ".join(domains.keys()) if domains else "no mapped domains"
		return f"Score {final_score:.2f} — driven mainly by {model_names} models (domains: {domain_names})."

	def calculate_confidence(self, tsn: float, sgcn: float, stgcn: float, fused: float) -> float:
		arr = np.array([tsn, sgcn, stgcn], dtype=np.float32)
		variance = float(np.var(arr))
		variance_conf = max(0.0, 1.0 - (variance / 5.0))
		deviation = float(np.mean(np.abs(arr - fused)))
		deviation_conf = 1.0 - (deviation / 5.0)
		conf = (variance_conf + deviation_conf) / 2.0
		return max(0.0, min(1.0, conf))

	def confidence_label(self, conf: float) -> str:
		if conf >= 0.8:
			return "High Confidence - Strong model agreement"
		if conf >= 0.6:
			return "Moderate Confidence - Reasonable model agreement"
		if conf >= 0.4:
			return "Low Confidence - Inconsistent model predictions"
		return "Very Low Confidence - High model disagreement"

	def generate_recommendations(self, risk_level_label: str, domains: List[str]) -> List[str]:
		base = {
			"Low Risk": [
				"Routine developmental monitoring recommended",
				"Continue with standard pediatric care",
			],
			"Mild Risk": [
				"Consider comprehensive developmental screening",
				"Monitor social communication milestones",
				"Discuss concerns with pediatrician",
			],
			"High Risk": [
				"Recommend comprehensive diagnostic assessment",
				"Early intervention services may be beneficial",
				"Refer to autism specialist for evaluation",
			],
			"Very High Risk": [
				"Urgent comprehensive evaluation needed",
				"Immediate referral to autism specialist",
				"Consider early intensive behavioral intervention",
			],
		}
		domain_advice = {
			"Social Communication & Interaction": [
				"Focus on social communication skills assessment",
				"Evaluate joint attention and social responsiveness",
				"Consider social skills training if indicated",
			],
			"Object & Repetitive Behaviors": [
				"Assess restricted and repetitive behaviors",
				"Evaluate impact on daily functioning",
				"Consider behavioral interventions for concerning patterns",
			],
			"Motor Development & Atypical Movements": [
				"Evaluate motor coordination and planning",
				"Assess movement quality and smoothness",
				"Consider occupational therapy assessment",
			],
			"Early Social Attention": [
				"Monitor attention to social stimuli",
				"Evaluate response to name and social bids",
				"Assess social engagement patterns",
			],
		}
		recs = list(base.get(risk_level_label, []))
		for d in domains:
			recs.extend(domain_advice.get(d, []))
		recs.append("Consult with healthcare provider for comprehensive assessment")
		return recs

	def build_guidance(self, tsn: float, sgcn: float, stgcn: float, fused: float, severity_label: str) -> Dict[str, object]:
		dominant = self.identify_dominant_models(tsn, sgcn, stgcn)
		domain_map = self.map_to_domains(dominant)
		base_text = self.generate_base_explanation(fused, dominant, domain_map)
		confidence = self.calculate_confidence(tsn, sgcn, stgcn, fused)
		confidence_text = self.confidence_label(confidence)
		recommendations = self.generate_recommendations(risk_level_label=severity_label, domains=list(domain_map.keys()))
		return {
			"base_explanation": base_text,
			"dominant_models": dominant,
			"domains": {k: vars(v) for k, v in domain_map.items()},
			"confidence": {
				"value": round(confidence, 3),
				"label": confidence_text,
			},
			"recommendations": recommendations,
		}


