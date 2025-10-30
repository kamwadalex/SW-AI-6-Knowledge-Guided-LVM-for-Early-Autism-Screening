from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Explanation:
	category: str
	description: str
	strength: float


class KnowledgeService:
	def __init__(self) -> None:
		# Minimal rules; extend with corpus-driven mappings
		self.rules = [
			("motor_patterns", "Repetitive motor patterns observed via motion cues.", "tsn"),
			("social_reciprocity", "Atypical social reciprocity inferred from skeletal dynamics.", "sgcn"),
			("joint_attention", "Joint attention deviations inferred from 3D pose temporal relations.", "stgcn"),
		]

	def explain(self, scores: Dict[str, float], fused_score: float) -> List[Explanation]:
		# Simple attribution: higher contributing modality -> higher strength
		total = sum(scores.values()) or 1.0
		explanations: List[Explanation] = []
		for key, desc, source in self.rules:
			strength = (scores.get(source, 0.0) / total)
			explanations.append(Explanation(category=key, description=desc, strength=strength))
		return explanations


