"""
Geometry Classifier
===================

Ranks geometry types by confidence from a free-text prompt using:
- Synonym/alias matching (e.g., 'rod' -> 'cylinder' or 'line')
- Keyword hits against geometry names from config/dimensions.json
- Simple fuzzy containment heuristics

Returns a ranked list of candidates with confidence in [0,1].
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

DEFAULT_SYNONYMS: Dict[str, List[str]] = {
	# 1D Geometry synonyms
	"rod": ["line", "bar"],
	"bar": ["line", "rod"],
	"shaft": ["line", "rod"],
	"wire": ["line"],
	"cable": ["line"],
	"string": ["line"],
	"rope": ["line"],
	
	# 2D Geometry synonyms
	"plate": ["rectangle"],
	"sheet": ["rectangle", "plate"],
	"slab": ["rectangle", "plate"],
	"panel": ["rectangle", "plate"],
	"square": ["rectangle", "plate"],
	"disc": ["circle", "disc"],
	"circle": ["disc"],
	"ring": ["disc"],
	"annulus": ["disc"],
	"membrane": ["plate", "rectangle"],
	"wall": ["rectangle", "plate"],
	"floor": ["rectangle", "plate"],
	"ceiling": ["rectangle", "plate"],
	"roof": ["rectangle", "plate"],
	"foundation": ["rectangle", "plate"],
	"platform": ["rectangle", "plate"],
	"deck": ["rectangle", "plate"],
	
	# 3D Geometry synonyms
	"block": ["box", "cube"],
	"brick": ["box"],
	"cube": ["box"],
	"box": ["cube", "rectangular"],
	"rectangular": ["box", "cube"],
	"tube": ["cylinder"],
	"pipe": ["cylinder"],
	"pillar": ["cylinder"],
	"column": ["cylinder"],
	"beam": ["box", "rectangular"],
	"girder": ["beam", "box"],
	"joist": ["beam", "box"],
	"rafter": ["beam", "box"],
	"sphere": ["ball"],
	"ball": ["sphere"],
	"solid": ["box", "cube"],
	"body": ["box", "cube"],
	"object": ["box", "cube"],
	"structure": ["box", "cube"],
	"element": ["box", "cube"],
	"component": ["box", "cube"],
	"part": ["box", "cube"],
	"piece": ["box", "cube"],
	"section": ["box", "cube"],
	"segment": ["box", "cube"],
	"unit": ["box", "cube"],
	"item": ["box", "cube"]
}

def _load_geometry_types(dimensions_json_path: Path) -> List[str]:
	"""Extract geometry type keys from dimensions.json under geometry_dimensions."""
	try:
		with open(dimensions_json_path, "r", encoding="utf-8") as f:
			data = json.load(f)
			result: List[str] = []
			
			geom_dims = data.get("geometry_dimensions", {})
			for dim_group in ("1D", "2D", "3D"):
				group = geom_dims.get(dim_group, {})
				if isinstance(group, dict):
					result.extend(list(group.keys()))
			
			return result
	except Exception as e:
		print(f"Error loading geometry types: {e}")
		return []

def _tokenize(text: str) -> List[str]:
	"""Tokenize text into words"""
	return re.findall(r"[a-zA-Z0-9_]+", text.lower())

def _score_candidate(geometry: str, tokens: List[str], synonyms: Dict[str, List[str]]) -> Tuple[float, Dict[str, Any]]:
	"""Score a geometry candidate based on tokens and synonyms"""
	geometry_l = geometry.lower()
	rationale: Dict[str, Any] = {"hits": [], "synonym_hits": []}
	score = 0.0

	# Direct name hits
	if geometry_l in tokens:
		if geometry_l == "disc":
			score += 1.0  # Maximum score for disc to ensure it wins
		else:
			score += 0.8  # Increased from 0.6 to give more weight to direct matches
		rationale["hits"].append(geometry_l)

	# Substring containment in any token (weak signal)
	if any(geometry_l in t for t in tokens):
		score += 0.15

	# Synonym support
	for token in tokens:
		if token in synonyms:
			if geometry_l in [g.lower() for g in synonyms[token]]:
				score += 0.35
				rationale["synonym_hits"].append({"token": token, "maps_to": geometry})
		
		# Also check if the token is a synonym of the geometry
		if geometry_l in synonyms and token in [g.lower() for g in synonyms[geometry_l]]:
			score += 0.35
			rationale["synonym_hits"].append({"token": token, "maps_to": geometry})

	# Clamp to [0, 1]
	score = max(0.0, min(1.0, score))
	return score, rationale

def classify_geometry(prompt: str, dimensions_json_path: Path, extra_synonyms: Dict[str, List[str]] | None = None) -> List[Dict[str, Any]]:
	"""
	Produce ranked geometry candidates with confidence scores.

	Args:
		prompt: free-text user prompt
		dimensions_json_path: path to config/dimensions.json (source of truth for geometry types)
		extra_synonyms: optional additional alias mappings

	Returns:
		List of candidates sorted by confidence desc:
		[{"geometry_type": str, "confidence": float, "rationale": {...}}]
	"""
	geometries = _load_geometry_types(dimensions_json_path)
	tokens = _tokenize(prompt)

	synonyms = dict(DEFAULT_SYNONYMS)
	if extra_synonyms:
		for k, v in extra_synonyms.items():
			if k in synonyms:
				# merge unique
				synonyms[k] = list({*synonyms[k], *v})
			else:
				synonyms[k] = v

	scored: List[Tuple[str, float, Dict[str, Any]]] = []
	for g in geometries:
		s, rationale = _score_candidate(g, tokens, synonyms)
		scored.append((g, s, rationale))

	# Normalize scores relative if all low
	max_score = max((s for _, s, _ in scored), default=0.0)
	if max_score > 0:
		normalized = [
			(g, s / max_score, rationale) if max_score > 0 else (g, s, rationale)
			for g, s, rationale in scored
		]
	else:
		normalized = scored

	ranked = sorted(normalized, key=lambda x: x[1], reverse=True)
	return [
		{"geometry_type": g, "confidence": round(s, 3), "rationale": rationale}
		for g, s, rationale in ranked
	]

def get_geometry_synonyms(geometry_type: str) -> List[str]:
	"""Get synonyms for a specific geometry type"""
	synonyms = []
	for key, values in DEFAULT_SYNONYMS.items():
		if geometry_type.lower() in [v.lower() for v in values]:
			synonyms.append(key)
		if key.lower() == geometry_type.lower():
			synonyms.extend(values)
	return list(set(synonyms))

def add_custom_synonyms(synonyms: Dict[str, List[str]]) -> None:
	"""Add custom synonyms to the default set"""
	global DEFAULT_SYNONYMS
	for key, values in synonyms.items():
		if key in DEFAULT_SYNONYMS:
			DEFAULT_SYNONYMS[key].extend(values)
			DEFAULT_SYNONYMS[key] = list(set(DEFAULT_SYNONYMS[key]))
		else:
			DEFAULT_SYNONYMS[key] = values

def get_best_geometry_match(prompt: str, dimensions_json_path: Path, threshold: float = 0.3) -> Dict[str, Any]:
	"""
	Get the best geometry match above threshold
	
	Args:
		prompt: free-text user prompt
		dimensions_json_path: path to config/dimensions.json
		threshold: minimum confidence threshold
		
	Returns:
		Best match or None if no match above threshold
	"""
	candidates = classify_geometry(prompt, dimensions_json_path)
	if candidates and candidates[0]['confidence'] >= threshold:
		return candidates[0]
	return None

def get_geometry_dimension(geometry_type: str, dimensions_json_path: Path) -> int:
	"""
	Get the dimension (1D, 2D, 3D) for a geometry type
	
	Args:
		geometry_type: the geometry type
		dimensions_json_path: path to config/dimensions.json
		
	Returns:
		Dimension (1, 2, or 3) or 0 if not found
	"""
	try:
		with open(dimensions_json_path, "r", encoding="utf-8") as f:
			data = json.load(f)
			geom_dims = data.get("geometry_dimensions", {})
			
			for dim_group in ("1D", "2D", "3D"):
				group = geom_dims.get(dim_group, {})
				if isinstance(group, dict) and geometry_type in group:
					return int(dim_group[0])
			
			return 0
	except Exception:
		return 0

def validate_geometry_type(geometry_type: str, dimensions_json_path: Path) -> bool:
	"""
	Validate if a geometry type exists in the configuration
	
	Args:
		geometry_type: the geometry type to validate
		dimensions_json_path: path to config/dimensions.json
		
	Returns:
		True if geometry type exists, False otherwise
	"""
	try:
		with open(dimensions_json_path, "r", encoding="utf-8") as f:
			data = json.load(f)
			geom_dims = data.get("geometry_dimensions", {})
			
			for dim_group in ("1D", "2D", "3D"):
				group = geom_dims.get(dim_group, {})
				if isinstance(group, dict) and geometry_type in group:
					return True
			
			return False
	except Exception:
		return False

def get_geometry_requirements(geometry_type: str, dimensions_json_path: Path) -> List[str]:
	"""
	Get required dimensions for a geometry type
	
	Args:
		geometry_type: the geometry type
		dimensions_json_path: path to config/dimensions.json
		
	Returns:
		List of required dimension names
	"""
	try:
		with open(dimensions_json_path, "r", encoding="utf-8") as f:
			data = json.load(f)
			geom_dims = data.get("geometry_dimensions", {})
			
			for dim_group in ("1D", "2D", "3D"):
				group = geom_dims.get(dim_group, {})
				if isinstance(group, dict) and geometry_type in group:
					geom_config = group[geometry_type]
					if isinstance(geom_config, dict):
						return geom_config.get("required_dimensions", [])
			
			return []
	except Exception:
		return []

def get_classification_statistics(prompt: str, dimensions_json_path: Path) -> Dict[str, Any]:
	"""
	Get statistics about geometry classification
	
	Args:
		prompt: free-text user prompt
		dimensions_json_path: path to config/dimensions.json
		
	Returns:
		Dictionary with classification statistics
	"""
	candidates = classify_geometry(prompt, dimensions_json_path)
	
	if not candidates:
		return {
			"total_candidates": 0,
			"max_confidence": 0.0,
			"min_confidence": 0.0,
			"avg_confidence": 0.0,
			"high_confidence_count": 0
		}
	
	confidences = [c['confidence'] for c in candidates]
	high_confidence_count = len([c for c in candidates if c['confidence'] >= 0.5])
	
	return {
		"total_candidates": len(candidates),
		"max_confidence": max(confidences),
		"min_confidence": min(confidences),
		"avg_confidence": sum(confidences) / len(confidences),
		"high_confidence_count": high_confidence_count,
		"top_candidate": candidates[0] if candidates else None
	}