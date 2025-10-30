"""
Boundary Location Classifier
============================

Ranks boundary locations by confidence from a free-text prompt using:
- Synonym/alias matching (e.g., 'one side' -> 'left', 'opposite side' -> 'right')
- Keyword hits against boundary names from geometry_boundaries.json
- Context-aware reasoning (temperature values, geometric conventions)

Returns a ranked list of candidates with confidence in [0,1].
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any

DEFAULT_BOUNDARY_SYNONYMS: Dict[str, List[str]] = {
	# Left side synonyms
	"left": ["left", "left side", "left end", "start", "beginning", "one side", "one end"],
	"one side": ["left", "one side", "one end", "start", "beginning"],
	"one end": ["left", "one end", "start", "beginning"],
	"start": ["left", "start", "beginning"],
	"beginning": ["left", "start", "beginning"],
	
	# Right side synonyms  
	"right": ["right", "right side", "right end", "end", "opposite side", "other side"],
	"opposite side": ["right", "opposite side", "other side", "opposite end", "other end"],
	"other side": ["right", "other side", "opposite side"],
	"opposite end": ["right", "opposite end", "other end"],
	"other end": ["right", "other end", "opposite end"],
	"end": ["right", "end"],
	
	# Top synonyms
	"top": ["top", "upper", "top surface", "upper surface", "ceiling"],
	"upper": ["top", "upper", "top surface"],
	"top surface": ["top", "top surface", "upper surface"],
	
	# Bottom synonyms
	"bottom": ["bottom", "lower", "bottom surface", "lower surface", "floor"],
	"lower": ["bottom", "lower", "bottom surface"],
	"bottom surface": ["bottom", "bottom surface", "lower surface"],
	
	# Front/back synonyms (3D)
	"front": ["front", "forward", "front face", "front surface"],
	"back": ["back", "rear", "back face", "back surface", "rear surface"],
	
	# Special boundaries
	"circumference": ["circumference", "perimeter", "edge", "boundary"],
	"surface": ["surface", "outer surface", "external surface"],
	"curved surface": ["curved surface", "side", "lateral surface"],
	
	# All boundaries
	"all": ["all", "all sides", "all ends", "entire boundary", "complete boundary"],
	"all sides": ["all", "all sides", "entire boundary"],
	"all ends": ["all", "all ends", "entire boundary"],
	"entire boundary": ["all", "entire boundary", "complete boundary"]
}

def _tokenize(prompt: str) -> List[str]:
	"""Extract meaningful tokens from prompt"""
	# Convert to lowercase and split on common delimiters
	tokens = re.findall(r'\b\w+\b', prompt.lower())
	
	# Remove common stop words that don't help with boundary detection
	stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'this', 'that', 'these', 'those'}
	
	return [t for t in tokens if t not in stop_words and len(t) > 1]

def _load_boundary_types(geometry_boundaries_json_path: Path) -> List[str]:
	"""Load available boundary types from geometry_boundaries.json"""
	try:
		with open(geometry_boundaries_json_path, 'r') as f:
			data = json.load(f)
		
		boundaries = set()
		# Extract all unique boundary names from the geometry configuration
		for dim_group in data.get('geometries', {}).values():
			for geometry in dim_group.values():
				if 'available_boundaries' in geometry:
					boundaries.update(geometry['available_boundaries'])
		
		return list(boundaries)
	except Exception as e:
		# Fallback to common boundaries
		return ['left', 'right', 'top', 'bottom', 'front', 'back', 'circumference', 'surface', 'curved surface']

def _score_candidate(boundary: str, tokens: List[str], synonyms: Dict[str, List[str]], prompt: str) -> Tuple[float, Dict[str, Any]]:
	"""Score a boundary candidate based on tokens, synonyms, and context"""
	boundary_l = boundary.lower()
	rationale: Dict[str, Any] = {"hits": [], "synonym_hits": [], "context_clues": []}
	score = 0.0

	# Direct name hits
	if boundary_l in tokens:
		score += 0.8
		rationale["hits"].append(boundary_l)

	# Substring containment in any token
	if any(boundary_l in t for t in tokens):
		score += 0.15

	# Synonym support
	for token in tokens:
		if token in synonyms:
			if boundary_l in [b.lower() for b in synonyms[token]]:
				score += 0.4
				rationale["synonym_hits"].append({"token": token, "maps_to": boundary})
		
		# Also check if the token is a synonym of the boundary
		if boundary_l in synonyms and token in [b.lower() for b in synonyms[boundary_l]]:
			score += 0.4
			rationale["synonym_hits"].append({"token": token, "maps_to": boundary})

	# Context-aware reasoning
	prompt_lower = prompt.lower()
	
	# Temperature-based reasoning
	if boundary == 'left':
		if any(temp in prompt_lower for temp in ['100', 'hot', 'high', 'warm', 'heated']):
			score += 0.3
			rationale["context_clues"].append("High temperature suggests left boundary")
	elif boundary == 'right':
		if any(temp in prompt_lower for temp in ['10', '0', 'cold', 'low', 'cool', 'cooled']):
			score += 0.3
			rationale["context_clues"].append("Low temperature suggests right boundary")
	
	# Geometric reasoning
	if 'one' in prompt_lower and boundary == 'left':
		score += 0.2
		rationale["context_clues"].append("'One side' typically refers to left")
	elif 'opposite' in prompt_lower and boundary == 'right':
		score += 0.2
		rationale["context_clues"].append("'Opposite side' typically refers to right")

	# Clamp to [0, 1]
	score = max(0.0, min(1.0, score))
	return score, rationale

def classify_boundary_locations(prompt: str, geometry_type: str, geometry_boundaries_json_path: Path, extra_synonyms: Dict[str, List[str]] | None = None) -> List[Dict[str, Any]]:
	"""
	Produce ranked boundary location candidates with confidence scores.

	Args:
		prompt: free-text user prompt
		geometry_type: detected geometry type (e.g., 'square', 'cylinder')
		geometry_boundaries_json_path: path to config/geometry_boundaries.json
		extra_synonyms: optional additional alias mappings

	Returns:
		List of candidates sorted by confidence desc:
		[{"boundary": str, "confidence": float, "rationale": {...}}]
	"""
	# Get available boundaries for this geometry type
	available_boundaries = _get_available_boundaries_for_geometry(geometry_type, geometry_boundaries_json_path)
	
	if not available_boundaries:
		return []
	
	tokens = _tokenize(prompt)
	
	synonyms = dict(DEFAULT_BOUNDARY_SYNONYMS)
	if extra_synonyms:
		for k, v in extra_synonyms.items():
			if k in synonyms:
				synonyms[k] = list({*synonyms[k], *v})
			else:
				synonyms[k] = v

	scored: List[Tuple[str, float, Dict[str, Any]]] = []
	for boundary in available_boundaries:
		s, rationale = _score_candidate(boundary, tokens, synonyms, prompt)
		scored.append((boundary, s, rationale))

	# Sort by confidence (descending)
	scored.sort(key=lambda x: x[1], reverse=True)

	# Return top candidates with confidence > 0.1
	results = []
	for boundary, confidence, rationale in scored:
		if confidence > 0.1:
			results.append({
				"boundary": boundary,
				"confidence": confidence,
				"rationale": rationale
			})

	return results

def _get_available_boundaries_for_geometry(geometry_type: str, geometry_boundaries_json_path: Path) -> List[str]:
	"""Get available boundaries for a specific geometry type"""
	try:
		with open(geometry_boundaries_json_path, 'r') as f:
			data = json.load(f)
		
		# Search in all dimension categories
		for dim_group in ['1d', '2d', '3d']:
			if dim_group in data.get('geometries', {}):
				if geometry_type in data['geometries'][dim_group]:
					return data['geometries'][dim_group][geometry_type].get('available_boundaries', [])
		
		# Fallback to common boundaries
		return ['left', 'right', 'top', 'bottom']
	except Exception:
		return ['left', 'right', 'top', 'bottom']
