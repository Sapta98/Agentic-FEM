#!/usr/bin/env python3
"""
Prompt Templates for Physics Simulation Natural Language Processing
"""

import json
import re
from typing import Dict, Any, Optional

class PromptManager:
	"""Prompt manager for natural language parsing of physics simulations"""

	def __init__(self, client, model: str = "gpt-4"):
		self.client = client
		self.model = model

	def _call_openai(self, prompt_content: str) -> Dict[str, Any]:
		"""Call OpenAI API with the given prompt"""
		try:
			# Add JSON format instruction to the prompt
			json_prompt = prompt_content + "\n\nIMPORTANT: Respond ONLY with valid JSON. Do not include any other text, explanations, or formatting."

			# GPT-5 doesn't support custom temperature, use default
			if "gpt-5" in self.model.lower():
				response = self.client.chat.completions.create(
					model=self.model,
					messages=[{"role": "user", "content": json_prompt}]
				)
			else:
				response = self.client.chat.completions.create(
					model=self.model,
					messages=[{"role": "user", "content": json_prompt}],
					temperature=0.1
				)

			content = response.choices[0].message.content.strip()

			# Try to parse JSON, handle cases where response might have extra text
			try:
				return json.loads(content)
			except json.JSONDecodeError:
				# Try to extract JSON from the response if it has extra text
				json_match = re.search(r'\{.*\}', content, re.DOTALL)
				if json_match:
					return json.loads(json_match.group())
				else:
					raise ValueError(f"Could not parse JSON from response: {content}")

		except Exception as e:
			return {"error": str(e)}

	def identify_physics_type(self, prompt: str) -> Dict[str, Any]:
		"""Function 1: Identify physics type from prompt"""
		return self._call_openai(self._get_physics_type_prompt(prompt))

	def identify_geometry_type(self, prompt: str, physics_type: str, context_summary: str) -> Dict[str, Any]:
		"""Function 2: Identify geometry type from prompt"""
		return self._call_openai(self._get_geometry_type_prompt(prompt, physics_type, context_summary))

	def identify_material_type(self, prompt: str, physics_type: str, context_summary: str) -> Dict[str, Any]:
		"""Function 3: Identify material type from prompt"""
		return self._call_openai(self._get_material_type_prompt(prompt, physics_type, context_summary))

	def analyze_boundary_locations(self, prompt: str, geometry_type: str, available_boundaries: list) -> Dict[str, Any]:
		"""Function 4a: Map vague boundary locations to specific boundaries for a geometry"""
		return self._call_openai(self._get_boundary_locations_prompt(prompt, geometry_type, available_boundaries))
	
	def analyze_boundary_condition_types(self, prompt: str, physics_type: str, boundary_locations: list) -> Dict[str, Any]:
		"""Function 4b: Analyze boundary condition types based on boundary locations"""
		return self._call_openai(self._get_boundary_condition_types_prompt(prompt, physics_type, boundary_locations))

	def analyze_external_loads(self, prompt: str, physics_type: str, context_summary: str) -> Dict[str, Any]:
		"""Function 5: Analyze external loads from prompt"""
		return self._call_openai(self._get_external_loads_prompt(prompt, physics_type, context_summary))

	def complete_data_structure(self, prompt: str, physics_type: str, material_type: str, geometry_type: str, boundary_conditions: dict, context_summary: str) -> Dict[str, Any]:
		"""Stage 3: Complete data structure with material coefficients, dimensions, and boundary conditions"""
		return self._call_openai(self._get_data_structure_completion_prompt(prompt, physics_type, material_type, geometry_type, boundary_conditions, context_summary))

	def parse_dimensions(self, prompt: str, geometry_type: str, required_dimensions: list) -> Dict[str, Any]:
		"""Parse geometry dimensions from prompt"""
		return self._call_openai(self._get_dimension_parsing_prompt(prompt, geometry_type, required_dimensions))

	def get_material_properties(self, material_type: str, physics_type: str) -> Dict[str, Any]:
		"""Get material properties for a given material type"""
		return self._call_openai(self._get_material_properties_prompt(material_type, physics_type))

	@staticmethod
	def _get_physics_type_prompt(prompt: str) -> str:
		"""Generate prompt for physics type identification"""
		return f"""
Analyze this prompt and determine if it's related to physics simulation:

Prompt: "{prompt}"

Respond with a JSON object containing:
1. "is_physics": true/false - whether this is a physics simulation prompt that we can handle
2. "confidence_scores": {{
	"heat_transfer": 0.0-1.0,
	"solid_mechanics": 0.0-1.0,
	"other_physics": 0.0-1.0
}}
3. "reasoning": brief explanation of the analysis

RULES:
- "heat_transfer": temperature, thermal, conduction, convection, heat flow, heat transfer
- "solid_mechanics": stress, strain, deformation, elasticity, forces, displacement, mechanical analysis
- "other_physics": fluid mechanics, electromagnetics, quantum mechanics, etc.

CRITICAL: Set "is_physics": true ONLY if the sum of heat_transfer + solid_mechanics confidence is >= 0.8.
If other_physics confidence is high (>0.5), set "is_physics": false.

UNSUPPORTED PHYSICS TERMS (should increase "other_physics" confidence):
- FLUID MECHANICS: vorticity, flow, fluid, velocity, pressure, turbulence, navier stokes, boundary layer, reynolds number, viscosity, laminar, turbulent, flow field, velocity field, streamlines, circulation, lift, drag, aerodynamics, hydrodynamics
- ELECTROMAGNETICS: electric field, magnetic field, EM, electromagnetic, maxwell equations, voltage, current, resistance, capacitance, inductance, wave propagation
- OTHER PHYSICS: quantum mechanics, quantum physics, particle physics, nuclear physics, relativity, gravitational waves, astrophysics, cosmology

Examples:
- "hi" -> {{"is_physics": false, "confidence_scores": {{"heat_transfer": 0.0, "solid_mechanics": 0.0, "other_physics": 0.0}}, "reasoning": "No physics content"}}
- "simulate heat transfer" -> {{"is_physics": true, "confidence_scores": {{"heat_transfer": 1.0, "solid_mechanics": 0.0, "other_physics": 0.0}}, "reasoning": "Explicit heat transfer simulation"}}
- "analyze stress in beam" -> {{"is_physics": true, "confidence_scores": {{"heat_transfer": 0.0, "solid_mechanics": 1.0, "other_physics": 0.0}}, "reasoning": "Structural mechanics analysis"}}
- "stress variation" -> {{"is_physics": true, "confidence_scores": {{"heat_transfer": 0.0, "solid_mechanics": 0.9, "other_physics": 0.1}}, "reasoning": "Stress analysis indicates solid mechanics"}}
- "thermal stress analysis" -> {{"is_physics": true, "confidence_scores": {{"heat_transfer": 0.4, "solid_mechanics": 0.4, "other_physics": 0.2}}, "reasoning": "Combines thermal and mechanical effects"}}
- "analyze vorticity" -> {{"is_physics": false, "confidence_scores": {{"heat_transfer": 0.0, "solid_mechanics": 0.0, "other_physics": 1.0}}, "reasoning": "Vorticity is fluid mechanics, not supported"}}
"""

	@staticmethod
	def _get_geometry_type_prompt(prompt: str, physics_type: str, context_summary: str) -> str:
		"""Generate prompt for geometry type identification"""
		return f"""
Identify the geometry type from this prompt:

Prompt: "{prompt}"
Physics Type: {physics_type}
Previous Context: {context_summary}

Look for geometry types such as:
- bar, beam, rod, shaft
- plate, sheet, slab, disc, circle
- cube, block, rectangular solid
- cylinder, pipe, tube
- sphere, ball
- complex shapes (specify if mentioned)

IMPORTANT: Handle compound phrases like "iron cylinder", "steel beam", "aluminum plate":
- "iron cylinder" -> geometry_type: "cylinder"
- "steel beam" -> geometry_type: "beam"
- "aluminum plate" -> geometry_type: "plate"
- "copper disc" -> geometry_type: "disc"

Respond with JSON:
{{
	"has_geometry_type": true/false,
	"geometry_type": "e.g., bar, beam, plate, cube, cylinder",
	"confidence": 0.0-1.0,
	"reasoning": "explanation of geometry type detection"
}}

Examples:
- "copper bar" -> {{"has_geometry_type": true, "geometry_type": "bar", "confidence": 0.9, "reasoning": "Explicitly mentions bar geometry"}}
- "steel beam" -> {{"has_geometry_type": true, "geometry_type": "beam", "confidence": 0.9, "reasoning": "Explicitly mentions beam geometry"}}
- "aluminum plate" -> {{"has_geometry_type": true, "geometry_type": "plate", "confidence": 0.9, "reasoning": "Explicitly mentions plate geometry"}}
- "cube" -> {{"has_geometry_type": true, "geometry_type": "cube", "confidence": 1.0, "reasoning": "Explicit cube geometry"}}
- "iron" -> {{"has_geometry_type": false, "geometry_type": null, "confidence": 0.0, "reasoning": "No geometry type mentioned"}}
"""

	@staticmethod
	def _get_material_type_prompt(prompt: str, physics_type: str, context_summary: str) -> str:
		"""Generate prompt for material type identification"""
		return f"""
Identify the material type from this prompt:

Prompt: "{prompt}"
Physics Type: {physics_type}
Previous Context: {context_summary}

Look for material types such as:
- Metals: steel, aluminum, copper, iron, brass, titanium, etc.
- Polymers: plastic, rubber, PVC, etc.
- Ceramics: concrete, glass, ceramic, etc.
- Composites: carbon fiber, fiberglass, etc.

IMPORTANT: Handle compound phrases like "iron cylinder", "steel beam", "aluminum plate":
- "iron cylinder" -> material_type: "iron"
- "steel beam" -> material_type: "steel"
- "aluminum plate" -> material_type: "aluminum"

Respond with JSON:
{{
	"has_material_type": true/false,
	"material_type": "e.g., steel, aluminum, copper, iron",
	"confidence": 0.0-1.0,
	"reasoning": "explanation of material type detection"
}}

Examples:
- "copper bar" -> {{"has_material_type": true, "material_type": "copper", "confidence": 0.9, "reasoning": "Explicitly mentions copper material"}}
- "steel beam" -> {{"has_material_type": true, "material_type": "steel", "confidence": 0.9, "reasoning": "Explicitly mentions steel material"}}
- "aluminum plate" -> {{"has_material_type": true, "material_type": "aluminum", "confidence": 0.9, "reasoning": "Explicitly mentions aluminum material"}}
- "bar" -> {{"has_material_type": false, "material_type": null, "confidence": 0.0, "reasoning": "No material type mentioned"}}
"""

	@staticmethod
	def _get_boundary_locations_prompt(prompt: str, geometry_type: str, available_boundaries: list) -> str:
		"""Generate prompt for mapping vague boundary locations to specific boundaries"""
		return f"""
Analyze the user's prompt and map vague boundary location descriptions to the SPECIFIC AVAILABLE BOUNDARIES for this geometry.

Prompt: "{prompt}"
Geometry Type: {geometry_type}
AVAILABLE BOUNDARIES (you MUST use ONLY these): {', '.join(available_boundaries)}

YOUR TASK:
1. Identify ALL boundary locations mentioned in the prompt (even if vague like "one side", "opposite side")
2. Map each vague location to ONE of the available boundaries listed above
3. Use intelligent reasoning based on:
   - Temperature/force values (higher → typically left/top, lower → right/bottom)
   - Context clues ("one side" vs "opposite side")
   - Geometric conventions

Respond with JSON:
{{
	"boundary_locations": [
		{{
			"vague_location": "user's vague description (e.g., 'one side', 'opposite side')",
			"specific_boundary": "ONE of {', '.join(available_boundaries)}",
			"value": "numerical value if mentioned",
			"confidence": 0.0-1.0
		}}
	],
	"reasoning": "explanation of how vague locations were mapped to specific boundaries"
}}

EXAMPLES:
- Prompt: "100°C at one side, 10°C at opposite side" for square with [left, right, top, bottom]
  → {{"boundary_locations": [{{"vague_location": "one side", "specific_boundary": "left", "value": 100, "confidence": 0.9}}, {{"vague_location": "opposite side", "specific_boundary": "right", "value": 10, "confidence": 0.9}}]}}

- Prompt: "100°C at one end, 10°C at other end" for disc with [circumference, center]
  → {{"boundary_locations": [{{"vague_location": "one end", "specific_boundary": "circumference", "value": 100, "confidence": 0.9}}, {{"vague_location": "other end", "specific_boundary": "center", "value": 10, "confidence": 0.9}}]}}

- Prompt: "insulated on top and bottom" for plate with [left, right, top, bottom]
  → {{"boundary_locations": [{{"vague_location": "top", "specific_boundary": "top", "value": null, "confidence": 1.0}}, {{"vague_location": "bottom", "specific_boundary": "bottom", "value": null, "confidence": 1.0}}]}}

GEOMETRY-SPECIFIC MAPPING RULES:
- For DISC: "one end" = circumference (outer edge), "other end" = center (inner point)
- For CYLINDER: "one end" = top, "other end" = bottom
- For LINE/ROD: "one end" = left, "other end" = right
- For RECTANGLE/PLATE: "one side" = left, "other side" = right
"""

	@staticmethod
	def _get_boundary_condition_types_prompt(prompt: str, physics_type: str, boundary_locations: list) -> str:
		"""Generate prompt for determining boundary condition types from boundary locations"""
		return f"""
Based on the boundary locations identified, determine the TYPE of boundary condition for each location.

Prompt: "{prompt}"
Physics Type: {physics_type}
Boundary Locations Identified: {boundary_locations}

BOUNDARY CONDITION TYPES (match to standard types from boundary_conditions.json):

For HEAT TRANSFER:
- Temperature values (°C, °F, K) → type: "temperature", bc_type: "dirichlet"
- "insulated", "no heat flux", "adiabatic" → type: "insulated", bc_type: "neumann" (value: 0)
- "heat flux", "thermal flux" → type: "flux", bc_type: "neumann"
- "convection", "cooling", "heating" → type: "convection", bc_type: "robin"

For SOLID MECHANICS:
- "fixed", "clamped", "cantilever", "built-in" → type: "fixed", bc_type: "dirichlet" (value: 0)
- "free", "unconstrained", "unrestrained" → type: "free", bc_type: "neumann" (value: 0)
- "force", "load" → type: "force", bc_type: "neumann"
- "pressure" → type: "pressure", bc_type: "neumann"
- "displacement", "deflection" → type: "displacement", bc_type: "dirichlet"
- "symmetry", "symmetric" → type: "symmetry", bc_type: "neumann" (value: 0)

Respond with JSON:
{{
	"boundary_conditions": [
		{{
			"location": "specific boundary from the identified locations",
			"type": "temperature/flux/convection/fixed/free/force/displacement/symmetry",
			"value": "numerical value",
			"bc_type": "dirichlet/neumann/robin",
			"confidence": 0.0-1.0
		}}
	],
	"reasoning": "explanation of boundary condition type classification"
}}
"""

	@staticmethod
	def _get_external_loads_prompt(prompt: str, physics_type: str, context_summary: str) -> str:
		"""Generate prompt for external loads analysis"""
		return f"""
Analyze external loads from this prompt:

Prompt: "{prompt}"
Physics Type: {physics_type}
Previous Context: {context_summary}

Look for load types:
- Point loads: "force", "load", "weight", "mass"
- Distributed loads: "pressure", "distributed", "uniform"
- Thermal loads: "heat", "temperature", "thermal"
- Body forces: "gravity", "acceleration"

Respond with JSON:
{{
	"has_external_loads": true/false,
	"external_loads": [
		{{
			"type": "point/distributed/thermal/body",
			"magnitude": "numerical value if specified",
			"direction": "x/y/z/radial/tangential",
			"location": "where applied",
			"confidence": 0.0-1.0
		}}
	],
	"reasoning": "explanation of load detection"
}}

Examples:
- "100 N force" -> {{"has_external_loads": true, "external_loads": [{{"type": "point", "magnitude": 100, "direction": "unspecified", "location": "unspecified", "confidence": 0.9}}], "reasoning": "Explicit force magnitude"}}
- "pressure 5 MPa" -> {{"has_external_loads": true, "external_loads": [{{"type": "distributed", "magnitude": 5e6, "direction": "normal", "location": "surface", "confidence": 0.9}}], "reasoning": "Pressure load specified"}}
- "gravity" -> {{"has_external_loads": true, "external_loads": [{{"type": "body", "magnitude": 9.81, "direction": "downward", "location": "all", "confidence": 0.8}}], "reasoning": "Gravity mentioned"}}
"""

	@staticmethod
	def _get_data_structure_completion_prompt(prompt: str, physics_type: str, material_type: str, geometry_type: str, boundary_conditions: dict, context_summary: str) -> str:
		"""Generate prompt for data structure completion"""
		return f"""
Complete the simulation data structure based on the provided information:

Prompt: "{prompt}"
Physics Type: {physics_type}
Material Type: {material_type}
Geometry Type: {geometry_type}
Boundary Conditions: {boundary_conditions}
Previous Context: {context_summary}

Fill in missing information and provide complete simulation configuration:

Respond with JSON:
{{
	"geometry_dimensions": {{
		"length": "value if specified",
		"width": "value if specified",
		"height": "value if specified",
		"radius": "value if specified"
	}},
	"material_properties": {{
		"density": "value if known",
		"thermal_conductivity": "value if heat transfer",
		"youngs_modulus": "value if solid mechanics",
		"poisson_ratio": "value if solid mechanics"
	}},
	"boundary_conditions": [
		{{
			"type": "fixed/free/temperature/force",
			"value": "numerical value",
			"location": "where applied"
		}}
	],
	"simulation_parameters": {{
		"mesh_quality": "coarse/medium/fine",
		"solver_type": "appropriate solver",
		"time_stepping": "if transient"
	}},
	"completeness": {{
		"complete": true/false,
		"missing_components": ["list of missing items"]
	}}
}}
"""

	@staticmethod
	def _get_dimension_parsing_prompt(prompt: str, geometry_type: str, required_dimensions: list) -> str:
		"""Generate prompt for dimension parsing"""
		return f"""
Parse geometry dimensions from this prompt:

Prompt: "{prompt}"
Geometry Type: {geometry_type}
Required Dimensions: {required_dimensions}

Extract numerical values for the required dimensions. Look for:
- Length: "length", "L", "long", "longitudinal"
- Width: "width", "W", "wide", "transverse"
- Height: "height", "H", "thick", "thickness"
- Radius: "radius", "R", "diameter/2"
- Diameter: "diameter", "D", "d"

Respond with JSON:
{{
	"dimensions": {{
		"length": "value if found",
		"width": "value if found",
		"height": "value if found",
		"radius": "value if found",
		"diameter": "value if found"
	}},
	"units": {{
		"length": "m/mm/cm/in",
		"width": "m/mm/cm/in",
		"height": "m/mm/cm/in",
		"radius": "m/mm/cm/in",
		"diameter": "m/mm/cm/in"
	}},
	"confidence": 0.0-1.0,
	"reasoning": "explanation of dimension extraction"
}}

Examples:
- "2m long, 0.1m wide" -> {{"dimensions": {{"length": 2.0, "width": 0.1}}, "units": {{"length": "m", "width": "m"}}, "confidence": 0.9, "reasoning": "Explicit dimensions provided"}}
- "radius 5cm" -> {{"dimensions": {{"radius": 5.0}}, "units": {{"radius": "cm"}}, "confidence": 0.9, "reasoning": "Radius explicitly specified"}}
- "10mm thick" -> {{"dimensions": {{"height": 10.0}}, "units": {{"height": "mm"}}, "confidence": 0.8, "reasoning": "Thickness implies height dimension"}}
"""

	@staticmethod
	def _get_material_properties_prompt(material_type: str, physics_type: str) -> str:
		"""Generate prompt for material properties"""
		return f"""
Provide material properties for the given material and physics type:

Material Type: {material_type}
Physics Type: {physics_type}

Respond with JSON containing typical material properties:

{{
	"material_type": "{material_type}",
	"physics_type": "{physics_type}",
	"properties": {{
		"density": "kg/m³",
		"thermal_conductivity": "W/(m·K) if heat transfer",
		"specific_heat": "J/(kg·K) if heat transfer",
		"youngs_modulus": "Pa if solid mechanics",
		"poisson_ratio": "dimensionless if solid mechanics",
		"yield_strength": "Pa if solid mechanics",
		"thermal_expansion": "1/K if both"
	}},
	"source": "reference source",
	"confidence": "0.0-1.0"
}}

Provide realistic values based on engineering materials database.
"""

	def get_available_models(self) -> list:
		"""Get list of available models"""
		return ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"]

	def validate_response(self, response: Dict[str, Any]) -> bool:
		"""Validate API response structure"""
		if "error" in response:
			return False
		
		# Check for required fields based on function type
		required_fields = ["reasoning"]
		return all(field in response for field in required_fields)

	def get_response_confidence(self, response: Dict[str, Any]) -> float:
		"""Get confidence score from response"""
		if "error" in response:
			return 0.0
		
		# Look for confidence field in response
		if "confidence" in response:
			return float(response["confidence"])
		
		# Look for confidence_scores
		if "confidence_scores" in response:
			scores = response["confidence_scores"]
			if isinstance(scores, dict):
				return max(scores.values()) if scores else 0.0
		
		return 0.5  # Default confidence

	def format_response(self, response: Dict[str, Any]) -> str:
		"""Format response for display"""
		if "error" in response:
			return f"Error: {response['error']}"
		
		formatted = []
		for key, value in response.items():
			if key != "reasoning":
				formatted.append(f"{key}: {value}")
		
		if "reasoning" in response:
			formatted.append(f"Reasoning: {response['reasoning']}")
		
		return "\n".join(formatted)