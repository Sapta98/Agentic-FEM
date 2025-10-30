"""
Central Configuration Manager
Single source of truth for all application configuration
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

logger = logging.getLogger(__name__)

Number = Union[int, float]

class ConfigManager:
	"""Central configuration manager for the entire application"""

	def __init__(self, config_file: str = "config/config.json"):
		# Make path absolute relative to project root
		if not Path(config_file).is_absolute():
			current_dir = Path(__file__).parent
			project_root = current_dir.parent if current_dir.name == "config" else current_dir
			self.config_file = project_root / config_file
		else:
			self.config_file = Path(config_file)

		self.config_file.parent.mkdir(exist_ok=True)
		self._config = self._load_config()
		self._dimensions_config = None
		self._boundary_conditions_config = None

	# -------------------- IO --------------------

	def _load_config(self) -> Dict[str, Any]:
		"""Load configuration from JSON file"""
		if self.config_file.exists():
			with open(self.config_file, 'r') as f:
				config = json.load(f)
				logger.info(f"Loaded config from {self.config_file}")
				return config
		logger.error(f"Config file {self.config_file} not found - configuration file is required")
		raise FileNotFoundError(f"Configuration file {self.config_file} is required but not found")

	def _resolve_sidecar_path(self, key: str, default_name: str) -> Path:
		"""
		Resolve sidecar json paths from main config:
		- looks in config["configuration_files"][key] if present
		- falls back to sibling file (e.g., boundary_conditions.json) next to main config
		"""
		cfg = self._config or {}
		sidecars = cfg.get("configuration_files", {})
		if isinstance(sidecars, dict) and key in sidecars and sidecars[key]:
			path = Path(sidecars[key])
			return path if path.is_absolute() else self.config_file.parent / path
		return self.config_file.parent / default_name

	def save_config(self) -> bool:
		"""Save configuration to JSON file"""
		# Add timestamp if simulation section exists
		sim = self._config.setdefault("simulation", {})
		sim["last_updated"] = datetime.now().isoformat()

		with open(self.config_file, 'w') as f:
			json.dump(self._config, f, indent=2)
			logger.info(f"Saved config to {self.config_file}")
			return True

	# -------------------- Safe getters/setters --------------------

	def get(self, key: str, default: Any = None) -> Any:
		"""Get configuration value by key (supports dot notation)"""
		value = self._config
		for k in key.split('.'):
			if not isinstance(value, dict) or k not in value:
				return default
			value = value[k]
		return value

	def set(self, key: str, value: Any) -> bool:
		"""Set configuration value by key (supports dot notation)"""
		keys = key.split('.')
		config = self._config
		for k in keys[:-1]:
			if k not in config or not isinstance(config[k], dict):
				config[k] = {}
			config = config[k]
		config[keys[-1]] = value
		return self.save_config()

	def get_all(self) -> Dict[str, Any]:
		"""Get entire configuration"""
		return json.loads(json.dumps(self._config))  # deep copy

	# -------------------- Geometry & materials --------------------

	def update_geometry(self, geometry_type: str, dimensions: Dict[str, float]) -> bool:
		self._config.setdefault("geometry", {})
		self._config["geometry"]["type"] = geometry_type
		self._config["geometry"]["dimensions"] = dimensions
		return self.save_config()

	def update_material(self, material_type: str, properties: Dict[str, float]) -> bool:
		self._config.setdefault("material", {})
		self._config["material"]["type"] = material_type
		self._config["material"]["properties"] = properties
		return self.save_config()

	# -------------------- PDE config (new) --------------------

	def get_pde_config(self) -> Dict[str, Any]:
		return self._config.setdefault("pde_config", {})

	def update_pde_config(self, physics_type: str, boundary_conditions: List[Dict[str, Any]], **kwargs) -> bool:
		"""
		Update PDE config (new schema). Accepts additional keyword settings like:
		family, degree, temp_degree, disp_degree, material, body_force, etc.
		"""
		pde = self._config.setdefault("pde_config", {})
		pde["physics_type"] = (physics_type or "").strip().lower()
		pde["boundary_conditions"] = boundary_conditions or []
		for k, v in kwargs.items():
			pde[k] = v
		# Back-compat mirror (optional)
		self._config["physics"] = {
			"type": pde["physics_type"],
			"boundary_conditions": pde["boundary_conditions"],
			"external_loads": kwargs.get("external_loads", []),
			"mesh_parameters": self._config.get("physics", {}).get("mesh_parameters", self._config.get("mesh", {}))
		}
		return self.save_config()

	# Backward-compatible method name (writes into pde_config)
	def update_physics(self, physics_type: str, boundary_conditions: list, external_loads: list) -> bool:
		return self.update_pde_config(physics_type, boundary_conditions, external_loads=external_loads)

	# -------------------- Sidecar loads --------------------

	def _load_dimensions_config(self) -> Dict[str, Any]:
		if self._dimensions_config is None:
			dimensions_file = self._resolve_sidecar_path("dimensions", "dimensions.json")
			if dimensions_file.exists():
				with open(dimensions_file, 'r') as f:
					self._dimensions_config = json.load(f)
					logger.info(f"Loaded dimensions config from {dimensions_file}")
			else:
				logger.warning(f"Dimensions config file {dimensions_file} not found")
				self._dimensions_config = {}
		return self._dimensions_config

	def _load_boundary_conditions_config(self) -> Dict[str, Any]:
		if self._boundary_conditions_config is None:
			bc_file = self._resolve_sidecar_path("boundary_conditions", "boundary_conditions.json")
			if bc_file.exists():
				with open(bc_file, 'r') as f:
					self._boundary_conditions_config = json.load(f)
					logger.info(f"Loaded boundary conditions config from {bc_file}")
			else:
				logger.warning(f"Boundary conditions config file {bc_file} not found")
				self._boundary_conditions_config = {}
		return self._boundary_conditions_config

	def get_dimensions_config(self) -> Dict[str, Any]:
		return self._load_dimensions_config()

	def get_boundary_conditions_config(self) -> Dict[str, Any]:
		return self._load_boundary_conditions_config()

	# -------------------- Geometry helpers --------------------

	def get_geometry_dimensions(self, geometry_type: str, dimension_type: str = None) -> Dict[str, Any]:
		dimensions_config = self.get_dimensions_config()
		for dim_type, geometries in dimensions_config.get('geometry_dimensions', {}).items():
			if geometry_type in geometries:
				geometry_config = geometries[geometry_type]
				return geometry_config.get(dimension_type, {}) if dimension_type else geometry_config
		return {}

	def get_dimension_units(self, geometry_type: str, dimension_name: str) -> list:
		geometry_config = self.get_geometry_dimensions(geometry_type)
		return geometry_config.get('dimension_units', {}).get(dimension_name, [])

	def convert_dimension_units(self, value: float, from_unit: str, to_unit: str) -> float:
		conversions = self.get_dimensions_config().get('unit_conversions', {})
		if from_unit in conversions and to_unit in conversions[from_unit]:
			return value * conversions[from_unit][to_unit]
		logger.warning(f"No conversion found from {from_unit} to {to_unit}")
		return value

	def validate_geometry_dimensions(self, geometry_type: str, dimensions: Dict[str, Any]) -> Dict[str, Any]:
		geometry_config = self.get_geometry_dimensions(geometry_type)
		required_dims = geometry_config.get('required_dimensions', [])
		result = {'valid': True, 'errors': [], 'warnings': [], 'converted_dimensions': {}}

		# Check required
		for req_dim in required_dims:
			if req_dim not in dimensions:
				result['errors'].append(f"Missing required dimension: {req_dim}")
				result['valid'] = False

		# Parse & convert
		for dim_name, dim_value in dimensions.items():
			if isinstance(dim_value, str):
				try:
					import re
					match = re.match(r'([\d.+-eE]+)\s*([a-zA-Z]+)?', dim_value.strip())
					if match:
						value, unit = float(match.group(1)), (match.group(2) or 'm')
						if unit != 'm':
							value = self.convert_dimension_units(value, unit, 'm')
						result['converted_dimensions'][dim_name] = value
					else:
						result['converted_dimensions'][dim_name] = float(dim_value)
				except ValueError:
					result['errors'].append(f"Invalid dimension value for {dim_name}: {dim_value}")
					result['valid'] = False
			else:
				result['converted_dimensions'][dim_name] = float(dim_value)
		return result

	# -------------------- Simulation packaging --------------------

	def get_geometry_for_mesh(self) -> Dict[str, Any]:
		return self.get("geometry", {})

	def get_full_simulation_config(self) -> Dict[str, Any]:
		return {
			"geometry": self.get("geometry", {}),
			"material": self.get("material", {}),
			"pde_config": self.get("pde_config", {}),
			"simulation": self.get("simulation", {})
		}

	# -------------------- Boundary templates & normalization --------------------

	def get_boundary_condition_templates(self, physics_type: Optional[str] = None) -> Dict[str, Any]:
		"""
		Return the template section(s) from boundary_conditions.json.
		- If physics_type is given: returns {"description":.., "templates":[...]}
		- Else: returns the whole {"physics_types":{...}, ...}
		"""
		bc = self.get_boundary_conditions_config()
		if physics_type:
			return bc.get("physics_types", {}).get(physics_type, {})
		return bc

	def get_available_physics_types(self) -> List[str]:
		bc = self.get_boundary_conditions_config()
		return list(bc.get("physics_types", {}).keys())

	# --- Normalization helpers ---

	def _canonical_location(self, label: str, mesh_dimensionality: int) -> str:
		"""Map user/human labels to canonical locations using the aliases in boundary_conditions.json."""
		bc = self.get_boundary_conditions_config()
		aliases = bc.get("aliases", {}).get(str(mesh_dimensionality) + "D", {})
		label_norm = (label or "").strip().lower()
		# direct canonical accepts
		canon = {
			"x_min","x_max","y_min","y_max","z_min","z_max","all_boundary"
		}
		if label_norm in canon:
			return label_norm
		return aliases.get(label_norm, "all_boundary")

	def _merge_components(self, d: Dict[str, Any], keys: Tuple[str, str, str], gdim: int) -> List[Number]:
		"""Collect *_x/_y/_z into a vector of length gdim; missing become 0.0."""
		out = [0.0]*gdim
		vals = []
		for k in keys[:gdim]:
			if k in d and d[k] != "":
				vals.append(float(d[k]))
		for i, v in enumerate(vals[:gdim]):
			out[i] = v
		return out

	def _convert_units_if_needed(self, value: Any, units: Any) -> Any:
		"""Currently: only °C → K. Extend as needed."""
		if isinstance(units, str) and units.lower() in ("c", "celsius", "degc", "°c"):
			# scalar
			if isinstance(value, (int, float)):
				return float(value) + 273.15
			# object with fields
			if isinstance(value, dict):
				v = dict(value)
				for key in ("T", "Tinf", "temperature"):
					if key in v:
						v[key] = float(v[key]) + 273.15
				return v
		return value

	def normalize_bc_entry(self, bc_in: Dict[str, Any], mesh_dimensionality: int, gdim: int) -> Dict[str, Any]:
		"""
		Turns a template/custom BC into solver-ready:
		{type, location, value}
		- maps 'boundary' -> 'location'
		- applies aliases
		- merges component fields
		- corrects 'pressure' vs 'neumann'
		- converts Celsius to Kelvin if units say so
		"""
		bc = dict(bc_in)  # copy

		# keys & type
		btype = (bc.get("type") or "").strip().lower()
		location = bc.get("location", bc.get("boundary", "all_boundary"))
		location = self._canonical_location(location, mesh_dimensionality)

		# value resolution
		value = bc.get("value", None)

		# Heat: dirichlet/neumann/robin
		if btype in ("dirichlet", "neumann", "robin"):
			# collect component values if present (force_x, displacement_x, etc.)
			if value is None:
				# try common component names
				comp = self._merge_components(
					bc, ("*_x","*_y","*_z"), gdim
				)  # placeholder, handled below
				# attempt known exact keys
				for group in (("force_x","force_y","force_z"),
				              ("displacement_x","displacement_y","displacement_z")):
					if any(k in bc for k in group):
						value = self._merge_components(bc, group, gdim)
						break

			# robin structured value (convection/radiation)
			if isinstance(bc.get("value"), dict):
				value = dict(bc["value"])

		# Solid: traction vs pressure
		if btype == "pressure":
			# scalar pressure
			if value is None:
				value = float(bc.get("pressure", 0.0))
		elif btype in ("traction", "neumann"):
			if value is None:
				value = self._merge_components(bc, ("force_x","force_y","force_z"), gdim)

		# Units conversion if present
		units = bc.get("units") or bc.get("parameters", {}).get("units")
		value = self._convert_units_if_needed(value, units)

		return {"type": btype, "location": location, "value": value}

	def normalize_all_bcs(self, mesh_dimensionality: int, gdim: int) -> List[Dict[str, Any]]:
		"""
		Normalize whatever is in config (pde_config.boundary_conditions OR legacy physics.boundary_conditions)
		into a solver-ready list.
		"""
		pde = self.get_pde_config()
		raw_bcs = pde.get("boundary_conditions")
		if raw_bcs is None:
			raw_bcs = self.get("physics.boundary_conditions", []) or []

		normalized: List[Dict[str, Any]] = []
		for bc in raw_bcs:
			try:
				normalized.append(self.normalize_bc_entry(bc, mesh_dimensionality, gdim))
			except Exception as e:
				logger.warning(f"Skipping BC due to normalization error: {e} | BC: {bc}")
		# write back
		pde["boundary_conditions"] = normalized
		self._config["pde_config"] = pde
		self.save_config()
		return normalized

	# -------------------- Templates -> Add BC --------------------

	def add_boundary_condition_from_template(self, physics_type: str, template_name: str, custom_params: Dict[str, Any] = None,
	                                         mesh_dimensionality: int = 3, gdim: int = 3) -> bool:
		"""
		Add a BC by finding a template (in boundary_conditions.json) and normalizing it.
		- physics_type: 'heat_transfer' | 'solid_mechanics'
		- template_name: the 'name' field inside the templates list
		- custom_params: fields to override in the template's 'parameters'
		"""
		bc_config = self.get_boundary_conditions_config()
		section = bc_config.get("physics_types", {}).get(physics_type, {})
		templates = section.get("templates", [])

		# find template by name
		tpl = next((t for t in templates if t.get("name") == template_name), None)
		if tpl is None:
			logger.error(f"Template {template_name!r} not found for physics type {physics_type!r}")
			return False

		# flatten template to runtime dict
		params = dict(tpl.get("parameters", {}))
		if custom_params:
			params.update(custom_params)

		runtime_bc = {
			"type": tpl.get("type"),
			"location": params.pop("location", params.pop("boundary", "all_boundary")),
			"value": params.pop("value", None),
			"units": params.pop("units", None)
		}
		# carry through remaining param fields for special types (e.g., robin dict)
		if runtime_bc["value"] is None and params:
			runtime_bc["value"] = params

		# normalize → solver-ready
		ready = self.normalize_bc_entry(runtime_bc, mesh_dimensionality, gdim)

		# append into pde_config.boundary_conditions
		pde = self.get_pde_config()
		pde.setdefault("boundary_conditions", []).append(ready)
		self._config["pde_config"] = pde
		return self.save_config()

	def reset_to_defaults(self) -> bool:
		"""Reset configuration to new defaults (pde_config schema)"""
		self._config = {
			"geometry": {"type": "beam", "dimensions": {}},
			"material": {"type": "steel", "properties": {}},
			"pde_config": {
				"physics_type": "heat_transfer",
				"family": "Lagrange",
				"degree": 1,
				"boundary_conditions": []
			},
			"simulation": {"solver": "fenics", "last_updated": datetime.now().isoformat()},
			"configuration_files": {
				"dimensions": "config/dimensions.json",
				"boundary_conditions": "config/boundary_conditions.json"
			}
		}
		return self.save_config()

# Global config manager instance
config_manager = ConfigManager()
