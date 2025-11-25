"""
Mesh Dimension Detector
=======================

Utilities for detecting appropriate mesh dimensions based on geometry
and physics simulation requirements.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
from ..config.mesh_config import mesh_config

def detect_mesh_dimensions(geometry_type: str, dimensions: Dict[str, float]) -> Dict[str, Any]:
	"""
	Detect appropriate mesh dimensions and configuration based on geometry only

	Args:
		geometry_type: Type of geometry (cube, beam, plate, etc.)
		dimensions: Geometry dimensions

	Returns:
		Dictionary with mesh configuration
	"""
	# Get mesh dimension from config
	mesh_dim = mesh_config.get_mesh_dimension(geometry_type)

	# Analyze geometry for mesh requirements
	geometry_analysis = analyze_geometry(geometry_type, dimensions)

	# Get geometry configuration
	geometry_config = mesh_config.get_geometry_config(geometry_type)

	# Determine mesh type and parameters
	mesh_type = determine_mesh_type(mesh_dim, geometry_type)
	mesh_parameters = determine_mesh_parameters(mesh_dim, geometry_analysis, geometry_config)

	return {
		'mesh_dimension': mesh_dim,
		'mesh_type': mesh_type,
		'geometry_type': geometry_type,
		'geometry_analysis': geometry_analysis,
		'mesh_parameters': mesh_parameters,
		'recommendations': get_mesh_recommendations(mesh_dim, geometry_type)
	}

def analyze_geometry(geometry_type: str, dimensions: Dict[str, float]) -> Dict[str, Any]:
	"""Analyze geometry for mesh generation requirements"""
	analysis = {
		'aspect_ratios': {},
		'symmetry': [],
		'complexity': 'simple',
		'special_features': []
	}

	# Calculate aspect ratios for different geometries
	if geometry_type in ['line', 'rod', 'bar']:
		length = dimensions.get('length', 1.0)
		analysis['aspect_ratios'] = {'length': length}
		analysis['complexity'] = '1d'

	elif geometry_type in ['beam']:
		length = dimensions.get('length', 1.0)
		width = dimensions.get('width', 0.1)
		height = dimensions.get('height', 0.1)

		analysis['aspect_ratios'] = {
			'length_width': length / width if width > 0 else 1.0,
			'length_height': length / height if height > 0 else 1.0,
			'width_height': width / height if height > 0 else 1.0
		}

		# Check if it's a thin beam (1D-like)
		if analysis['aspect_ratios']['length_width'] > 10 or analysis['aspect_ratios']['length_height'] > 10:
			analysis['complexity'] = '1d_approximation'
			analysis['special_features'].append('thin_structure')

	elif geometry_type in ['plate', 'membrane', 'square']:
		length = dimensions.get('length', 1.0)
		width = dimensions.get('width', length)  # For square, width = length
		thickness = dimensions.get('thickness', 0.01)

		analysis['aspect_ratios'] = {
			'length_width': length / width if width > 0 else 1.0,
			'length_thickness': length / thickness if thickness > 0 else 1.0,
			'width_thickness': width / thickness if thickness > 0 else 1.0
		}

		# Check if it's a thin plate (2D-like)
		if (analysis['aspect_ratios']['length_thickness'] > 20 or
			analysis['aspect_ratios']['width_thickness'] > 20):
			analysis['complexity'] = '2d_approximation'
			analysis['special_features'].append('thin_plate')

	elif geometry_type in ['cylinder']:
		radius = dimensions.get('radius', 0.5)
		# Handle both 'length' and 'height' for cylinder (height is preferred)
		height = dimensions.get('height', dimensions.get('length', 1.0))

		analysis['aspect_ratios'] = {
			'height_radius': height / radius if radius > 0 else 1.0
		}

		# Check for axisymmetric approximation
		if analysis['aspect_ratios']['height_radius'] > 5:
			analysis['symmetry'].append('axisymmetric')
			analysis['special_features'].append('long_cylinder')

	elif geometry_type in ['cube', 'box', 'solid', 'rectangular']:
		length = dimensions.get('length', 1.0)
		width = dimensions.get('width', 1.0)
		height = dimensions.get('height', 1.0)

		analysis['aspect_ratios'] = {
			'length_width': length / width if width > 0 else 1.0,
			'length_height': length / height if height > 0 else 1.0,
			'width_height': width / height if height > 0 else 1.0
		}

		# Check for symmetry
		if abs(analysis['aspect_ratios']['length_width'] - 1.0) < 0.1:
			analysis['symmetry'].append('cubic')
			if abs(analysis['aspect_ratios']['length_height'] - 1.0) < 0.1:
				analysis['symmetry'].append('square_section')

	elif geometry_type in ['disc']:
		radius = dimensions.get('radius', 1.0)
		analysis['aspect_ratios'] = {'radius': radius}
		analysis['symmetry'].append('circular')
		analysis['complexity'] = '2d'

	elif geometry_type in ['sphere']:
		radius = dimensions.get('radius', 1.0)
		analysis['aspect_ratios'] = {'radius': radius}
		analysis['symmetry'].append('spherical')
		analysis['complexity'] = '3d'

	return analysis

def determine_mesh_type(mesh_dim: int, geometry_type: str) -> str:
	"""Determine the appropriate mesh type"""
	if mesh_dim == 1:
		return 'line_1d'
	elif mesh_dim == 2:
		if geometry_type in ['plate', 'membrane', 'square']:
			return 'shell_2d'
		else:
			return 'planar_2d'
	else:  # mesh_dim == 3
		return 'volumetric_3d'

def determine_mesh_parameters(mesh_dim: int, geometry_analysis: Dict[str, Any],
		geometry_config: Dict[str, Any]) -> Dict[str, Any]:
	"""Determine mesh generation parameters"""
	params = {
		'resolution': mesh_config.get_mesh_quality()['resolution'],
		'element_type': 'linear',
		'special_requirements': [],
		'gmsh_type': geometry_config.get('gmsh_type', 'volume'),
		'mesh_size': mesh_config.get_mesh_quality()['gmsh_mesh_size']
	}

	# Adjust resolution based on complexity
	if geometry_analysis.get('complexity') == 'simple':
		params['resolution'] = int(params['resolution'] * 0.8)
	elif geometry_analysis.get('complexity') in ['1d_approximation', '2d_approximation']:
		params['resolution'] = int(params['resolution'] * 1.2)

	# Add geometry-specific requirements
	if geometry_analysis.get('special_features'):
		params['special_requirements'].extend(geometry_analysis['special_features'])

	# Determine element type based on mesh dimension
	if mesh_dim == 1:
		params['element_type'] = 'line'
	elif mesh_dim == 2:
		params['element_type'] = 'triangle'
	else:  # mesh_dim == 3
		params['element_type'] = 'tetra'

	return params

def get_mesh_recommendations(mesh_dim: int, geometry_type: str) -> List[str]:
	"""Get recommendations for mesh generation"""
	recommendations = []

	if mesh_dim == 1:
		recommendations.extend([
			"Use linear 1D elements",
			"Consider beam theory approximations",
			"Ensure sufficient element density for accurate results",
			"Use GMSH for high-quality 1D meshing"
		])
	elif mesh_dim == 2:
		recommendations.extend([
			"Use triangular or quadrilateral elements",
			"Consider shell/membrane theory for thin structures",
			"Apply appropriate boundary conditions",
			"Use GMSH Delaunay algorithm for 2D meshing"
		])
	else:  # mesh_dim == 3
		recommendations.extend([
			"Use tetrahedral or hexahedral elements",
			"Consider mesh quality metrics",
			"Apply proper 3D boundary conditions",
			"Use GMSH with quality optimization for 3D meshing"
		])

	# Add GMSH-specific recommendations
	recommendations.extend([
		"Use GMSH for professional mesh generation",
		"Enable mesh optimization and smoothing",
		"Consider element quality metrics (gamma, eta)"
	])

	return recommendations

def validate_geometry_dimensions(geometry_type: str, dimensions: Dict[str, float]) -> Dict[str, Any]:
	"""Validate geometry dimensions and provide feedback"""
	result = {
		'valid': True,
		'errors': [],
		'warnings': [],
		'suggestions': []
	}

	# Get required dimensions
	required_dims = mesh_config.get_required_dimensions(geometry_type)
	
	# Get geometry config to check for alternative dimensions
	geometry_config = mesh_config.get_geometry_config(geometry_type)
	alternative_dims = geometry_config.get('alternative_dimensions', {})
	
	# Check for missing dimensions, considering alternatives
	# Convert dimensions to numbers first to avoid type comparison errors
	missing_dims = []
	for dim in required_dims:
		dim_value = dimensions.get(dim)
		# Convert to number if it's a string
		try:
			if isinstance(dim_value, str):
				dim_value = float(dim_value.strip()) if dim_value.strip() else None
			elif dim_value is not None and not isinstance(dim_value, (int, float)):
				dim_value = float(dim_value)
		except (ValueError, TypeError):
			dim_value = None
		
		# Update dimensions dict with converted value if conversion succeeded
		if dim_value is not None:
			dimensions[dim] = dim_value
		
		if dim not in dimensions or dim_value is None or dim_value <= 0:
			# Check if there's an alternative dimension name
			alternatives = alternative_dims.get(dim, [])
			has_alternative = False
			for alt in alternatives:
				if alt in dimensions:
					alt_value = dimensions[alt]
					try:
						if isinstance(alt_value, str):
							alt_value = float(alt_value.strip()) if alt_value.strip() else None
						elif alt_value is not None and not isinstance(alt_value, (int, float)):
							alt_value = float(alt_value)
						# Update dimensions dict with converted value
						if alt_value is not None:
							dimensions[alt] = alt_value
						if alt_value is not None and alt_value > 0:
							has_alternative = True
							break
					except (ValueError, TypeError):
						continue
			if not has_alternative:
				missing_dims.append(dim)
	
	if missing_dims:
		result['valid'] = False
		result['errors'].append(f"Missing required dimensions: {missing_dims}")

	# Check for invalid dimension values (ensure they're numbers, not strings)
	for dim_name, value in dimensions.items():
		if value is None:
			result['valid'] = False
			result['errors'].append(f"Dimension {dim_name} is None")
		else:
			# Convert string values to float if needed
			try:
				if isinstance(value, str):
					value = float(value.strip()) if value.strip() else None
				elif not isinstance(value, (int, float)):
					value = float(value)
				
				if value is None:
					result['valid'] = False
					result['errors'].append(f"Dimension {dim_name} is None or empty")
				elif value <= 0:
					result['valid'] = False
					result['errors'].append(f"Dimension {dim_name} must be positive, got {value}")
				elif value < 0.001:
					result['warnings'].append(f"Dimension {dim_name} is very small ({value}), consider scaling")
				# Update the dimensions dict with the converted value
				dimensions[dim_name] = value
			except (ValueError, TypeError) as e:
				result['valid'] = False
				result['errors'].append(f"Dimension {dim_name} has invalid value '{value}' (type: {type(value).__name__}): {e}")

	# Geometry-specific validations (skip if any dimensions are None)
	# Check if all required dimensions are present and not None before doing comparisons
	try:
		if geometry_type in ['beam']:
			length = dimensions.get('length')
			width = dimensions.get('width')
			height = dimensions.get('height')
			
			# Only validate if all dimensions are valid numbers (not None)
			if length is not None and width is not None and height is not None:
				if length > 0 and width > 0 and height > 0:
					aspect_ratio = length / min(width, height)
					if aspect_ratio > 50:
						result['suggestions'].append("Very high aspect ratio detected - consider 1D approximation")
					elif aspect_ratio < 0.1:
						result['warnings'].append("Very low aspect ratio - may cause mesh quality issues")

		elif geometry_type in ['plate', 'membrane', 'square']:
			thickness = dimensions.get('thickness')
			length = dimensions.get('length')
			width = dimensions.get('width')
			# For square, if width is None, use length as default
			if width is None and length is not None:
				width = length
			
			# Only validate if all dimensions are valid numbers (not None)
			if thickness is not None and length is not None and width is not None:
				if thickness > 0 and length > 0 and width > 0:
					thin_ratio = min(length, width) / thickness
					if thin_ratio > 20:
						result['suggestions'].append("Very thin structure - consider 2D shell approximation")
	except (TypeError, ValueError):
		# Skip geometry-specific validations if there are type errors
		pass

	return result

def get_optimal_mesh_size(geometry_type: str, dimensions: Dict[str, float], quality_level: str = 'medium') -> float:
	"""Get optimal mesh size for geometry"""
	# Get base mesh size from quality level
	quality_settings = mesh_config.get_mesh_quality(quality_level)
	base_mesh_size = quality_settings.get('gmsh_mesh_size', 0.2)

	# Adjust based on geometry characteristics
	geometry_analysis = analyze_geometry(geometry_type, dimensions)
	
	# For thin structures, use smaller mesh size
	if 'thin_structure' in geometry_analysis.get('special_features', []):
		base_mesh_size *= 0.5
	elif 'thin_plate' in geometry_analysis.get('special_features', []):
		base_mesh_size *= 0.7

	# For symmetric geometries, can use slightly larger mesh
	if geometry_analysis.get('symmetry'):
		base_mesh_size *= 1.1

	return max(0.01, base_mesh_size)  # Minimum mesh size

def get_mesh_complexity_score(geometry_type: str, dimensions: Dict[str, float]) -> float:
	"""Get complexity score for mesh generation (0-1, higher = more complex)"""
	analysis = analyze_geometry(geometry_type, dimensions)
	
	# Base complexity by dimension
	mesh_dim = mesh_config.get_mesh_dimension(geometry_type)
	base_score = mesh_dim / 3.0  # 0.33 for 1D, 0.67 for 2D, 1.0 for 3D

	# Adjust for special features
	if 'thin_structure' in analysis.get('special_features', []):
		base_score += 0.2
	if 'thin_plate' in analysis.get('special_features', []):
		base_score += 0.1
	if 'long_cylinder' in analysis.get('special_features', []):
		base_score += 0.1

	# Adjust for aspect ratios
	aspect_ratios = analysis.get('aspect_ratios', {})
	for ratio_name, ratio_value in aspect_ratios.items():
		if ratio_value > 10:
			base_score += 0.1
		elif ratio_value < 0.1:
			base_score += 0.1

	return min(1.0, base_score)

def get_mesh_generation_strategy(geometry_type: str, dimensions: Dict[str, float]) -> Dict[str, Any]:
	"""Get comprehensive mesh generation strategy"""
	mesh_dim = mesh_config.get_mesh_dimension(geometry_type)
	analysis = analyze_geometry(geometry_type, dimensions)
	complexity_score = get_mesh_complexity_score(geometry_type, dimensions)
	optimal_mesh_size = get_optimal_mesh_size(geometry_type, dimensions)

	return {
		'mesh_dimension': mesh_dim,
		'complexity_score': complexity_score,
		'optimal_mesh_size': optimal_mesh_size,
		'geometry_analysis': analysis,
		'recommended_algorithm': 'Delaunay' if mesh_dim <= 2 else 'Delaunay',
		'quality_optimization': complexity_score > 0.7,
		'special_handling': analysis.get('special_features', []),
		'expected_elements': _estimate_element_count(mesh_dim, dimensions, optimal_mesh_size)
	}

def _estimate_element_count(mesh_dim: int, dimensions: Dict[str, float], mesh_size: float) -> int:
	"""Estimate number of elements for mesh"""
	if mesh_dim == 1:
		length = dimensions.get('length', 1.0)
		return max(10, int(length / mesh_size))
	elif mesh_dim == 2:
		length = dimensions.get('length', 1.0)
		width = dimensions.get('width', 1.0)
		area = length * width
		return max(50, int(area / (mesh_size ** 2)))
	else:  # mesh_dim == 3
		length = dimensions.get('length', 1.0)
		width = dimensions.get('width', 1.0)
		height = dimensions.get('height', 1.0)
		volume = length * width * height
		return max(200, int(volume / (mesh_size ** 3)))