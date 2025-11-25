"""
GMSH Mesh Generator
===================

Professional mesh generation using GMSH library exclusively.
Provides high-quality meshes with proper physical groups for field variables.
"""

import logging
import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class GMSHGenerator:
	"""Professional mesh generator using GMSH exclusively"""
	
	def __init__(self):
		self.gmsh_available = self._check_gmsh_availability()
		self.gmsh_model = None  # Store the GMSH model object
		self.gmsh_initialized = False
		self._geometry_boundaries_config = None  # Cache for geometry_boundaries.json
		if self.gmsh_available:
			logger.debug("GMSH mesh generator initialized")
		else:
			logger.error("GMSH not available - GMSH is required for mesh generation")
    
	def _check_gmsh_availability(self) -> bool:
		"""Check if GMSH is available"""
		try:
			import gmsh
			return True
		except (ImportError, OSError) as e:
			# ImportError: GMSH Python package not found
			# OSError: GMSH library found but shared libraries are missing (e.g., libXft.so.2)
			logger.warning(f"GMSH not available: {e}")
			return False

	def initialize_gmsh(self, mesh_size: float = 0.2, resolution: int = 50) -> bool:
		"""Initialize GMSH with mesh parameters"""
		if not self.gmsh_available:
			return False
		
		try:
			import gmsh
			
			# Initialize GMSH
			gmsh.initialize()
			gmsh.option.setNumber("General.Terminal", 0)
			gmsh.option.setNumber("General.NumThreads", 0)  # Use all available cores
			gmsh.option.setNumber("General.Verbosity", 0)
			
			# Configure GMSH mesh settings
			gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
			gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
			gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
			gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay 3D (reliable)
			gmsh.option.setNumber("Mesh.Optimize", 1)
			gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)  # Additional optimization
			gmsh.option.setNumber("Mesh.ElementOrder", 2)  # Second-order elements for curved surfaces
			gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)  # Keep curved geometry (not linear)
			gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 12)  # Better curved surface handling
			gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 6)  # Minimum elements for curves
			gmsh.option.setNumber("Mesh.HighOrderOptimize", 1)  # Optimize high-order elements
			
			self.gmsh_initialized = True
			self.gmsh_model = gmsh.model  # Store reference to GMSH model
			logger.debug("GMSH initialized successfully")
			return True
			
		except Exception as e:
			logger.error(f"Failed to initialize GMSH: {e}")
			return False

	def get_gmsh_model(self):
		"""Get the GMSH model object"""
		if not self.gmsh_initialized:
			logger.warning("GMSH not initialized. Call initialize_gmsh() first.")
			return None
		return self.gmsh_model

	def finalize_gmsh(self):
		"""Finalize GMSH and clean up resources"""
		if self.gmsh_initialized:
			try:
				import gmsh
				# Clear the GMSH model completely
				gmsh.clear()
				# Finalize GMSH
				gmsh.finalize()
				logger.debug("GMSH finalized and cleared successfully")
			except Exception as e:
				logger.error(f"Error finalizing GMSH: {e}")
			finally:
				self.gmsh_initialized = False
				self.gmsh_model = None

	def is_gmsh_initialized(self) -> bool:
		"""Check if GMSH is currently initialized"""
		return self.gmsh_initialized

	def force_clear_gmsh(self):
		"""Force clear GMSH model regardless of initialization state"""
		try:
			import gmsh
			if hasattr(gmsh, "isInitialized"):
				if gmsh.isInitialized():
					gmsh.clear()
					logger.debug("Force cleared GMSH model")
				else:
					logger.debug("GMSH is not initialized; nothing to clear")
			else:
				# Older versions may not expose isInitialized; attempt clear and ignore specific error
				try:
					gmsh.clear()
					logger.debug("Force cleared GMSH model")
				except Exception as clear_err:
					if "not been initialized" in str(clear_err):
						logger.debug("GMSH not initialized; clear skipped")
					else:
						raise
		except Exception as e:
			logger.warning(f"Could not force clear GMSH model: {e}")

	def cleanup_gmsh(self):
		"""Manually finalize GMSH when needed (for cleanup)"""
		try:
			import gmsh
			if hasattr(gmsh, "isInitialized") and not gmsh.isInitialized():
				logger.debug("GMSH already uninitialized; skipping cleanup clear")
			else:
				gmsh.clear()
			logger.debug("Cleared GMSH model during cleanup")
		except Exception as e:
			logger.warning(f"Could not clear GMSH model: {e}")
		
		if self.gmsh_initialized:
			self.finalize_gmsh()
			logger.info("GMSH cleaned up manually")

	def _load_geometry_boundaries_config(self) -> Dict[str, Any]:
		"""Load geometry boundaries configuration from JSON file"""
		if self._geometry_boundaries_config is not None:
			return self._geometry_boundaries_config
		
		try:
			# Find config file relative to project root
			# From mesh/utils/gmsh_generator.py, go up 2 levels to project root
			current_dir = Path(__file__).parent.parent.parent
			config_path = current_dir / "config" / "geometry_boundaries.json"
			
			if config_path.exists():
				with open(config_path, 'r') as f:
					self._geometry_boundaries_config = json.load(f)
					logger.debug(f"Loaded geometry boundaries config from {config_path}")
					return self._geometry_boundaries_config
			else:
				logger.warning(f"Geometry boundaries config not found at {config_path}, using empty config")
				self._geometry_boundaries_config = {}
				return {}
		except Exception as e:
			logger.error(f"Error loading geometry boundaries config: {e}")
			self._geometry_boundaries_config = {}
			return {}

	def _get_expected_boundaries(self, geometry_type: str) -> List[str]:
		"""Get expected boundary names for a geometry type from geometry_boundaries.json"""
		config = self._load_geometry_boundaries_config()
		geometries = config.get("geometries", {})
		
		# Search in all dimension categories
		for dim_key in ["1d", "2d", "3d"]:
			dim_geometries = geometries.get(dim_key, {})
			if geometry_type in dim_geometries:
				boundaries = dim_geometries[geometry_type].get("available_boundaries", [])
				logger.debug(f"Expected boundaries for {geometry_type}: {boundaries}")
				return boundaries
		
		logger.warning(f"No expected boundaries found for geometry type: {geometry_type}")
		return []

	def _identify_cylinder_boundaries(self, boundaries: List[Tuple[int, int]], dimensions: Dict[str, float]) -> Dict[int, str]:
		"""Identify cylinder boundaries (top, bottom, curved surface) using geometric predicates.
		Returns names matching JSON config exactly: "top", "bottom", "curved surface"."""
		import gmsh
		
		height = dimensions.get('height', dimensions.get('length', 1.0))
		surface_names = {}
		
		for boundary in boundaries:
			surface_tag = boundary[1]
			abs_tag = abs(surface_tag)
			
			try:
				# Get bounding box of the surface
				bbox = gmsh.model.getBoundingBox(2, abs_tag)
				z_min = bbox[2]
				z_max = bbox[5]
				z_range = abs(z_max - z_min)
				
				# Get surface center for radius check
				center = gmsh.model.occ.getCenterOfMass(2, abs_tag)
				x_center, y_center = center[0], center[1]
				radius_from_center = np.sqrt(x_center**2 + y_center**2)
				
				logger.debug(f"Surface {surface_tag}: z_min={z_min:.6f}, z_max={z_max:.6f}, z_range={z_range:.6f}, radius={radius_from_center:.6f}")
				
				# Flat surfaces have very small z-range
				# Bottom surface: z is close to 0
				if z_range < 1e-6 and abs(z_min) < 1e-6:
					surface_names[surface_tag] = "bottom"
				# Top surface: z is close to height
				elif z_range < 1e-6 and abs(z_max - height) < 1e-6:
					surface_names[surface_tag] = "top"
				# Curved surface: z varies significantly (spans full height)
				# Use exact JSON format: "curved surface" (with space, not underscore)
				elif z_range > height * 0.1:
					surface_names[surface_tag] = "curved surface"
				else:
					logger.debug(f"Could not identify surface {surface_tag}")
			except Exception as e:
				logger.warning(f"Failed to identify cylinder surface {surface_tag}: {e}")
		
		return surface_names

	def _identify_rectangular_boundaries_2d(self, boundaries: List[Tuple[int, int]], dimensions: Dict[str, float]) -> Dict[int, str]:
		"""Identify 2D rectangular boundaries (left, right, top, bottom) using geometric predicates"""
		import gmsh
		
		length = dimensions.get('length', 1.0)
		width = dimensions.get('width', length)
		
		surface_names = {}
		boundary_names = ["left", "bottom", "right", "top"]  # Standard GMSH order
		
		# Get coordinates of boundary curves to identify them
		for i, boundary in enumerate(boundaries):
			curve_tag = boundary[1]
			abs_tag = abs(curve_tag)
			
			try:
				# Get bounding box of the curve
				bbox = gmsh.model.getBoundingBox(1, abs_tag)
				x_min, y_min = bbox[0], bbox[1]
				x_max, y_max = bbox[3], bbox[4]
				
				# Get center point
				center = gmsh.model.occ.getCenterOfMass(1, abs_tag)
				x_center, y_center = center[0], center[1]
				
				logger.debug(f"Curve {curve_tag}: x_center={x_center:.6f}, y_center={y_center:.6f}, x_range={x_max-x_min:.6f}, y_range={y_max-y_min:.6f}")
				
				# Identify based on position and orientation
				tolerance = 1e-6
				
				# Left: x is close to 0, y varies
				if abs(x_center) < tolerance and (y_max - y_min) > tolerance:
					surface_names[curve_tag] = "left"
				# Right: x is close to length, y varies
				elif abs(x_center - length) < tolerance and (y_max - y_min) > tolerance:
					surface_names[curve_tag] = "right"
				# Bottom: y is close to 0, x varies
				elif abs(y_center) < tolerance and (x_max - x_min) > tolerance:
					surface_names[curve_tag] = "bottom"
				# Top: y is close to width, x varies
				elif abs(y_center - width) < tolerance and (x_max - x_min) > tolerance:
					surface_names[curve_tag] = "top"
				else:
					# Fallback to standard order if geometric identification fails
					if i < len(boundary_names):
						surface_names[curve_tag] = boundary_names[i]
			except Exception as e:
				logger.warning(f"Failed to identify rectangular boundary {curve_tag}: {e}")
				# Fallback to standard order
				if i < len(boundary_names):
					surface_names[curve_tag] = boundary_names[i]
		
		return surface_names

	def _identify_rectangular_boundaries_3d(self, boundaries: List[Tuple[int, int]], dimensions: Dict[str, float]) -> Dict[int, str]:
		"""Identify 3D rectangular boundaries (left, right, top, bottom, front, back) using geometric predicates"""
		import gmsh
		
		length = dimensions.get('length', 1.0)
		width = dimensions.get('width', length)
		height = dimensions.get('height', length)
		
		surface_names = {}
		
		for boundary in boundaries:
			surface_tag = boundary[1]
			abs_tag = abs(surface_tag)
			
			try:
				# Get bounding box of the surface
				bbox = gmsh.model.getBoundingBox(2, abs_tag)
				x_min, y_min, z_min = bbox[0], bbox[1], bbox[2]
				x_max, y_max, z_max = bbox[3], bbox[4], bbox[5]
				
				# Get center point
				center = gmsh.model.occ.getCenterOfMass(2, abs_tag)
				x_center, y_center, z_center = center[0], center[1], center[2]
				
				tolerance = 1e-6
				
				# Identify based on position
				# Left: x is close to 0
				if abs(x_center) < tolerance:
					surface_names[surface_tag] = "left"
				# Right: x is close to length
				elif abs(x_center - length) < tolerance:
					surface_names[surface_tag] = "right"
				# Front: y is close to 0
				elif abs(y_center) < tolerance:
					surface_names[surface_tag] = "front"
				# Back: y is close to width
				elif abs(y_center - width) < tolerance:
					surface_names[surface_tag] = "back"
				# Bottom: z is close to 0
				elif abs(z_center) < tolerance:
					surface_names[surface_tag] = "bottom"
				# Top: z is close to height
				elif abs(z_center - height) < tolerance:
					surface_names[surface_tag] = "top"
			except Exception as e:
				logger.warning(f"Failed to identify 3D rectangular boundary {surface_tag}: {e}")
		
		return surface_names
    
	def generate_mesh(self, geometry_type: str, dimensions: Dict[str, float], mesh_parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate mesh using GMSH with proper physical groups and boundary identification - GMSH model included by default"""
		if not self.gmsh_available:
			return self._create_error_mesh("GMSH not available")
		
		# Store geometry_type for use in extraction methods
		self.geometry_type = geometry_type
		
		# Show loading message
		logger.debug("Generating mesh with GMSH...")
		
		mesh_data = None
		try:
			# Always clear any previous model to avoid duplicate entities
			# First, finalize and clear any existing GMSH instance completely
			if self.gmsh_initialized:
				self.finalize_gmsh()
			else:
				self.force_clear_gmsh()
			
			# Normalize dimensions: convert strings to numbers (UI often sends strings)
			# This ensures all dimension values are numeric before any calculations
			normalized_dimensions = {}
			for key, value in dimensions.items():
				if value is None:
					continue  # Skip None values
				try:
					if isinstance(value, str):
						value_str = value.strip()
						if value_str:
							normalized_dimensions[key] = float(value_str)
					elif isinstance(value, (int, float)):
						normalized_dimensions[key] = float(value)
					else:
						normalized_dimensions[key] = float(value)
				except (ValueError, TypeError) as e:
					logger.warning(f"Could not normalize dimension '{key}' with value '{value}': {e}")
					# Use default value for this dimension if conversion fails
					normalized_dimensions[key] = 1.0
			
			# Use normalized dimensions (or original if normalization failed completely)
			dimensions = normalized_dimensions if normalized_dimensions else dimensions
			
			# Set mesh parameters
			mesh_size = mesh_parameters.get('mesh_size', 0.2)
			resolution = mesh_parameters.get('resolution', 50)
			
			# Always initialize GMSH fresh (we finalized it above if it was initialized)
			if not self.initialize_gmsh(mesh_size, resolution):
				return self._create_error_mesh("Failed to initialize GMSH")
			
			# Generate geometry based on type with boundary identification
			if geometry_type in ['line', 'rod', 'bar']:
				mesh_data = self._generate_line_mesh_gmsh(dimensions, mesh_size, resolution)
			elif geometry_type in ['plate', 'membrane', 'disc', 'rectangle', 'square']:
				mesh_data = self._generate_2d_mesh_gmsh(geometry_type, dimensions, mesh_size, resolution)
			elif geometry_type in ['cube', 'box', 'beam', 'cylinder', 'sphere', 'solid', 'rectangular']:
				mesh_data = self._generate_3d_mesh_gmsh(geometry_type, dimensions, mesh_size, resolution)
			else:
				return self._create_error_mesh(f"Unsupported geometry type: {geometry_type}")
			
			# Export mesh to .msh file for downstream consumption
			msh_file = self._export_msh_file(geometry_type)
			if msh_file:
				mesh_data['msh_file'] = msh_file
				logger.info(f"Exported GMSH mesh to {msh_file}")
			else:
				logger.error("Failed to export GMSH mesh to .msh file")
			
			mesh_data['gmsh_model_was_available'] = bool(msh_file)
			
			return mesh_data
			
		except Exception as e:
			logger.error(f"GMSH mesh generation error: {e}")
			return self._create_error_mesh(str(e))
		finally:
			# Always finalize the current model so subsequent meshes start clean
			self.cleanup_gmsh()

	def _export_msh_file(self, geometry_type: str) -> Optional[str]:
		"""Export the current GMSH model to a temporary .msh file"""
		try:
			import gmsh
			fd, path = tempfile.mkstemp(prefix=f"{geometry_type}_", suffix=".msh")
			os.close(fd)
			gmsh.write(path)
			return path
		except Exception as e:
			logger.error(f"Failed to export mesh to .msh file: {e}")
			return None

	def _generate_line_mesh_gmsh(self, dimensions: Dict[str, float], mesh_size: float, resolution: int) -> Dict[str, Any]:
		"""Generate 1D line mesh using GMSH with enhanced boundary identification"""
		try:
			import gmsh

			length = dimensions.get('length', 1.0)

			# Create line geometry
			p1 = gmsh.model.occ.addPoint(0, 0, 0, mesh_size)
			p2 = gmsh.model.occ.addPoint(length, 0, 0, mesh_size)
			line = gmsh.model.occ.addLine(p1, p2)
			gmsh.model.occ.synchronize()
			
			# Create physical groups with exact JSON names
			# Left boundary (x=0) - use "left" to match JSON config exactly
			pg_tag = gmsh.model.addPhysicalGroup(0, [p1], 1)
			gmsh.model.setPhysicalName(0, pg_tag, "left")
			# Right boundary (x=length) - use "right" to match JSON config exactly
			pg_tag = gmsh.model.addPhysicalGroup(0, [p2], 2)
			gmsh.model.setPhysicalName(0, pg_tag, "right")
			# Line domain
			pg_tag = gmsh.model.addPhysicalGroup(1, [line], 3)
			gmsh.model.setPhysicalName(1, pg_tag, "domain")
			
			# Generate mesh
			gmsh.model.mesh.generate(1)
			
			# Extract mesh data with boundary information
			vertices, faces, cells, physical_groups = self._extract_mesh_data_gmsh()
			
			# Add boundary identification metadata
			boundary_info = self._identify_boundaries_1d(vertices, physical_groups)
			
			# Count 1D elements (line and line_2nd)
			num_cells = sum(len(cell_list) for cell_type, cell_list in cells.items() if cell_type in ['line', 'line_2nd'])
			cell_types_1d = [ct for ct in cells.keys() if ct in ['line', 'line_2nd']]
			
			return {
				'vertices': vertices,
				'faces': faces,
				'cells': cells,
				'physical_groups': physical_groups,
				'boundary_info': boundary_info,
				'type': '1d_mesh',
				'geometry_type': 'line',
				'dimensions': dimensions,
				'mesh_dimension': 1,
				'mesh_stats': {
					'num_vertices': len(vertices),
					'num_faces': len(faces),
					'num_cells': num_cells,
					'cell_types': cell_types_1d
				},
				'success': True
			}

		except Exception as e:
			logger.error(f"Error generating 1D mesh: {e}")
			return self._create_error_mesh(f"1D mesh generation failed: {e}")

	def _generate_2d_mesh_gmsh(self, geometry_type: str, dimensions: Dict[str, float], mesh_size: float, resolution: int) -> Dict[str, Any]:
		"""Generate 2D mesh using GMSH with enhanced boundary identification"""
		try:
			import gmsh
			
			if geometry_type in ['plate', 'membrane', 'rectangle', 'square']:
				# Rectangular geometry
				length = dimensions.get('length', 1.0)
				width = dimensions.get('width', length)  # For square, width = length

				# Create rectangle
				rect = gmsh.model.occ.addRectangle(0, 0, 0, length, width)
				gmsh.model.occ.synchronize()

				# Get expected boundaries from config
				expected_boundaries = self._get_expected_boundaries(geometry_type)
				
				# Create physical groups with automatic boundary identification
				boundaries = gmsh.model.getBoundary([(2, rect)])
				identified_names = self._identify_rectangular_boundaries_2d(boundaries, dimensions)
				
				physical_group_id = 1
				for boundary in boundaries:
					curve_tag = boundary[1]
					abs_tag = abs(curve_tag)
					
					# Use identified name, or fallback to expected_boundaries order
					boundary_name = identified_names.get(curve_tag, "")
					if not boundary_name and physical_group_id <= len(expected_boundaries):
						boundary_name = expected_boundaries[physical_group_id - 1]
					
					# Create physical group with proper name
					if boundary_name:
						pg_tag = gmsh.model.addPhysicalGroup(1, [abs_tag], physical_group_id)
						gmsh.model.setPhysicalName(1, pg_tag, boundary_name)
						logger.debug(f"Created physical group {pg_tag} for curve {curve_tag} (abs: {abs_tag}) with name '{boundary_name}'")
					else:
						gmsh.model.addPhysicalGroup(1, [abs_tag], physical_group_id)
						logger.warning(f"Created physical group {physical_group_id} for curve {curve_tag} without name")
					physical_group_id += 1
				
				# Domain
				pg_tag = gmsh.model.addPhysicalGroup(2, [rect], physical_group_id)
				gmsh.model.setPhysicalName(2, pg_tag, "domain")
				
				# CRITICAL: Synchronize after setting all physical names to ensure they're registered
				gmsh.model.occ.synchronize()
				
			elif geometry_type == 'disc':
				# Circular geometry
				radius = dimensions.get('radius', 1.0)
				logger.debug(f"Creating disc with radius: {radius}")

				# Create disc
				disc = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
				gmsh.model.occ.synchronize()
				logger.debug(f"Created disc geometry: {disc}")

				# Get expected boundaries from config (should be ["circumference", "center"])
				expected_boundaries = self._get_expected_boundaries('disc')
				
				physical_group_id = 1

				# Create physical groups - for disc, all boundary curves should be in one group
				boundaries = gmsh.model.getBoundary([(2, disc)])
				logger.debug(f"Disc boundaries: {boundaries}")
				if boundaries:
					# Group all boundary curves into a single physical group with name "circumference"
					boundary_curves = [abs(boundary[1]) for boundary in boundaries]
					name = "circumference" if "circumference" in expected_boundaries else ""
					if name:
						pg_tag = gmsh.model.addPhysicalGroup(1, boundary_curves, physical_group_id)
						# Set the name explicitly using setPhysicalName (more reliable than name parameter)
						gmsh.model.setPhysicalName(1, pg_tag, name)
						logger.debug(f"Created physical group {pg_tag} for disc boundary curves with name '{name}'")
						physical_group_id += 1
					else:
						gmsh.model.addPhysicalGroup(1, boundary_curves, physical_group_id)
						logger.warning(f"Created physical group {physical_group_id} for disc boundary curves without name")
						physical_group_id += 1
				
				# Domain
				pg_tag = gmsh.model.addPhysicalGroup(2, [disc], physical_group_id)
				gmsh.model.setPhysicalName(2, pg_tag, "domain")
				logger.debug("Added physical groups for disc")
				physical_group_id += 1  # CRITICAL: Increment after domain creation
				
				# CRITICAL: Synchronize after setting all physical names to ensure they're registered
				gmsh.model.occ.synchronize()

			# Generate mesh
			logger.debug("Generating 2D mesh...")
			gmsh.model.mesh.generate(2)
			logger.debug("2D mesh generation completed")
			
			# Create center physical group AFTER meshing by finding closest element (triangle)
			if "center" in expected_boundaries:
				try:
					# Get all 2D elements (triangles) and their node coordinates
					element_types, element_tags, node_tags = gmsh.model.mesh.getElements(2, -1)
					if element_types is not None and len(element_types) > 0:
						# Find triangle element type (type 2 for P1, type 9 for P2)
						# Prefer P2 (type 9) if available, otherwise use P1 (type 2)
						triangle_type = None
						nodes_per_triangle = None
						for elem_type in element_types:
							if elem_type == 9:  # 6-node triangle (P2) - prefer this
								triangle_type = elem_type
								nodes_per_triangle = 6
								break
						# If P2 not found, check for P1
						if triangle_type is None:
							for elem_type in element_types:
								if elem_type == 2:  # 3-node triangle (P1)
									triangle_type = elem_type
									nodes_per_triangle = 3
									break
						
						if triangle_type is not None:
							# Get triangle elements
							triangle_idx = list(element_types).index(triangle_type)
							triangle_tags = element_tags[triangle_idx]
							triangle_node_tags = node_tags[triangle_idx]
							
							# Get all node coordinates
							all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes(-1, -1)
							node_coord_dict = {}
							num_nodes = len(all_node_tags)
							coords_flat = all_node_coords.reshape((num_nodes, 3))
							for i, tag in enumerate(all_node_tags):
								node_coord_dict[int(tag)] = coords_flat[i]
							
							# Find triangle with centroid closest to (0, 0, 0)
							center_target = np.array([0.0, 0.0, 0.0])
							min_distance = float('inf')
							closest_element_tag = None
							closest_centroid = None
							
							# Process triangles
							num_triangles = len(triangle_tags)
							for i in range(num_triangles):
								# Get node tags for this triangle
								tri_node_tags = triangle_node_tags[i*nodes_per_triangle:(i+1)*nodes_per_triangle]
								# For centroid, use only corner nodes (first 3 nodes for both P1 and P2)
								corner_nodes = tri_node_tags[:3]
								# Get coordinates of triangle corner vertices
								tri_coords = np.array([node_coord_dict[int(tag)] for tag in corner_nodes])
								# Calculate centroid
								centroid = np.mean(tri_coords, axis=0)
								# Calculate distance to center
								distance = np.linalg.norm(centroid - center_target)
								if distance < min_distance:
									min_distance = distance
									closest_element_tag = int(triangle_tags[i])
									closest_centroid = centroid
									closest_tri_nodes = tri_node_tags
							
							if closest_element_tag is not None:
								logger.info(f"Found closest triangle to center: element_tag={closest_element_tag}, centroid={closest_centroid}, distance={min_distance:.6f}")
								
								# Tag the element by storing its node tags
								# GMSH cannot create dim=0 physical groups with mesh node tags directly
								# We'll add this manually to the physical_groups dict after extraction
								node_tag_list = [int(tag) for tag in closest_tri_nodes]
								logger.info(f"Identified center element {closest_element_tag} with {len(node_tag_list)} nodes. Storing for manual physical group creation.")
								# Store for manual addition after _extract_physical_groups_gmsh
								self._center_node_tags = node_tag_list
								self._center_element_tag = closest_element_tag
							else:
								logger.warning("No triangle elements found - cannot create center physical group")
						else:
							logger.warning("No triangle element type found in mesh")
					else:
						logger.warning("No 2D elements found - cannot create center physical group")
				except Exception as e:
					logger.error(f"Failed to create center element physical group for disc: {e}")
					import traceback
					logger.error(f"Traceback: {traceback.format_exc()}")
			
			# Verify physical names are set (for debugging)
			try:
				for dim in [0, 1, 2]:
					names = gmsh.model.getPhysicalNames(dim)
					if names:
						logger.debug(f"Physical names for dim={dim} after mesh generation: {names}")
			except Exception as e:
				logger.debug(f"Could not verify physical names: {e}")

			# Extract mesh data
			vertices, faces, cells, physical_groups = self._extract_mesh_data_gmsh()
			logger.debug(f"Extracted mesh data: {len(vertices)} vertices, {len(faces)} faces, cells: {list(cells.keys())}")
			
			# Add boundary identification metadata
			boundary_info = self._identify_boundaries_2d(vertices, physical_groups, geometry_type)
			
			# Count 2D elements only (exclude 1D boundary elements for reporting)
			num_2d_cells = 0
			cell_types_2d = []
			for cell_type, cell_list in cells.items():
				if cell_type not in ['line', 'line_2nd']:  # Exclude 1D boundary elements
					num_2d_cells += len(cell_list)
					cell_types_2d.append(cell_type)
			
			return {
				'vertices': vertices,
				'faces': faces,
				'cells': cells,
				'physical_groups': physical_groups,
				'boundary_info': boundary_info,
				'type': '2d_mesh',
				'geometry_type': geometry_type,
				'dimensions': dimensions,
				'mesh_dimension': 2,
				'mesh_stats': {
					'num_vertices': len(vertices),
					'num_faces': len(faces),
					'num_cells': num_2d_cells,
					'cell_types': cell_types_2d  # Only 2D cell types (triangles, quads)
				},
				'success': True
			}

		except Exception as e:
			logger.error(f"Error generating 2D mesh: {e}")
			return self._create_error_mesh(f"2D mesh generation failed: {e}")

	def _generate_3d_mesh_gmsh(self, geometry_type: str, dimensions: Dict[str, float], mesh_size: float, resolution: int) -> Dict[str, Any]:
		"""Generate 3D mesh using GMSH with enhanced boundary identification"""
		try:
			import gmsh
			
			if geometry_type in ['cube', 'box', 'beam', 'solid', 'rectangular']:
				# Box geometry
				length = dimensions.get('length', 1.0)
				width = dimensions.get('width', 1.0)
				height = dimensions.get('height', 1.0)

				if geometry_type == 'cube':
					width = height = length  # Cube has equal dimensions

				# Create box
				box = gmsh.model.occ.addBox(0, 0, 0, length, width, height)
				gmsh.model.occ.synchronize()
				
				# Get expected boundaries from config
				expected_boundaries = self._get_expected_boundaries(geometry_type)
				
				# Create physical groups with automatic boundary identification
				boundaries = gmsh.model.getBoundary([(3, box)])
				identified_names = self._identify_rectangular_boundaries_3d(boundaries, dimensions)
				
				physical_group_id = 1
				for boundary in boundaries:
					surface_tag = boundary[1]
					abs_tag = abs(surface_tag)
					
					# Use identified name, or fallback to expected_boundaries order
					boundary_name = identified_names.get(surface_tag, "")
					if not boundary_name and physical_group_id <= len(expected_boundaries):
						boundary_name = expected_boundaries[physical_group_id - 1]
					
					# Create physical group with proper name
					if boundary_name:
						pg_tag = gmsh.model.addPhysicalGroup(2, [abs_tag], physical_group_id)
						gmsh.model.setPhysicalName(2, pg_tag, boundary_name)
						logger.debug(f"Created physical group {pg_tag} for surface {surface_tag} (abs: {abs_tag}) with name '{boundary_name}'")
					else:
						gmsh.model.addPhysicalGroup(2, [abs_tag], physical_group_id)
						logger.warning(f"Created physical group {physical_group_id} for surface {surface_tag} without name")
					physical_group_id += 1
				
				# Domain
				pg_tag = gmsh.model.addPhysicalGroup(3, [box], physical_group_id)
				gmsh.model.setPhysicalName(3, pg_tag, "domain")
				
				# CRITICAL: Synchronize after setting all physical names to ensure they're registered
				gmsh.model.occ.synchronize()
				
			elif geometry_type == 'cylinder':
				# Cylindrical geometry
				radius = dimensions.get('radius', 1.0)
				# Handle both 'length' and 'height' for cylinder (height is preferred)
				height = dimensions.get('height', dimensions.get('length', 1.0))

				# Create cylinder
				cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, radius)
				gmsh.model.occ.synchronize()

				# Set moderate mesh size constraints for curved surfaces (higher-order elements need less refinement)
				# Get all curves (edges) of the cylinder
				curves = gmsh.model.getBoundary([(3, cylinder)], False, False, True)
				for curve in curves:
					# Set moderate mesh size for curved boundaries (higher-order elements handle curvature better)
					curve_mesh_size = mesh_size * 0.5  # Moderate refinement on curves
					gmsh.model.mesh.setSize([curve], curve_mesh_size)
				
				# Also set mesh size on the curved surfaces (not just edges)
				surfaces = gmsh.model.getBoundary([(3, cylinder)], False, True, False)
				for surface in surfaces:
					# Set moderate mesh size on curved surfaces
					surface_mesh_size = mesh_size * 0.7  # Moderate refinement on surfaces
					gmsh.model.mesh.setSize([surface], surface_mesh_size)

				# Get expected boundaries from config
				expected_boundaries = self._get_expected_boundaries('cylinder')
				
				# Create physical groups with automatic boundary identification
				boundaries = gmsh.model.getBoundary([(3, cylinder)])
				logger.debug(f"Cylinder boundaries: {boundaries}")
				
				# Identify boundaries using geometric predicates
				identified_names = self._identify_cylinder_boundaries(boundaries, dimensions)
				logger.info(f"Cylinder boundary identification results: {identified_names}")
				
				# ALWAYS create physical groups for ALL boundaries
				# Use identified names - they should match expected_boundaries from JSON config exactly
				physical_group_id = 1
				for boundary in boundaries:
					surface_tag = boundary[1]
					abs_tag = abs(surface_tag)
					
					# Use identified name - should already match JSON format exactly
					name = identified_names.get(surface_tag, "")
					
					# Verify name is in expected boundaries (should always match if identification worked)
					if name and name in expected_boundaries:
						pg_tag = gmsh.model.addPhysicalGroup(2, [abs_tag], physical_group_id)
						gmsh.model.setPhysicalName(2, pg_tag, name)
					else:
						# Fallback: use expected_boundaries in order if identification failed
						if expected_boundaries and physical_group_id <= len(expected_boundaries):
							name = expected_boundaries[physical_group_id - 1]
							pg_tag = gmsh.model.addPhysicalGroup(2, [abs_tag], physical_group_id)
							gmsh.model.setPhysicalName(2, pg_tag, name)
							logger.warning(f"Boundary identification failed for surface {surface_tag}, using expected name '{name}'")
						else:
							gmsh.model.addPhysicalGroup(2, [abs_tag], physical_group_id)
							logger.warning(f"Created physical group {physical_group_id} for surface {surface_tag} without name")
					# Always increment physical_group_id after creating a group
					physical_group_id += 1
				
				# Domain
				pg_tag = gmsh.model.addPhysicalGroup(3, [cylinder], physical_group_id)
				gmsh.model.setPhysicalName(3, pg_tag, "domain")
				
				# CRITICAL: Synchronize after setting all physical names to ensure they're registered
				gmsh.model.occ.synchronize()
				
			elif geometry_type == 'sphere':
				# Spherical geometry
				radius = dimensions.get('radius', 1.0)

				# Create sphere
				sphere = gmsh.model.occ.addSphere(0, 0, 0, radius)
				gmsh.model.occ.synchronize()

				# Set moderate mesh size constraints for curved surfaces (higher-order elements need less refinement)
				# Get all curves (edges) of the sphere
				curves = gmsh.model.getBoundary([(3, sphere)], False, False, True)
				for curve in curves:
					# Set moderate mesh size for curved boundaries (higher-order elements handle curvature better)
					curve_mesh_size = mesh_size * 0.5  # Moderate refinement on curves
					gmsh.model.mesh.setSize([curve], curve_mesh_size)
				
				# Also set mesh size on the curved surfaces (not just edges)
				surfaces = gmsh.model.getBoundary([(3, sphere)], False, True, False)
				for surface in surfaces:
					# Set moderate mesh size on curved surfaces
					surface_mesh_size = mesh_size * 0.7  # Moderate refinement on surfaces
					gmsh.model.mesh.setSize([surface], surface_mesh_size)

				# Get expected boundaries from config (should be ["center", "surface"])
				expected_boundaries = self._get_expected_boundaries('sphere')
				
				physical_group_id = 1
				
				# Create physical groups with proper names for surface boundaries
				boundaries = gmsh.model.getBoundary([(3, sphere)])
				
				# For sphere, all boundaries are the surface (there's only one)
				for boundary in boundaries:
					surface_tag = boundary[1]
					abs_tag = abs(surface_tag)
					
					# Use "surface" as the name (from expected_boundaries)
					name = "surface" if "surface" in expected_boundaries else ""
					
					if name:
						pg_tag = gmsh.model.addPhysicalGroup(2, [abs_tag], physical_group_id)
						gmsh.model.setPhysicalName(2, pg_tag, name)
						logger.debug(f"Created physical group {pg_tag} for surface {surface_tag} (abs: {abs_tag}) with name '{name}'")
					else:
						gmsh.model.addPhysicalGroup(2, [abs_tag], physical_group_id)
						logger.warning(f"Created physical group {physical_group_id} for surface {surface_tag} without name")
					physical_group_id += 1
				
				# Domain
				pg_tag = gmsh.model.addPhysicalGroup(3, [sphere], physical_group_id)
				gmsh.model.setPhysicalName(3, pg_tag, "domain")
				physical_group_id += 1  # CRITICAL: Increment after domain creation
				
				# CRITICAL: Synchronize after setting all physical names to ensure they're registered
				gmsh.model.occ.synchronize()

			# Generate mesh
			logger.debug("Generating 3D mesh...")
			gmsh.model.mesh.generate(3)
			logger.debug("3D mesh generation completed")
			
			# Create center physical group AFTER meshing by finding closest element (tetrahedron)
			if "center" in expected_boundaries:
				try:
					# Get all 3D elements (tetrahedra) and their node coordinates
					element_types, element_tags, node_tags = gmsh.model.mesh.getElements(3, -1)
					if element_types is not None and len(element_types) > 0:
						# Find tetrahedron element type (type 4 for P1, type 11 for P2)
						# Prefer P2 (type 11) if available, otherwise use P1 (type 4)
						tet_type = None
						nodes_per_tet = None
						for elem_type in element_types:
							if elem_type == 11:  # 10-node tetrahedron (P2) - prefer this
								tet_type = elem_type
								nodes_per_tet = 10
								break
						# If P2 not found, check for P1
						if tet_type is None:
							for elem_type in element_types:
								if elem_type == 4:  # 4-node tetrahedron (P1)
									tet_type = elem_type
									nodes_per_tet = 4
									break
						
						if tet_type is not None:
							# Get tetrahedron elements
							tet_idx = list(element_types).index(tet_type)
							tet_tags = element_tags[tet_idx]
							tet_node_tags = node_tags[tet_idx]
							
							# Get all node coordinates
							all_node_tags, all_node_coords, _ = gmsh.model.mesh.getNodes(-1, -1)
							node_coord_dict = {}
							num_nodes = len(all_node_tags)
							coords_flat = all_node_coords.reshape((num_nodes, 3))
							for i, tag in enumerate(all_node_tags):
								node_coord_dict[int(tag)] = coords_flat[i]
							
							# Find tetrahedron with centroid closest to (0, 0, 0)
							center_target = np.array([0.0, 0.0, 0.0])
							min_distance = float('inf')
							closest_element_tag = None
							closest_centroid = None
							
							# Process tetrahedra
							num_tets = len(tet_tags)
							for i in range(num_tets):
								# Get node tags for this tetrahedron
								tet_node_tags_list = tet_node_tags[i*nodes_per_tet:(i+1)*nodes_per_tet]
								# For centroid, use only corner nodes (first 4 nodes for both P1 and P2)
								corner_nodes = tet_node_tags_list[:4]
								# Get coordinates of tetrahedron corner vertices
								tet_coords = np.array([node_coord_dict[int(tag)] for tag in corner_nodes])
								# Calculate centroid
								centroid = np.mean(tet_coords, axis=0)
								# Calculate distance to center
								distance = np.linalg.norm(centroid - center_target)
								if distance < min_distance:
									min_distance = distance
									closest_element_tag = int(tet_tags[i])
									closest_centroid = centroid
									closest_tet_nodes = tet_node_tags_list
							
							if closest_element_tag is not None:
								logger.info(f"Found closest tetrahedron to center: element_tag={closest_element_tag}, centroid={closest_centroid}, distance={min_distance:.6f}")
								
								# Tag the element by storing its node tags
								# GMSH cannot create dim=0 physical groups with mesh node tags directly
								# We'll add this manually to the physical_groups dict after extraction
								node_tag_list = [int(tag) for tag in closest_tet_nodes]
								logger.info(f"Identified center element {closest_element_tag} with {len(node_tag_list)} nodes. Storing for manual physical group creation.")
								# Store for manual addition after _extract_physical_groups_gmsh
								self._center_node_tags = node_tag_list
								self._center_element_tag = closest_element_tag
							else:
								logger.warning("No tetrahedron elements found - cannot create center physical group")
						else:
							logger.warning("No tetrahedron element type found in mesh")
					else:
						logger.warning("No 3D elements found - cannot create center physical group")
				except Exception as e:
					logger.error(f"Failed to create center element physical group for sphere: {e}")
					import traceback
					logger.error(f"Traceback: {traceback.format_exc()}")
			
			# DEBUG: Count facets and DOFs in GMSH for each physical group after mesh generation
			
			# Verify physical names are set (for debugging)
			try:
				for dim in [0, 1, 2, 3]:
					names = gmsh.model.getPhysicalNames(dim)
					if names:
						logger.debug(f"Physical names for dim={dim} after mesh generation: {names}")
			except Exception as e:
				logger.debug(f"Could not verify physical names: {e}")
			
			# Check if mesh was actually generated
			try:
				node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
				logger.debug(f"Generated {len(node_tags)} nodes")
			except Exception as e:
				logger.error(f"Failed to get nodes after mesh generation: {e}")
			
			# Extract mesh data
			vertices, faces, cells, physical_groups = self._extract_mesh_data_gmsh()
			
			logger.debug(f"Returning mesh data: {len(vertices)} vertices, {len(faces)} faces, {len(cells)} cell types")
			logger.debug(f"Cell types: {list(cells.keys())}")
			if 'tetrahedron' in cells:
				logger.debug(f"Tetrahedrons: {len(cells['tetrahedron'])}")
			if 'triangle' in cells:
				logger.debug(f"Triangles: {len(cells['triangle'])}")
			if 'quad' in cells:
				logger.debug(f"Quads: {len(cells['quad'])}")
			
			# Add boundary identification metadata
			boundary_info = self._identify_boundaries_3d(vertices, physical_groups, geometry_type)
			
			# Count 3D elements only (exclude 2D surface and 1D edge boundary elements for reporting)
			num_3d_cells = 0
			cell_types_3d = []
			for cell_type, cell_list in cells.items():
				if cell_type not in ['triangle', 'triangle_2nd', 'quad', 'line', 'line_2nd']:  # Exclude boundary elements
					num_3d_cells += len(cell_list)
					cell_types_3d.append(cell_type)
			
			return {
				'vertices': vertices,
				'faces': faces,
				'cells': cells,
				'physical_groups': physical_groups,
				'boundary_info': boundary_info,
				'type': '3d_mesh',
				'geometry_type': geometry_type,
				'dimensions': dimensions,
				'mesh_dimension': 3,
				'mesh_stats': {
					'num_vertices': len(vertices),
					'num_faces': len(faces),
					'num_cells': num_3d_cells,
					'cell_types': cell_types_3d  # Only 3D cell types (tetrahedrons, hexahedrons)
				},
				'success': True
			}

		except Exception as e:
			logger.error(f"Error generating 3D mesh: {e}")
			return self._create_error_mesh(f"3D mesh generation failed: {e}")

	def _extract_mesh_data_gmsh(self) -> Tuple[List[List[float]], List[List[int]], Dict[str, List[List[int]]], Dict[str, Any]]:
		"""Extract mesh data from GMSH"""
		try:
			import gmsh
			
			# Get vertices and build tag->index mapping to ensure correct connectivity
			node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
			vertices = []
			tag_to_index = {}
			for vi in range(0, len(node_coords), 3):
				vertex_index = vi // 3
				vertices.append([node_coords[vi], node_coords[vi+1], node_coords[vi+2]])
				tag_to_index[int(node_tags[vertex_index])] = vertex_index

			# Initialize cells dictionary
			cells = {}
			
			# Get faces (2D elements)
			faces = []
			try:
				elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2)
				logger.debug(f"2D element types found: {elem_types}")
				for elem_type, elem_node_tags in zip(elem_types, elem_node_tags):
					logger.debug(f"Processing element type {elem_type} with {len(elem_node_tags)} node tags")
					if elem_type == 2:  # First-order triangle
						cells['triangle'] = []
						for i in range(0, len(elem_node_tags), 3):
							# Map gmsh node tags to contiguous vertex indices
							triangle = [
								tag_to_index[int(elem_node_tags[i])],
								tag_to_index[int(elem_node_tags[i+1])],
								tag_to_index[int(elem_node_tags[i+2])]
							]
							faces.append(triangle)
							cells['triangle'].append(triangle)
						logger.debug(f"Created {len(cells['triangle'])} first-order triangles")
					elif elem_type == 9:  # Second-order triangle (6 nodes: 3 corners + 3 mid-edges)
						cells['triangle_2nd'] = []
						for i in range(0, len(elem_node_tags), 6):
							# Use all 6 nodes for curved surface representation
							triangle_2nd = [
								tag_to_index[int(elem_node_tags[i])],      # Corner 1
								tag_to_index[int(elem_node_tags[i+1])],    # Corner 2
								tag_to_index[int(elem_node_tags[i+2])],    # Corner 3
								tag_to_index[int(elem_node_tags[i+3])],    # Mid-edge 1
								tag_to_index[int(elem_node_tags[i+4])],    # Mid-edge 2
								tag_to_index[int(elem_node_tags[i+5])]     # Mid-edge 3
							]
							# For faces, use corner nodes only (for compatibility)
							triangle_corners = triangle_2nd[:3]
							faces.append(triangle_corners)
							cells['triangle_2nd'].append(triangle_2nd)
						logger.debug(f"Created {len(cells['triangle_2nd'])} second-order triangles (full curved elements)")
					elif elem_type == 3:  # First-order quad
						cells['quad'] = []
						for i in range(0, len(elem_node_tags), 4):
							# Map gmsh node tags to contiguous vertex indices
							quad = [
								tag_to_index[int(elem_node_tags[i])],
								tag_to_index[int(elem_node_tags[i+1])],
								tag_to_index[int(elem_node_tags[i+2])],
								tag_to_index[int(elem_node_tags[i+3])]
							]
							faces.append(quad)
							cells['quad'].append(quad)
						logger.debug(f"Created {len(cells['quad'])} first-order quads")
					elif elem_type == 10:  # Second-order quad (9 nodes: 4 corners + 4 mid-edges + 1 center)
						cells['quad'] = []
						for i in range(0, len(elem_node_tags), 9):
							# Extract only corner nodes (first 4) for surface representation
							quad = [
								tag_to_index[int(elem_node_tags[i])],
								tag_to_index[int(elem_node_tags[i+1])],
								tag_to_index[int(elem_node_tags[i+2])],
								tag_to_index[int(elem_node_tags[i+3])]
							]
							faces.append(quad)
							cells['quad'].append(quad)
						logger.debug(f"Created {len(cells['quad'])} second-order quads (using corner nodes)")
			except Exception as e:
				logger.error(f"Error extracting 2D elements: {e}")
				pass  # No 2D elements

			# Get cells (3D elements)
			try:
				elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
				logger.debug(f"3D element types found: {elem_types}")
				for elem_type, elem_node_tags in zip(elem_types, elem_node_tags):
					logger.debug(f"Processing 3D element type {elem_type} with {len(elem_node_tags)} node tags")
					if elem_type == 4:  # First-order tetrahedron
						cells['tetrahedron'] = []
						for i in range(0, len(elem_node_tags), 4):
							cells['tetrahedron'].append([
								tag_to_index[int(elem_node_tags[i])],
								tag_to_index[int(elem_node_tags[i+1])],
								tag_to_index[int(elem_node_tags[i+2])],
								tag_to_index[int(elem_node_tags[i+3])]
							])
						logger.debug(f"Created {len(cells['tetrahedron'])} first-order tetrahedrons")
					elif elem_type == 11:  # Second-order tetrahedron (10 nodes: 4 corners + 6 mid-edges)
						cells['tetrahedron_2nd'] = []
						for i in range(0, len(elem_node_tags), 10):
							# Use all 10 nodes for curved volume representation
							tetrahedron_2nd = [
								tag_to_index[int(elem_node_tags[i])],      # Corner 1
								tag_to_index[int(elem_node_tags[i+1])],    # Corner 2
								tag_to_index[int(elem_node_tags[i+2])],    # Corner 3
								tag_to_index[int(elem_node_tags[i+3])],    # Corner 4
								tag_to_index[int(elem_node_tags[i+4])],    # Mid-edge 1
								tag_to_index[int(elem_node_tags[i+5])],    # Mid-edge 2
								tag_to_index[int(elem_node_tags[i+6])],    # Mid-edge 3
								tag_to_index[int(elem_node_tags[i+7])],    # Mid-edge 4
								tag_to_index[int(elem_node_tags[i+8])],    # Mid-edge 5
								tag_to_index[int(elem_node_tags[i+9])]     # Mid-edge 6
							]
							cells['tetrahedron_2nd'].append(tetrahedron_2nd)
						logger.debug(f"Created {len(cells['tetrahedron_2nd'])} second-order tetrahedrons (full curved elements)")
			except Exception as e:
				logger.warning(f"No 3D elements found: {e}")
				pass  # No 3D elements

			# Handle 1D elements
			try:
				elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(1)
				logger.debug(f"1D element types found: {elem_types}")
				for elem_type, elem_node_tags in zip(elem_types, elem_node_tags):
					logger.debug(f"Processing 1D element type {elem_type} with {len(elem_node_tags)} node tags")
					if elem_type == 1:  # First-order line
						if 'line' not in cells:
							cells['line'] = []
						for i in range(0, len(elem_node_tags), 2):
							cells['line'].append([
								tag_to_index[int(elem_node_tags[i])],
								tag_to_index[int(elem_node_tags[i+1])]
							])
					elif elem_type == 8:  # Second-order line
						if 'line_2nd' not in cells:
							cells['line_2nd'] = []
						for i in range(0, len(elem_node_tags), 3):
							cells['line_2nd'].append([
								tag_to_index[int(elem_node_tags[i])],
								tag_to_index[int(elem_node_tags[i+1])],
								tag_to_index[int(elem_node_tags[i+2])]
							])
			except Exception as e:
				logger.warning(f"No 1D elements found: {e}")
				pass  # No 1D elements

			# Extract physical groups
			physical_groups = self._extract_physical_groups_gmsh()
			
			# Manually add center physical group if it wasn't created in GMSH
			if hasattr(self, '_center_node_tags') and self._center_node_tags:
				from types import SimpleNamespace
				# Create a PhysicalGroupWrapper manually for the center
				class PhysicalGroupWrapper:
					def __init__(self, dim, tag, entities, name, entity_coordinates=None, node_tags=None):
						self.dim = dim
						self.tag = tag
						self.entities = entities
						self.name = name
						self.dimension = dim
						self.entity_coordinates = entity_coordinates or []
						self.node_tags = node_tags or []
						self._data = {
							'dim': dim,
							'tag': tag,
							'entities': entities,
							'name': name,
							'dimension': dim,
							'entity_coordinates': entity_coordinates or [],
							'node_tags': node_tags or []
						}
					def __getitem__(self, key):
						return self._data[key]
					def __contains__(self, key):
						return key in self._data
					def get(self, key, default=None):
						return self._data.get(key, default)
					def __repr__(self):
						return f"PhysicalGroup(dim={self.dim}, tag={self.tag}, name='{self.name}')"
				
				# Use a high tag number that won't conflict
				center_tag = 9999
				# Note: Using dim=0 because the solver expects dim=0 for "center" locations
				# (see physical_groups.py line 276: dim_to_check = 0 if location_key in center_aliases)
				# Even though we're conceptually tagging an element by its nodes, dim=0 is required
				# for the dimension check to pass. The actual DOF resolution uses node_tags, not dim.
				center_pg = PhysicalGroupWrapper(
					dim=0,  # Required by solver for "center" lookup, even though we're tagging element nodes
					tag=center_tag,
					entities=self._center_node_tags,  # Store node tags as entities
					name="center",
					entity_coordinates=[],
					node_tags=self._center_node_tags
				)
				physical_groups["center"] = center_pg
				logger.info(f"Manually added center physical group with {len(self._center_node_tags)} node tags (element_tag={getattr(self, '_center_element_tag', 'unknown')})")
				# Clean up
				delattr(self, '_center_node_tags')
				if hasattr(self, '_center_element_tag'):
					delattr(self, '_center_element_tag')

			return vertices, faces, cells, physical_groups
						
		except Exception as e:
			logger.error(f"Error extracting mesh data: {e}")
			return [], [], {}, {}

	def _extract_physical_groups_gmsh(self) -> Dict[str, Any]:
		"""Extract physical groups from GMSH with their actual names"""
		try:
			import gmsh
			from types import SimpleNamespace
			
			physical_groups = {}
			
			# Get all physical groups
			physical_groups_data = gmsh.model.getPhysicalGroups()
			logger.info(f"Found {len(physical_groups_data)} physical groups in GMSH: {physical_groups_data}")
			
			# Get physical names - GMSH Python API uses getPhysicalName(dim, tag) for individual groups
			# We need to iterate through physical groups and get each name
			all_physical_names = {}
			try:
				# Iterate through all physical groups and get their names
				for dim, tag in physical_groups_data:
					try:
						# GMSH Python API: getPhysicalName(dim, tag) returns the name string
						physical_name = gmsh.model.getPhysicalName(dim, tag)
						if physical_name:
							all_physical_names[(dim, tag)] = physical_name
							logger.info(f"Found physical name for dim={dim}, tag={tag}: '{physical_name}'")
						else:
							logger.warning(f"Physical group dim={dim}, tag={tag} has no name")
					except Exception as e:
						logger.warning(f"Could not get physical name for dim={dim}, tag={tag}: {e}")
			except Exception as e:
				logger.warning(f"Error getting physical names: {e}")
				import traceback
				logger.debug(f"Traceback: {traceback.format_exc()}")
			
			
			
			# Create a wrapper class that can be accessed with .dim and .tag for compatibility
			# But is also JSON-serializable (has __dict__ and _data)
			class PhysicalGroupWrapper:
				def __init__(self, dim, tag, entities, name, entity_coordinates=None, node_tags=None):
					self.dim = dim
					self.tag = tag
					self.entities = entities
					self.name = name
					self.dimension = dim  # Also store as 'dimension' for backwards compatibility
					self.entity_coordinates = entity_coordinates or []
					self.node_tags = node_tags or []  # Store GMSH node tags for this physical group
					self._data = {
						'dim': dim,
					'tag': tag,
						'entities': entities,
						'name': name,
						'dimension': dim,
						'entity_coordinates': entity_coordinates or [],
						'node_tags': node_tags or []
					}
				def __getitem__(self, key):
					return self._data[key]
				def __contains__(self, key):
					return key in self._data
				def get(self, key, default=None):
					return self._data.get(key, default)
				def __repr__(self):
					return f"PhysicalGroup(dim={self.dim}, tag={self.tag}, name='{self.name}')"
			
			for dim, tag in physical_groups_data:
				entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
				entity_coordinates = None
				node_tags_set = set()  # Collect all unique node tags for this physical group
				
				# For surface physical groups (dim=2), collect all node tags from mesh elements
				if dim == 2:
					for entity_tag in entities:
						try:
							element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim, entity_tag)
							# node_tags is a list of arrays (one per element type)
							# Collect all node tags from all elements on this entity
							for node_tags_array in node_tags:
								for node_tag in node_tags_array:
									node_tags_set.add(int(node_tag))
						except Exception as e:
							logger.debug(f"Could not get elements for entity {entity_tag} in physical group tag={tag}: {e}")
				
				# For curve physical groups (dim=1), collect all node tags from mesh elements
				if dim == 1:
					for entity_tag in entities:
						try:
							element_types, element_tags, node_tags = gmsh.model.mesh.getElements(dim, entity_tag)
							# node_tags is a list of arrays (one per element type)
							# Collect all node tags from all elements on this entity
							for node_tags_array in node_tags:
								for node_tag in node_tags_array:
									node_tags_set.add(int(node_tag))
						except Exception as e:
							logger.debug(f"Could not get elements for entity {entity_tag} in physical group tag={tag}: {e}")
				
				# For point physical groups (dim=0), collect node tags and coordinates
				if dim == 0:
					entity_coordinates = []
					# If entities list is empty or invalid, the physical group might have been created with node tags directly
					# In that case, we need to get the node tags from the physical group's entities differently
					if len(entities) == 0:
						# Try to get nodes directly - if the physical group was created with node tags,
						# the entities might be the node tags themselves
						logger.debug(f"Physical group dim={dim}, tag={tag} has no geometric entities - may have been created with node tags directly")
						# We'll rely on the node tags being stored elsewhere or handle this in the creation code
					else:
						for entity in entities:
							try:
								node_tags_array, node_coords, _ = gmsh.model.mesh.getNodes(dim, entity)
								if node_tags_array is not None and len(node_tags_array) > 0:
									# Collect node tags for this point entity
									for node_tag in node_tags_array:
										node_tags_set.add(int(node_tag))
								if node_coords is None or len(node_coords) == 0:
									continue
								num_points = len(node_coords) // 3
								for i in range(num_points):
									start = 3 * i
									entity_coordinates.append([
										float(node_coords[start]),
										float(node_coords[start + 1]),
										float(node_coords[start + 2]),
									])
							except Exception as coord_err:
								logger.debug(f"Could not extract coordinates for physical group dim={dim}, tag={tag}, entity={entity}: {coord_err}")
								# If getNodes fails, the entity might actually be a node tag itself
								# Try treating the entity as a node tag
								try:
									node_tags_set.add(int(entity))
									logger.debug(f"Treated entity {entity} as node tag for dim=0 physical group")
								except:
									pass
				
				# Get the actual name of the physical group
				physical_name = all_physical_names.get((dim, tag))
				
				if not physical_name:
					logger.warning(f"Physical group dim={dim}, tag={tag} has no name in GMSH. Available names: {list(all_physical_names.keys())}")
				
				# Convert node_tags_set to sorted list for JSON serialization
				node_tags_list = sorted(list(node_tags_set)) if node_tags_set else []
				group_obj = PhysicalGroupWrapper(dim, tag, entities, physical_name, entity_coordinates, node_tags_list)
				if len(node_tags_list) == 0:
					if dim == 2:
						logger.warning(f"Physical group '{physical_name}' (tag={tag}): No node tags collected for dim=2 surface group!")
					elif dim == 1:
						logger.warning(f"Physical group '{physical_name}' (tag={tag}): No node tags collected for dim=1 curve group!")
					elif dim == 0:
						logger.warning(f"Physical group '{physical_name}' (tag={tag}): No node tags collected for dim=0 point group!")
				
				# CRITICAL: Physical groups MUST have names from geometry_boundaries.json config
				if not physical_name:
					logger.error(f"Physical group dim={dim}, tag={tag} has no name! Skipping.")
					continue
				
				# Store with the name key - this is what the solver expects
				logger.info(f"Extracting physical group: name='{physical_name}', dim={dim}, tag={tag}, node_tags={len(node_tags_list)}")
				physical_groups[physical_name] = group_obj
			
			return physical_groups
		
		except Exception as e:
			logger.error(f"Error extracting physical groups: {e}")
			return {}

	def _create_error_mesh(self, error_message: str) -> Dict[str, Any]:
		"""Create error mesh"""
		logger.error(error_message)
		return {
			'vertices': [],
			'faces': [],
			'cells': {},
			'physical_groups': {},
			'type': 'error',
			'mesh_dimension': 0,
			'bounds': {'min': [0, 0, 0], 'max': [0, 0, 0]},
			'mesh_stats': {'vertices': 0, 'faces': 0, 'cells': 0, 'cell_types': []},
			'success': False,
			'error': error_message
		}

	def get_supported_geometries(self) -> List[str]:
		"""Get list of supported geometry types"""
		return [
			# 1D
			'line', 'rod', 'bar',
			# 2D
			'plate', 'membrane', 'disc', 'rectangle',
			# 3D
			'cube', 'box', 'beam', 'cylinder', 'sphere', 'solid', 'rectangular'
		]

	def get_mesh_info(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
		"""Get information about the generated mesh"""
		if not mesh_data.get('success', False):
			return {'error': mesh_data.get('error', 'Unknown error')}

		stats = mesh_data.get('mesh_stats', {})
		physical_groups = mesh_data.get('physical_groups', {})
		
		return {
			'type': mesh_data.get('type', 'unknown'),
			'geometry': mesh_data.get('geometry_type', 'unknown'),
			'vertices': stats.get('vertices', 0),
			'faces': stats.get('faces', 0),
			'cells': stats.get('cells', 0),
			'cell_types': stats.get('cell_types', []),
			'physical_groups': len(physical_groups),
			'dimensions': mesh_data.get('dimensions', {}),
			'success': True
		}

	def _identify_boundaries_1d(self, vertices: List[List[float]], physical_groups: Dict[str, Any]) -> Dict[str, Any]:
		"""Identify boundaries for 1D geometry"""
		try:
			import numpy as np
			
			vertices_array = np.array(vertices)
			x_coords = vertices_array[:, 0]
			
			boundary_info = {
				'boundaries': {},
				'domain_bounds': {
					'x_min': float(np.min(x_coords)),
					'x_max': float(np.max(x_coords))
				}
			}
			
			# Identify left and right boundaries based on x-coordinates
			tolerance = 1e-10
			left_vertices = np.where(np.abs(x_coords - np.min(x_coords)) < tolerance)[0]
			right_vertices = np.where(np.abs(x_coords - np.max(x_coords)) < tolerance)[0]
			
			boundary_info['boundaries']['left'] = {
				'type': 'point',
				'vertex_indices': left_vertices.tolist(),
				'coordinates': vertices_array[left_vertices].tolist(),
				'description': 'Left boundary (x=0)'
			}
			
			boundary_info['boundaries']['right'] = {
				'type': 'point',
				'vertex_indices': right_vertices.tolist(),
				'coordinates': vertices_array[right_vertices].tolist(),
				'description': 'Right boundary (x=length)'
			}
			
			return boundary_info
			
		except Exception as e:
			logger.error(f"Error identifying 1D boundaries: {e}")
			return {'boundaries': {}, 'domain_bounds': {}}

	def _identify_boundaries_2d(self, vertices: List[List[float]], physical_groups: Dict[str, Any], geometry_type: str) -> Dict[str, Any]:
		"""Identify boundaries for 2D geometry"""
		try:
			import numpy as np
			
			vertices_array = np.array(vertices)
			x_coords = vertices_array[:, 0]
			y_coords = vertices_array[:, 1]
			
			boundary_info = {
				'boundaries': {},
				'domain_bounds': {
					'x_min': float(np.min(x_coords)),
					'x_max': float(np.max(x_coords)),
					'y_min': float(np.min(y_coords)),
					'y_max': float(np.max(y_coords))
				}
			}
			
			if geometry_type in ['plate', 'membrane', 'rectangle', 'square']:
				# Rectangular boundaries
				tolerance = 1e-10
				
				# Left boundary (x = x_min)
				left_vertices = np.where(np.abs(x_coords - np.min(x_coords)) < tolerance)[0]
				boundary_info['boundaries']['left'] = {
					'type': 'line',
					'vertex_indices': left_vertices.tolist(),
					'coordinates': vertices_array[left_vertices].tolist(),
					'description': 'Left boundary (x=0)'
				}
				
				# Right boundary (x = x_max)
				right_vertices = np.where(np.abs(x_coords - np.max(x_coords)) < tolerance)[0]
				boundary_info['boundaries']['right'] = {
					'type': 'line',
					'vertex_indices': right_vertices.tolist(),
					'coordinates': vertices_array[right_vertices].tolist(),
					'description': 'Right boundary (x=length)'
				}
				
				# Bottom boundary (y = y_min)
				bottom_vertices = np.where(np.abs(y_coords - np.min(y_coords)) < tolerance)[0]
				boundary_info['boundaries']['bottom'] = {
					'type': 'line',
					'vertex_indices': bottom_vertices.tolist(),
					'coordinates': vertices_array[bottom_vertices].tolist(),
					'description': 'Bottom boundary (y=0)'
				}
				
				# Top boundary (y = y_max)
				top_vertices = np.where(np.abs(y_coords - np.max(y_coords)) < tolerance)[0]
				boundary_info['boundaries']['top'] = {
					'type': 'line',
					'vertex_indices': top_vertices.tolist(),
					'coordinates': vertices_array[top_vertices].tolist(),
					'description': 'Top boundary (y=width)'
				}
				
			elif geometry_type == 'disc':
				# Circular boundary
				tolerance = 1e-10
				center = np.array([0.0, 0.0])
				radius = np.sqrt(np.sum(vertices_array**2, axis=1))
				expected_radius = np.max(radius)
				
				# Find vertices on the circular boundary
				circular_vertices = np.where(np.abs(radius - expected_radius) < tolerance)[0]
				boundary_info['boundaries']['circular'] = {
					'type': 'curve',
					'vertex_indices': circular_vertices.tolist(),
					'coordinates': vertices_array[circular_vertices].tolist(),
					'description': 'Circular boundary',
					'radius': float(expected_radius)
				}
			
			return boundary_info
			
		except Exception as e:
			logger.error(f"Error identifying 2D boundaries: {e}")
			return {'boundaries': {}, 'domain_bounds': {}}

	def _identify_boundaries_3d(self, vertices: List[List[float]], physical_groups: Dict[str, Any], geometry_type: str) -> Dict[str, Any]:
		"""Identify boundaries for 3D geometry"""
		try:
			import numpy as np
			
			vertices_array = np.array(vertices)
			x_coords = vertices_array[:, 0]
			y_coords = vertices_array[:, 1]
			z_coords = vertices_array[:, 2]
			tolerance = 1e-10
			
			boundary_info = {
				'boundaries': {},
				'domain_bounds': {
					'x_min': float(np.min(x_coords)),
					'x_max': float(np.max(x_coords)),
					'y_min': float(np.min(y_coords)),
					'y_max': float(np.max(y_coords)),
					'z_min': float(np.min(z_coords)),
					'z_max': float(np.max(z_coords))
				}
			}
			
			if geometry_type in ['cube', 'box', 'beam', 'solid', 'rectangular']:
				# Box boundaries
				tolerance = 1e-10
				
				# Left boundary (x = x_min)
				left_vertices = np.where(np.abs(x_coords - np.min(x_coords)) < tolerance)[0]
				boundary_info['boundaries']['left'] = {
					'type': 'surface',
					'vertex_indices': left_vertices.tolist(),
					'coordinates': vertices_array[left_vertices].tolist(),
					'description': 'Left boundary (x=0)'
				}
				
				# Right boundary (x = x_max)
				right_vertices = np.where(np.abs(x_coords - np.max(x_coords)) < tolerance)[0]
				boundary_info['boundaries']['right'] = {
					'type': 'surface',
					'vertex_indices': right_vertices.tolist(),
					'coordinates': vertices_array[right_vertices].tolist(),
					'description': 'Right boundary (x=length)'
				}
				
				# Front boundary (y = y_min)
				front_vertices = np.where(np.abs(y_coords - np.min(y_coords)) < tolerance)[0]
				boundary_info['boundaries']['front'] = {
					'type': 'surface',
					'vertex_indices': front_vertices.tolist(),
					'coordinates': vertices_array[front_vertices].tolist(),
					'description': 'Front boundary (y=0)'
				}
				
				# Back boundary (y = y_max)
				back_vertices = np.where(np.abs(y_coords - np.max(y_coords)) < tolerance)[0]
				boundary_info['boundaries']['back'] = {
					'type': 'surface',
					'vertex_indices': back_vertices.tolist(),
					'coordinates': vertices_array[back_vertices].tolist(),
					'description': 'Back boundary (y=width)'
				}
				
				# Bottom boundary (z = z_min)
				bottom_vertices = np.where(np.abs(z_coords - np.min(z_coords)) < tolerance)[0]
				boundary_info['boundaries']['bottom'] = {
					'type': 'surface',
					'vertex_indices': bottom_vertices.tolist(),
					'coordinates': vertices_array[bottom_vertices].tolist(),
					'description': 'Bottom boundary (z=0)'
				}
				
				# Top boundary (z = z_max)
				top_vertices = np.where(np.abs(z_coords - np.max(z_coords)) < tolerance)[0]
				boundary_info['boundaries']['top'] = {
					'type': 'surface',
					'vertex_indices': top_vertices.tolist(),
					'coordinates': vertices_array[top_vertices].tolist(),
					'description': 'Top boundary (z=height)'
				}
				
			elif geometry_type == 'cylinder':
				# Cylindrical boundaries
				radius = np.sqrt(vertices_array[:, 0]**2 + vertices_array[:, 1]**2)
				expected_radius = np.max(radius)
				z_min, z_max = np.min(z_coords), np.max(z_coords)
				
				# Cylindrical surface
				cylindrical_vertices = np.where(np.abs(radius - expected_radius) < tolerance)[0]
				boundary_info['boundaries']['cylindrical'] = {
					'type': 'surface',
					'vertex_indices': cylindrical_vertices.tolist(),
					'coordinates': vertices_array[cylindrical_vertices].tolist(),
					'description': 'Cylindrical boundary',
					'radius': float(expected_radius)
				}
				
				# Bottom face (z = z_min)
				bottom_vertices = np.where(np.abs(z_coords - z_min) < tolerance)[0]
				boundary_info['boundaries']['bottom'] = {
					'type': 'surface',
					'vertex_indices': bottom_vertices.tolist(),
					'coordinates': vertices_array[bottom_vertices].tolist(),
					'description': 'Bottom face (z=0)'
				}
				
				# Top face (z = z_max)
				top_vertices = np.where(np.abs(z_coords - z_max) < tolerance)[0]
				boundary_info['boundaries']['top'] = {
					'type': 'surface',
					'vertex_indices': top_vertices.tolist(),
					'coordinates': vertices_array[top_vertices].tolist(),
					'description': 'Top face (z=height)'
				}
				
			elif geometry_type == 'sphere':
				# Spherical boundary
				radius = np.sqrt(np.sum(vertices_array**2, axis=1))
				expected_radius = np.max(radius)
				
				spherical_vertices = np.where(np.abs(radius - expected_radius) < tolerance)[0]
				boundary_info['boundaries']['spherical'] = {
					'type': 'surface',
					'vertex_indices': spherical_vertices.tolist(),
					'coordinates': vertices_array[spherical_vertices].tolist(),
					'description': 'Spherical boundary',
					'radius': float(expected_radius)
				}
			
			return boundary_info
			
		except Exception as e:
			logger.error(f"Error identifying 3D boundaries: {e}")
			return {'boundaries': {}, 'domain_bounds': {}}