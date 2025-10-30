"""
GMSH Mesh Generator
===================

Professional mesh generation using GMSH library exclusively.
Provides high-quality meshes with proper physical groups for field variables.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class GMSHGenerator:
	"""Professional mesh generator using GMSH exclusively"""
	
	def __init__(self):
		self.gmsh_available = self._check_gmsh_availability()
		self.gmsh_model = None  # Store the GMSH model object
		self.gmsh_initialized = False
		if self.gmsh_available:
			logger.info("✅ GMSH mesh generator initialized")
		else:
			logger.error("❌ GMSH not available - GMSH is required for mesh generation")
    
	def _check_gmsh_availability(self) -> bool:
		"""Check if GMSH is available"""
		try:
			import gmsh
			return True
		except ImportError:
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
			logger.info("✅ GMSH initialized successfully")
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
				logger.info("✅ GMSH finalized and cleared successfully")
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
			gmsh.clear()
			logger.info("Force cleared GMSH model")
		except Exception as e:
			logger.warning(f"Could not force clear GMSH model: {e}")

	def cleanup_gmsh(self):
		"""Manually finalize GMSH when needed (for cleanup)"""
		try:
			import gmsh
			# Clear any existing GMSH model
			gmsh.clear()
			logger.info("Cleared GMSH model during cleanup")
		except Exception as e:
			logger.warning(f"Could not clear GMSH model: {e}")
		
		if self.gmsh_initialized:
			self.finalize_gmsh()
			logger.info("GMSH cleaned up manually")


    
	def generate_mesh(self, geometry_type: str, dimensions: Dict[str, float], mesh_parameters: Dict[str, Any]) -> Dict[str, Any]:
		"""Generate mesh using GMSH with proper physical groups and boundary identification - GMSH model included by default"""
		if not self.gmsh_available:
			return self._create_error_mesh("GMSH not available")
		
		# Show loading message
		logger.info("Loading visualization... Generating mesh with GMSH...")
		
		try:
			# Set mesh parameters
			mesh_size = mesh_parameters.get('mesh_size', 0.2)
			resolution = mesh_parameters.get('resolution', 50)
			
			# Always force clear any existing GMSH model before generating new mesh
			self.force_clear_gmsh()
			
			# Initialize GMSH if not already initialized
			if not self.gmsh_initialized:
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
			
			# Always include GMSH model reference in mesh data (default behavior)
			if self.gmsh_initialized:
				mesh_data['gmsh_model'] = self.gmsh_model
				mesh_data['gmsh_initialized'] = True
				mesh_data['gmsh_available'] = True
			else:
				mesh_data['gmsh_model'] = None
				mesh_data['gmsh_initialized'] = False
				mesh_data['gmsh_available'] = self.gmsh_available
			
			return mesh_data
			
		except Exception as e:
			logger.error(f"GMSH mesh generation error: {e}")
			return self._create_error_mesh(str(e))
		# Note: GMSH is NOT finalized by default to keep the model available for FEniCS backend

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
			
			# Create physical groups with enhanced boundary identification
			# Left boundary (x=0)
			gmsh.model.addPhysicalGroup(0, [p1], 1, name="left_boundary")
			# Right boundary (x=length)
			gmsh.model.addPhysicalGroup(0, [p2], 2, name="right_boundary")
			# Line domain
			gmsh.model.addPhysicalGroup(1, [line], 3, name="domain")
			
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

				# Create physical groups with enhanced boundary identification
				boundaries = gmsh.model.getBoundary([(2, rect)])
				boundary_names = ["left", "bottom", "right", "top"]  # Standard boundary names
				for i, boundary in enumerate(boundaries):
					boundary_name = boundary_names[i % len(boundary_names)]
					gmsh.model.addPhysicalGroup(1, [boundary[1]], i + 1, name=f"{boundary_name}_boundary")
				gmsh.model.addPhysicalGroup(2, [rect], 5, name="domain")
				
			elif geometry_type == 'disc':
				# Circular geometry
				radius = dimensions.get('radius', 1.0)
				logger.info(f"Creating disc with radius: {radius}")

				# Create disc
				disc = gmsh.model.occ.addDisk(0, 0, 0, radius, radius)
				gmsh.model.occ.synchronize()
				logger.info(f"Created disc geometry: {disc}")

				# Create physical groups - for disc, all boundary curves should be in one group
				boundaries = gmsh.model.getBoundary([(2, disc)])
				logger.info(f"Disc boundaries: {boundaries}")
				if boundaries:
					# Group all boundary curves into a single physical group
					boundary_curves = [boundary[1] for boundary in boundaries]
					gmsh.model.addPhysicalGroup(1, boundary_curves, 1, name="circular_boundary")
				gmsh.model.addPhysicalGroup(2, [disc], 2, name="domain")
				logger.info("Added physical groups for disc")

			# Generate mesh
			logger.info("Generating 2D mesh...")
			gmsh.model.mesh.generate(2)
			logger.info("2D mesh generation completed")

			# Extract mesh data
			vertices, faces, cells, physical_groups = self._extract_mesh_data_gmsh()
			logger.info(f"Extracted mesh data: {len(vertices)} vertices, {len(faces)} faces, cells: {list(cells.keys())}")
			
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
				
				# Create physical groups with enhanced boundary identification
				boundaries = gmsh.model.getBoundary([(3, box)])
				boundary_names = ["left", "right", "front", "back", "bottom", "top"]  # Standard boundary names
				for i, boundary in enumerate(boundaries):
					boundary_name = boundary_names[i % len(boundary_names)]
					gmsh.model.addPhysicalGroup(2, [boundary[1]], i + 1, name=f"{boundary_name}_boundary")
				gmsh.model.addPhysicalGroup(3, [box], 7, name="domain")
				
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

				# Create physical groups with meaningful names for cylinder
				boundaries = gmsh.model.getBoundary([(3, cylinder)])
				logger.info(f"Cylinder boundaries: {boundaries}")
				
				# Try to identify which boundary is which, but always create physical groups for ALL boundaries
				# GMSH creates cylinder with: bottom face, top face, and curved surface
				
				# Map of surface tags to their identified names (if we can identify them)
				surface_names = {}
				
				# Try to identify surfaces by their geometric properties
				# GMSH may use negative tags to represent orientation, use absolute value for queries
				for boundary in boundaries:
					surface_tag = boundary[1]
					abs_tag = abs(surface_tag)
					
					try:
						# Get bounding box of the surface (use absolute tag for query)
						bbox = gmsh.model.getBoundingBox(2, abs_tag)
						z_min = bbox[2]
						z_max = bbox[5]
						
						# Calculate the z-range to determine if surface is flat or curved
						z_range = abs(z_max - z_min)
						
						logger.info(f"Surface {surface_tag} (abs: {abs_tag}): z_min={z_min:.6f}, z_max={z_max:.6f}, z_range={z_range:.6f}")
						
						# Flat surfaces have very small z-range (bottom and top faces)
						# Bottom surface: z is close to 0 and z_range is small
						if z_range < 1e-6 and abs(z_min) < 1e-6:
							surface_names[surface_tag] = "bottom"
							logger.info(f"  → Identified as bottom")
						# Top surface: z is close to height and z_range is small
						elif z_range < 1e-6 and abs(z_max - height) < 1e-6:
							surface_names[surface_tag] = "top"
							logger.info(f"  → Identified as top")
						# Curved surface: z varies significantly (z_range is large)
						elif z_range > height * 0.1:  # Significant variation means curved
							surface_names[surface_tag] = "curved_surface"
							logger.info(f"  → Identified as curved_surface")
						else:
							# Couldn't determine - leave unnamed
							logger.warning(f"  → Could not identify surface {surface_tag}")
					except Exception as e:
						logger.warning(f"Failed to identify surface {surface_tag}: {e}")
						# Leave unnamed - will be created but without a name
				
				# ALWAYS create physical groups for ALL boundaries (ensures we have 3 groups)
				# Use meaningful names where we identified them
				# GMSH may return negative tags for orientation, but physical groups should use absolute values
				physical_group_id = 1
				for i, boundary in enumerate(boundaries):
					surface_tag = boundary[1]
					abs_tag = abs(surface_tag)
					name = surface_names.get(surface_tag, "")
					try:
						# Use absolute tag for physical group creation (GMSH handles orientation internally)
						gmsh.model.addPhysicalGroup(2, [abs_tag], physical_group_id, name=name if name else "")
						logger.info(f"Created physical group {physical_group_id} for surface {surface_tag} (abs: {abs_tag}) with name '{name}'")
						physical_group_id += 1
					except Exception as e:
						logger.warning(f"Failed to create physical group for boundary {i+1} (surface {surface_tag}): {e}")
				
				# Domain
				try:
					gmsh.model.addPhysicalGroup(3, [cylinder], physical_group_id, name="domain")
				except Exception as e:
					logger.warning(f"Failed to create domain physical group: {e}")
				
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

				# Create physical groups
				boundaries = gmsh.model.getBoundary([(3, sphere)])
				for i, boundary in enumerate(boundaries):
					gmsh.model.addPhysicalGroup(2, [boundary[1]], i + 1)
				gmsh.model.addPhysicalGroup(3, [sphere], 2)  # Domain

			# Generate mesh
			logger.info("Generating 3D mesh...")
			gmsh.model.mesh.generate(3)
			logger.info("3D mesh generation completed")
			
			# Check if mesh was actually generated
			try:
				node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
				logger.info(f"Generated {len(node_tags)} nodes")
			except Exception as e:
				logger.error(f"Failed to get nodes after mesh generation: {e}")
			
			# Extract mesh data
			vertices, faces, cells, physical_groups = self._extract_mesh_data_gmsh()
			
			logger.info(f"Returning mesh data: {len(vertices)} vertices, {len(faces)} faces, {len(cells)} cell types")
			logger.info(f"Cell types: {list(cells.keys())}")
			if 'tetrahedron' in cells:
				logger.info(f"Tetrahedrons: {len(cells['tetrahedron'])}")
			if 'triangle' in cells:
				logger.info(f"Triangles: {len(cells['triangle'])}")
			if 'quad' in cells:
				logger.info(f"Quads: {len(cells['quad'])}")
			
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
				logger.info(f"2D element types found: {elem_types}")
				for elem_type, elem_node_tags in zip(elem_types, elem_node_tags):
					logger.info(f"Processing element type {elem_type} with {len(elem_node_tags)} node tags")
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
						logger.info(f"Created {len(cells['triangle'])} first-order triangles")
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
						logger.info(f"Created {len(cells['triangle_2nd'])} second-order triangles (full curved elements)")
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
						logger.info(f"Created {len(cells['quad'])} first-order quads")
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
						logger.info(f"Created {len(cells['quad'])} second-order quads (using corner nodes)")
			except Exception as e:
				logger.error(f"Error extracting 2D elements: {e}")
				pass  # No 2D elements

			# Get cells (3D elements)
			try:
				elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
				logger.info(f"3D element types found: {elem_types}")
				for elem_type, elem_node_tags in zip(elem_types, elem_node_tags):
					logger.info(f"Processing 3D element type {elem_type} with {len(elem_node_tags)} node tags")
					if elem_type == 4:  # First-order tetrahedron
						cells['tetrahedron'] = []
						for i in range(0, len(elem_node_tags), 4):
							cells['tetrahedron'].append([
								tag_to_index[int(elem_node_tags[i])],
								tag_to_index[int(elem_node_tags[i+1])],
								tag_to_index[int(elem_node_tags[i+2])],
								tag_to_index[int(elem_node_tags[i+3])]
							])
						logger.info(f"Created {len(cells['tetrahedron'])} first-order tetrahedrons")
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
						logger.info(f"Created {len(cells['tetrahedron_2nd'])} second-order tetrahedrons (full curved elements)")
			except Exception as e:
				logger.warning(f"No 3D elements found: {e}")
				pass  # No 3D elements

			# Handle 1D elements
			try:
				elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(1)
				logger.info(f"1D element types found: {elem_types}")
				for elem_type, elem_node_tags in zip(elem_types, elem_node_tags):
					logger.info(f"Processing 1D element type {elem_type} with {len(elem_node_tags)} node tags")
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

			return vertices, faces, cells, physical_groups
						
		except Exception as e:
			logger.error(f"Error extracting mesh data: {e}")
			return [], [], {}, {}

	def _extract_physical_groups_gmsh(self) -> Dict[str, Any]:
		"""Extract physical groups from GMSH"""
		try:
			import gmsh
			
			physical_groups = {}
			
			# Get all physical groups
			physical_groups_data = gmsh.model.getPhysicalGroups()
			
			for dim, tag in physical_groups_data:
				entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
				physical_groups[f'group_{tag}'] = {
					'dimension': dim,
					'tag': tag,
					'entities': entities
				}
			
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