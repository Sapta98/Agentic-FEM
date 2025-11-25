"""
Frontend Manager
================

Main coordinator for all frontend components.
Manages the integration of terminal interface, visualizers, and UI components.
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from .components.terminal_interface import TerminalInterface
from .components.config_panels import ConfigPanels
from .components.input_fields import InputFields
from .components.result_display import ResultDisplay
from .visualizers.mesh_visualizer import MeshVisualizer
from .visualizers.field_visualizer import FieldVisualizer

logger = logging.getLogger(__name__)

class FrontendManager:
	"""Main frontend manager coordinating all components"""

	def __init__(self, config_manager=None, static_dir: str = "frontend/static"):
		"""
		Initialize frontend manager

		Args:
		config_manager: Configuration manager instance
		static_dir: Directory for static assets
		"""
		self.config_manager = config_manager
		self.static_dir = Path(static_dir)
		self.static_dir.mkdir(parents=True, exist_ok=True)

		# Initialize components
		self.terminal_interface = TerminalInterface()
		self.config_panels = ConfigPanels(config_manager)
		self.input_fields = InputFields()
		self.result_display = ResultDisplay(str(self.static_dir))
		self.mesh_visualizer = MeshVisualizer(str(self.static_dir))
		self.field_visualizer = FieldVisualizer(str(self.static_dir))

		# Frontend settings
		self.settings = {
			'theme': 'dark',
			'auto_save': True,
			'auto_refresh': False,
			'show_debug_info': False,
			'enable_animations': True,
			'default_mesh_resolution': 50,
			'default_field_colormap': 'viridis'
		}

		# Current state
		self.current_context = {}
		self.current_results = {}
		self.is_initialized = False

	def initialize(self) -> bool:
		"""
		Initialize the frontend system

		Returns:
		True if initialization successful
		"""
		# Validate template
		validation = self.terminal_interface.validate_template()
		if not validation['valid']:
			logger.error(f"Template validation failed: {validation.get('issues', validation.get('error', 'Unknown error'))}")
			return False

		# Setup component integration
		self._setup_component_integration()

		# Initialize visualizers
		self._initialize_visualizers()

		self.is_initialized = True
		logger.debug("Frontend manager initialized successfully")
		return True

	def _setup_component_integration(self) -> None:
		"""Setup integration between components"""
		# Don't update API endpoints - they are now dynamically constructed in JavaScript
		# using window.location.origin to work with any host (AWS IP, domain, localhost)
		# self.terminal_interface.update_api_endpoints()  # DISABLED - using dynamic URLs

		# Don't load external CSS files to preserve terminal interface styling
		# The terminal interface has its own embedded CSS with black background and green styling
		pass

		# Don't load external JavaScript files to preserve terminal interface functionality
		# The terminal interface has its own embedded JavaScript
		pass

	def _initialize_visualizers(self) -> None:
		"""Initialize visualization components"""
		# Set default settings for visualizers
		self.mesh_visualizer.update_settings({
			'camera_position': [5, 5, 5],
			'background_color': '#1a1a1a',
			'mesh_color': '#00ff00',
			'show_wireframe': True,
			'show_axes': True
		})

		self.field_visualizer.update_settings({
			'colormap': self.settings['default_field_colormap'],
			'show_contours': True,
			'show_colorbar': True,
			'opacity': 0.8
		})


	def get_main_interface(self) -> str:
		"""
		Get the main terminal interface HTML

		Returns:
		HTML content for the main interface
		"""
		if not self.is_initialized:
			self.initialize()

		return self.terminal_interface.get_html()

	def create_mesh_visualization(self, mesh_data: Dict[str, Any]) -> str:
		"""
		Create mesh visualization

		Args:
		mesh_data: Mesh geometry data

		Returns:
		URL to mesh visualization
		"""
		visualization_url = self.mesh_visualizer.create_mesh_visualization(mesh_data)
		logger.info(f"Created mesh visualization: {visualization_url}")
		return visualization_url

	def create_field_visualization(self, mesh_data: Dict[str, Any], field_data: Dict[str, Any]) -> str:
		"""
		Create field visualization

		Args:
		mesh_data: Mesh geometry data
		field_data: Field solution data

		Returns:
		URL to field visualization
		"""
		visualization_url = self.field_visualizer.create_field_visualization(mesh_data, field_data)
		logger.info(f"Created field visualization: {visualization_url}")
		return visualization_url


	def generate_configuration_panels(self, context: Dict[str, Any]) -> str:
		"""
		Generate configuration panels HTML

		Args:
		context: Current simulation context

		Returns:
		HTML content for configuration panels
		"""
		self.current_context = context
		panels_html = self.config_panels.generate_config_html(context)
		logger.info("Generated configuration panels")
		return panels_html

	def display_simulation_results(self, results: Dict[str, Any]) -> str:
		"""
		Display simulation results

		Args:
		results: Simulation results data

		Returns:
		HTML content for result display
		"""
		self.current_results = results
		results_html = self.result_display.display_simulation_results(results)
		logger.info("Generated simulation results display")
		return results_html

	def create_results_report(self, results: Dict[str, Any]) -> str:
		"""
		Create comprehensive results report

		Args:
		results: Simulation results data

		Returns:
		URL to results report
		"""
		report_url = self.result_display.create_results_report(results)
		logger.info(f"Created results report: {report_url}")
		return report_url

	def update_settings(self, new_settings: Dict[str, Any]) -> None:
		"""
		Update frontend settings

		Args:
		new_settings: New settings to apply
		"""
		self.settings.update(new_settings)

		# Update component settings
		if 'default_field_colormap' in new_settings:
			self.field_visualizer.update_settings({
				'colormap': new_settings['default_field_colormap']
			})

		logger.debug(f"Updated frontend settings: {new_settings}")

	def customize_theme(self, theme_config: Dict[str, str]) -> None:
		"""
		Customize the visual theme

		Args:
		theme_config: Theme configuration
		"""
		self.terminal_interface.customize_theme(theme_config)
		logger.info(f"Applied theme customization: {theme_config}")

	def validate_configuration(self, context: Dict[str, Any]) -> Dict[str, Any]:
		"""
		Validate current configuration

		Args:
		context: Configuration context to validate

		Returns:
		Validation results
		"""
		return self.config_panels.validate_configuration(context)

	def cleanup_old_files(self, max_age_hours: int = 24) -> Dict[str, int]:
		"""
		Clean up old visualization and report files

		Args:
		max_age_hours: Maximum age of files in hours

		Returns:
		Dictionary with cleanup counts
		"""
		cleanup_counts = {
			'mesh_visualizations': self.mesh_visualizer.cleanup_old_visualizations(max_age_hours),
			'field_visualizations': self.field_visualizer.cleanup_old_visualizations(max_age_hours),
			'reports': self.result_display.cleanup_old_reports(max_age_hours)
		}

		total_cleaned = sum(cleanup_counts.values())
		logger.info(f"Cleaned up {total_cleaned} old files: {cleanup_counts}")

		return cleanup_counts

	def get_system_status(self) -> Dict[str, Any]:
		"""
		Get frontend system status

		Returns:
		System status information
		"""
		return {
			'initialized': self.is_initialized,
			'components': {
				'terminal_interface': self.terminal_interface.get_configuration_data(),
				'config_panels': self.config_panels.settings,
				'mesh_visualizer': self.mesh_visualizer.get_current_mesh_info(),
				'field_visualizer': self.field_visualizer.settings,
				'result_display': self.result_display.settings
			},
			'current_context': self.current_context,
			'current_results': bool(self.current_results),
			'settings': self.settings,
			'static_dir': str(self.static_dir)
		}