"""
Result Display Component
========================

Component for displaying simulation results, including
mesh visualizations, field plots, and analysis data.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class ResultDisplay:
	"""Result display component"""

	def __init__(self, static_dir: str = "frontend/static"):
		"""
		Initialize result display component

		Args:
		static_dir: Directory for static assets
		"""
		self.static_dir = Path(static_dir)
		self.static_dir.mkdir(parents=True, exist_ok=True)

		# Display settings
		self.settings = {
			'show_mesh_stats': True,
			'show_field_stats': True,
			'show_solver_info': True,
			'show_performance_metrics': True,
			'auto_refresh': False,
			'refresh_interval': 5000
		}

		# Current results
		self.current_results = {}
		self.current_mesh_url = None
		self.current_field_url = None

	def display_simulation_results(self, results: Dict[str, Any]) -> str:
		"""
		Display simulation results

		Args:
		results: Simulation results data

		Returns:
		HTML content for result display
		"""
		self.current_results = results

		# Generate unique filename
		import time
		timestamp = int(time.time() * 1000)
		filename = f"results_display_{timestamp}.html"
		filepath = self.static_dir / filename

		# Create results HTML
		html_content = self._generate_results_html(results)

		# Save to static directory
		with open(filepath, 'w', encoding='utf-8') as f:
			f.write(html_content)

		logger.info(f"Created results display: {filepath}")
		return str(filepath)

	def create_results_report(self, results: Dict[str, Any]) -> str:
		"""
		Create comprehensive results report

		Args:
		results: Simulation results data

		Returns:
		URL to results report
		"""
		# Generate unique filename
		import time
		timestamp = int(time.time() * 1000)
		filename = f"results_report_{timestamp}.html"
		filepath = self.static_dir / filename

		# Create comprehensive report HTML
		html_content = self._generate_results_html(results, is_report=True)

		# Save to static directory
		with open(filepath, 'w', encoding='utf-8') as f:
			f.write(html_content)

		logger.info(f"Created results report: {filepath}")
		return f"http://localhost:8080/static/{filename}"

	def _generate_results_html(self, results: Dict[str, Any], is_report: bool = False) -> str:
		"""
		Generate HTML content for results display

		Args:
		results: Simulation results data
		is_report: Whether this is a comprehensive report

		Returns:
		HTML content string
		"""
		# Prepare data for template
		status = results.get('status', 'unknown')
		status_class = 'success' if status == 'completed' else 'error' if status == 'failed' else 'warning'
		timestamp = results.get('timestamp', '')
		timestamp_html = f'<div class="timestamp">Completed: {timestamp}</div>' if timestamp else ''

		# Generate sections
		mesh_visualization_html = self._generate_mesh_visualization_section(results)
		field_visualization_html = self._generate_field_visualization_section(results)
		results_summary_html = self._generate_results_summary(results)
		mesh_statistics_html = self._generate_mesh_statistics(results) if self.settings['show_mesh_stats'] else ''
		field_statistics_html = self._generate_field_statistics(results) if self.settings['show_field_stats'] else ''
		solver_info_html = self._generate_solver_information(results) if self.settings['show_solver_info'] else ''
		performance_metrics_html = self._generate_performance_metrics(results) if self.settings['show_performance_metrics'] else ''
		export_options_html = self._generate_export_options(results) if is_report else ''

		# Load template file
		template_path = Path(__file__).parent.parent / "templates" / "result_display.html"
		
		with open(template_path, 'r', encoding='utf-8') as f:
			template_content = f.read()

		# Replace placeholders in template
		html_content = template_content.format(
			status_class=status_class,
			status=status.title(),
			timestamp_html=timestamp_html,
			mesh_visualization_html=mesh_visualization_html,
			field_visualization_html=field_visualization_html,
			results_summary_html=results_summary_html,
			mesh_statistics_html=mesh_statistics_html,
			field_statistics_html=field_statistics_html,
			solver_info_html=solver_info_html,
			performance_metrics_html=performance_metrics_html,
			export_options_html=export_options_html
		)

		return html_content

	def _generate_mesh_visualization_section(self, results: Dict[str, Any]) -> str:
		"""Generate mesh visualization section"""
		if 'mesh_visualization_url' not in results:
			return ''

		mesh_url = results['mesh_visualization_url']
		mesh_type = results.get('mesh_type', 'unknown')
		mesh_dimension = results.get('mesh_dimension', 'unknown')

		self.current_mesh_url = mesh_url

		return f"""
		<div class="visualization-section mesh-section">
		<h3>3D Mesh Visualization</h3>
		<div class="mesh-info">
		<span class="mesh-type">Type: {mesh_type}</span>
		<span class="mesh-dimension">Dimension: {mesh_dimension}D</span>
		</div>
		<div class="visualization-container">
		<iframe id="mesh-iframe"
		src="{mesh_url}"
		width="100%"
		height="500px"
		frameborder="0"
		style="border: 2px solid #28a745; border-radius: 8px;">
		</iframe>
		<div class="visualization-controls">
		<button onclick="openInNewTab('{mesh_url}')" class="btn btn-secondary">Open in New Tab</button>
		<button onclick="downloadMesh()" class="btn btn-primary">Download Mesh</button>
		</div>
		</div>
		</div>
		"""

	def _generate_field_visualization_section(self, results: Dict[str, Any]) -> str:
		"""Generate field visualization section"""
		if 'field_visualization_url' not in results:
			return ''

		field_url = results['field_visualization_url']
		field_name = results.get('field_name', 'Field')
		field_units = results.get('field_units', '')

		self.current_field_url = field_url

		return f"""
		<div class="visualization-section field-section">
		<h3>Field Visualization: {field_name}</h3>
		<div class="field-info">
		<span class="field-name">Field: {field_name}</span>
		<span class="field-units">Units: {field_units}</span>
		</div>
		<div class="visualization-container">
		<iframe id="field-iframe"
		src="{field_url}"
		width="100%"
		height="500px"
		frameborder="0"
		style="border: 2px solid #007bff; border-radius: 8px;">
		</iframe>
		<div class="visualization-controls">
		<button onclick="openInNewTab('{field_url}')" class="btn btn-secondary">Open in New Tab</button>
		<button onclick="downloadField()" class="btn btn-primary">Download Field Data</button>
		</div>
		</div>
		</div>
		"""

	def _generate_results_summary(self, results: Dict[str, Any]) -> str:
		"""Generate results summary section"""
		summary = results.get('summary', {})
		
		if not summary:
			return ''

		summary_items = ''
		for key, value in summary.items():
			summary_items += f"""
			<div class="summary-item">
			<span class="summary-label">{key.replace('_', ' ').title()}:</span>
			<span class="summary-value">{value}</span>
			</div>
			"""

		return f"""
		<div class="results-summary">
		<h3>Results Summary</h3>
		<div class="summary-grid">
		{summary_items}
		</div>
		</div>
		"""

	def _generate_mesh_statistics(self, results: Dict[str, Any]) -> str:
		"""Generate mesh statistics section"""
		mesh_stats = results.get('mesh_stats', {})
		
		if not mesh_stats:
			return ''

		stats_items = ''
		for key, value in mesh_stats.items():
			stats_items += f"""
			<div class="stat-item">
			<span class="stat-label">{key.replace('_', ' ').title()}:</span>
			<span class="stat-value">{value}</span>
			</div>
			"""

		return f"""
		<div class="mesh-statistics">
		<h3>Mesh Statistics</h3>
		<div class="stats-grid">
		{stats_items}
		</div>
		</div>
		"""

	def _generate_field_statistics(self, results: Dict[str, Any]) -> str:
		"""Generate field statistics section"""
		field_stats = results.get('field_stats', {})
		
		if not field_stats:
			return ''

		stats_items = ''
		for key, value in field_stats.items():
			stats_items += f"""
			<div class="stat-item">
			<span class="stat-label">{key.replace('_', ' ').title()}:</span>
			<span class="stat-value">{value}</span>
			</div>
			"""

		return f"""
		<div class="field-statistics">
		<h3>Field Statistics</h3>
		<div class="stats-grid">
		{stats_items}
		</div>
		</div>
		"""

	def _generate_solver_information(self, results: Dict[str, Any]) -> str:
		"""Generate solver information section"""
		solver_info = results.get('solver_info', {})
		
		if not solver_info:
			return ''

		info_items = ''
		for key, value in solver_info.items():
			info_items += f"""
			<div class="info-item">
			<span class="info-label">{key.replace('_', ' ').title()}:</span>
			<span class="info-value">{value}</span>
			</div>
			"""

		return f"""
		<div class="solver-information">
		<h3>Solver Information</h3>
		<div class="info-grid">
		{info_items}
		</div>
		</div>
		"""

	def _generate_performance_metrics(self, results: Dict[str, Any]) -> str:
		"""Generate performance metrics section"""
		performance = results.get('performance', {})
		
		if not performance:
			return ''

		metrics_items = ''
		for key, value in performance.items():
			metrics_items += f"""
			<div class="metric-item">
			<span class="metric-label">{key.replace('_', ' ').title()}:</span>
			<span class="metric-value">{value}</span>
			</div>
			"""

		return f"""
		<div class="performance-metrics">
		<h3>Performance Metrics</h3>
		<div class="metrics-grid">
		{metrics_items}
		</div>
		</div>
		"""

	def _generate_export_options(self, results: Dict[str, Any]) -> str:
		"""Generate export options section"""
		return """
		<div class="export-options">
		<h3>Export Options</h3>
		<div class="export-buttons">
		<button onclick="exportResults('pdf')" class="btn btn-export">Export as PDF</button>
		<button onclick="exportResults('html')" class="btn btn-export">Export as HTML</button>
		<button onclick="exportResults('json')" class="btn btn-export">Export as JSON</button>
		</div>
		</div>
		"""

	def update_settings(self, new_settings: Dict[str, Any]) -> None:
		"""
		Update display settings

		Args:
		new_settings: New settings to apply
		"""
		self.settings.update(new_settings)
		logger.info(f"Updated result display settings: {new_settings}")

	def cleanup_old_reports(self, max_age_hours: int = 24) -> int:
		"""
		Clean up old result reports

		Args:
		max_age_hours: Maximum age of files in hours

		Returns:
		Number of files cleaned up
		"""
		import time

		current_time = time.time()
		max_age_seconds = max_age_hours * 3600
		cleaned_count = 0

		for file_path in self.static_dir.glob("results_*.html"):
			file_age = current_time - file_path.stat().st_mtime
			if file_age > max_age_seconds:
				file_path.unlink()
				cleaned_count += 1
				logger.info(f"Cleaned up old result report: {file_path}")

		return cleaned_count