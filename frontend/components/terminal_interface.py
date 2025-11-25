"""
Terminal Interface Component
============================

Main terminal interface for the FEM simulation application.
Handles user input, command processing, and result display.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

class TerminalInterface:
	"""Main terminal interface component"""

	def __init__(self, template_path: str = None):
		"""
		Initialize terminal interface

		Args:
			template_path: Path to HTML template file
		"""
		if template_path is None:
			# Use the original terminal frontend template
			template_path = Path(__file__).parent.parent.parent / "main_app" / "terminal_frontend.html"

		self.template_path = Path(template_path)
		self.html_content = self._load_template()

	def _load_template(self) -> str:
		"""Load HTML template content"""
		with open(self.template_path, 'r', encoding='utf-8') as f:
			return f.read()

	def get_html(self) -> str:
		"""Get HTML content for serving"""
		return self.html_content

	def update_api_endpoints(self, base_url: str = "http://localhost:8080") -> None:
		"""
		DEPRECATED: This method is no longer used.
		API endpoints are now dynamically constructed in JavaScript using window.location.origin.
		This ensures URLs work correctly on any host (localhost, AWS IP, custom domain).
		
		Args:
		base_url: Base URL for API endpoints (ignored - kept for backward compatibility)
		"""
		# Method intentionally left empty - endpoints are now handled dynamically in JavaScript
		# All fetch() calls in terminal_frontend.html use window.location.origin + endpoint
		pass

	def customize_theme(self, theme_config: Dict[str, str]) -> None:
		"""
		Customize the visual theme

		Args:
		theme_config: Dictionary with theme settings
		"""
		if 'primary_color' in theme_config:
			self.html_content = self.html_content.replace(
				'var(--primary-color, #007bff)',
				f'var(--primary-color, {theme_config["primary_color"]})'
			)

		if 'background_color' in theme_config:
			self.html_content = self.html_content.replace(
				'var(--bg-color, #1a1a1a)',
				f'var(--bg-color, {theme_config["background_color"]})'
			)

	def add_custom_css(self, css_content: str) -> None:
		"""
		Add custom CSS to the template

		Args:
		css_content: CSS content to add
		"""
		css_tag = f'<style>\n{css_content}\n</style>'
		self.html_content = self.html_content.replace('</head>', f'{css_tag}\n</head>')

	def add_custom_js(self, js_content: str) -> None:
		"""
		Add custom JavaScript to the template

		Args:
		js_content: JavaScript content to add
		"""
		js_tag = f'<script>\n{js_content}\n</script>'
		self.html_content = self.html_content.replace('</body>', f'{js_tag}\n</body>')

	def get_configuration_data(self) -> Dict[str, Any]:
		"""
		Get configuration data from the template

		Returns:
		Dictionary with configuration settings
		"""
		return {
			'template_path': str(self.template_path),
			'has_mesh_visualizer': 'meshContainer' in self.html_content,
			'has_config_panels': 'right-panel' in self.html_content,
			'api_endpoints': self._extract_api_endpoints()
		}

	def _extract_api_endpoints(self) -> list:
		"""Extract API endpoints from the template"""
		import re
		endpoints = re.findall(r'["\']([^"\']*/(?:simulation|mesh|solve-pde)[^"\']*)["\']', self.html_content)
		return list(set(endpoints))

	def validate_template(self) -> Dict[str, Any]:
		"""
		Validate the terminal interface template
		
		Returns:
			Dictionary with validation results
		"""
		try:
			# Check if template is loaded
			if not self.html_content:
				return {
					'valid': False,
					'error': 'Template not loaded',
					'issues': ['No template content found']
				}
			
			# Check for required HTML elements (more lenient for terminal interface)
			required_elements = ['<body>']
			missing_elements = []
			
			for element in required_elements:
				if element not in self.html_content:
					missing_elements.append(element)
			
			# Check for API endpoints
			api_endpoints = self._extract_api_endpoints()
			
			# Check for basic structure
			has_basic_structure = len(missing_elements) == 0
			
			validation_result = {
				'valid': has_basic_structure,
				'issues': missing_elements,
				'api_endpoints': api_endpoints,
				'template_size': len(self.html_content),
				'has_required_elements': has_basic_structure
			}
			
			if not has_basic_structure:
				validation_result['error'] = f'Missing required elements: {missing_elements}'
			
			return validation_result
			
		except Exception as e:
			return {
				'valid': False,
				'error': f'Validation failed: {str(e)}',
				'issues': [str(e)]
			}

	def load_external_css(self, css_files: list) -> None:
		"""
		Load external CSS files into the template
		
		Args:
			css_files: List of CSS file paths or URLs
		"""
		try:
			css_links = []
			for css_file in css_files:
				if css_file.startswith('http'):
					# External URL
					css_links.append(f'<link rel="stylesheet" href="{css_file}">')
				else:
					# Local file
					css_links.append(f'<link rel="stylesheet" href="{css_file}">')
			
			# Insert CSS links before closing head tag
			css_content = '\n'.join(css_links)
			self.html_content = self.html_content.replace('</head>', f'{css_content}\n</head>')
			
		except Exception as e:
			logger.error(f"Error loading external CSS: {e}")

	def load_external_js(self, js_files: list) -> None:
		"""
		Load external JavaScript files into the template
		
		Args:
			js_files: List of JavaScript file paths or URLs
		"""
		try:
			js_links = []
			for js_file in js_files:
				if js_file.startswith('http'):
					# External URL
					js_links.append(f'<script src="{js_file}"></script>')
				else:
					# Local file
					js_links.append(f'<script src="{js_file}"></script>')
			
			# Insert JS links before closing body tag
			js_content = '\n'.join(js_links)
			self.html_content = self.html_content.replace('</body>', f'{js_content}\n</body>')
			
		except Exception as e:
			logger.error(f"Error loading external JS: {e}")