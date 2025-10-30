"""
Input Fields Component
======================

Reusable input field components for various data types.
Handles validation, formatting, and user interaction.
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class InputFields:
	"""Reusable input field components"""

	def __init__(self):
		"""Initialize input fields component"""
		self.field_types = {
			'number': self._create_number_field,
			'text': self._create_text_field,
			'select': self._create_select_field,
			'checkbox': self._create_checkbox_field,
			'range': self._create_range_field,
			'textarea': self._create_textarea_field,
			'file': self._create_file_field,
			'color': self._create_color_field,
			'date': self._create_date_field,
			'time': self._create_time_field
		}

		# Validation rules
		self.validation_rules = {
			'required': lambda value: value is not None and str(value).strip() != '',
			'min': lambda value, min_val: float(value) >= float(min_val),
			'max': lambda value, max_val: float(value) <= float(max_val),
			'min_length': lambda value, min_len: len(str(value)) >= int(min_len),
			'max_length': lambda value, max_len: len(str(value)) <= int(max_len),
			'numeric': lambda value: str(value).replace('.', '').replace('-', '').isdigit()
		}

	def create_field(self, field_config: Dict[str, Any]) -> str:
		"""
		Create an input field based on configuration

		Args:
		field_config: Field configuration dictionary

		Returns:
		HTML string for the input field
		"""
		field_type = field_config.get('type', 'text')

		if field_type not in self.field_types:
			logger.warning(f"Unknown field type: {field_type}")
			field_type = 'text'

		return self.field_types[field_type](field_config)

	def _create_number_field(self, config: Dict[str, Any]) -> str:
		"""Create number input field"""
		field_id = config.get('id', '')
		field_name = config.get('name', '')
		field_label = config.get('label', field_name)
		field_value = config.get('value', '')
		field_placeholder = config.get('placeholder', f'Enter {field_label.lower()}')
		field_unit = config.get('unit', '')
		field_step = config.get('step', 'any')
		field_min = config.get('min', '')
		field_max = config.get('max', '')
		field_required = config.get('required', False)
		field_disabled = config.get('disabled', False)
		field_onchange = config.get('onchange', '')

		# Build attributes
		attributes = []
		if field_step: attributes.append(f'step="{field_step}"')
		if field_min: attributes.append(f'min="{field_min}"')
		if field_max: attributes.append(f'max="{field_max}"')
		if field_required: attributes.append('required')
		if field_disabled: attributes.append('disabled')
		if field_onchange: attributes.append(f'onchange="{field_onchange}"')

		attributes_str = ' '.join(attributes)

		html = f"""
		<div class="input-field number-field">
		<label for="{field_id}">{field_label}{' *' if field_required else ''}</label>
		<div class="input-with-unit">
		<input type="number"
		id="{field_id}"
		name="{field_name}"
		value="{field_value}"
		placeholder="{field_placeholder}"
		{attributes_str}>
		{f'<span class="unit">{field_unit}</span>' if field_unit else ''}
		</div>
		</div>
		"""

		return html

	def _create_text_field(self, config: Dict[str, Any]) -> str:
		"""Create text input field"""
		field_id = config.get('id', '')
		field_name = config.get('name', '')
		field_label = config.get('label', field_name)
		field_value = config.get('value', '')
		field_placeholder = config.get('placeholder', f'Enter {field_label.lower()}')
		field_required = config.get('required', False)
		field_disabled = config.get('disabled', False)
		field_onchange = config.get('onchange', '')
		field_pattern = config.get('pattern', '')

		# Build attributes
		attributes = []
		if field_required: attributes.append('required')
		if field_disabled: attributes.append('disabled')
		if field_onchange: attributes.append(f'onchange="{field_onchange}"')
		if field_pattern: attributes.append(f'pattern="{field_pattern}"')

		attributes_str = ' '.join(attributes)

		html = f"""
		<div class="input-field text-field">
		<label for="{field_id}">{field_label}{' *' if field_required else ''}</label>
		<input type="text"
		id="{field_id}"
		name="{field_name}"
		value="{field_value}"
		placeholder="{field_placeholder}"
		{attributes_str}>
		</div>
		"""

		return html

	def _create_select_field(self, config: Dict[str, Any]) -> str:
		"""Create select dropdown field"""
		field_id = config.get('id', '')
		field_name = config.get('name', '')
		field_label = config.get('label', field_name)
		field_value = config.get('value', '')
		field_options = config.get('options', [])
		field_required = config.get('required', False)
		field_disabled = config.get('disabled', False)
		field_onchange = config.get('onchange', '')

		# Build attributes
		attributes = []
		if field_required: attributes.append('required')
		if field_disabled: attributes.append('disabled')
		if field_onchange: attributes.append(f'onchange="{field_onchange}"')

		attributes_str = ' '.join(attributes)

		# Build options
		options_html = f'<option value="">Select {field_label.lower()}...</option>'
		for option in field_options:
			if isinstance(option, dict):
				option_value = option.get('value', '')
				option_label = option.get('label', option_value)
				option_selected = 'selected' if option_value == field_value else ''
				options_html += f'<option value="{option_value}" {option_selected}>{option_label}</option>'
			else:
				option_selected = 'selected' if option == field_value else ''
				options_html += f'<option value="{option}" {option_selected}>{option}</option>'

		html = f"""
		<div class="input-field select-field">
		<label for="{field_id}">{field_label}{' *' if field_required else ''}</label>
		<select id="{field_id}"
		name="{field_name}"
		{attributes_str}>
		{options_html}
		</select>
		</div>
		"""

		return html

	def _create_checkbox_field(self, config: Dict[str, Any]) -> str:
		"""Create checkbox field"""
		field_id = config.get('id', '')
		field_name = config.get('name', '')
		field_label = config.get('label', field_name)
		field_value = config.get('value', False)
		field_disabled = config.get('disabled', False)
		field_onchange = config.get('onchange', '')

		# Build attributes
		attributes = []
		if field_value: attributes.append('checked')
		if field_disabled: attributes.append('disabled')
		if field_onchange: attributes.append(f'onchange="{field_onchange}"')

		attributes_str = ' '.join(attributes)

		html = f"""
		<div class="input-field checkbox-field">
		<label for="{field_id}">
		<input type="checkbox"
		id="{field_id}"
		name="{field_name}"
		{attributes_str}>
		{field_label}
		</label>
		</div>
		"""

		return html

	def _create_range_field(self, config: Dict[str, Any]) -> str:
		"""Create range slider field"""
		field_id = config.get('id', '')
		field_name = config.get('name', '')
		field_label = config.get('label', field_name)
		field_value = config.get('value', 50)
		field_min = config.get('min', 0)
		field_max = config.get('max', 100)
		field_step = config.get('step', 1)
		field_disabled = config.get('disabled', False)
		field_onchange = config.get('onchange', '')

		# Build attributes
		attributes = []
		if field_disabled: attributes.append('disabled')
		if field_onchange: attributes.append(f'onchange="{field_onchange}"')

		attributes_str = ' '.join(attributes)

		html = f"""
		<div class="input-field range-field">
		<label for="{field_id}">{field_label}</label>
		<input type="range"
		id="{field_id}"
		name="{field_name}"
		value="{field_value}"
		min="{field_min}"
		max="{field_max}"
		step="{field_step}"
		{attributes_str}>
		<span class="range-value">{field_value}</span>
		</div>
		"""

		return html

	def _create_textarea_field(self, config: Dict[str, Any]) -> str:
		"""Create textarea field"""
		field_id = config.get('id', '')
		field_name = config.get('name', '')
		field_label = config.get('label', field_name)
		field_value = config.get('value', '')
		field_placeholder = config.get('placeholder', f'Enter {field_label.lower()}...')
		field_rows = config.get('rows', 4)
		field_cols = config.get('cols', 50)
		field_required = config.get('required', False)
		field_disabled = config.get('disabled', False)
		field_onchange = config.get('onchange', '')

		# Build attributes
		attributes = []
		if field_required: attributes.append('required')
		if field_disabled: attributes.append('disabled')
		if field_onchange: attributes.append(f'onchange="{field_onchange}"')

		attributes_str = ' '.join(attributes)

		html = f"""
		<div class="input-field textarea-field">
		<label for="{field_id}">{field_label}{' *' if field_required else ''}</label>
		<textarea id="{field_id}"
		name="{field_name}"
		rows="{field_rows}"
		cols="{field_cols}"
		placeholder="{field_placeholder}"
		{attributes_str}>{field_value}</textarea>
		</div>
		"""

		return html

	def _create_file_field(self, config: Dict[str, Any]) -> str:
		"""Create file input field"""
		field_id = config.get('id', '')
		field_name = config.get('name', '')
		field_label = config.get('label', field_name)
		field_accept = config.get('accept', '')
		field_multiple = config.get('multiple', False)
		field_required = config.get('required', False)
		field_disabled = config.get('disabled', False)
		field_onchange = config.get('onchange', '')

		# Build attributes
		attributes = []
		if field_accept: attributes.append(f'accept="{field_accept}"')
		if field_multiple: attributes.append('multiple')
		if field_required: attributes.append('required')
		if field_disabled: attributes.append('disabled')
		if field_onchange: attributes.append(f'onchange="{field_onchange}"')

		attributes_str = ' '.join(attributes)

		html = f"""
		<div class="input-field file-field">
		<label for="{field_id}">{field_label}{' *' if field_required else ''}</label>
		<input type="file"
		id="{field_id}"
		name="{field_name}"
		{attributes_str}>
		</div>
		"""

		return html

	def _create_color_field(self, config: Dict[str, Any]) -> str:
		"""Create color picker field"""
		field_id = config.get('id', '')
		field_name = config.get('name', '')
		field_label = config.get('label', field_name)
		field_value = config.get('value', '#000000')
		field_disabled = config.get('disabled', False)
		field_onchange = config.get('onchange', '')

		# Build attributes
		attributes = []
		if field_disabled: attributes.append('disabled')
		if field_onchange: attributes.append(f'onchange="{field_onchange}"')

		attributes_str = ' '.join(attributes)

		html = f"""
		<div class="input-field color-field">
		<label for="{field_id}">{field_label}</label>
		<input type="color"
		id="{field_id}"
		name="{field_name}"
		value="{field_value}"
		{attributes_str}>
		</div>
		"""

		return html

	def _create_date_field(self, config: Dict[str, Any]) -> str:
		"""Create date input field"""
		field_id = config.get('id', '')
		field_name = config.get('name', '')
		field_label = config.get('label', field_name)
		field_value = config.get('value', '')
		field_required = config.get('required', False)
		field_disabled = config.get('disabled', False)
		field_onchange = config.get('onchange', '')

		# Build attributes
		attributes = []
		if field_required: attributes.append('required')
		if field_disabled: attributes.append('disabled')
		if field_onchange: attributes.append(f'onchange="{field_onchange}"')

		attributes_str = ' '.join(attributes)

		html = f"""
		<div class="input-field date-field">
		<label for="{field_id}">{field_label}{' *' if field_required else ''}</label>
		<input type="date"
		id="{field_id}"
		name="{field_name}"
		value="{field_value}"
		{attributes_str}>
		</div>
		"""

		return html

	def _create_time_field(self, config: Dict[str, Any]) -> str:
		"""Create time input field"""
		field_id = config.get('id', '')
		field_name = config.get('name', '')
		field_label = config.get('label', field_name)
		field_value = config.get('value', '')
		field_required = config.get('required', False)
		field_disabled = config.get('disabled', False)
		field_onchange = config.get('onchange', '')

		# Build attributes
		attributes = []
		if field_required: attributes.append('required')
		if field_disabled: attributes.append('disabled')
		if field_onchange: attributes.append(f'onchange="{field_onchange}"')

		attributes_str = ' '.join(attributes)

		html = f"""
		<div class="input-field time-field">
		<label for="{field_id}">{field_label}{' *' if field_required else ''}</label>
		<input type="time"
		id="{field_id}"
		name="{field_name}"
		value="{field_value}"
		{attributes_str}>
		</div>
		"""

		return html

	def validate_field(self, field_config: Dict[str, Any], value: Any) -> Dict[str, Any]:
		"""
		Validate field value against validation rules

		Args:
		field_config: Field configuration
		value: Value to validate

		Returns:
		Validation result dictionary
		"""
		validation_rules = field_config.get('validation', [])
		result = {
			'valid': True,
			'errors': []
		}

		for rule in validation_rules:
			rule_name = rule.get('name', '')
			rule_value = rule.get('value', '')
			
			if rule_name in self.validation_rules:
				try:
					if not self.validation_rules[rule_name](value, rule_value):
						result['valid'] = False
						result['errors'].append(rule.get('message', f'Validation failed for {rule_name}'))
				except Exception as e:
					result['valid'] = False
					result['errors'].append(f'Validation error: {str(e)}')

		return result

	def create_form(self, form_config: Dict[str, Any]) -> str:
		"""
		Create a complete form with multiple fields

		Args:
		form_config: Form configuration dictionary

		Returns:
		HTML string for the complete form
		"""
		form_id = form_config.get('id', 'form')
		form_action = form_config.get('action', '')
		form_method = form_config.get('method', 'post')
		form_fields = form_config.get('fields', [])
		form_title = form_config.get('title', '')

		# Generate fields HTML
		fields_html = ''
		for field_config in form_fields:
			fields_html += self.create_field(field_config)

		# Generate form HTML
		html = f"""
		<form id="{form_id}" action="{form_action}" method="{form_method}">
		{f'<h3>{form_title}</h3>' if form_title else ''}
		{fields_html}
		<div class="form-actions">
		<button type="submit" class="btn btn-primary">Submit</button>
		<button type="reset" class="btn btn-secondary">Reset</button>
		</div>
		</form>
		"""

		return html