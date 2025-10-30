// Main JavaScript for FEM Simulation Frontend

// Global variables
let currentContext = {};
let isProcessing = false;

// API Configuration
const API_BASE_URL = window.location.origin;

// Lightweight logging (disabled by default for clean console)
const DEBUG = false;
const log = (...args) => { if (DEBUG) console.log(...args); };

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
	log('Frontend initialized');
    initializeFrontend();
    setupEventListeners();
    ensureMeshContainer();
});

// Initialize frontend components
function initializeFrontend() {
    // Set up initial UI state
    updateUIState('ready');
    
    // Initialize mesh container
    ensureMeshContainer();
    
    log('Frontend components initialized');
}

// Setup event listeners
function setupEventListeners() {
    // Command submission
    const commandForm = document.querySelector('#command-form');
    if (commandForm) {
        commandForm.addEventListener('submit', handleCommandSubmit);
    }
    
    // Window events
    window.addEventListener('resize', handleWindowResize);
    window.addEventListener('beforeunload', handleBeforeUnload);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
}

// Ensure mesh container exists
function ensureMeshContainer() {
    const centerContent = document.getElementById('centerContent');
    if (centerContent && !document.getElementById('meshContainer')) {
        const meshContainer = document.createElement('div');
        meshContainer.id = 'meshContainer';
        meshContainer.style.cssText = 'width: 100%; height: 100%; position: relative;';
        centerContent.appendChild(meshContainer);
        log('Mesh container created');
    }
}

// Handle command submission
async function handleCommandSubmit(event) {
    event.preventDefault();
    
    const commandInput = document.querySelector('#command-input');
    if (!commandInput) return;
    
    const prompt = commandInput.value.trim();
    if (!prompt) return;
    
    log('Processing command:', prompt);
    
    try {
        isProcessing = true;
        updateUIState('processing');
        
        // Send to simulation parser
        const response = await fetch('/simulation/parse', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                context: currentContext
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        log('Received response:', data);
        
        // Display results
        displayResults(data);
        
        // Clear input
        commandInput.value = '';
        
    } catch (error) {
        console.error('Error processing command:', error);
        displayError('Failed to process command: ' + error.message);
    } finally {
        isProcessing = false;
        updateUIState('ready');
    }
}

// Display simulation results
function displayResults(data) {
    log('Displaying results:', data);
    
    // Store context for editing
    currentContext = data.context || {};
    
    // Update right panel with configuration
    refreshRightPanel();
    
    // Display mesh if available
    if (data.mesh_visualization_url) {
        displayMeshVisualization(data.mesh_visualization_url);
    } else if (currentContext.geometry_type && currentContext.geometry_dimensions) {
        // Generate mesh preview if we have geometry info
        generateMeshPreviewInCenter();
    }
    
    // Display field visualization if available
    if (data.field_visualization_url) {
        displayFieldVisualization(data.field_visualization_url);
    }
    
    // Show success message
    if (data.success) {
        showNotification('Simulation completed successfully!', 'success');
    }
}

// Refresh right panel with current context
function refreshRightPanel() {
    const rightPanel = document.querySelector('.right-panel');
    if (!rightPanel) return;
    
    log('Refreshing right panel with context:', currentContext);
    
    // Generate configuration panels HTML
    const configHTML = generateConfigurationPanels(currentContext);
    
    // Update right panel content
    rightPanel.innerHTML = `
        <div class="context-section">
            <h3>Simulation Context</h3>
            ${configHTML}
        </div>
    `;
    
    // Re-attach event listeners
    attachConfigEventListeners();
}

// Generate configuration panels HTML
function generateConfigurationPanels(context) {
    let html = '';
    
    // Physics Type
    html += `
        <div class="config-group">
            <label for="physics-type">Physics Type:</label>
            <select id="physics-type" onchange="updateConfig('physics_type', this.value)">
                <option value="">Select physics...</option>
                <option value="heat_transfer" ${context.physics_type === 'heat_transfer' ? 'selected' : ''}>Heat Transfer</option>
                <option value="solid_mechanics" ${context.physics_type === 'solid_mechanics' ? 'selected' : ''}>Solid Mechanics</option>
            </select>
        </div>
    `;
    
    // Geometry Type
    html += `
        <div class="config-group">
            <label for="geometry-type">Geometry Type:</label>
            <select id="geometry-type" onchange="updateConfig('geometry_type', this.value)">
                <option value="">Select geometry...</option>
                <option value="line" ${context.geometry_type === 'line' ? 'selected' : ''}>Line (1D)</option>
                <option value="plate" ${context.geometry_type === 'plate' ? 'selected' : ''}>Plate (2D)</option>
                <option value="membrane" ${context.geometry_type === 'membrane' ? 'selected' : ''}>Membrane (2D)</option>
                <option value="disc" ${context.geometry_type === 'disc' ? 'selected' : ''}>Disc (2D)</option>
                <option value="rectangle" ${context.geometry_type === 'rectangle' ? 'selected' : ''}>Rectangle (2D)</option>
                <option value="cube" ${context.geometry_type === 'cube' ? 'selected' : ''}>Cube (3D)</option>
                <option value="box" ${context.geometry_type === 'box' ? 'selected' : ''}>Box (3D)</option>
                <option value="beam" ${context.geometry_type === 'beam' ? 'selected' : ''}>Beam (3D)</option>
                <option value="cylinder" ${context.geometry_type === 'cylinder' ? 'selected' : ''}>Cylinder (3D)</option>
                <option value="sphere" ${context.geometry_type === 'sphere' ? 'selected' : ''}>Sphere (3D)</option>
                <option value="solid" ${context.geometry_type === 'solid' ? 'selected' : ''}>Solid (3D)</option>
            </select>
        </div>
    `;
    
    // Geometry Dimensions
    if (context.geometry_type) {
        const dimensionFields = getDimensionFieldsForGeometry(context.geometry_type);
        dimensionFields.forEach(field => {
            const value = context.geometry_dimensions?.[field.name] || '';
            html += `
                <div class="config-group">
                    <label for="dim-${field.name}">${field.label} (${field.unit}):</label>
                    <input type="number" 
                           id="dim-${field.name}" 
                           step="any" 
                           placeholder="Enter ${field.label.toLowerCase()}"
                           value="${value}"
                           onchange="updateConfig('geometry_dimensions.${field.name}', this.value)">
                </div>
            `;
        });
    }
    
    // Material Type
    html += `
        <div class="config-group">
            <label for="material-type">Material Type:</label>
            <select id="material-type" onchange="updateConfig('material_type', this.value)">
                <option value="">Select material...</option>
                <option value="aluminum" ${context.material_type === 'aluminum' ? 'selected' : ''}>Aluminum</option>
                <option value="steel" ${context.material_type === 'steel' ? 'selected' : ''}>Steel</option>
                <option value="copper" ${context.material_type === 'copper' ? 'selected' : ''}>Copper</option>
                <option value="concrete" ${context.material_type === 'concrete' ? 'selected' : ''}>Concrete</option>
                <option value="wood" ${context.material_type === 'wood' ? 'selected' : ''}>Wood</option>
                <option value="plastic" ${context.material_type === 'plastic' ? 'selected' : ''}>Plastic</option>
                <option value="custom" ${context.material_type === 'custom' ? 'selected' : ''}>Custom</option>
            </select>
        </div>
    `;
    
    return html;
}

// Get dimension fields for geometry type
function getDimensionFieldsForGeometry(geometryType) {
    const dimensionMapping = {
        'line': [{name: 'length', label: 'Length', unit: 'm'}],
        'plate': [
            {name: 'length', label: 'Length', unit: 'm'},
            {name: 'width', label: 'Width', unit: 'm'},
            {name: 'thickness', label: 'Thickness', unit: 'm'}
        ],
        'membrane': [
            {name: 'length', label: 'Length', unit: 'm'},
            {name: 'width', label: 'Width', unit: 'm'}
        ],
        'disc': [{name: 'radius', label: 'Radius', unit: 'm'}],
        'rectangle': [
            {name: 'length', label: 'Length', unit: 'm'},
            {name: 'width', label: 'Width', unit: 'm'}
        ],
        'cube': [{name: 'length', label: 'Side Length', unit: 'm'}],
        'box': [
            {name: 'length', label: 'Length', unit: 'm'},
            {name: 'width', label: 'Width', unit: 'm'},
            {name: 'height', label: 'Height', unit: 'm'}
        ],
        'beam': [
            {name: 'length', label: 'Length', unit: 'm'},
            {name: 'width', label: 'Width', unit: 'm'},
            {name: 'height', label: 'Height', unit: 'm'}
        ],
        'cylinder': [
            {name: 'radius', label: 'Radius', unit: 'm'},
            {name: 'length', label: 'Length', unit: 'm'}
        ],
        'sphere': [{name: 'radius', label: 'Radius', unit: 'm'}],
        'solid': [
            {name: 'length', label: 'Length', unit: 'm'},
            {name: 'width', label: 'Width', unit: 'm'},
            {name: 'height', label: 'Height', unit: 'm'}
        ]
    };
    
    return dimensionMapping[geometryType] || [];
}

// Get default dimensions for geometry type
function getDefaultDimensionsForGeometry(geometryType) {
    const defaultDimensions = {
        'line': {length: 1.0},
        'plate': {length: 1.0, width: 1.0, thickness: 0.1},
        'membrane': {length: 1.0, width: 1.0},
        'disc': {radius: 0.5},
        'rectangle': {length: 1.0, width: 1.0},
        'cube': {length: 1.0},
        'box': {length: 1.0, width: 1.0, height: 1.0},
        'beam': {length: 1.0, width: 0.1, height: 0.1},
        'cylinder': {radius: 0.5, length: 1.0},
        'sphere': {radius: 0.5},
        'solid': {length: 1.0, width: 1.0, height: 1.0},
        'rod': {length: 1.0}
    };
    
    return defaultDimensions[geometryType] || {};
}

// Update configuration
function updateConfig(key, value) {
    log(`Updating config: ${key} = ${value}`);
    
    // Parse the key to update nested objects
    const keys = key.split('.');
    let current = currentContext;
    
    // Navigate to the parent object
    for (let i = 0; i < keys.length - 1; i++) {
        if (!current[keys[i]]) {
            current[keys[i]] = {};
        }
        current = current[keys[i]];
    }
    
    // Set the value
    current[keys[keys.length - 1]] = value;
    
    // Handle geometry type change - update dimensions immediately
    if (key === 'geometry_type') {
        log(`Geometry type changed to: ${value}`);
        
        // Clear existing dimensions
        currentContext.geometry_dimensions = {};
        
        // Set default dimensions for the new geometry type
        const defaultDimensions = getDefaultDimensionsForGeometry(value);
        if (defaultDimensions) {
            currentContext.geometry_dimensions = defaultDimensions;
            log(`Set default dimensions for ${value}:`, defaultDimensions);
        }
        
        // Immediately refresh the right panel to show new dimension fields
        setTimeout(() => {
            refreshRightPanel();
        }, 50);
    }
    
    // Trigger mesh update if geometry changed
    if (key.startsWith('geometry_type') || key.startsWith('geometry_dimensions')) {
        setTimeout(() => {
            generateMeshPreviewInCenter();
        }, 500);
    }
    
    // Refresh right panel if needed
    if (key.startsWith('material_type')) {
        setTimeout(() => {
            refreshRightPanel();
        }, 100);
    }
}

// Generate mesh preview in center panel
async function generateMeshPreviewInCenter() {
    if (!currentContext.geometry_type || !currentContext.geometry_dimensions) {
        log('Insufficient geometry information for mesh preview');
        return;
    }
    
    // Validate dimensions
    const dimensions = currentContext.geometry_dimensions;
    const hasValidDimensions = Object.values(dimensions).some(value => 
        value && !isNaN(parseFloat(value)) && parseFloat(value) > 0
    );
    
    if (!hasValidDimensions) {
        log('No valid dimensions found');
        return;
    }
    
    log('Generating mesh preview with:', {
        geometry_type: currentContext.geometry_type,
        dimensions: dimensions
    });
    
    try {
        // Send request to mesh preview endpoint
        const response = await fetch('/mesh/preview', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                geometry_type: currentContext.geometry_type,
                dimensions: dimensions
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        log('Mesh preview response:', data);
        
        if (data.success && data.mesh_visualization_url) {
            displayMeshVisualization(data.mesh_visualization_url);
        } else {
            console.error('Mesh preview failed:', data.error || 'Unknown error');
            displayError('Failed to generate mesh preview: ' + (data.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Error generating mesh preview:', error);
        displayError('Error generating mesh preview: ' + error.message);
    }
}

// Display mesh visualization
function displayMeshVisualization(url) {
    const centerContent = document.getElementById('centerContent');
    
    // Hide placeholder
    const placeholder = document.getElementById('placeholder');
    if (placeholder) {
        placeholder.style.display = 'none';
    }
    
    // Ensure mesh container exists
    let meshContainer = document.getElementById('meshContainer');
    if (!meshContainer) {
        meshContainer = document.createElement('div');
        meshContainer.id = 'meshContainer';
        meshContainer.style.cssText = 'width: 100%; height: 100%; position: relative;';
        centerContent.appendChild(meshContainer);
    }
    
    // Show mesh container
    meshContainer.style.display = 'block';
    
    // Create or update iframe
    let iframe = meshContainer.querySelector('iframe');
    if (!iframe) {
        iframe = document.createElement('iframe');
        iframe.style.cssText = 'width: 100%; height: 100%; border: none; border-radius: 8px;';
        meshContainer.appendChild(iframe);
    }
    
    // Update iframe source
    iframe.src = url;
    
    // Add success indicator
    iframe.style.border = '2px solid #28a745';
    
    log('VTK mesh visualization updated:', url);
}

// Display field visualization
function displayFieldVisualization(url) {
    const centerContent = document.getElementById('centerContent');
    
    // Ensure mesh container exists
    let meshContainer = document.getElementById('meshContainer');
    if (!meshContainer) {
        meshContainer = document.createElement('div');
        meshContainer.id = 'meshContainer';
        meshContainer.style.cssText = 'width: 100%; height: 100%; position: relative;';
        centerContent.appendChild(meshContainer);
    }
    
    // Create or update iframe
    let iframe = meshContainer.querySelector('iframe');
    if (!iframe) {
        iframe = document.createElement('iframe');
        iframe.style.cssText = 'width: 100%; height: 100%; border: none; border-radius: 8px;';
        meshContainer.appendChild(iframe);
    }
    
    // Update iframe source
    iframe.src = url;
    
    // Add field visualization indicator
    iframe.style.border = '2px solid #007bff';
    
    log('Field visualization updated:', url);
}

// Solve PDE
async function solvePDE() {
    if (isProcessing) return;
    
    log('Solving PDE with context:', currentContext);
    
    try {
        isProcessing = true;
        updateUIState('processing');
        
        const response = await fetch('/simulation/solve', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                context: currentContext
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        log('PDE solution response:', data);
        
        if (data.success) {
            displayResults(data);
            showNotification('PDE solved successfully!', 'success');
        } else {
            displayError('Failed to solve PDE: ' + (data.error || 'Unknown error'));
        }
        
    } catch (error) {
        console.error('Error solving PDE:', error);
        displayError('Error solving PDE: ' + error.message);
    } finally {
        isProcessing = false;
        updateUIState('ready');
    }
}

// Attach configuration event listeners
function attachConfigEventListeners() {
    // This function would attach specific event listeners to configuration elements
    // For now, we rely on inline event handlers in the generated HTML
}

// Update UI state
function updateUIState(state) {
    const submitBtn = document.querySelector('.submit-btn');
    const commandInput = document.querySelector('#command-input');
    
    switch (state) {
        case 'ready':
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = 'Submit';
            }
            if (commandInput) {
                commandInput.disabled = false;
            }
            break;
            
        case 'processing':
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.textContent = 'Processing...';
            }
            if (commandInput) {
                commandInput.disabled = true;
            }
            break;
    }
}

// Display error message
function displayError(message) {
    console.error(message);
    showNotification(message, 'error');
    
    // Display error in center panel
    const centerContent = document.getElementById('centerContent');
    if (centerContent) {
        centerContent.innerHTML = `
            <div class="error">
                <h3>Error</h3>
                <p>${message}</p>
            </div>
        `;
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Style the notification
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 4px;
        color: white;
        font-weight: bold;
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    
    // Set background color based on type
    switch (type) {
        case 'success':
            notification.style.backgroundColor = '#28a745';
            break;
        case 'error':
            notification.style.backgroundColor = '#dc3545';
            break;
        case 'warning':
            notification.style.backgroundColor = '#ffc107';
            notification.style.color = '#000';
            break;
        default:
            notification.style.backgroundColor = '#17a2b8';
    }
    
    // Add to document
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

// Handle window resize
function handleWindowResize() {
    // Adjust layout if needed
    log('Window resized');
}

// Handle before unload
function handleBeforeUnload(event) {
    if (isProcessing) {
        event.preventDefault();
        event.returnValue = 'Simulation is still processing. Are you sure you want to leave?';
        return event.returnValue;
    }
}

// Handle keyboard shortcuts
function handleKeyboardShortcuts(event) {
    // Ctrl/Cmd + Enter to submit
    if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        const commandForm = document.querySelector('#command-form');
        if (commandForm) {
            commandForm.dispatchEvent(new Event('submit'));
        }
    }
    
    // Escape to clear input
    if (event.key === 'Escape') {
        const commandInput = document.querySelector('#command-input');
        if (commandInput) {
            commandInput.value = '';
        }
    }
}

// Utility functions
function openInNewTab(url) {
    window.open(url, '_blank');
}

function clearResults() {
    const centerContent = document.getElementById('centerContent');
    if (centerContent) {
        // Preserve mesh container if it exists
        const meshContainer = document.getElementById('meshContainer');
        if (meshContainer) {
            centerContent.innerHTML = '';
            centerContent.appendChild(meshContainer);
        } else {
            centerContent.innerHTML = `
                <div class="placeholder">
                    <h3>FEM Simulation Interface</h3>
                    <p>Enter a physics simulation prompt to get started.</p>
                </div>
            `;
        }
    }
}

// Export functions for global access
window.updateConfig = updateConfig;
window.generateMeshPreviewInCenter = generateMeshPreviewInCenter;
window.displayMeshVisualization = displayMeshVisualization;
window.displayFieldVisualization = displayFieldVisualization;
window.solvePDE = solvePDE;
window.openInNewTab = openInNewTab;
window.clearResults = clearResults;
window.displayResults = displayResults;
window.refreshRightPanel = refreshRightPanel;
