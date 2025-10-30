// Visualization JavaScript for FEM Simulation Frontend

// VTK.js utilities
const VTKUtils = {
    // Check if VTK.js is loaded
    isLoaded: function() {
        return typeof vtk !== 'undefined';
    },
    
    // Create VTK render window
    createRenderWindow: function(canvas, options = {}) {
        if (!this.isLoaded()) {
            throw new Error('VTK.js is not loaded');
        }
        
        const renderWindow = vtk.Rendering.Core.vtkRenderWindow.newInstance();
        const renderer = vtk.Rendering.Core.vtkRenderer.newInstance();
        const interactor = vtk.Rendering.Core.vtkRenderWindowInteractor.newInstance();
        
        // Configure renderer
        if (options.background) {
            renderer.setBackground(...this.hexToRgb(options.background));
        }
        
        // Configure interactor
        interactor.setView(canvas);
        interactor.initialize();
        interactor.bindEvents(canvas);
        
        // Set up render window
        renderWindow.addRenderer(renderer);
        renderWindow.setInteractor(interactor);
        
        return { renderWindow, renderer, interactor };
    },
    
    // Create polydata from mesh data
    createPolyData: function(meshData) {
        if (!this.isLoaded()) {
            throw new Error('VTK.js is not loaded');
        }
        
        const polydata = vtk.Common.DataModel.vtkPolyData.newInstance();
        
        // Set points
        if (meshData.vertices && meshData.vertices.length > 0) {
            const points = vtk.Common.Core.vtkPoints.newInstance();
            const flatVertices = meshData.vertices.flat();
            points.setData(new Float32Array(flatVertices), 3);
            polydata.setPoints(points);
        }
        
        // Set cells (faces)
        if (meshData.faces && meshData.faces.length > 0) {
            const cells = vtk.Common.DataModel.vtkCellArray.newInstance();
            
            for (let i = 0; i < meshData.faces.length; i++) {
                const face = meshData.faces[i];
                if (face.length === 3) {
                    // Triangle
                    cells.insertNextCell([face[0], face[1], face[2]]);
                } else if (face.length === 4) {
                    // Quad - convert to two triangles
                    cells.insertNextCell([face[0], face[1], face[2]]);
                    cells.insertNextCell([face[0], face[2], face[3]]);
                }
            }
            
            polydata.setPolys(cells);
        }
        
        return polydata;
    },
    
    // Add scalar field data to polydata
    addScalarField: function(polydata, fieldData) {
        if (!this.isLoaded()) {
            throw new Error('VTK.js is not loaded');
        }
        
        const scalars = vtk.Common.Core.vtkDataArray.newInstance({
            name: fieldData.name || 'Field',
            numberOfComponents: 1,
            values: new Float32Array(fieldData.values)
        });
        
        polydata.getPointData().setScalars(scalars);
        return scalars;
    },
    
    // Create color lookup table
    createLookupTable: function(fieldValues, colormap = 'viridis') {
        if (!this.isLoaded()) {
            throw new Error('VTK.js is not loaded');
        }
        
        const lookupTable = vtk.Rendering.Core.vtkColorTransferFunction.newInstance();
        const minVal = Math.min(...fieldValues);
        const maxVal = Math.max(...fieldValues);
        
        // Apply colormap
        switch (colormap) {
            case 'viridis':
                lookupTable.addRGBPoint(minVal, 0.267, 0.004, 0.329);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.2, 0.282, 0.140, 0.457);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.4, 0.253, 0.265, 0.529);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.6, 0.206, 0.371, 0.553);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.8, 0.163, 0.471, 0.558);
                lookupTable.addRGBPoint(maxVal, 0.993, 0.906, 0.143);
                break;
            case 'plasma':
                lookupTable.addRGBPoint(minVal, 0.050, 0.029, 0.527);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.2, 0.238, 0.024, 0.527);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.4, 0.459, 0.043, 0.486);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.6, 0.682, 0.134, 0.384);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.8, 0.926, 0.298, 0.207);
                lookupTable.addRGBPoint(maxVal, 0.988, 0.998, 0.645);
                break;
            case 'jet':
                lookupTable.addRGBPoint(minVal, 0.0, 0.0, 0.5);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.2, 0.0, 0.5, 1.0);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.4, 0.0, 1.0, 1.0);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.6, 0.5, 1.0, 0.5);
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.8, 1.0, 1.0, 0.0);
                lookupTable.addRGBPoint(maxVal, 1.0, 0.0, 0.0);
                break;
            default: // rainbow
                lookupTable.addRGBPoint(minVal, 0.0, 0.0, 1.0); // Blue
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.25, 0.0, 1.0, 1.0); // Cyan
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.5, 0.0, 1.0, 0.0); // Green
                lookupTable.addRGBPoint(minVal + (maxVal - minVal) * 0.75, 1.0, 1.0, 0.0); // Yellow
                lookupTable.addRGBPoint(maxVal, 1.0, 0.0, 0.0); // Red
        }
        
        return lookupTable;
    },
    
    // Create mesh actor
    createMeshActor: function(polydata, options = {}) {
        if (!this.isLoaded()) {
            throw new Error('VTK.js is not loaded');
        }
        
        const mapper = vtk.Rendering.Core.vtkMapper.newInstance();
        mapper.setInputData(polydata);
        
        const actor = vtk.Rendering.Core.vtkActor.newInstance();
        actor.setMapper(mapper);
        
        // Set properties
        const props = actor.getProperty();
        if (options.color) {
            props.setColor(...this.hexToRgb(options.color));
        }
        if (options.opacity !== undefined) {
            props.setOpacity(options.opacity);
        }
        if (options.wireframe !== undefined) {
            props.setRepresentation(options.wireframe ? 1 : 0);
        }
        
        return { actor, mapper };
    },
    
    // Create axes actor
    createAxesActor: function(options = {}) {
        if (!this.isLoaded()) {
            throw new Error('VTK.js is not loaded');
        }
        
        const axes = vtk.Rendering.Core.vtkAxesActor.newInstance();
        
        const size = options.size || 2;
        axes.setTotalLength(size, size, size);
        axes.setShaftType(options.shaftType || 0); // 0 = line, 1 = cylinder
        axes.setAxisLabels(options.showLabels ? 1 : 0);
        
        return axes;
    },
    
    // Convert hex color to RGB array
    hexToRgb: function(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? [
            parseInt(result[1], 16) / 255,
            parseInt(result[2], 16) / 255,
            parseInt(result[3], 16) / 255
        ] : [0, 0, 0];
    },
    
    // Handle window resize
    handleResize: function(renderWindow, canvas) {
        const container = canvas.parentElement;
        const size = container.getBoundingClientRect();
        
        canvas.width = size.width;
        canvas.height = size.height;
        
        renderWindow.getViews()[0].setSize(size.width, size.height);
        renderWindow.render();
    }
};

// VTK Mesh Visualization class
class VTKMeshVisualization {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            background: '#1a1a1a',
            meshColor: '#00ff00',
            wireframeColor: '#ffffff',
            showWireframe: false,
            showAxes: false,
            autoRotate: false,
            opacity: 0.8,
            ...options
        };
        
        this.renderWindow = null;
        this.renderer = null;
        this.interactor = null;
        this.meshActor = null;
        this.meshMapper = null;
        this.axesActor = null;
        this.isInitialized = false;
        this.autoRotateInterval = null;
        
        this.init();
    }
    
    init() {
        try {
            if (!VTKUtils.isLoaded()) {
                throw new Error('VTK.js is not loaded');
            }
            
            // Create VTK components
            const canvas = this.container.querySelector('#canvas') || this.container;
            const vtkComponents = VTKUtils.createRenderWindow(canvas, {
                background: this.options.background
            });
            
            this.renderWindow = vtkComponents.renderWindow;
            this.renderer = vtkComponents.renderer;
            this.interactor = vtkComponents.interactor;
            
            // Handle window resize
            window.addEventListener('resize', () => this.handleResize());
            
            console.log('VTK mesh visualization initialized');
            this.isInitialized = true;
        } catch (error) {
            console.error('Failed to initialize VTK mesh visualization:', error);
            throw error;
        }
    }
    
    loadMesh(meshData) {
        try {
            if (!this.isInitialized) {
                throw new Error('VTK visualization not initialized');
            }
            
            if (!meshData.vertices || meshData.vertices.length === 0) {
                throw new Error('No mesh data available');
            }
            
            // Remove existing mesh
            if (this.meshActor) {
                this.renderer.removeActor(this.meshActor);
            }
            
            // Create polydata
            const polydata = VTKUtils.createPolyData(meshData);
            
            // Create mesh actor
            const meshComponents = VTKUtils.createMeshActor(polydata, {
                color: this.options.meshColor,
                opacity: this.options.opacity,
                wireframe: this.options.showWireframe
            });
            
            this.meshActor = meshComponents.actor;
            this.meshMapper = meshComponents.mapper;
            
            // Add to renderer
            this.renderer.addActor(this.meshActor);
            
            // Reset camera to fit the data
            this.renderer.resetCamera();
            
            // Render
            this.renderWindow.render();
            
            console.log('Mesh loaded successfully');
        } catch (error) {
            console.error('Failed to load mesh:', error);
            throw error;
        }
    }
    
    loadField(meshData, fieldData) {
        try {
            if (!this.isInitialized) {
                throw new Error('VTK visualization not initialized');
            }
            
            // Create polydata with field data
            const polydata = VTKUtils.createPolyData(meshData);
            
            // Add scalar field
            VTKUtils.addScalarField(polydata, fieldData);
            
            // Create color lookup table
            const lookupTable = VTKUtils.createLookupTable(fieldData.values, this.options.colormap);
            
            // Create mesh actor with field visualization
            const meshComponents = VTKUtils.createMeshActor(polydata, {
                opacity: this.options.opacity,
                wireframe: this.options.showWireframe
            });
            
            this.meshActor = meshComponents.actor;
            this.meshMapper = meshComponents.mapper;
            
            // Configure mapper for scalar visualization
            this.meshMapper.setScalarVisibility(true);
            this.meshMapper.setScalarModeToUsePointData();
            this.meshMapper.setLookupTable(lookupTable);
            
            // Add to renderer
            this.renderer.addActor(this.meshActor);
            
            // Reset camera
            this.renderer.resetCamera();
            
            // Render
            this.renderWindow.render();
            
            console.log('Field visualization loaded successfully');
        } catch (error) {
            console.error('Failed to load field visualization:', error);
            throw error;
        }
    }
    
    toggleWireframe() {
        if (this.meshActor) {
            const props = this.meshActor.getProperty();
            const currentRep = props.getRepresentation();
            props.setRepresentation(currentRep === 0 ? 1 : 0);
            this.renderWindow.render();
        }
    }
    
    toggleAxes() {
        if (this.options.showAxes) {
            this.removeAxes();
            this.options.showAxes = false;
        } else {
            this.addAxes();
            this.options.showAxes = true;
        }
    }
    
    addAxes() {
        this.removeAxes(); // Remove existing axes
        
        this.axesActor = VTKUtils.createAxesActor({
            size: 2,
            shaftType: 0,
            showLabels: false
        });
        
        this.renderer.addActor(this.axesActor);
        this.renderWindow.render();
    }
    
    removeAxes() {
        if (this.axesActor) {
            this.renderer.removeActor(this.axesActor);
            this.axesActor = null;
            this.renderWindow.render();
        }
    }
    
    startAutoRotate() {
        if (this.autoRotateInterval) return;
        
        this.autoRotateInterval = setInterval(() => {
            if (this.renderer) {
                const camera = this.renderer.getActiveCamera();
                camera.azimuth(1);
                this.renderWindow.render();
            }
        }, 50); // 20 FPS
    }
    
    stopAutoRotate() {
        if (this.autoRotateInterval) {
            clearInterval(this.autoRotateInterval);
            this.autoRotateInterval = null;
        }
    }
    
    setBackground(color) {
        if (this.renderer) {
            this.renderer.setBackground(...VTKUtils.hexToRgb(color));
            this.renderWindow.render();
        }
    }
    
    updateOptions(newOptions) {
        this.options = { ...this.options, ...newOptions };
        
        if (this.renderer) {
            this.renderer.setBackground(...VTKUtils.hexToRgb(this.options.background));
        }
        
        if (this.meshActor) {
            const props = this.meshActor.getProperty();
            props.setColor(...VTKUtils.hexToRgb(this.options.meshColor));
            props.setOpacity(this.options.opacity);
            props.setRepresentation(this.options.showWireframe ? 1 : 0);
        }
    }
    
    handleResize() {
        if (this.isInitialized && this.renderWindow) {
            const canvas = this.container.querySelector('#canvas') || this.container;
            VTKUtils.handleResize(this.renderWindow, canvas);
        }
    }
    
    dispose() {
        this.stopAutoRotate();
        
        if (this.interactor) {
            this.interactor.delete();
        }
        
        window.removeEventListener('resize', this.handleResize);
        
        console.log('VTK mesh visualization disposed');
    }
}

// Legacy Three.js utilities (kept for compatibility)
const VisualizationUtils = {
    // Create Three.js scene
    createScene: function(background = '#1a1a1a') {
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(background);
        return scene;
    },
    
    // Create camera
    createCamera: function(width, height, position = [5, 5, 5]) {
        const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        camera.position.set(...position);
        return camera;
    },
    
    // Create renderer
    createRenderer: function(width, height, container) {
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        if (container) {
            container.appendChild(renderer.domElement);
        }
        
        return renderer;
    },
    
    // Create orbit controls
    createControls: function(camera, renderer) {
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        return controls;
    },
    
    // Create axes helper
    createAxes: function(size = 2) {
        return new THREE.AxesHelper(size);
    },
    
    // Create lighting
    createLighting: function() {
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(10, 10, 5);
        directionalLight.castShadow = true;
        
        return { ambientLight, directionalLight };
    }
};

// Mesh visualization class
class MeshVisualization {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            background: '#1a1a1a',
            meshColor: '#00ff00',
            wireframeColor: '#ffffff',
            showWireframe: true,
            showAxes: true,
            autoRotate: false,
            ...options
        };
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.mesh = null;
        this.wireframe = null;
        this.axes = null;
        this.animationId = null;
        
        this.init();
    }
    
    init() {
        try {
            const rect = this.container.getBoundingClientRect();
            
            // Create scene
            this.scene = VisualizationUtils.createScene(this.options.background);
            
            // Create camera
            this.camera = VisualizationUtils.createCamera(rect.width, rect.height);
            
            // Create renderer
            this.renderer = VisualizationUtils.createRenderer(rect.width, rect.height, this.container);
            
            // Create controls
            this.controls = VisualizationUtils.createControls(this.camera, this.renderer);
            
            // Create lighting
            const { ambientLight, directionalLight } = VisualizationUtils.createLighting();
            this.scene.add(ambientLight);
            this.scene.add(directionalLight);
            
            // Create axes
            this.axes = VisualizationUtils.createAxes();
            this.axes.visible = this.options.showAxes;
            this.scene.add(this.axes);
            
            // Handle window resize
            window.addEventListener('resize', () => this.handleResize());
            
            console.log('Mesh visualization initialized');
        } catch (error) {
            console.error('Failed to initialize mesh visualization:', error);
            throw error;
        }
    }
    
    loadMesh(meshData) {
        try {
            if (!meshData.vertices || meshData.vertices.length === 0) {
                throw new Error('No mesh data available');
            }
            
            // Remove existing mesh
            if (this.mesh) {
                this.scene.remove(this.mesh);
            }
            if (this.wireframe) {
                this.scene.remove(this.wireframe);
            }
            
            // Create geometry
            const geometry = new THREE.BufferGeometry();
            
            // Set vertices
            const vertices = new Float32Array(meshData.vertices.flat());
            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            
            // Set faces (if available)
            if (meshData.faces && meshData.faces.length > 0) {
                const indices = new Uint32Array(meshData.faces.flat());
                geometry.setIndex(new THREE.BufferAttribute(indices, 1));
            }
            
            // Compute normals
            geometry.computeVertexNormals();
            
            // Create material
            const material = new THREE.MeshLambertMaterial({
                color: this.options.meshColor,
                side: THREE.DoubleSide
            });
            
            // Create mesh
            this.mesh = new THREE.Mesh(geometry, material);
            this.mesh.castShadow = true;
            this.mesh.receiveShadow = true;
            this.scene.add(this.mesh);
            
            // Create wireframe
            const wireframeGeometry = geometry.clone();
            const wireframeMaterial = new THREE.LineBasicMaterial({
                color: this.options.wireframeColor,
                wireframe: true
            });
            this.wireframe = new THREE.LineSegments(
                new THREE.WireframeGeometry(wireframeGeometry),
                wireframeMaterial
            );
            this.wireframe.visible = this.options.showWireframe;
            this.scene.add(this.wireframe);
            
            // Center and scale mesh
            geometry.computeBoundingBox();
            const center = geometry.boundingBox.getCenter(new THREE.Vector3());
            const size = geometry.boundingBox.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2 / maxDim;
            
            this.mesh.scale.setScalar(scale);
            this.wireframe.scale.setScalar(scale);
            this.mesh.position.sub(center.multiplyScalar(scale));
            this.wireframe.position.copy(this.mesh.position);
            
            // Start animation
            this.animate();
            
            console.log('Mesh loaded successfully');
        } catch (error) {
            console.error('Failed to load mesh:', error);
            throw error;
        }
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        this.controls.update();
        
        if (this.options.autoRotate && this.mesh) {
            this.mesh.rotation.y += 0.005;
            this.wireframe.rotation.y += 0.005;
        }
        
        this.renderer.render(this.scene, this.camera);
    }
    
    handleResize() {
        const rect = this.container.getBoundingClientRect();
        
        this.camera.aspect = rect.width / rect.height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(rect.width, rect.height);
    }
    
    updateOptions(newOptions) {
        this.options = { ...this.options, ...newOptions };
        
        if (this.scene) {
            this.scene.background = new THREE.Color(this.options.background);
        }
        
        if (this.axes) {
            this.axes.visible = this.options.showAxes;
        }
        
        if (this.wireframe) {
            this.wireframe.visible = this.options.showWireframe;
        }
        
        if (this.mesh) {
            this.mesh.material.color.set(this.options.meshColor);
        }
    }
    
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        window.removeEventListener('resize', this.handleResize);
        
        console.log('Mesh visualization disposed');
    }
}

// Field visualization class
class FieldVisualization {
    constructor(container, options = {}) {
        this.container = container;
        this.options = {
            background: '#1a1a1a',
            colormap: 'viridis',
            showContours: true,
            showColorbar: true,
            opacity: 0.8,
            ...options
        };
        
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.mesh = null;
        this.contours = null;
        this.animationId = null;
        this.fieldMin = 0;
        this.fieldMax = 1;
        this.fieldMean = 0.5;
        
        this.colormaps = {
            viridis: ['#440154', '#482777', '#3f4a8a', '#31678e', '#26838f', 
                     '#1f9d8a', '#6cce5a', '#b6de2b', '#fee825'],
            plasma: ['#0c0786', '#6a00a8', '#b02a8f', '#e16462', '#fca636', '#f0f921'],
            inferno: ['#000004', '#1b0c42', '#4a0c6b', '#781c6d', '#a52c60', 
                     '#d14e53', '#f37651', '#feb078', '#f0f921'],
            magma: ['#000004', '#1c1044', '#4c1a6b', '#7c1d70', '#ad2268', 
                   '#dd4968', '#f66e5c', '#fca636', '#f0f921'],
            jet: ['#000080', '#0000ff', '#0080ff', '#00ffff', '#80ff00', 
                 '#ffff00', '#ff8000', '#ff0000', '#800000']
        };
        
        this.init();
    }
    
    init() {
        try {
            const rect = this.container.getBoundingClientRect();
            
            // Create scene
            this.scene = VisualizationUtils.createScene(this.options.background);
            
            // Create camera
            this.camera = VisualizationUtils.createCamera(rect.width, rect.height);
            
            // Create renderer
            this.renderer = VisualizationUtils.createRenderer(rect.width, rect.height, this.container);
            
            // Create controls
            this.controls = VisualizationUtils.createControls(this.camera, this.renderer);
            
            // Create lighting
            const { ambientLight, directionalLight } = VisualizationUtils.createLighting();
            this.scene.add(ambientLight);
            this.scene.add(directionalLight);
            
            // Handle window resize
            window.addEventListener('resize', () => this.handleResize());
            
            console.log('Field visualization initialized');
        } catch (error) {
            console.error('Failed to initialize field visualization:', error);
            throw error;
        }
    }
    
    loadField(meshData, fieldData) {
        try {
            if (!meshData.vertices || meshData.vertices.length === 0) {
                throw new Error('No mesh data available');
            }
            
            if (!fieldData.values || fieldData.values.length === 0) {
                throw new Error('No field data available');
            }
            
            // Calculate field statistics
            this.calculateFieldStats(fieldData.values);
            
            // Remove existing mesh
            if (this.mesh) {
                this.scene.remove(this.mesh);
            }
            if (this.contours) {
                this.scene.remove(this.contours);
            }
            
            // Create geometry
            const geometry = new THREE.BufferGeometry();
            
            // Set vertices
            const vertices = new Float32Array(meshData.vertices.flat());
            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            
            // Set faces (if available)
            if (meshData.faces && meshData.faces.length > 0) {
                const indices = new Uint32Array(meshData.faces.flat());
                geometry.setIndex(new THREE.BufferAttribute(indices, 1));
            }
            
            // Compute normals
            geometry.computeVertexNormals();
            
            // Create colors based on field values
            const colors = this.createFieldColors(fieldData.values);
            geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
            
            // Create material
            const material = new THREE.MeshLambertMaterial({
                vertexColors: true,
                transparent: true,
                opacity: this.options.opacity
            });
            
            // Create mesh
            this.mesh = new THREE.Mesh(geometry, material);
            this.mesh.castShadow = true;
            this.mesh.receiveShadow = true;
            this.scene.add(this.mesh);
            
            // Create contours if enabled
            if (this.options.showContours) {
                this.createContours(geometry);
            }
            
            // Center and scale mesh
            geometry.computeBoundingBox();
            const center = geometry.boundingBox.getCenter(new THREE.Vector3());
            const size = geometry.boundingBox.getSize(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z);
            const scale = 2 / maxDim;
            
            this.mesh.scale.setScalar(scale);
            this.mesh.position.sub(center.multiplyScalar(scale));
            
            // Start animation
            this.animate();
            
            console.log('Field loaded successfully');
        } catch (error) {
            console.error('Failed to load field:', error);
            throw error;
        }
    }
    
    calculateFieldStats(values) {
        this.fieldMin = Math.min(...values);
        this.fieldMax = Math.max(...values);
        this.fieldMean = values.reduce((a, b) => a + b, 0) / values.length;
    }
    
    createFieldColors(values) {
        const colors = new Float32Array(values.length * 3);
        const colormap = this.colormaps[this.options.colormap];
        
        for (let i = 0; i < values.length; i++) {
            const value = values[i];
            const normalized = (value - this.fieldMin) / (this.fieldMax - this.fieldMin);
            const color = this.getColorFromColormap(normalized, colormap);
            
            colors[i * 3] = color.r;
            colors[i * 3 + 1] = color.g;
            colors[i * 3 + 2] = color.b;
        }
        
        return colors;
    }
    
    getColorFromColormap(t, colormap) {
        const n = colormap.length - 1;
        const scaled = t * n;
        const index = Math.floor(scaled);
        const fraction = scaled - index;
        
        if (index >= n) {
            return this.hexToRgb(colormap[n]);
        }
        
        const color1 = this.hexToRgb(colormap[index]);
        const color2 = this.hexToRgb(colormap[index + 1]);
        
        return {
            r: color1.r + fraction * (color2.r - color1.r),
            g: color1.g + fraction * (color2.g - color1.g),
            b: color1.b + fraction * (color2.b - color1.b)
        };
    }
    
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16) / 255,
            g: parseInt(result[2], 16) / 255,
            b: parseInt(result[3], 16) / 255
        } : {r: 0, g: 0, b: 0};
    }
    
    createContours(geometry) {
        const wireframeGeometry = geometry.clone();
        const wireframeMaterial = new THREE.LineBasicMaterial({
            color: 0xffffff,
            wireframe: true,
            transparent: true,
            opacity: 0.3
        });
        this.contours = new THREE.LineSegments(
            new THREE.WireframeGeometry(wireframeGeometry),
            wireframeMaterial
        );
        this.contours.visible = this.options.showContours;
        this.scene.add(this.contours);
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    handleResize() {
        const rect = this.container.getBoundingClientRect();
        
        this.camera.aspect = rect.width / rect.height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(rect.width, rect.height);
    }
    
    updateOptions(newOptions) {
        this.options = { ...this.options, ...newOptions };
        
        if (this.scene) {
            this.scene.background = new THREE.Color(this.options.background);
        }
        
        if (this.mesh) {
            this.mesh.material.opacity = this.options.opacity;
        }
        
        if (this.contours) {
            this.contours.visible = this.options.showContours;
        }
    }
    
    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        if (this.renderer) {
            this.renderer.dispose();
        }
        
        window.removeEventListener('resize', this.handleResize);
        
        console.log('Field visualization disposed');
    }
}


// Export classes for global access
window.VisualizationUtils = VisualizationUtils;
window.MeshVisualization = MeshVisualization;
window.FieldVisualization = FieldVisualization;
