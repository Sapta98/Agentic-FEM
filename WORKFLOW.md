# Workflow

## Overview

This document describes the workflow for the Agentic FEM application, including the complete simulation workflow, parsing workflow, and integration workflows.

## Complete Simulation Workflow

### Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Prompt                               │
│                  (Natural Language)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Parsing Workflow                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │ Physics  │  │ Geometry │  │ Material │  (Parallel)    │
│  │  Agent   │  │  Agent   │  │  Agent   │                │
│  └──────────┘  └──────────┘  └──────────┘                │
│         │            │            │                        │
│         └────────────┴────────────┘                        │
│                         │                                   │
│                         ▼                                   │
│  ┌──────────┐  ┌──────────┐                              │
│  │Dimension │  │Boundary  │  (Sequential)                │
│  │  Agent   │  │  Agent   │                              │
│  └──────────┘  └──────────┘                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Simulation Context                                │
│  - Physics Type                                             │
│  - Geometry Type & Dimensions                               │
│  - Material Properties                                      │
│  - Boundary Conditions                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Mesh Generation                                 │
│                  Mesh Agent                                  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              PDE Solving                                     │
│                 Solver Agent                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Visualization                                   │
│              Field Visualizer                                │
└─────────────────────────────────────────────────────────────┘
```

### Workflow Steps

1. **User Input**: User provides natural language prompt
2. **Parsing**: ParserAgent orchestrates specialized agents to extract simulation context
3. **Mesh Generation**: MeshAgent generates mesh from geometry
4. **PDE Solving**: SolverAgent solves PDE using FEniCS
5. **Visualization**: FieldVisualizer creates visualization

## Parsing Workflow

### Stage A: Primary Identification (Parallel)

**Agents**: PhysicsAgent, GeometryAgent, MaterialAgent

**Execution**: Parallel execution using ThreadPoolExecutor

**Purpose**: Identify physics type, geometry type, and material type simultaneously

**Workflow**:
```python
# Parallel execution
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = {
        "physics": executor.submit(physics_agent.identify_physics_type, prompt),
        "geometry": executor.submit(geometry_agent.identify_geometry, prompt),
        "material": executor.submit(material_agent.identify_material, prompt),
    }
    results = {key: future.result() for key, future in futures.items()}
```

### Stage B: Build Expectations

**Purpose**: Determine required dimensions and available boundaries

**Workflow**:
```python
# Get required dimensions from template
required_dimensions = template_manager.get_geometry_dimension_requirements(
    physics_type, geometry_type
)

# Get available boundaries from geometry
available_boundaries = context_parser._get_available_boundaries(geometry_type)
```

### Stage C: Clean Prompt

**Purpose**: Remove already-identified sections from prompt

**Workflow**:
```python
# Remove geometry, physics, material, and dimension mentions
cleaned_prompt = context_parser._extract_and_remove_parsed_sections(
    prompt, geometry_type, physics_type, material_type, dimensions
)
```

### Stage D: Extract Dimensions and Boundary Conditions

**Agents**: DimensionAgent, BoundaryConditionAgent

**Execution**: Sequential execution

**Purpose**: Extract dimensions and boundary conditions from cleaned prompt

**Workflow**:
```python
# Extract dimensions
dimensions_result = dimension_agent.execute_task(
    "extract_dimensions",
    {"prompt": cleaned_prompt, "geometry_type": geometry_type, "required_dimensions": required_dimensions}
)

# Extract boundary conditions
boundary_result = boundary_agent.execute_task(
    "extract_boundary_conditions",
    {"prompt": cleaned_prompt, "physics_type": physics_type, "geometry_type": geometry_type}
)
```

### Stage E: Determine Completeness

**Purpose**: Check if simulation context is complete

**Workflow**:
```python
# Check completeness
completeness = context_parser.check_completeness(physics_type, context)

# Build response
if completeness.get("complete"):
    action = "simulation_ready"
    simulation_config = context_parser._create_simulation_config()
else:
    action = "request_info"
    missing = completeness.get("missing", [])
```

## Mesh Generation Workflow

### Workflow Steps

1. **Geometry Validation**: Validate geometry type and dimensions
2. **Mesh Generator Selection**: Select appropriate mesh generator
3. **Mesh Generation**: Generate mesh using GMSH
4. **Mesh Validation**: Validate mesh quality
5. **Mesh Visualization**: Create mesh visualization

### Workflow

```python
# Mesh generation workflow
def generate_mesh(context):
    # Step 1: Validate geometry
    geometry_type = context.get("geometry_type")
    dimensions = context.get("geometry_dimensions")
    
    # Step 2: Select mesh generator
    mesh_generator = mesh_viewer.mesh_generator
    generator = mesh_generator.get_generator(geometry_type, dimensions)
    
    # Step 3: Generate mesh
    mesh_data = generator.generate_mesh(dimensions)
    
    # Step 4: Validate mesh
    if not mesh_data.get("success"):
        return {"success": False, "error": "Mesh generation failed"}
    
    # Step 5: Create visualization
    mesh_visualization_url = mesh_viewer.create_visualization(mesh_data)
    
    return {
        "success": True,
        "mesh_data": mesh_data,
        "mesh_visualization_url": mesh_visualization_url
    }
```

## PDE Solving Workflow

### Workflow Steps

1. **Build Simulation Config**: Build simulation configuration from context
2. **Validate Config**: Validate simulation configuration
3. **Solve PDE**: Solve PDE using FEniCS
4. **Extract Solution Data**: Extract solution data from FEniCS solution
5. **Create Visualization**: Create field visualization

### Workflow

```python
# PDE solving workflow
def solve_pde(context):
    # Step 1: Build simulation config
    simulation_config = build_simulation_config(context)
    
    # Step 2: Validate config
    if not validate_config(simulation_config):
        return {"success": False, "error": "Invalid configuration"}
    
    # Step 3: Solve PDE
    solver = fenics_solver
    solution_result = solver.solve(simulation_config)
    
    # Step 4: Extract solution data
    solution_data = solver.extract_solution_data(solution_result)
    
    # Step 5: Create visualization
    field_visualization_url = field_visualizer.create_visualization(
        mesh_data, solution_data
    )
    
    return {
        "success": True,
        "solution_data": solution_data,
        "field_visualization_url": field_visualization_url
    }
```

## Integration Workflows

### Feature 1: Transient Heat Transfer Workflow

**Purpose**: Handle transient heat transfer with time-stepping

**Workflow**:
1. **User Input**: User specifies transient heat transfer in prompt
2. **Parser**: Detects transient mode and sets `time_stepping` config
3. **Solver**: Detects transient mode and calls `_solve_heat_transfer_transient()`
4. **Time-Stepping**: Solver solves at multiple time steps
5. **Solution Data**: Returns time-series solution data
6. **Visualization**: Field visualizer detects transient flag and shows time controls
7. **UI**: User can play/pause/seek through time steps

### Feature 2: Deflection Visualization Workflow

**Purpose**: Display deflection for solid mechanics

**Workflow**:
1. **User Input**: User specifies solid mechanics simulation
2. **Solver**: Computes displacement vector field
3. **Solution Data**: Computes deflection as magnitude
4. **Visualization**: Field visualizer displays "Deflection Field"
5. **UI**: User sees "Deflection Field" in dropdown and visualization

### Feature 3: Field Type Display Workflow

**Purpose**: Display field type in UI

**Workflow**:
1. **Solution Data**: Solver includes `field_name` in solution data
2. **Field Visualizer**: Extracts field name and passes to template
3. **Template**: Displays "{field_name} Field" in dropdown
4. **UI**: User sees specific field type in dropdown and visualization title

## Agent Communication Workflow

### AgentBus Communication

**Purpose**: Agents communicate via AgentBus (Event Bus Pattern)

**Workflow**:
1. **Agent sends message**: Agent publishes message to AgentBus
2. **AgentBus routes message**: AgentBus routes message to subscribed agents
3. **Agent receives message**: Subscribed agents receive message
4. **Agent processes message**: Agent processes message and responds
5. **Agent sends response**: Agent sends response via AgentBus

### Message Types

1. **Task Request**: Request for task execution
2. **Task Result**: Result of task execution
3. **State Update**: Update to agent state
4. **Coordination**: Coordination message between agents
5. **Error**: Error message

### Workflow Example

```python
# Agent communication workflow
def execute_task(self, task, context):
    # Step 1: Send task request
    self._send_message(
        receiver="master_agent",
        message_type=MessageType.TASK_REQUEST,
        payload={"task": task, "context": context}
    )
    
    # Step 2: Wait for response
    response = self._wait_for_response(correlation_id)
    
    # Step 3: Process response
    if response.get("success"):
        return response.get("result")
    else:
        return {"success": False, "error": response.get("error")}
```

## Error Handling Workflow

### Error Handling Strategy

**Purpose**: Handle errors gracefully and provide recovery mechanisms

**Workflow**:
1. **Error Detection**: Detect error in agent execution
2. **Error Isolation**: Isolate error to specific agent
3. **Error Reporting**: Report error via AgentBus
4. **Error Recovery**: Attempt to recover from error
5. **Fallback Mechanism**: Use fallback mechanism if recovery fails

### Error Handling Example

```python
# Error handling workflow
def execute_task(self, task, context):
    try:
        # Execute task
        result = self._execute_task_internal(task, context)
        return {"success": True, "result": result}
    except Exception as e:
        # Error detection
        logger.error(f"Error in {self.name}: {e}")
        
        # Error isolation
        error_message = {
            "agent": self.name,
            "task": task,
            "error": str(e)
        }
        
        # Error reporting
        self._send_message(
            receiver="master_agent",
            message_type=MessageType.ERROR,
            payload=error_message
        )
        
        # Error recovery
        fallback_result = self._fallback_mechanism(task, context)
        if fallback_result:
            return {"success": True, "result": fallback_result}
        
        # Return error
        return {"success": False, "error": str(e)}
```

## State Management Workflow

### State Management Strategy

**Purpose**: Manage agent state and context sharing

**Workflow**:
1. **Local State**: Each agent maintains its own state
2. **State Updates**: Agents broadcast state updates via AgentBus
3. **State Synchronization**: Agents synchronize state via messages
4. **State Recovery**: Agents can recover state from messages

### State Management Example

```python
# State management workflow
def update_state(self, updates):
    # Update local state
    self.state.update(updates)
    
    # Broadcast state update
    self._send_state_update(updates)
    
    # Other agents receive update and adapt
    # geometry_agent._handle_state_update(message)
```

## Testing Workflow

### Testing Strategy

**Purpose**: Test agents independently and together

**Workflow**:
1. **Unit Testing**: Test individual agents in isolation
2. **Integration Testing**: Test agents together
3. **Mock Agents**: Mock agents for testing
4. **End-to-End Testing**: Test complete workflow

### Testing Example

```python
# Testing workflow
def test_physics_agent():
    # Unit test
    agent = PhysicsAgent(agent_bus, prompt_manager)
    result = agent.execute_task("identify_physics_type", {
        "prompt": "stress in a beam"
    })
    assert result["physics_type"] == "solid_mechanics"

def test_integration():
    # Integration test
    master_agent = MasterAgent(agent_bus, prompt_manager, ...)
    result = master_agent.execute_task("parse_simulation", {
        "prompt": "heat transfer in a rod",
        "context": {}
    })
    assert result["success"] == True
    assert result["updated_context"]["physics_type"] == "heat_transfer"
```

## Performance Optimization Workflow

### Performance Optimization Strategy

**Purpose**: Optimize workflow performance

**Workflow**:
1. **Parallel Execution**: Execute independent tasks in parallel
2. **Caching**: Cache frequently used data
3. **Lazy Loading**: Load data on demand
4. **Resource Management**: Manage resources efficiently

### Performance Optimization Example

```python
# Performance optimization workflow
def parse_simulation(self, context):
    # Parallel execution
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "physics": executor.submit(physics_agent.identify_physics_type, prompt),
            "geometry": executor.submit(geometry_agent.identify_geometry, prompt),
            "material": executor.submit(material_agent.identify_material, prompt),
        }
        results = {key: future.result() for key, future in futures.items()}
    
    # Cache results
    self._cache_results(results)
    
    # Lazy load additional data
    if results["physics"]["physics_type"] == "heat_transfer":
        material_properties = self._lazy_load_material_properties(results["material"]["material_type"])
    
    return results
```

## Conclusion

The workflow for the Agentic FEM application is designed to be:
- **Modular**: Each stage is independent and can be tested separately
- **Parallel**: Independent tasks can run in parallel
- **Resilient**: Errors are isolated and handled gracefully
- **Scalable**: Workflow can scale horizontally and vertically
- **Extensible**: New workflows can be added easily

The workflow provides a clear, structured approach to simulation processing, from user input to visualization, with proper error handling, state management, and performance optimization.

