#!/usr/bin/env python3
"""
Test script to verify boundary conditions are in the API response
"""
import requests
import json

# Test the API endpoint
url = "http://localhost:8000/simulation/parse"
prompt = "deformation in a 1d rod"

payload = {
    "prompt": prompt,
    "context": {}
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    
    print("=== API RESPONSE ===")
    print(f"Success: {data.get('success')}")
    print(f"Action: {data.get('action')}")
    print(f"Message: {data.get('message')}")
    
    print("\n=== CONTEXT ===")
    context = data.get('context', {})
    print(f"Context keys: {list(context.keys())}")
    print(f"Physics type: {context.get('physics_type')}")
    print(f"Geometry type: {context.get('geometry_type')}")
    print(f"Material type: {context.get('material_type')}")
    print(f"Geometry dimensions: {context.get('geometry_dimensions')}")
    
    print("\n=== BOUNDARY CONDITIONS IN CONTEXT ===")
    bcs_in_context = context.get('boundary_conditions')
    print(f"Boundary conditions in context: {bcs_in_context}")
    print(f"Type: {type(bcs_in_context)}")
    if isinstance(bcs_in_context, list):
        print(f"Length: {len(bcs_in_context)}")
        for i, bc in enumerate(bcs_in_context):
            print(f"  BC {i}: {bc}")
    else:
        print("Boundary conditions not found or not a list!")
    
    print("\n=== SIMULATION CONFIG ===")
    sim_config = data.get('simulation_config')
    if sim_config:
        print(f"Simulation config keys: {list(sim_config.keys())}")
        pde_config = sim_config.get('pde_config', {})
        if pde_config:
            print(f"PDE config keys: {list(pde_config.keys())}")
            bcs_in_pde = pde_config.get('boundary_conditions')
            print(f"Boundary conditions in pde_config: {bcs_in_pde}")
            if isinstance(bcs_in_pde, list):
                print(f"Length: {len(bcs_in_pde)}")
                for i, bc in enumerate(bcs_in_pde):
                    print(f"  BC {i}: {bc}")
        
        required_components = sim_config.get('required_components', {})
        if required_components:
            print(f"Required components keys: {list(required_components.keys())}")
            bcs_in_required = required_components.get('boundary_conditions')
            print(f"Boundary conditions in required_components: {bcs_in_required}")
            if isinstance(bcs_in_required, list):
                print(f"Length: {len(bcs_in_required)}")
                for i, bc in enumerate(bcs_in_required):
                    print(f"  BC {i}: {bc}")
    else:
        print("No simulation_config in response!")
    
    print("\n=== FULL RESPONSE (JSON) ===")
    print(json.dumps(data, indent=2, default=str))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

