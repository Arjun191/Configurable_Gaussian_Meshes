import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
import time



def create_simple_proxy_collision(mesh_path, physicsClient):
    """Create a simpler collision shape for physics simulation"""
    # Load the mesh for visual purposes only
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[1, 1, 1],
        rgbaColor=[0.8, 0.8, 0.8, 1]
    )
    
    # Create a simple box for collision
    # First, calculate bounds of the mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()
    
    # Size and center of box
    size = max_bound - min_bound
    center = (min_bound + max_bound) / 2
    
    # Create box collision shape
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[size[0]/2, size[1]/2, size[2]/2]
    )
    
    # Create the multibody
    body_id = p.createMultiBody(
        baseMass=0.0,  # Static object
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[center[0], center[1], center[2]]
    )
    
    return body_id

def load_fixed_mesh_in_pybullet(mesh_path):
    # Connect to PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # Try the following approaches in sequence until one works
    try:
        print("Trying to load mesh directly...")
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[1, 1, 1]
        )
        
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[1, 1, 1],
            rgbaColor=[0.8, 0.8, 0.8, 1]
        )
        
        body_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[0, 0, 1]
        )
        print("Successfully loaded mesh directly")
    except Exception as e:
        print(f"Direct loading failed: {e}")
        print("Creating simple proxy collision shape...")
        body_id = create_simple_proxy_collision(mesh_path, physicsClient)
        print("Using proxy collision shape")
    
    return physicsClient, body_id



mesh_path = "meshes/playroom/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj"

client, body_id = load_fixed_mesh_in_pybullet(mesh_path)
        
# Run simulation
for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()