import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
import time
import os

def create_rotation_quaternion(axis, angle_degrees):
    """
    Create a quaternion representing rotation around an arbitrary axis.
    
    Parameters:
        axis (list or ndarray): The [x, y, z] components of the rotation axis (doesn't need to be normalized)
        angle_degrees (float): The rotation angle in degrees
        
    Returns:
        tuple: The quaternion as (x, y, z, w)
    """
    # Convert angle to radians
    angle_radians = angle_degrees * np.pi / 180.0
    
    # Normalize the axis
    axis = np.array(axis, dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-10:
        # Handle zero vector case - return identity quaternion
        return (0.0, 0.0, 0.0, 1.0)
    
    axis = axis / axis_norm
    
    # Calculate quaternion components
    sin_half_angle = np.sin(angle_radians / 2.0)
    cos_half_angle = np.cos(angle_radians / 2.0)
    
    x = axis[0] * sin_half_angle
    y = axis[1] * sin_half_angle
    z = axis[2] * sin_half_angle
    w = cos_half_angle
    
    # Return in PyBullet order (x, y, z, w)
    return (x, y, z, w)


def analyze_mesh(mesh_path):
    """Analyze mesh properties to debug rendering issues"""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Check if mesh is loaded properly
    if not mesh.has_vertices():
        print("ERROR: Mesh has no vertices!")
        return None
    
    # # Fix inside-out mesh by flipping the normals
    # mesh.triangles = np.asarray([np.flip(triangle) for triangle in mesh.triangles])
    # mesh.compute_vertex_normals()
    
    # Save the fixed mesh
    fixed_mesh_path = mesh_path.replace(".obj", "_fixed.obj")
    o3d.io.write_triangle_mesh(fixed_mesh_path, mesh)
    print(f"Fixed mesh saved to {fixed_mesh_path}")

    # Get mesh properties
    print(f"Mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    min_bound = mesh.get_min_bound()
    max_bound = mesh.get_max_bound()
    size = max_bound - min_bound
    center = (min_bound + max_bound) / 2
    
    print(f"Mesh bounds: Min {min_bound}, Max {max_bound}")
    print(f"Mesh size: {size}")
    print(f"Mesh center: {center}")
    
    return mesh, min_bound, max_bound, center, size

def load_mesh_with_rotation(mesh_path):
    """Direct approach that loads original mesh and applies transforms in PyBullet"""
    # Connect to PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Enhanced visualization settings
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    
    # Load ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # Analyze the mesh
    result = analyze_mesh(mesh_path)
    if result is None:
        return physicsClient, None
    
    mesh, min_bound, max_bound, center, size = result
    
    # Keep original scale
    scale = [-1.0, 1.0, 1.0]
    
    # For a room mesh, align the floor with the ground plane
    floor_offset = min_bound[2]
    
    # Position with floor aligned to ground and centered in xy
    position = [-center[0], -center[1], -floor_offset]
    
    print(f"Positioning mesh at {position}")
    
    # Calculate rotation quaternion for your desired rotation
    # orientation = p.getQuaternionFromEuler([-np.pi/2 + np.pi/3, -np.pi/2, np.pi/6])

    # orientation_2 = create_rotation_quaternion([1, 1/3, 1/6], -125)  # Example rotation around x-axis
    # orientation = orientation_2
    orientation = p.getQuaternionFromEuler([-np.pi/2, 0, 0])  # Example rotation around x-axis
    
    try:
        print("Loading mesh with direct transform approach...")
        
        # First create visual shape
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=scale,
            rgbaColor=[0.8, 0.8, 0.8, 1]  # Default color if mesh doesn't have its own
        )
        
        # Then create collision shape
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=scale,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        
        # Create the body with proper rotation
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position,
            baseOrientation=orientation
        )
        
        print(f"Successfully loaded mesh body ID: {body_id}")
        
        # Try to enable double-sided rendering to fix the inverted normals issue
        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
        try:
            # This may fail on some PyBullet versions
            p.changeVisualShape(body_id, -1, flags=p.VISUAL_SHAPE_DOUBLE_SIDED)
            print("Enabled double-sided rendering")
        except:
            print("Could not enable double-sided rendering (this is okay)")
        
    except Exception as e:
        print(f"Direct loading failed: {e}")
        
        try:
            print("Falling back to simplified collision shape...")
            
            # Create a simple box for collision
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[size[0]/2, size[1]/2, size[2]/2]
            )
            
            # Visual shape uses the mesh
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=mesh_path,
                meshScale=scale,
                rgbaColor=[0.8, 0.8, 0.8, 1]
            )
            
            # Create the body
            box_position = [position[0], position[1], position[2] + size[2]/2]
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=box_position,
                baseOrientation=orientation
            )
            
            print(f"Successfully loaded mesh with box collision, body ID: {body_id}")
            
        except Exception as e:
            print(f"All approaches failed: {e}")
            body_id = None
    
    # Add test objects if mesh loaded successfully
    if body_id is not None:
        # Add a sphere that falls from ceiling
        sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        sphere_pos = [0, 0, max_bound[2] - floor_offset - 2]
        p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=sphere_id,
            basePosition=sphere_pos
        )
        
        # Add a box that falls from ceiling
        box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
        box_pos = [1, 1, max_bound[2] - floor_offset - 2]
        p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=box_id,
            basePosition=box_pos
        )
    
    return physicsClient, body_id

# Main code
def main():
    # mesh_path = "meshes/playroom/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj"
    mesh_path = "meshes/fixed_playroom_mesh/fixed_mesh.obj"
    
    print("Loading mesh into PyBullet...")
    client, body_id = load_mesh_with_rotation(mesh_path)
    
    if body_id is not None:
        print("Mesh loaded successfully! Running simulation...")
        
        # Set better camera view for room interior
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=45,
            cameraPitch=-20,
            cameraTargetPosition=[0, 0, 0]
        )
        
        # Run simulation
        for i in range(10000):
            p.stepSimulation()
            time.sleep(1./240.)
    else:
        print("Failed to load mesh!")
    
    p.disconnect()

if __name__ == "__main__":
    main()