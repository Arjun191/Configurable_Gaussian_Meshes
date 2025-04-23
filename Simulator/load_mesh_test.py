import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
import time

def analyze_mesh(mesh_path):
    """Analyze mesh properties to debug rendering issues"""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Check if mesh is loaded properly
    if not mesh.has_vertices():
        print("ERROR: Mesh has no vertices!")
        return None
    
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

def fix_mesh_normals(mesh_path):
    """Fix inverted normals and rotate the mesh while preserving colors"""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # Store original colors if they exist
    has_vertex_colors = mesh.has_vertex_colors()
    has_triangle_uvs = mesh.has_triangle_uvs()
    has_textures = mesh.has_textures()
    
    if has_vertex_colors:
        original_colors = np.asarray(mesh.vertex_colors).copy()
        print("Preserving vertex colors...")
    
    # Fix normals - force them to point outward
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print("Flipping mesh normals to fix inverted walls...")
    mesh.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh.triangles)[:, ::-1]  # Reverse triangle vertex order to flip normals
    )
    mesh.compute_vertex_normals()
    
    # Rotate mesh 
    print("Rotating mesh ...")
    R = mesh.get_rotation_matrix_from_xyz([-np.pi/2 + np.pi/3, -np.pi/2, np.pi/6])
    mesh.rotate(R, center=mesh.get_center())
    
    # Restore vertex colors if they existed
    if has_vertex_colors:
        mesh.vertex_colors = o3d.utility.Vector3dVector(original_colors)
    
    # Save the fixed mesh
    fixed_path = "fixed_room_mesh.obj"
    
    # Use PLY format instead of OBJ to better preserve color information
    if has_vertex_colors or has_textures:
        fixed_path = "fixed_room_mesh.ply"
        print(f"Using PLY format to preserve colors")
    
    o3d.io.write_triangle_mesh(fixed_path, mesh)
    print(f"Saved fixed mesh to {fixed_path}")
    
    return fixed_path

def load_fixed_mesh_in_pybullet(mesh_path):
    # Connect to PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Enhanced visualization settings
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    
    # Load ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # Fix the mesh first (rotate and fix normals)
    fixed_path = fix_mesh_normals(mesh_path)
    
    # Analyze the fixed mesh
    result = analyze_mesh(fixed_path)
    if result is None:
        return physicsClient, None
    
    mesh, min_bound, max_bound, center, size = result
    
    # Keep original scale
    scale = [1.0, 1.0, 1.0]
    
    # For a room mesh, align the floor with the ground plane
    floor_offset = min_bound[2]-1
    
    # Position with floor aligned to ground and centered in xy
    position = [-center[0], -center[1], -floor_offset]
    
    print(f"Positioning mesh at {position} to align floor with ground plane")
    
    try:
        print("Attempting to load as concave trimesh...")
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName=fixed_path,
            meshScale=scale,
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName=fixed_path,
            meshScale=scale,
            rgbaColor=[0.8, 0.8, 0.8, 1]
        )
        
        body_id = p.createMultiBody(
            baseMass=0.0,  # Static object - 0
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )
        print("Successfully loaded mesh as concave trimesh")
        
    except Exception as e:
        print(f"Concave trimesh failed: {e}")
        try:
            print("Attempting to load as convex hull...")
            collision_shape_id = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=fixed_path,
                meshScale=scale
            )
            
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=fixed_path,
                meshScale=scale,
                rgbaColor=[0.8, 0.8, 0.8, 1]
            )
            
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=position
            )
            print("Successfully loaded mesh as convex hull")
            
        except Exception as e:
            print(f"Convex hull failed: {e}")
            try:
                print("Falling back to box approximation...")
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=fixed_path,
                    meshScale=scale,
                    rgbaColor=[0.8, 0.8, 0.8, 1]
                )
                
                collision_shape_id = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[size[0]/2, size[1]/2, size[2]/2]
                )
                
                box_position = [position[0], position[1], position[2] + size[2]/2]
                
                body_id = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision_shape_id,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=box_position
                )
                print("Using box approximation for collision")
                
            except Exception as e:
                print(f"All approaches failed: {e}")
                body_id = None
    
    # Add test objects
    if body_id is not None:
        # Create a sphere that will fall onto the floor of the room
        sphere_id = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
        sphere_pos = [0, 0, max_bound[2] - floor_offset - 2]  # 2 units below ceiling
        sphere_body = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=sphere_id,
            basePosition=sphere_pos
        )
        
        # Create a box as well
        box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2])
        box_pos = [1, 1, max_bound[2] - floor_offset - 2]
        box_body = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=box_id,
            basePosition=box_pos
        )
    
    return physicsClient, body_id

# Main code
mesh_path = "meshes/playroom/sugarfine_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000_normalconsistency01_gaussperface1.obj"

print("Loading mesh into PyBullet...")
client, body_id = load_fixed_mesh_in_pybullet(mesh_path)

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