import numpy as np
import open3d as o3d
import pybullet as p
import pybullet_data
import time

def repair_and_convert_gaussian_ply(ply_path, output_path):
    # Load the point cloud
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        return None
    
    # Check if point cloud has points
    if len(pcd.points) == 0:
        print("Point cloud has no points!")
        return None
    
    print(f"Loaded point cloud with {len(pcd.points)} points")
    
    # STEP 1: More aggressive cleaning to remove problematic points
    # Remove statistical outliers
    print("Removing outliers...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)  # More strict filtering
    pcd = pcd.select_by_index(ind)
    print(f"After outlier removal: {len(pcd.points)} points")
    
    # Voxel downsampling to make the point cloud more uniform
    print("Downsampling point cloud...")
    pcd = pcd.voxel_down_sample(voxel_size=0.02)  # Adjust voxel size as needed
    print(f"After downsampling: {len(pcd.points)} points")
    
    # STEP 2: Try a simpler reconstruction method first - Alpha Shape
    print("Trying Alpha Shape reconstruction...")
    try:
        # Start with a conservative alpha value
        alpha = 0.1  # Adjust based on point cloud density
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        print(f"Alpha Shape created mesh with {len(mesh.triangles)} triangles")
        
        if len(mesh.triangles) < 100:  # If too few triangles were created
            print("Too few triangles, trying with smaller alpha...")
            alpha = 0.05  # Try a smaller alpha value
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
            print(f"New Alpha Shape created mesh with {len(mesh.triangles)} triangles")
    except Exception as e:
        print(f"Alpha Shape failed: {e}")
        print("Falling back to convex hull...")
        # As a last resort, try a simple convex hull
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_convex_hull(pcd)
        print(f"Convex hull created mesh with {len(mesh.triangles)} triangles")
    
    # Clean up the mesh
    print("Cleaning up mesh...")
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    
    # Save the processed mesh
    o3d.io.write_triangle_mesh(output_path, mesh)
    print(f"Saved repaired mesh to {output_path}")
    
    return output_path

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
            baseMass=0.0,
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

# Example usage
if __name__ == "__main__":
    gaussian_ply_path = "sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000.ply"
    mesh_output_path = "simplified_mesh.obj"
    
    # Convert the PLY to a simplified mesh
    repaired_path = repair_and_convert_gaussian_ply(gaussian_ply_path, mesh_output_path)
    
    if repaired_path:
        # Load in PyBullet
        client, body_id = load_fixed_mesh_in_pybullet(repaired_path)
        
        # Run simulation
        for i in range(10000):
            p.stepSimulation()
            time.sleep(1./240.)
        
        p.disconnect()
    else:
        print("Failed to repair mesh. Check your PLY file for serious corruption.")