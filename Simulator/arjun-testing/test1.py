import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d

# Step 1: Convert Gaussian Splatting .ply to standard mesh
def convert_gaussian_ply_to_mesh(ply_path, output_path):
    # Load the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(ply_path)
    
    # Create a mesh from the point cloud using Poisson surface reconstruction
    # You may need to adjust the depth parameter based on your point cloud density
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # Optional: Remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Save as OBJ file
    o3d.io.write_triangle_mesh(output_path, mesh)
    
    return output_path

# Step 2: Load the converted mesh into PyBullet
def load_mesh_in_pybullet(mesh_path):
    # Connect to PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # Load the converted mesh
    # Using createCollisionShape and createMultiBody for better control
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
        basePosition=[10, 10, 10]
    )
    
    return physicsClient, body_id

# Example usage
if __name__ == "__main__":
    gaussian_ply_path = "playroom/sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000.ply"
    mesh_output_path = "converted_mesh.obj"
    
    # Convert the PLY to OBJ
    converted_path = convert_gaussian_ply_to_mesh(gaussian_ply_path, mesh_output_path)
    
    # Load in PyBullet
    client, body_id = load_mesh_in_pybullet(converted_path)
    
    # Run simulation
    for i in range(10000):
        p.stepSimulation()
        import time
        time.sleep(1./240.)
    
    p.disconnect()