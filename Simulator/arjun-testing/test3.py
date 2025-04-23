import open3d as o3d
import pybullet as p
import pybullet_data
import time

def load_ply_mesh_directly(ply_path, output_obj_path=None):
    # Load the PLY file
    print(f"Loading PLY file: {ply_path}")
    mesh = o3d.io.read_triangle_mesh(ply_path)
    
    # Check if it actually contains triangle data
    if len(mesh.triangles) > 0:
        print(f"Successfully loaded mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        
        # Optional: Save as OBJ for PyBullet (sometimes more compatible)
        if output_obj_path:
            o3d.io.write_triangle_mesh(output_obj_path, mesh)
            print(f"Converted PLY to OBJ: {output_obj_path}")
            return output_obj_path
        return ply_path
    else:
        print("The PLY file does not contain triangle data, it might be just a point cloud")
        return None

def load_mesh_in_pybullet(mesh_path):
    # Connect to PyBullet
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load ground plane
    planeId = p.loadURDF("plane.urdf")
    
    # Load the mesh
    print(f"Loading mesh into PyBullet: {mesh_path}")
    
    # Try loading with GEOM_MESH
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
        baseMass=0.0,  # Static object
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[0, 0, 1]
    )
    
    print(f"Successfully loaded mesh as body ID: {body_id}")
    return physicsClient, body_id

# Example usage
if __name__ == "__main__":
    gaussian_ply_path = "sugarmesh_3Dgs7000_densityestim02_sdfnorm02_level03_decim1000000.ply"
    output_obj_path = "converted_mesh.obj"  # Optional OBJ output
    
    # Try to load the PLY directly as a mesh
    mesh_path = load_ply_mesh_directly(gaussian_ply_path, output_obj_path)
    
    if mesh_path:
        # Load in PyBullet
        client, body_id = load_mesh_in_pybullet(mesh_path)
        
        # Run simulation
        for i in range(10000):
            p.stepSimulation()
            time.sleep(1./240.)
        
        p.disconnect()
    else:
        print("The PLY file doesn't contain a valid mesh. Consider using point cloud conversion methods.")