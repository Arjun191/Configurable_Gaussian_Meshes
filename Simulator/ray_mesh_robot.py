import pybullet as p
import pybullet_data
import time
import numpy as np
import math

def setup_environment():
    """Initialize PyBullet and set up the simulation environment"""
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    # Load plane
    planeId = p.loadURDF("plane.urdf")
    
    return physicsClient
ra
def create_scene(physics_client):
    """Create objects in the scene and return a list of their IDs"""
    shapes = []
    
    # Box
    box_pos = [2, 3, 0.5]
    box_ori = p.getQuaternionFromEuler([0, 0, 0])
    box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=[1, 0, 0, 1])
    box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_col, 
                               baseVisualShapeIndex=box_vis, basePosition=box_pos, baseOrientation=box_ori)
    shapes.append(box_id)
    
    # Sphere
    sphere_pos = [3, 1, 0.5]
    sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5)
    sphere_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=[0, 1, 0, 1])
    sphere_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=sphere_col, 
                                 baseVisualShapeIndex=sphere_vis, basePosition=sphere_pos)
    shapes.append(sphere_id)
    
    # Cylinder
    cylinder_pos = [0, 1, 0.5]
    cylinder_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.5, height=1.0)
    cylinder_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, length=1.0, rgbaColor=[0, 0, 1, 1])
    cylinder_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cylinder_col, 
                                   baseVisualShapeIndex=cylinder_vis, basePosition=cylinder_pos)
    shapes.append(cylinder_id)
    
    return shapes

def create_robot(position=[0, 0, 0.1], orientation=[0, 0, 0]):
    """Create and return a simple robot"""
    robot_ori = p.getQuaternionFromEuler(orientation)
    
    # Simple robot with collision detection
    robot_body_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.05])
    robot_body_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.1, 0.05], rgbaColor=[0.8, 0.8, 0.8, 1])
    robot_id = p.createMultiBody(baseMass=1, baseCollisionShapeIndex=robot_body_col, 
                                baseVisualShapeIndex=robot_body_vis, basePosition=position, baseOrientation=robot_ori)
    
    return robot_id

def visualize_ray(from_pos, to_pos, hit=False, lifetime=0.05):
    """Visualize a ray, with color indicating whether it hit something"""
    # Red if hit, green otherwise
    color = [1, 0, 0, 1] if hit else [0, 1, 0, 1]
    p.addUserDebugLine(from_pos, to_pos, color, 2, lifetime)

def cast_rays(robot_id, num_rays=16, ray_length=5.0):
    """Cast rays from the robot and return ray origins, endpoints, and results"""
    robot_pos, robot_ori = p.getBasePositionAndOrientation(robot_id)
    robot_euler = p.getEulerFromQuaternion(robot_ori)
    robot_yaw = robot_euler[2]
    
    ray_froms = []
    ray_tos = []
    
    # Create rays in all directions
    for j in range(num_rays):
        angle = 2 * math.pi * j / num_rays + robot_yaw
        
        # Calculate ray direction
        ray_dx = math.cos(angle)
        ray_dy = math.sin(angle)
        
        from_pos = [robot_pos[0], robot_pos[1], robot_pos[2] + 0.05]
        to_pos = [
            from_pos[0] + ray_dx * ray_length,
            from_pos[1] + ray_dy * ray_length,
            from_pos[2]
        ]
        
        ray_froms.append(from_pos)
        ray_tos.append(to_pos)
    
    # Perform ray tests
    results = p.rayTestBatch(ray_froms, ray_tos)
    
    return ray_froms, ray_tos, results

def visualize_rays(ray_froms, ray_tos, results):
    """Visualize all rays based on intersection results"""
    for j, result in enumerate(results):
        hit_obj_id = result[0]
        hit_fraction = result[2]
        
        if hit_obj_id >= 0:  # If ray hit something
            # Calculate actual hit position
            hit_pos = [
                ray_froms[j][0] + (ray_tos[j][0] - ray_froms[j][0]) * hit_fraction,
                ray_froms[j][1] + (ray_tos[j][1] - ray_froms[j][1]) * hit_fraction,
                ray_froms[j][2] + (ray_tos[j][2] - ray_froms[j][2]) * hit_fraction
            ]
            # Use red for intersecting rays
            visualize_ray(ray_froms[j], hit_pos, hit=True)
        else:
            visualize_ray(ray_froms[j], ray_tos[j], hit=False)

def check_collision(position, shapes, robot_radius=0.25):
    """Check if a position would cause a collision with any shape"""
    for shape_id in shapes:
        shape_pos, _ = p.getBasePositionAndOrientation(shape_id)
        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(position, shape_pos)]))
        
        # We use a conservative collision radius
        collision_radius = 0.9  # Assuming obstacles are ~0.5 units plus robot radius
        if distance < collision_radius:
            return True
    return False

def simple_motion_planning(robot_pos, goal_pos, ray_results, num_rays, ray_length):
    """Simple motion planning: go straight to goal unless obstacle detected"""
    
    # Calculate vector and angle to goal
    goal_direction = [goal_pos[0] - robot_pos[0], goal_pos[1] - robot_pos[1]]
    goal_distance = math.sqrt(goal_direction[0]**2 + goal_direction[1]**2)
    
    if goal_distance > 0:
        # Normalize
        goal_direction = [goal_direction[0]/goal_distance, goal_direction[1]/goal_distance]
    
    goal_angle = math.atan2(goal_direction[1], goal_direction[0])
    
    # Find which ray points most directly to the goal
    goal_ray_index = int(((goal_angle + 2*math.pi) % (2*math.pi)) / (2*math.pi) * num_rays) % num_rays
    
    # Check if there's an obstacle directly in front (in the direction of the goal)
    obstacle_in_path = False
    detection_distance = 1.0  # Only respond to obstacles within this distance
    
    # Check rays near the goal direction (the main ray and adjacent rays)
    for offset in range(-2, 3):
        ray_index = (goal_ray_index + offset) % num_rays
        result = ray_results[ray_index]
        
        if result[0] >= 0:  # If ray hit something
            hit_distance = result[2] * ray_length
            if hit_distance < detection_distance:
                obstacle_in_path = True
                break
    
    if obstacle_in_path:
        # Find clearest path by checking all directions
        best_direction = None
        max_clearance = 0
        
        for i in range(num_rays):
            result = ray_results[i]
            
            # Calculate distance to obstacle in this direction
            if result[0] >= 0:  # If ray hit something
                clearance = result[2] * ray_length
            else:
                clearance = ray_length  # Max clearance if no hit
            
            # Calculate angular difference from goal direction
            ray_angle = 2 * math.pi * i / num_rays
            angle_diff = min((ray_angle - goal_angle) % (2*math.pi), (goal_angle - ray_angle) % (2*math.pi))
            
            # Prefer directions close to goal direction but with good clearance
            # Weight based on clearance and closeness to goal direction
            direction_score = clearance * (1 - angle_diff / math.pi)
            
            if direction_score > max_clearance:
                max_clearance = direction_score
                best_direction = ray_angle
        
        return best_direction
    else:
        # No obstacles, go straight to goal
        return goal_angle

def navigate_to_goal(start_pos, goal_pos, max_steps=1000):
    """Navigate robot from start to goal using simple line-to-goal with obstacle avoidance"""
    # Setup environment
    physics_client = setup_environment()
    
    # Create scene and robot
    scene_shapes = create_scene(physics_client)
    robot_id = create_robot(start_pos)
    
    # Navigation parameters
    num_rays = 10
    ray_length = 1.0
    step_size = 0.1
    goal_threshold = 0.3
    
    # Display goal
    goal_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 1, 0, 1])
    goal_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=goal_visual, basePosition=goal_pos)
    
    # Visualize straight line path to goal
    p.addUserDebugLine(start_pos, goal_pos, [0, 0, 1, 0.5], 1, 0)
    
    try:
        for step in range(max_steps):
            # Get current robot position
            robot_pos, robot_ori = p.getBasePositionAndOrientation(robot_id)
            
            # Check if we've reached the goal
            distance_to_goal = math.sqrt(sum([(a - b) ** 2 for a, b in zip(robot_pos, goal_pos)]))
            if distance_to_goal < goal_threshold:
                print("Goal reached!")
                break
            
            # Cast rays to detect obstacles
            ray_froms, ray_tos, ray_results = cast_rays(robot_id, num_rays, ray_length)
            
            # Visualize the rays
            visualize_rays(ray_froms, ray_tos, ray_results)
            
            # Use simple motion planning
            move_angle = simple_motion_planning(robot_pos, goal_pos, ray_results, num_rays, ray_length)
            
            # Calculate next position
            next_x = robot_pos[0] + math.cos(move_angle) * step_size
            next_y = robot_pos[1] + math.sin(move_angle) * step_size
            next_pos = [next_x, next_y, robot_pos[2]]
            
            # Final collision check
            if not check_collision(next_pos, scene_shapes):
                next_ori = p.getQuaternionFromEuler([0, 0, move_angle])
                p.resetBasePositionAndOrientation(robot_id, next_pos, next_ori)
            else:
                # If still would collide, try to find a better direction
                for angle_offset in [30, 60, 90, 120, 150, 180]:
                    for sign in [1, -1]:
                        test_angle = move_angle + sign * math.radians(angle_offset)
                        test_x = robot_pos[0] + math.cos(test_angle) * step_size
                        test_y = robot_pos[1] + math.sin(test_angle) * step_size
                        test_pos = [test_x, test_y, robot_pos[2]]
                        
                        if not check_collision(test_pos, scene_shapes):
                            next_ori = p.getQuaternionFromEuler([0, 0, test_angle])
                            p.resetBasePositionAndOrientation(robot_id, test_pos, next_ori)
                            break
                    else:
                        continue
                    break
            
            # Step the simulation
            p.stepSimulation()
            time.sleep(0.01)
            
        print(f"Navigation completed after {step+1} steps")
        
    except KeyboardInterrupt:
        print("Navigation interrupted")
    finally:
        # Keep the simulation open for a while to see the final state
        time.sleep(2)
        p.disconnect()

def main():
    """Main function to run the simulation"""
    # Define start and goal positions
    start_pos = [-3, -3, 0.1]
    goal_pos = [3, 3, 0.1]
    
    # Run navigation
    navigate_to_goal(start_pos, goal_pos)

if __name__ == "__main__":
    main()