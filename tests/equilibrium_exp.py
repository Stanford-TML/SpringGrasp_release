import pybullet as pb
import numpy as np
import time
from utils.rigidBodySento import create_primitive_shape, apply_external_world_force_on_local_point, move_object_local_frame

def construct_triangle(a,b,c):
    offset = np.array([0., 0., 0.05])
    vertices = np.array([a-offset,b-offset,c-offset,a+offset,b+offset,c+offset])
    triangles = np.array([[2,1,0],[3,4,5],
                          [0,1,4],[0,4,3],
                          [1,2,5],[1,5,4],
                          [2,0,3],[2,3,5]])
    vis_id = pb.createVisualShape(shapeType=pb.GEOM_MESH, 
                                  vertices=vertices, 
                                  indices=triangles.flatten(), 
                                  meshScale=[1,1,1])
    col_id = pb.createCollisionShape(shapeType=pb.GEOM_MESH, 
                                     vertices=vertices, 
                                     indices=triangles.flatten(), 
                                     meshScale=[1,1,1])
    o = pb.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id)
    return o

c = pb.connect(pb.GUI)

obj = construct_triangle(np.array([-1.0,1.0,0.0]),np.array([1.0,1.0,0.0]),np.array([0.0, 0.0, 0.0]))


poses = np.array([[-0.5, 0.5, 0.0],[0.5, 0.5, 0.0],[0.0, 1.0, 0.0]])
ref = np.array([[0.05, 0.0, 0.0],[-0.05, 0.0, 0.0],[0.0, 0.15, 0.0],[0.0, -0.15, -0.0]])
kp = 2.0 / 0.05
kd = 2 * np.sqrt(kp)
vis_sp = []
for i in range(4):
    vis_sp.append(create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, dim=(0.01,), color=(0.3,1,0.3,1), init_xyz=poses[i], collidable=False))
move_object_local_frame(pb, obj, vis_sp, poses)
input()
for s in range(200):
    # Need map force to world frame
    world_coords, world_vels = move_object_local_frame(pb, obj, vis_sp, poses)
    forces = (ref - world_coords) * kp - world_vels * kd
    #print(forces)
    for i in range(4):
        apply_external_world_force_on_local_point(pb, obj, -1, forces[i], poses[i])
    
    pb.stepSimulation()
    time.sleep(0.1)