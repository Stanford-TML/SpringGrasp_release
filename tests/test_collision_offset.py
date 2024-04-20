import pybullet as pb
import numpy as np
import torch
from rigidBodySento import create_primitive_shape
from differentiable_robot_model import DifferentiableRobotModel
from scipy.spatial.transform import Rotation 

pb.connect(pb.GUI)

joint_angles = np.array([-0.12999636, 0.46788138, 0.438807, 0.48968481,
                        -0.02283426, 0.26199859, 0.73503519, 0.75897687,
                        0.12591666, 0.27977724, 0.65864516, 0.69471026,
                        1.55866289, 0.16972215, -0.15359271,  1.68753028])

urdf_link = "../pybullet_robot/src/pybullet_robot/robots/allegro_hand/models/allegro_hand_description_left.urdf"
init_pose = [-0.03, 0.03, 0.05, -np.pi/2,-np.pi/4, 0.0]
ori = Rotation.from_euler("XYZ", init_pose[3:])
print(ori.as_matrix())
r = pb.loadURDF(urdf_link, useFixedBase=True, basePosition=init_pose[:3], baseOrientation=ori.as_quat(),  flags=pb.URDF_MERGE_FIXED_LINKS)

for i, angle in enumerate(joint_angles):
    pb.resetJointState(r, i, angle)

robot_model = DifferentiableRobotModel(urdf_link, device="cpu")

collisions_links = ["link_11.0_tip", "link_7.0_tip", "link_3.0_tip","link_15.0_tip"]
collisions_offsets = [[0.0, 0.0, 0.0]] * 4

poses = robot_model.compute_forward_kinematics(torch.tensor(joint_angles), collisions_links, offsets=collisions_offsets, recursive=True)[0].view(-1, 3)

color_code = [[1,0,0,0.7],[0,1,0,0.7],[0,0,1,0.7],[0,1,1,0.7]]

for i, (link, offset) in enumerate(zip(collisions_links, collisions_offsets)):
    o = create_primitive_shape(pb, 0.0, shape=pb.GEOM_SPHERE, dim=(0.02,), collidable=False, color=color_code[i])
    pb.resetBasePositionAndOrientation(o, poses[i], [0,0,0,1])


input("Press Enter to end")


