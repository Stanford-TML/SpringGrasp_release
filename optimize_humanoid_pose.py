
import numpy as np
import matplotlib.pyplot as plt
from gpis.gpis import GPIS
import torch
from utils import robot_configs
from utils.pb_grasp_visualizer import HumanoidVisualizer
from utils.create_arrow import create_direct_arrow

from spring_grasp_planner.optimizers import HumanoidOptimizer

device = torch.device("cpu")


if __name__ == "__main__":
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=2000)
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()

    init_joint_angles = torch.zeros(29).unsqueeze(0).float().to(device)
    init_joint_angles[:,[0,6]] = -0.312
    init_joint_angles[:,[3,9]] = 0.669
    init_joint_angles[:,[4,10]] = -0.363

    humanoid_optimizer = HumanoidOptimizer(ref_q = np.zeros(29), num_iters=args.num_iters)
    num_guesses = 30
    init_joint_angles = init_joint_angles.repeat_interleave(num_guesses,dim=0)
    init_joint_angles += torch.randn_like(init_joint_angles)*0.2

    target_pose = torch.tensor([
        [[0.5, 0.3, 0.2],[0.4, -0.2, 0.2]]])

    opt_joint_angles, opt_root_pos, opt_root_rot = humanoid_optimizer.optimize(init_joint_angles,target_pose, verbose=True)
    opt_eef_pose,_,_ = humanoid_optimizer.forward_kinematics(opt_joint_angles, opt_root_pos, opt_root_rot)
    #print("init joint angles:",init_joint_angles)
    # Visualize target and tip pose
    
    
    grasp_vis = HumanoidVisualizer(robot_urdf="assets/g1/g1_29dof_rev_1_0.urdf")

    # Visualize grasp in pybullet
    grasp_vis.visualize_robot(opt_joint_angles.detach().cpu().numpy(), 
                              np.concatenate([opt_root_pos.detach().cpu().numpy(), opt_root_rot.detach().cpu().numpy()]), 
                              target_pose[0].detach().cpu().numpy())
    grasp_vis.cleanup()
    
