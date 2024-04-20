import numpy as np
from torchsdf import compute_sdf
import torch
import time
from functools import partial
from differentiable_robot_model.robot_model import DifferentiableRobotModel
from utils.math_utils import minimum_wrench_reward, euler_angles_to_matrix

from .initial_guesses import *
from .metric import *

device = torch.device("cpu")

z_margin = 0.3
FINGERTIP_LB = [-0.2, -0.2,   0.015,   -0.2, -0.2,      0.015,  -0.2, -0.2,      0.015, -0.2, -0.2, 0.015]
FINGERTIP_UB = [0.2,  0.2,  z_margin,  0.2,   0.2,   z_margin,   0.2,  0.2,  z_margin,   0.2,  0.2,  z_margin]

class KinGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names,
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=[-0.01, 0.015, 0.12],
                 num_iters=1000, optimize_target=False,
                 ref_q=None,
                 mass=0.1, com = [0.0,0.0,0.0],
                 gravity=True,
                 uncertainty=0.0):
        self.ref_q = torch.tensor(ref_q).cpu()
        self.robot_model = DifferentiableRobotModel(robot_urdf, device="cpu:0")
        self.num_iters = num_iters
        self.ee_link_names = ee_link_names
        self.ee_link_offsets = ee_link_offsets
        self.palm_offset = torch.tensor(palm_offset).cpu()
        self.optimize_target = optimize_target
        self.gravity = gravity
        self.mass = mass
        self.com = com

    def forward_kinematics(self, joint_angles):
        """
        :params: joint_angles: [num_envs, num_dofs]
        :return: tip_poses: [num_envs * num_fingers, 3]
        """
        tip_poses = self.robot_model.compute_forward_kinematics(joint_angles, self.ee_link_names, 
                                                           offsets=self.ee_link_offsets, recursive=True)[0].view(-1,3) + self.palm_offset
        return tip_poses.view(-1,3)

    def optimize(self, joint_angles, target_pose, compliance, friction_mu, object_mesh, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        Params:
        joint_angles: [num_envs, num_dofs]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        opt_mask: [num_envs, num_fingers]
        """
        joint_angles = joint_angles.clone().requires_grad_(True)
        compliance = compliance.clone().requires_grad_(True)
        triangles = np.asarray(object_mesh.triangles)
        vertices = np.asarray(object_mesh.vertices)
        face_vertices = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cpu().float()
        object_mesh.scale(0.9, center=[0,0,0])
        vertices = np.asarray(object_mesh.vertices)
        face_vertices_deflate = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cpu().float()
        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            optim = torch.optim.Adam([{"params":joint_angles, "lr":2e-3},
                                        {"params":target_pose, "lr":1e-5},
                                        {"params":compliance, "lr":0.2}])
        else:
            optim = torch.optim.Adam([{"params":joint_angles, "lr":1e-2}, # Directly optimizing joint angles can result in highly non-linear manifold..
                                        {"params":compliance, "lr":0.2}])
        opt_joint_angle = joint_angles.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(joint_angles.shape[0]).cpu()
        normal = None
        opt_margin = None
        for _ in range(self.num_iters):
            optim.zero_grad()
            all_tip_pose = self.forward_kinematics(joint_angles)
            _,sign1,current_normal1,_ = compute_sdf(all_tip_pose, face_vertices_deflate)
            dist,sign2,current_normal2,_ = compute_sdf(all_tip_pose, face_vertices)
            tar_dist, tar_sign, _, _ = compute_sdf(target_pose.view(-1,3), face_vertices)
            # Note: normal direction will flip when tip is inside the object, normal vector at surface is not defined.
            current_normal = 0.5 * sign1.unsqueeze(1) * current_normal1 + 0.5 * sign2.unsqueeze(1) * current_normal2
            current_normal = current_normal / current_normal.norm(dim=1).unsqueeze(1)
            task_reward, margin, force_norm = force_eq_reward(all_tip_pose.view(target_pose.shape), 
                                target_pose, 
                                compliance, 
                                friction_mu, 
                                current_normal.view(target_pose.shape),
                                mass = self.mass, COM=self.com,
                                gravity=10.0 if self.gravity else None)
            c = -task_reward * 5.0
            center_tip = all_tip_pose.view(target_pose.shape).mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0
            ref_cost = (joint_angles - self.ref_q).norm(dim=1) * 10.0

            dist_cost = 1000 * torch.sqrt(dist).view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 * (tar_sign * torch.sqrt(tar_dist).view(target_pose.shape[0], target_pose.shape[1])).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost + ref_cost # Encourage target pose to stay inside the object
            l.sum().backward()
            if verbose:
                print("Loss:",float(l.sum()), compliance)
            if torch.isnan(l.sum()):
                print(dist, all_tip_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum():
                    normal = current_normal
                    opt_margin = margin
                    opt_value[update_flag] = l[update_flag]
                    opt_joint_angle[update_flag] = joint_angles[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
        if verbose:
            print(opt_margin, normal)
        flag = (opt_margin > 0.0).all()
        return opt_joint_angle, opt_compliance, opt_target_pose, flag

class SDFGraspOptimizer:
    def __init__(self, tip_bounding_box, num_iters=2000, optimize_target=False, mass=0.1, com=[0.0, 0.0, 0.0], gravity=True, uncertainty=0.0):
        """
        tip_bounding_box: [lb [num_finger, 3], ub [num_finger, 3]]
        """
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cpu().view(-1,3), torch.tensor(tip_bounding_box[1]).cpu().view(-1,3)]
        self.num_iters = num_iters
        self.optimize_target = optimize_target
        self.mass = mass
        self.com = com
        self.gravity = gravity

    def optimize(self, tip_pose, target_pose, compliance, friction_mu, object_mesh, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        TODO: Add a penalty term to encourage target pose stay inside the object.
        Params:
        tip_pose: [num_envs, num_fingers, 3]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        opt_mask: [num_envs, num_fingers]
        """
        tip_pose = tip_pose.clone().requires_grad_(True)
        compliance = compliance.clone().requires_grad_(True)
        triangles = np.asarray(object_mesh.triangles)
        vertices = np.asarray(object_mesh.vertices)
        face_vertices = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cpu().float()
        object_mesh.scale(0.9, center=[0,0,0])
        vertices = np.asarray(object_mesh.vertices)
        face_vertices_deflate = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cpu().float()
        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            optim = torch.optim.RMSprop([{"params":tip_pose, "lr":1e-3},
                                        {"params":target_pose, "lr":1e-3}, 
                                        {"params":compliance, "lr":0.2}])
        else:
            optim = torch.optim.RMSprop([{"params":tip_pose, "lr":1e-3},
                                        {"params":compliance, "lr":0.2}])
        opt_tip_pose = tip_pose.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(tip_pose.shape[0]).cpu()
        normal = None
        opt_margin = None
        for _ in range(self.num_iters):
            optim.zero_grad()
            all_tip_pose = tip_pose.view(-1,3)
            _,sign1,current_normal1,_ = compute_sdf(all_tip_pose, face_vertices_deflate)
            dist,sign2,current_normal2,_ = compute_sdf(all_tip_pose, face_vertices)
            tar_dist, tar_sign, _, _ = compute_sdf(target_pose.view(-1,3), face_vertices)
            # Note: normal direction will flip when tip is inside the object, normal vector at surface is not defined.
            current_normal = 0.5 * sign1.unsqueeze(1) * current_normal1 + 0.5 * sign2.unsqueeze(1) * current_normal2
            current_normal = current_normal / current_normal.norm(dim=1).unsqueeze(1)
            task_reward, margin, force_norm = force_eq_reward(tip_pose, 
                                target_pose, 
                                compliance, 
                                friction_mu, 
                                current_normal.view(tip_pose.shape[0], tip_pose.shape[1], tip_pose.shape[2]),
                                mass = self.mass,
                                COM = self.com,
                                gravity = 10.0 if self.gravity else None)
            c = -task_reward * 5.0
            center_tip = tip_pose.mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0

            dist_cost = 1000 * torch.sqrt(dist).view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 *(torch.sqrt(tar_dist).view(tip_pose.shape[0], tip_pose.shape[1]) * tar_sign).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost # Encourage target pose to stay inside the object
            l.sum().backward()
            if verbose:
                print("Loss:",float(l.sum()), float(force_cost.sum()), float(center_cost.sum()), float(dist_cost.sum()), float(tar_dist_cost.sum()))
            if torch.isnan(l.sum()):
                print(dist, tip_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum():
                    normal = current_normal
                    opt_margin = margin
                    opt_value[update_flag] = l[update_flag]
                    opt_tip_pose[update_flag] = tip_pose[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                tip_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
                target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        if verbose:
            print(opt_margin, normal)
        flag = (opt_margin > 0.0).all()
        return opt_tip_pose, opt_compliance, opt_target_pose, flag

class GPISGraspOptimizer:
    def __init__(self, tip_bounding_box, num_iters=2000, optimize_target=False, mass=0.1, com=[0.0, 0.0, 0.0], gravity=True, uncertainty=20.0):
        """
        tip_bounding_box: [lb [num_finger, 3], ub [num_finger, 3]]
        """
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cpu().view(-1,3), torch.tensor(tip_bounding_box[1]).cpu().view(-1,3)]
        self.num_iters = num_iters
        self.optimize_target = optimize_target
        self.mass = mass
        self.com = com
        self.gravity = gravity
        self.uncertainty = uncertainty

    def optimize(self, tip_pose, target_pose, compliance, friction_mu, gpis, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        TODO: Add a penalty term to encourage target pose stay inside the object.
        Params:
        tip_pose: [num_envs, num_fingers, 3]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        """
        tip_pose = tip_pose.clone().requires_grad_(True)
        compliance = compliance.clone().requires_grad_(True)
        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            optim = torch.optim.RMSprop([{"params":tip_pose, "lr":1e-3},
                                        {"params":target_pose, "lr":1e-3}, 
                                        {"params":compliance, "lr":0.2}])
        else:
            optim = torch.optim.RMSprop([{"params":tip_pose, "lr":1e-3},
                                        {"params":compliance, "lr":0.2}])
        opt_tip_pose = tip_pose.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(tip_pose.shape[0]).double().cpu()
        normal = None
        opt_margin = None
        for _ in range(self.num_iters):
            optim.zero_grad()
            all_tip_pose = tip_pose.view(-1,3)
            dist, var = gpis.pred(all_tip_pose)
            tar_dist, _ = gpis.pred(target_pose.view(-1,3))
            current_normal = gpis.compute_normal(all_tip_pose)
            task_reward, margin, force_norm = force_eq_reward(tip_pose, 
                                target_pose, 
                                compliance, 
                                friction_mu, 
                                current_normal.view(tip_pose.shape[0], tip_pose.shape[1], tip_pose.shape[2]),
                                mass = self.mass,
                                COM = self.com,
                                gravity = 10.0 if self.gravity else None)
            c = -task_reward * 25.0
            center_tip = tip_pose.mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0

            dist_cost = 1000 * torch.abs(dist).view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 *tar_dist.view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            variance_cost = self.uncertainty * var.view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost + variance_cost # Encourage target pose to stay inside the object
            l.sum().backward()
            if verbose:
                print("Loss:",float(l.sum()), float(force_cost.sum()), float(center_cost.sum()), float(dist_cost.sum()), float(tar_dist_cost.sum()), float(variance_cost.sum()))
            if torch.isnan(l.sum()):
                print("Loss NaN trace:",dist, tip_pose, target_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum():
                    normal = current_normal
                    opt_margin = margin
                    opt_value[update_flag] = l[update_flag]
                    opt_tip_pose[update_flag] = tip_pose[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                tip_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
                compliance.clamp_(min=40.0) # prevent negative compliance
                #target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        if verbose:
            print(margin, normal)
        flag = (opt_margin > 0.0).all()
        return opt_tip_pose, opt_compliance, opt_target_pose, flag

class KinGPISGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names,
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=WRIST_OFFSET,
                 num_iters=1000, optimize_target=False,
                 ref_q=None,
                 tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB],
                 mass=0.1, com=[0.0, 0.0, 0.0], gravity=True, uncertainty=10.0):
        self.ref_q = torch.tensor(ref_q).cpu()
        self.robot_model = DifferentiableRobotModel(robot_urdf, device="cpu:0")
        self.num_iters = num_iters
        self.ee_link_names = ee_link_names
        self.ee_link_offsets = ee_link_offsets
        self.palm_offset = torch.tensor(palm_offset).double().cpu()
        self.optimize_target = optimize_target
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cpu().view(-1,3), torch.tensor(tip_bounding_box[1]).cpu().view(-1,3)]
        self.mass = mass
        self.com = com
        self.gravity = gravity
        self.uncertainty = uncertainty

    def forward_kinematics(self, joint_angles):
        """
        :params: joint_angles: [num_envs, num_dofs]
        :return: tip_poses: [num_envs * num_fingers, 3]
        """
        tip_poses = self.robot_model.compute_forward_kinematics(joint_angles, self.ee_link_names,
                                                                offsets=self.ee_link_offsets, recursive=True)[0].view(-1,3) + self.palm_offset
        return tip_poses.view(-1,3)
    
    def optimize(self, joint_angles, target_pose, compliance, friction_mu, gpis, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        Params:
        joint_angles: [num_envs, num_dofs]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        opt_mask: [num_envs, num_fingers]
        """
        joint_angles = joint_angles.clone().requires_grad_(True)
        compliance = compliance.clone().requires_grad_(True)

        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            optim = torch.optim.RMSprop([{"params":joint_angles, "lr":2e-3},
                                        {"params":target_pose, "lr":1e-3},
                                        {"params":compliance, "lr":0.2}])
        else:
            optim = torch.optim.RMSprop([{"params":joint_angles, "lr":1e-2}, # Directly optimizing joint angles can result in highly non-linear manifold..
                                        {"params":compliance, "lr":0.2}])
        opt_joint_angle = joint_angles.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(joint_angles.shape[0]).double().cpu()
        normal = None
        opt_margin = None
        for _ in range(self.num_iters):
            optim.zero_grad()
            all_tip_pose = self.forward_kinematics(joint_angles)
            dist, var = gpis.pred(all_tip_pose)
            tar_dist, _ = gpis.pred(target_pose.view(-1,3))
            current_normal = gpis.compute_normal(all_tip_pose)
            task_reward, margin, force_norm = force_eq_reward(all_tip_pose.view(target_pose.shape), 
                                target_pose, 
                                compliance, 
                                friction_mu, 
                                current_normal.view(target_pose.shape),
                                mass = self.mass,
                                COM = self.com,
                                gravity = 10.0 if self.gravity else None)
            c = -task_reward * 5.0
            center_tip = all_tip_pose.view(target_pose.shape).mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0
            ref_cost = (joint_angles - self.ref_q).norm(dim=1) * 20.0
            variance_cost = self.uncertainty * torch.log(100 * var).view(target_pose.shape[0], target_pose.shape[1])
            dist_cost = 1000 * torch.abs(dist).view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 * tar_dist.view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost + ref_cost + variance_cost.max(dim=1)[0] # Encourage target pose to stay inside the object
            l.sum().backward()
            if verbose:
                print("Loss:",float(l.sum()), variance_cost.detach())
            if torch.isnan(l.sum()):
                print(dist, all_tip_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum():
                    normal = current_normal
                    opt_margin = margin
                    opt_value[update_flag] = l[update_flag]
                    opt_joint_angle[update_flag] = joint_angles[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                compliance.clamp_(min=40.0) # prevent negative compliance
                target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        if verbose:
            print(opt_margin, normal)
        flag = (opt_margin > 0.0).all()
        return opt_joint_angle, opt_compliance, opt_target_pose, flag
    
class FCGPISGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names,
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=WRIST_OFFSET,
                 num_iters=1000, 
                 optimize_target=False, # dummy
                 ref_q=None,
                 min_force = 2.0,
                 anchor_link_names = None,
                 anchor_link_offsets = None,
                 collision_pairs = None,
                 collision_pair_threshold = 0.02,
                 pregrasp_coefficients = [0.8, 0.8, 0.8, 0.8],
                 mass=0.1, com=[0.0, 0.0, 0.0], gravity=True,
                 uncertainty=0.0,
                 optimize_palm=False):
        self.ref_q = torch.tensor(ref_q).to(device)
        self.robot_model = DifferentiableRobotModel(robot_urdf, device=device)
        self.num_iters = num_iters
        self.ee_link_names = ee_link_names
        self.ee_link_offsets = ee_link_offsets
        self.palm_offset = torch.tensor(palm_offset).double().to(device)
        self.min_force = min_force
        self.mass = mass
        self.com = com
        self.uncertainty = uncertainty
        self.gravity = gravity
        self.optimize_palm = optimize_palm
        self.anchor_link_names = anchor_link_names
        self.anchor_link_offsets = anchor_link_offsets
        collision_pairs = torch.tensor(collision_pairs).long().to(device)
        self.collision_pair_left = collision_pairs[:,0]
        self.collision_pair_right = collision_pairs[:,1]
        self.collision_pair_threshold = collision_pair_threshold
        self.pregrasp_coefficients = torch.tensor(pregrasp_coefficients).to(device)

    def forward_kinematics(self, joint_angles, palm_poses=None, link_names=None):
        """
        :params: joint_angles: [num_envs, num_dofs]
        :params: palm_offset: [num_envs, 3]
        :return: tip_poses: [num_envs * num_fingers, 3]
        """
        if palm_poses is None:
            palm_poses = self.palm_offset
        if link_names is None:
            link_names = self.ee_link_names
            offsets = self.ee_link_offsets
        else:
            offsets = [[0.0, 0.0, 0.0]] * len(link_names)
        tip_poses = self.robot_model.compute_forward_kinematics(joint_angles.float(), link_names,
                                                                offsets=offsets, recursive=False)[0].double().view(-1,len(link_names),3)
        R = euler_angles_to_matrix(palm_poses[:,3:], convention="XYZ") #[num_envs, 3, 3]
        tip_poses = torch.bmm(R, tip_poses.transpose(1,2)).transpose(1,2) + palm_poses[:,:3].unsqueeze(1)
        return tip_poses
    
    def compute_collision_loss(self, joint_angles, palm_poses=None):
        if palm_poses is None:
            palm_poses = self.palm_offset
        anchor_pose = self.robot_model.compute_forward_kinematics(joint_angles.float(), 
                                                                  self.anchor_link_names, 
                                                                  offsets=self.anchor_link_offsets, 
                                                                  recursive=False)[0].double().view(-1,len(self.anchor_link_names),3)
        R = euler_angles_to_matrix(palm_poses[:,3:], convention="XYZ") #[num_envs, 3, 3]
        anchor_pose = torch.bmm(R, anchor_pose.transpose(1,2)).transpose(1,2) + palm_poses[:,:3].unsqueeze(1)
        collision_pair_left = anchor_pose[:,self.collision_pair_left]
        collision_pair_right = anchor_pose[:,self.collision_pair_right]
        dist = torch.norm(collision_pair_left - collision_pair_right, dim=2) # [num_collision_pairs]
        # Add pairwise collision cost
        mask = dist < self.collision_pair_threshold
        inverse_dist = 1.0 / dist
        inverse_dist[~mask] *= 0.0
        cost =  inverse_dist.sum(dim=1)
        # Add ground collision cost
        z_mask = anchor_pose[:,:,2] < 0.02
        z_dist_cost = 1/(anchor_pose[:,:,2]) * 0.1
        z_dist_cost[~z_mask] *= 0.0
        z_cost = z_dist_cost.sum(dim=1)
        cost += z_cost
        # add palm-floor collision cost
        if self.optimize_palm:
            palm_z_mask = palm_poses[:,2] < 0.02
            palm_z_dist_cost = 1/(palm_poses[:,2])
            palm_z_dist_cost[~palm_z_mask] *= 0.0
            palm_z_cost = palm_z_dist_cost
            cost += palm_z_cost
        return cost

    def optimize(self, init_joint_angles, target_pose, compliance, friction_mu, gpis, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        Params:
        joint_angles: [num_envs, num_dofs]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        opt_mask: [num_envs, num_fingers]
        """
        joint_angles = init_joint_angles.clone().requires_grad_(True)
        #compliance = compliance.clone().requires_grad_(True)
        params_list = [{"params":joint_angles, "lr":2e-3}]
        palm_poses = self.palm_offset[:,:3].clone().requires_grad_(self.optimize_palm)
        palm_oris = self.palm_offset[:,3:].clone().requires_grad_(self.optimize_palm)
        if self.optimize_palm:
            params_list.append({"params":palm_poses, "lr":1e-3})
            params_list.append({"params":palm_oris, "lr":1e-3})

        optim = torch.optim.RMSprop(params_list)
        num_envs = init_joint_angles.shape[0]
        
        opt_joint_angle = joint_angles.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_palm_poses = self.palm_offset.clone()
        opt_value = torch.inf * torch.ones(num_envs).double().to(device)
        opt_margin = torch.zeros(num_envs, 4).double().to(device)
        opt_task_cost = torch.inf * torch.ones(num_envs).double().to(device)
        time_ts = time.time()
        for s in range(self.num_iters):
            optim.zero_grad()
            palm_posori = torch.hstack([palm_poses, palm_oris])
            all_tip_pose = self.forward_kinematics(joint_angles, palm_posori) # [num_envs, num_fingers, 3]

            # Target pose are passively determined.
            # Pregrasp pose may not align with direction of force as we don't know target direction.
            dist, var = gpis.pred(all_tip_pose)
            current_normal = gpis.compute_normal(all_tip_pose)
            margin = []
            forces = []
            target_pose = []
            task_cost = []
            # Re-scale the compliance with surface uncertainty
            #compliance = (0.25 * 1/var).float().detach() # Should be with in reasonable range
            # Using wrench closure formulation, we cannot align multiple contact points with same pregrasp.
            for i in range(num_envs):
                task_cost_, margin_, forces_ = minimum_wrench_reward(all_tip_pose[i], 
                                                                     current_normal[i], 
                                                                     friction_mu, min_force=self.min_force)
                # Compute target positions, we shouldn't tune contact pose in order to produce a target 
                target_pose.append(all_tip_pose[i] + forces_ / compliance[i].unsqueeze(1))
                task_cost.append(task_cost_)
                margin.append(margin_)
                forces.append(forces_)
            target_pose = torch.stack(target_pose)
            margin = torch.stack(margin).squeeze(1)
            forces = torch.stack(forces)
            task_cost = torch.stack(task_cost)
            force_norm = forces.norm(dim=2).view(target_pose.shape[0], target_pose.shape[1])
            forces = forces.view(target_pose.shape[0], target_pose.shape[1], 3)
            
            c = task_cost * 1000.0
            center_tip = all_tip_pose.view(target_pose.shape).mean(dim=1)
            center_target = target_pose.mean(dim=1) # Become a regularization term
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0
            ref_cost = (joint_angles - self.ref_q).norm(dim=1) * 20.0
            variance_cost = self.uncertainty * torch.log(100 * var).view(target_pose.shape[0], target_pose.shape[1])
            dist_cost = 1000 * torch.abs(dist).view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
            l = c + dist_cost + center_cost + force_cost + ref_cost + variance_cost.max(dim=1)[0] # Encourage target pose to stay inside the object
            # Add kinematic constraint loss
            if self.optimize_palm:
                palm_dist,_ = gpis.pred(palm_poses)
                palm_dist = palm_dist
                l += 1/palm_dist # Need to ensure palm is outside the object.
                #print("palm dist:", float(palm_dist.mean()))
            l += self.compute_collision_loss(joint_angles, palm_posori)
            # link penetration loss
            link_pos = self.forward_kinematics(joint_angles, palm_posori, link_names=["link_1.0", "link_2.0", "link_3.0",
                                                                                  "link_5.0", "link_6.0", "link_7.0",
                                                                                  "link_9.0", "link_10.0", "link_11.0"])
            link_dist,_ = gpis.pred(link_pos)
            l += (1/link_dist).sum(dim=1) * 0.01
            l.sum().backward()
            if verbose:
                print("Loss:",float(l.sum()), float(task_cost.sum()))
            if torch.isnan(l.sum()):
                print(dist, all_tip_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum() and s > 20:
                    opt_margin[update_flag] = margin[update_flag].clone()
                    opt_task_cost[update_flag] = task_cost[update_flag].clone()
                    opt_value[update_flag] = l[update_flag].clone()
                    opt_joint_angle[update_flag] = joint_angles[update_flag].clone()
                    opt_target_pose[update_flag] = target_pose[update_flag].clone()
                    opt_compliance[update_flag] = compliance[update_flag].clone() # dummy
                    opt_palm_poses[update_flag] = torch.hstack([palm_poses, palm_oris])[update_flag].clone()
            optim.step()
        print("Time taken:", time.time() - time_ts)
        if verbose:
            print(opt_margin)
        opt_tip_pose = self.forward_kinematics(opt_joint_angle, opt_palm_poses)
        # Make it pregrasp
        opt_tip_pose = opt_target_pose + (opt_tip_pose - opt_target_pose) * (1/self.pregrasp_coefficients).view(1,-1,1) #[num_envs, num_fingers, 3]
        task_cost_flag = opt_task_cost < 1e-5
        print(task_cost_flag)
        opt_margin[~task_cost_flag] = -1.0 # Ensure that if task cost is not zero, margin is negative.
        return opt_tip_pose, opt_compliance, opt_target_pose, opt_palm_poses, opt_margin, opt_joint_angle

class SpringGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names,
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=WRIST_OFFSET, # Should be a [num_envs, 6] matrix, 
                 num_iters=1000, optimize_target=False,
                 ref_q=None,
                 pregrasp_coefficients = [[0.7,0.7,0.7,0.7]],
                 pregrasp_weights = [1.0],
                 anchor_link_names = None,
                 anchor_link_offsets = None,
                 collision_pairs = None,
                 collision_pair_threshold = 0.02,
                 mass = 0.1, com = [0.0, 0.0, 0.0], gravity=True,
                 optimize_palm = False,
                 num_samples = 10,
                 weight_config = None):
        self.ref_q = torch.tensor(ref_q).to(device)
        self.robot_model = DifferentiableRobotModel(robot_urdf, device=device)
        self.num_iters = num_iters
        self.ee_link_names = ee_link_names
        self.ee_link_offsets = ee_link_offsets
        print("Wrist offset:", palm_offset)
        self.palm_offset = torch.from_numpy(palm_offset).double().to(device)
        self.optimize_target = optimize_target
        self.optimize_palm = optimize_palm
        self.pregrasp_coefficients = torch.tensor(pregrasp_coefficients).to(device)
        self.pregrasp_weights = torch.tensor(pregrasp_weights).double().to(device)
        self.anchor_link_names = anchor_link_names
        self.anchor_link_offsets = anchor_link_offsets
        collision_pairs = torch.tensor(collision_pairs).long().to(device)
        self.collision_pair_left = collision_pairs[:,0]
        self.collision_pair_right = collision_pairs[:,1]
        self.collision_pair_threshold = collision_pair_threshold
        self.mass = mass
        self.com = com
        self.gravity = gravity
        self.num_samples = num_samples
        # Different weights, shoudl load from config.
        if weight_config is None:
            self.w_sp = 200.0
            self.w_dist = 10000.0
            self.w_uncer = 20.0
            self.w_gain = 0.5
            self.w_tar = 1000.0
            self.w_col = 1.0
            self.w_reg = 10.0
            self.w_force = 200.0
        else:
            self.w_sp = weight_config["w_sp"]
            self.w_dist = weight_config["w_dist"]
            self.w_uncer = weight_config["w_uncer"]
            self.w_gain = weight_config["w_gain"]
            self.w_tar = weight_config["w_tar"]
            self.w_col = weight_config["w_col"]
            self.w_reg = weight_config["w_reg"]
            self.w_force = weight_config["w_force"]
        # Print out the weight configuration
        print("========Weight Configuration:========")
        print("w_sp:", self.w_sp)
        print("w_dist:", self.w_dist)
        print("w_uncer:", self.w_uncer)
        print("w_gain:", self.w_gain)
        print("w_tar:", self.w_tar)
        print("w_col:", self.w_col)
        print("w_reg:", self.w_reg)
        print("w_force:", self.w_force)
        print("=====================================")

    def forward_kinematics(self, joint_angles, palm_poses=None, link_names=None):
        """
        :params: joint_angles: [num_envs, num_dofs]
        :params: palm_offset: [num_envs, 3]
        :return: tip_poses: [num_envs * num_fingers, 3]
        """
        if palm_poses is None:
            palm_poses = self.palm_offset
        if link_names is None:
            link_names = self.ee_link_names
            offsets = self.ee_link_offsets
        else:
            offsets = [[0.0, 0.0, 0.0]] * len(link_names)
        tip_poses = self.robot_model.compute_forward_kinematics(joint_angles.float(), link_names,
                                                                offsets=offsets, recursive=False)[0].double().view(-1,len(link_names),3)
        R = euler_angles_to_matrix(palm_poses[:,3:], convention="XYZ") #[num_envs, 3, 3]
        tip_poses = torch.bmm(R, tip_poses.transpose(1,2)).transpose(1,2) + palm_poses[:,:3].unsqueeze(1)
        return tip_poses
    
    def compute_collision_loss(self, joint_angles, palm_poses=None):
        if palm_poses is None:
            palm_poses = self.palm_offset
        anchor_pose = self.robot_model.compute_forward_kinematics(joint_angles.float(), 
                                                                  self.anchor_link_names, 
                                                                  offsets=self.anchor_link_offsets, 
                                                                  recursive=False)[0].double().view(-1,len(self.anchor_link_names),3)
        R = euler_angles_to_matrix(palm_poses[:,3:], convention="XYZ") #[num_envs, 3, 3]
        anchor_pose = torch.bmm(R, anchor_pose.transpose(1,2)).transpose(1,2) + palm_poses[:,:3].unsqueeze(1)
        collision_pair_left = anchor_pose[:,self.collision_pair_left]
        collision_pair_right = anchor_pose[:,self.collision_pair_right]
        dist = torch.norm(collision_pair_left - collision_pair_right, dim=2) # [num_collision_pairs]
        # Add pairwise collision cost
        mask = dist < self.collision_pair_threshold
        inverse_dist = 1.0 / dist
        inverse_dist[~mask] *= 0.0
        cost =  inverse_dist.sum(dim=1)
        # Add ground collision cost
        #z_mask = anchor_pose[:,:,2] < 0.02
        z_dist_cost = 1/(anchor_pose[:,:,2]) * 0.1
        #z_dist_cost[~z_mask] *= 0.0
        z_cost = z_dist_cost.sum(dim=1)
        cost += z_cost
        # add palm-floor collision cost
        if self.optimize_palm:
            palm_z_mask = palm_poses[:,2] < 0.02
            palm_z_dist_cost = 1/(palm_poses[:,2])
            palm_z_dist_cost[~palm_z_mask] *= 0.0
            palm_z_cost = palm_z_dist_cost
            cost += palm_z_cost
        return cost
    
    def compute_contact_margin(self, tip_pose, target_pose, current_normal, friction_mu):
        force_dir = tip_pose - target_pose
        force_dir = force_dir / force_dir.norm(dim=2, keepdim=True)
        ang_diff = torch.einsum("ijk,ijk->ij",force_dir, current_normal)
        cos_mu = torch.sqrt(1/(1+torch.tensor(friction_mu)**2))
        margin = (ang_diff - cos_mu).clamp(-0.999)
        reward = (0.2 * torch.log(ang_diff+1)+ 0.8*torch.log(margin+1)).sum(dim=1)
        return reward

    # assume all_tip_pose has same shape as target_pose
    def compute_loss(self, all_tip_pose, joint_angles, target_pose, compliance, friction_mu, gpis):
        # All tip pose should be [num_envs*1, 4, 3]
        # Should use pregrasp tip_pose for sampling
        dist, var = gpis.pred(all_tip_pose)
        tar_dist, _ = gpis.pred(target_pose)
        current_normal = gpis.compute_normal(all_tip_pose)
        task_reward, margin, force_norm, R, t = force_eq_reward(
                            all_tip_pose,
                            target_pose,
                            compliance,
                            friction_mu, 
                            current_normal.view(target_pose.shape),
                            mass=self.mass,
                            COM = self.com,
                            gravity=10.0 if self.gravity else None)

        # initial feasibility should be equally important as task reward.
        c = -task_reward - self.compute_contact_margin(all_tip_pose, target_pose, current_normal, friction_mu=friction_mu)/2
        offsets = torch.tensor([0.0, 0.0, 0.0]).to(device)
        reg_cost = (torch.bmm(R,all_tip_pose.transpose(1,2)).transpose(1,2) + t.unsqueeze(1) - all_tip_pose - offsets).norm(dim=2).sum(dim=1) * 200.0
        reg_cost += (joint_angles - self.ref_q).norm(dim=1)
        force_cost = -force_norm.clamp(max=2.0).mean(dim=1)
        
        contact_prob = 1.0/(torch.sqrt(var))*torch.exp(-dist**2/(2*var))
        #variance_cost = self.uncertainty * torch.log(100 * var).max(dim=1)[0]
        #print(float(variance_cost.max(dim=1)[0]))
        dist_cost = torch.abs(dist).sum(dim=1)
        tar_dist_cost = tar_dist.sum(dim=1) # 30
        l = c * self.w_sp + dist_cost * self.w_dist + tar_dist_cost * self.w_tar + \
            force_cost * self.w_force + reg_cost * self.w_reg + self.w_gain * compliance.sum(dim=1)
        #print("All costs:", float(c.mean()), float(dist_cost.mean()), float(tar_dist_cost.mean()), float(center_cost.mean()), float(force_cost.mean()), float(ref_cost.mean()), float(variance_cost.mean()), float(reg_cost.mean()))
        return l, margin, R, t, contact_prob

    def closure(self, joint_angles, compliance, target_pose, palm_poses, palm_oris, friction_mu, gpis, num_envs):
        self.optim.zero_grad()
        palm_posori = torch.hstack([palm_poses, palm_oris])
        self.pregrasp_tip_pose = self.forward_kinematics(joint_angles, palm_posori)
    
        
        # Compute expected loss
        target_pose_extended = target_pose.repeat(len(self.pregrasp_coefficients),1,1)
        pregrasp_tip_pose_extended = self.pregrasp_tip_pose.repeat(len(self.pregrasp_coefficients),1,1) #[e1,e2,e3,e4,e1,e2,e3,e4, ...]
        pregrasp_coeffs = self.pregrasp_coefficients.repeat_interleave(num_envs,dim=0)
        all_tip_pose = target_pose_extended + pregrasp_coeffs.view(-1, 4, 1) * (pregrasp_tip_pose_extended - target_pose_extended)
        l, margin, R, t, contact_prob = self.compute_loss(all_tip_pose, # [num_envs*num_coeffs, 4, 3]
                                      joint_angles.repeat(len(self.pregrasp_coefficients), 1), 
                                      target_pose_extended, 
                                      compliance.repeat(len(self.pregrasp_coefficients), 1), friction_mu, gpis)

        interp = torch.linspace(0, 1, self.num_samples).to(device).view(1, self.num_samples, 1, 1)
        delta_vector = (self.pregrasp_tip_pose - target_pose).unsqueeze(1).repeat(1, self.num_samples, 1, 1) * interp
        sample_points = target_pose.unsqueeze(1).repeat(1, self.num_samples, 1, 1) + delta_vector
        sample_dist, sample_var = gpis.pred(sample_points.view(-1,4,3))
        sample_prob = (1/torch.sqrt(sample_var))*torch.exp(-sample_dist**2 / (2 * sample_var)) #[num_envs * num_samples, 4, 1]
        sample_prob = sample_prob.view(-1, self.num_samples, 4, 1)
        # Normalize sample probability
        normalization_factor = sample_prob.sum(dim=1) # [num_envs, 4, 1]
        sample_prob = sample_prob / normalization_factor.unsqueeze(1) 
        
        contact_prob = (contact_prob.unsqueeze(2) / normalization_factor).unsqueeze(1) # [num_envs, 1, 4, 1]
        total_prominence = (contact_prob - sample_prob).sum(dim=1).squeeze(-1) # [num_envs, 4]
        prominence_loss = total_prominence.sum(dim=1) # NOTE: Variance loss
        total_loss = -prominence_loss * self.w_uncer # NOTE: Variance loss
        self.R, self.t = R, t
        task_loss = l
        total_loss += task_loss
        self.total_margin = (self.pregrasp_weights.view(-1,1,1) * margin.view(-1, num_envs, 4)).sum(dim=0)
        pre_dist, _ = gpis.pred(self.pregrasp_tip_pose)
        pre_dist_loss = -pre_dist.sum(dim=1) * 50.0 # NOTE: Experimental
        total_loss += pre_dist_loss # Ignored for now
        # Palm to object distance
        if self.optimize_palm:
            palm_dist,_ = gpis.pred(palm_poses)
            palm_dist = palm_dist
            palm_dist_loss = 1/palm_dist # Need to ensure palm is outside the object.
            total_loss += palm_dist_loss
            #print("palm dist:", float(palm_dist.mean()))
        # Hand object collision loss
        link_pos = self.forward_kinematics(joint_angles, palm_posori, link_names=["link_1.0", "link_2.0", "link_3.0",
                                                                                  "link_5.0", "link_6.0", "link_7.0",
                                                                                  "link_9.0", "link_10.0", "link_11.0"])
        link_dist,_ = gpis.pred(link_pos)
        link_dist_loss = (1/link_dist).sum(dim=1) * 0.01
        total_loss += link_dist_loss * self.w_col
        
        total_loss += self.compute_collision_loss(joint_angles, palm_posori) * self.w_col
        self.total_loss = total_loss
        loss = total_loss.sum() # TODO: TO BE FINISHED
        loss.backward()
        return loss

    def optimize(self, init_joint_angles, target_pose, compliance, friction_mu, gpis, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        Params:
        joint_angles: [num_envs, num_dofs]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        opt_mask: [num_envs, num_fingers]
        """
        joint_angles = init_joint_angles.clone().requires_grad_(True)
        compliance = compliance.clone().requires_grad_(True)
        params_list = [{"params":joint_angles, "lr":2e-3},
                       {"params":compliance, "lr":0.5}]
        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            params_list.append({"params":target_pose, "lr":2e-3})
        
        palm_poses = self.palm_offset[:,:3].clone().requires_grad_(self.optimize_palm)
        palm_oris = self.palm_offset[:,3:].clone().requires_grad_(self.optimize_palm)
        if self.optimize_palm:
            params_list.append({"params":palm_poses, "lr":1e-3})
            params_list.append({"params":palm_oris, "lr":1e-3})

        self.optim = torch.optim.AdamW(params_list)

        num_envs = init_joint_angles.shape[0]
        opt_joint_angle = init_joint_angles.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(num_envs).double().to(device)
        opt_margin = torch.zeros(num_envs, 4).double().to(device)
        opt_palm_poses = self.palm_offset.clone()
        opt_R, opt_t = torch.zeros(num_envs, 3, 3).double().to(device), torch.zeros(num_envs, 3).double().to(device)
        start_ts = time.time()
        for s in range(self.num_iters):
            if isinstance(self.optim, torch.optim.LBFGS):
                self.optim.step(partial(self.closure, joint_angles=joint_angles,
                                                      compliance=compliance, 
                                                      target_pose=target_pose, 
                                                      palm_poses=palm_poses, 
                                                      palm_oris=palm_oris, 
                                                      friction_mu=friction_mu, 
                                                      gpis=gpis, num_envs=num_envs))
            else:
                loss = self.closure(joint_angles, compliance, target_pose, palm_poses, palm_oris, friction_mu, gpis, num_envs)
            if verbose:
                print(f"Step {s} Loss:",float(self.total_loss.mean()))
            if torch.isnan(self.total_loss.sum()):
                print("NaN detected:", self.pregrasp_tip_pose, self.total_margin)
            with torch.no_grad():
                update_flag = self.total_loss < opt_value
                if update_flag.sum() and s>20:
                    opt_value[update_flag] = self.total_loss[update_flag].clone()
                    opt_margin[update_flag] = self.total_margin[update_flag].clone()
                    opt_joint_angle[update_flag] = joint_angles[update_flag].clone()
                    opt_target_pose[update_flag] = target_pose[update_flag].clone()
                    opt_compliance[update_flag] = compliance[update_flag].clone()
                    opt_palm_poses[update_flag] = torch.hstack([palm_poses, palm_oris])[update_flag].clone()
                    opt_R[update_flag] = self.R[update_flag].clone()
                    opt_t[update_flag] = self.t[update_flag].clone()
            if not isinstance(self.optim, torch.optim.LBFGS):
                self.optim.step()
            with torch.no_grad():
                compliance.clamp_(min=40.0) # prevent negative compliance
        print("Optimization time:", time.time() - start_ts)
        print("Margin:",opt_margin)
        return opt_joint_angle, opt_compliance, opt_target_pose, opt_palm_poses, opt_margin, opt_R, opt_t
            
