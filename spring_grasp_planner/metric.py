import torch

device = torch.device("cpu")

def optimal_transformation_batch(S1, S2, weight):
    """
    S1: [num_envs, num_points, 3]
    S2: [num_envs, num_points, 3]
    weight: [num_envs, num_points]
    """
    weight = torch.nn.functional.normalize(weight, dim=1, p=1.0)
    weight = weight.unsqueeze(2) # [num_envs, num_points, 1]
    c1 = S1.mean(dim=1, keepdim=True) # [num_envs, 3]
    c2 = S2.mean(dim=1, keepdim=True)
    H = (weight * (S1 - c1)).transpose(1,2) @ (weight * (S2 - c2))
    U, _, Vh = torch.linalg.svd(H+1e-8*torch.rand_like(H).to(device))
    V = Vh.mH
    R_ = V @ U.transpose(1,2)
    mask = R_.det() < 0.0
    sign = torch.ones(R_.shape[0], 3, 3).to(device)
    sign[mask,:, -1] *= -1.0
    R = (V*sign) @ U.transpose(1,2)
    t = (weight * (S2 - (R@S1.transpose(1,2)).transpose(1,2))).sum(dim=1) / weight.sum(dim=1)
    return R, t

# Assume friction is uniform
# Differentiable 
def force_eq_reward(tip_pose, target_pose, compliance, friction_mu, current_normal, mass=0.4, gravity=None, M=0.2, COM=[0.0, 0.05, 0.0]):
    """
    Params:
    tip_pose: world frame [num_envs, num_fingers, 3]
    target_pose: world frame [num_envs, num_fingers, 3]
    compliance: [num_envs, num_fingers]
    friction_mu: scaler
    current_normal: world frame [num_envs, num_fingers, 3]
    
    Returns:
    reward: [num_envs]
    """
    # Prepare dummy gravity
    # Assume COM is at center of target
    if COM is not None:
        com = torch.tensor(COM).to(device).double()
    if gravity is not None:
        dummy_tip = torch.zeros(tip_pose.shape[0], 1, 3).cpu()
        dummy_tip[:,0,:] = target_pose.mean(dim=1) if COM is None else com
        dummy_target = torch.zeros(target_pose.shape[0], 1, 3).cpu()
        dummy_target[:,0,2] = -M # much lower than center of mass
        dummy_compliance = gravity * mass/M * torch.ones(compliance.shape[0], 1).cpu()
        R,t = optimal_transformation_batch(torch.cat([tip_pose, dummy_tip], dim=1), 
                                           torch.cat([target_pose, dummy_target], dim=1), 
                                           torch.cat([compliance, dummy_compliance], dim=1))
    else:
        R,t = optimal_transformation_batch(tip_pose, target_pose, compliance)
    # tip position at equilirium
    tip_pose_eq = (R@tip_pose.transpose(1,2)).transpose(1,2) + t.unsqueeze(1)
    diff_vec = tip_pose_eq - target_pose
    force = compliance.unsqueeze(2) * (-diff_vec)
    dir_vec = diff_vec / diff_vec.norm(dim=2).unsqueeze(2)
    # Rotate local norm to equilibrium pose
    normal_eq = (R @ current_normal.transpose(1,2)).transpose(1,2)
    # measure cos similarity between force direction and surface normal
    # dir_vec: [num_envs, num_fingers, 3]
    # normal_eq: [num_envs, num_fingers, 3]
    ang_diff =  torch.einsum("ijk,ijk->ij",dir_vec, normal_eq)
    cos_mu = torch.sqrt(1/(1+torch.tensor(friction_mu)**2))
    margin = (ang_diff - cos_mu).clamp(min=-0.9999)
    # if (margin == -0.999).any():
    #     print("Debug:",dir_vec, normal_eq)
    # we hope margin to be as large as possible, never below zero
    force_norm = force.norm(dim=2)
    reward = (0.2 * torch.log(ang_diff+1)+ 0.8 * torch.log(margin+1)).sum(dim=1)
    return reward , margin, force_norm, R, t
    
def check_force_closure(tip_pose, target_pose, compliance, R, t, COM, mass, gravity):
    #R, t = optimal_transformation_batch(tip_pose.cpu().unsqueeze(0), target_pose.cpu().unsqueeze(0), compliance.cpu().unsqueeze(0))
    tip_pose = tip_pose.cpu().detach()
    target_pose = target_pose.cpu().detach()
    compliance = compliance.cpu().detach()
    R = R.squeeze().cpu().detach()
    t = t.squeeze().cpu().detach()
    
    tip_pose_after = (R@tip_pose.transpose(0,1)).transpose(0,1) + t
    force = compliance.unsqueeze(1) * (target_pose - tip_pose_after)
    gravity_force = (torch.tensor([0.0, 0.0, -0.2])- R@COM - t) * gravity * mass/0.2
    COM_after = (R@COM + t).cpu().detach()
    total_torque = ((tip_pose_after - COM_after).cross(force)).sum(dim=0)
    # total_torque += gravity_force.cross(torch.from_numpy(COM))
    total_force = force.sum(dim=0) + gravity_force
    print(total_torque, total_force)