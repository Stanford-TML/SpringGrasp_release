import torch

def optimal_transformation_batch(S1, S2, weight):
    """
    S1: [num_envs, num_points, 3]
    S2: [num_envs, num_points, 3]
    weight: [num_envs, num_points]
    """
    #weight = torch.nn.functional.normalize(weight, dim=1, p=1.0)
    weight = weight.unsqueeze(2) # [num_envs, num_points, 1]
    c1 = (weight*S1).mean(dim=1, keepdim=True) # [num_envs, 3]
    c2 = (weight*S2).mean(dim=1, keepdim=True)
    #c1 = S1.mean(dim=1, keepdim=True)
    #c2 = S2.mean(dim=1, keepdim=True)
    H = (weight * (S1 - c1)).transpose(1,2) @ (weight * (S2 - c2))
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.mH
    R_ = V @ U.transpose(1,2)
    mask = R_.det() < 0.0
    sign = torch.ones(R_.shape[0], 3, 3).cuda()
    sign[mask,:, -1] *= -1.0
    R = (V*sign) @ U.transpose(1,2)
    t = (weight * (S2 - (R@S1.transpose(1,2)).transpose(1,2))).sum(dim=1) / weight.sum(dim=1)
    return R, t

offsets = torch.tensor([12.0, 20.0, 1.0]).cuda()
offsets2 = torch.tensor([20.0, 5.0, 1.0]).cuda()

tip_poses = torch.randn(1,4,3).cuda() + offsets
tar_poses = torch.randn(1,4,3).cuda() * 0.2 + offsets2
print(tip_poses, tar_poses)
compliance = torch.tensor([[10.0, 100.0, 10.0, 200.0]]).cuda()
R,t = optimal_transformation_batch(tip_poses, tar_poses, compliance)
new_tip = (R@tip_poses.transpose(1,2)).transpose(1,2)+t
force = (new_tip - tar_poses) * compliance.unsqueeze(2)

total_force = force.sum(dim=1)
total_torque = (new_tip - tar_poses).cross(force).sum(dim=1)
print(total_force, total_torque)