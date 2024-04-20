import pybullet as pb
import numpy as np
import open3d as o3d
import torch
import time
from optimize_pregrasp import optimal_transformation_batch, force_eq_reward
from math_utils import solve_minimum_wrench
from scipy.spatial.transform import Rotation

pb.connect(pb.DIRECT)

def construct_triangle(a,b,c):
    offset = np.array([0., 0., 0.1])
    vertices = np.array([a,b,c,a+offset,b+offset,c+offset])
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

def construct_simple_arm(anchor, theta):
    quat = Rotation.from_euler("XYZ", np.array([0.0, 0.0, theta])).as_quat()
    r = pb.loadURDF("simple_robot.urdf", basePosition=anchor, baseOrientation=quat, useFixedBase=True)
    return r
    
def solve_arm_ik(ee_pos, r_id):
    joint_angles = pb.calculateInverseKinematics(r_id, 2, ee_pos, maxNumIterations=100)
    # Verify IK solution
    current_joint_angles = get_joint_angles(r_id)
    set_joint_angles(r_id, joint_angles)
    solved_ee_pose = np.array(pb.getLinkState(r_id, 2)[0])
    error = np.linalg.norm(solved_ee_pose - ee_pos)
    if error > 1e-3:
        set_joint_angles(r_id, current_joint_angles)
        return None
    set_joint_angles(r_id, current_joint_angles)
    return np.array(joint_angles)

def get_joint_angles(r_id):
    joint_angles = []
    for i in range(2):
        joint_angles.append(pb.getJointState(r_id, i)[0])
    return np.array(joint_angles)

def set_joint_angles(r_id, joint_angles):
    for i in range(len(joint_angles)):
        pb.resetJointState(r_id, i, joint_angles[i])

def solve_inv_jac(r_id, joint_angles):
    jac,_ = pb.calculateJacobian(r_id, 2, [0,0,0], joint_angles.tolist(), [0.0, 0.0], [0.0, 0.0])
    jac_inv = np.linalg.pinv(jac)
    return jac_inv

def compute_contact_margin(tip_pose, target_pose, current_normal, friction_mu):
        force_dir = tip_pose - target_pose
        force_dir = force_dir / force_dir.norm(dim=2, keepdim=True)
        ang_diff = torch.einsum("ijk,ijk->ij",force_dir, current_normal)
        cos_mu = torch.sqrt(1/(1+torch.tensor(friction_mu)**2))
        margin = (ang_diff - cos_mu).clamp(-0.999)
        reward = (0.2 * torch.log(ang_diff+1)+ 0.8*torch.log(margin+1)).sum(dim=1)
        return reward

def optimize_target_compliance(tip_poses, target_poses, tip_normals, compliance, friction_mu, min_force):
    """
    tip_poses: [n_batch, 3, 3] np.ndarray
    target_poses: [n_batch, 3, 3] np.ndarray
    tip_normals: [n_batch, 3, 3] np.ndarray
    compliance: [n_batch, 3], np.ndarray
    friction_mu: scaler
    """
    # debug shape
    tip_poses = torch.from_numpy(tip_poses).cuda().double()
    target_poses_xy = torch.from_numpy(target_poses[:,:,:2]).cuda().requires_grad_(True).double()
    target_poses_z = torch.from_numpy(target_poses[:,:,[2]]).cuda().double()
    tip_normals = torch.from_numpy(tip_normals).cuda().double()
    compliance = torch.from_numpy(compliance).cuda().requires_grad_(True).double()
    optimizer = torch.optim.RMSprop([{"params":target_poses_xy, "lr":0.02},
                                     {"params":compliance, "lr":0.2}])
    opt_target_poses = torch.cat([target_poses_xy, target_poses_z], dim=2).clone()
    opt_compliance = compliance.clone()
    opt_value = torch.inf * torch.ones(tip_poses.shape[0], device=tip_poses.device).double()
    opt_margin = torch.zeros(tip_poses.shape[0], 3).cuda().double()
    opt_R = torch.zeros(tip_poses.shape[0], 3, 3).cuda().double()
    opt_t = torch.zeros(tip_poses.shape[0], 3).cuda().double()
    for i in range(200):
        optimizer.zero_grad()
        target_poses = torch.cat([target_poses_xy, target_poses_z], dim=2)
        reward, margin, force_norm, R, t = force_eq_reward(tip_poses, target_poses, compliance, friction_mu, tip_normals)
        init_reward = compute_contact_margin(tip_poses, target_poses, tip_normals, friction_mu)
        loss = (-reward - init_reward) * 10.0
        loss += -force_norm.clamp(max=min_force).sum(dim=1)

        loss.sum().backward()
        with torch.no_grad():
            update_flag = (loss < opt_value)
            if update_flag.sum() > 0:
                opt_target_poses[update_flag] = target_poses[update_flag].clone()
                opt_compliance[update_flag] = compliance[update_flag].clone()
                opt_value[update_flag] = reward[update_flag].clone()
                opt_margin[update_flag] = margin[update_flag].clone()
                opt_R[update_flag] = R[update_flag].clone()
                opt_t[update_flag] = t[update_flag].clone()
        optimizer.step()
        with torch.no_grad():
            compliance.clamp_(min=5.0)
    opt_margin = opt_margin.detach().cpu().numpy()
    feasible_flag = (opt_margin > 0.0).all(axis=1)
    opt_target_poses = opt_target_poses.detach().cpu().numpy()[feasible_flag]
    opt_compliance = opt_compliance.detach().cpu().numpy()[feasible_flag]
    opt_margin = opt_margin[feasible_flag]
    opt_tip_pose = tip_poses[feasible_flag]
    opt_tip_pose_after = torch.bmm(opt_R[feasible_flag], opt_tip_pose.transpose(1,2)).transpose(1,2) + opt_t[feasible_flag].unsqueeze(1)
    return opt_target_poses, opt_compliance, opt_margin, opt_tip_pose.detach().cpu().numpy(), opt_tip_pose_after.detach().cpu().numpy()

a = np.array([0.0, 0.0, 0.0])
b = np.array([1.0, 0.0, 0.0])
c = np.array([0.5, 1.0, 0.0])
# a = np.array([0.0, 0.0, 0.0])
# b = np.array([1.0, 0.0, 0.0])
# c = np.array([1.0, 1.0, 0.0])
n1 = np.array([0.0, -1.0, 0.0])
n2 = np.array([2/np.sqrt(5), 1/np.sqrt(5), 0.0])
n3 = np.array([-2/np.sqrt(5), 1/np.sqrt(5), 0.0])
#n2 = np.array([1.0, 0.0, 0.0])
#n3 = np.array([-np.sqrt(2)/2, np.sqrt(2)/2, 0.0])
torque_max = np.array([12.0, 12.0])
# Solve target pose and compliance
num_envs = 4000
num_exp = 100

feasible_num = 0
fc_num = 0

r1 = construct_simple_arm(np.array([0.5, -1, -0.05]), np.pi/2)
r2 = construct_simple_arm(np.array([0, 1, -0.05]), -np.pi/2)
r3 = construct_simple_arm(np.array([1, 1, -0.05]), -np.pi/2)
o = construct_triangle(a,b,c)

min_force = 5.0
num_fk_feasible = 0
while num_fk_feasible < 100:
    x = float(np.random.random(1) - 0.5)
    y = float(np.random.random(1) - 0.5)
    t = np.array([x, y, 0.0])
    theta = float(np.random.random(1) - 0.5) * np.pi/2
    R = Rotation.from_euler("XYZ", np.array([0.0, 0.0, theta])).as_matrix()


    # tip_poses = np.array([[0.8, 0.0, 0.0],[0.2, 0.4, 0.0],[0.6, 0.8, 0.0]])
    tip_poses = np.array([[0.5, 0.0, 0.0],[0.8, 0.4, 0.0],[0.2, 0.4, 0.0]])
    for i in range(3):
        tip_poses[i] = R@tip_poses[i] + t
    tip_poses = tip_poses.reshape(1,3,3).repeat(num_envs, axis=0)
    #target_poses = tip_poses.mean(axis=1, keepdims=True).repeat(3, axis=1)
    target_poses = np.random.random((num_envs-1,3,3))
    target_poses = np.concatenate([target_poses, tip_poses[0].mean(axis=0, keepdims=True).repeat(3,axis=0).reshape(1,3,3)])
    for e in range(num_envs):
        for i in range(3):
            target_poses[e,i] = R@target_poses[e,i] + t

    tip_normals = np.stack([R@n1,R@n2,R@n3]).reshape(1,3,3).repeat(num_envs, axis=0) # [n_batch, 3, 3]
    compliance = np.array([[10.0, 10.0, 10.0]] * num_envs)

    # Check ik feasibility before optimization

    # draw normal in pybullet
    # for i in range(3):
    #     pb.addUserDebugLine(tip_poses[0,i], tip_poses[0,i]+tip_normals[0,i]*0.1, [1,0,0], lineWidth=3)
    pb.resetBasePositionAndOrientation(o, t, Rotation.from_matrix(R).as_quat())

    tip_pose = tip_poses[0]
    joint_angles_1 = solve_arm_ik(tip_pose[0], r1)
    joint_angles_2 = solve_arm_ik(tip_pose[1], r2)
    joint_angles_3 = solve_arm_ik(tip_pose[2], r3)

    if joint_angles_1 is None or joint_angles_2 is None or joint_angles_3 is None:
        #print("IK failed!", joint_angles_1, joint_angles_2, joint_angles_3)
        continue
    else:
        num_fk_feasible += 1
    # set_joint_angles(r1, joint_angles_1)
    # set_joint_angles(r2, joint_angles_2)
    # set_joint_angles(r3, joint_angles_3)
    # input()

    opt_target, opt_compliance, opt_margin, opt_tip_pose, opt_tip_pose_after = optimize_target_compliance(tip_poses, target_poses, tip_normals, compliance, 0.5, min_force)

    # Get jacobians pseudo inverse
    num_active_envs = opt_margin.shape[0]
    tip_pose = tip_poses[0]
    jac_inv1 = solve_inv_jac(r1, joint_angles_1).reshape(1,-1,3).repeat(num_active_envs,axis=0) # [num_envs, 2, 3]
    jac_inv2 = solve_inv_jac(r2, joint_angles_2).reshape(1,-1,3).repeat(num_active_envs,axis=0)
    jac_inv3 = solve_inv_jac(r3, joint_angles_3).reshape(1,-1,3).repeat(num_active_envs,axis=0)

    F_before = (opt_target - opt_tip_pose) * opt_compliance.reshape(num_active_envs,3,1)
    F_after = (opt_target - opt_tip_pose_after) * opt_compliance.reshape(num_active_envs,3,1)

    torque1 = np.einsum("bij,bjk->bik", jac_inv1, F_before[:,0].reshape(num_active_envs,3,1)).reshape(num_active_envs,2) # [num_envs, 2]
    torque2 = np.einsum("bij,bjk->bik", jac_inv2, F_before[:,1].reshape(num_active_envs,3,1)).reshape(num_active_envs,2)
    torque3 = np.einsum("bij,bjk->bik", jac_inv3, F_before[:,2].reshape(num_active_envs,3,1)).reshape(num_active_envs,2)
    


    flag_1 = (np.abs(torque1) < torque_max).all(axis=1) * (np.linalg.norm(F_after[:,0],axis=1) > min_force)
    flag_2 = (np.abs(torque2) < torque_max).all(axis=1) * (np.linalg.norm(F_after[:,1],axis=1) > min_force)
    flag_3 = (np.abs(torque3) < torque_max).all(axis=1) * (np.linalg.norm(F_after[:,2],axis=1) > min_force)
    flag = flag_1 * flag_2 * flag_3
    if flag.sum() == 0:
        continue

    opt_tip_pose_after = opt_tip_pose_after[flag]
    post_success = False
    for i in range(len(opt_tip_pose_after)):
        joint_angles_1_after = solve_arm_ik(opt_tip_pose_after[i,0], r1)
        joint_angles_2_after = solve_arm_ik(opt_tip_pose_after[i,1], r2)
        joint_angles_3_after = solve_arm_ik(opt_tip_pose_after[i,2], r3)
        if joint_angles_1_after is None or joint_angles_2_after is None or joint_angles_3_after is None:
            continue
        jac_inv1_after = solve_inv_jac(r1, joint_angles_1_after)
        jac_inv2_after = solve_inv_jac(r2, joint_angles_2_after)
        jac_inv3_after = solve_inv_jac(r3, joint_angles_3_after)
        torque1_after = jac_inv1_after @ F_after[i,0]
        torque2_after = jac_inv2_after @ F_after[i,1]
        torque3_after = jac_inv3_after @ F_after[i,2]
        if (np.abs(torque1_after) > torque_max).all() or (np.abs(torque2_after) > torque_max).all() or (np.abs(torque3_after) > torque_max).all():
            continue
        else:
            post_success = True
            break
    if not post_success:
        continue
    # requires new jacobian inverse each env may different
    #torque1_after = np.einsum("bij,bjk->bik", jac_inv1, F_after[:,0].reshape(num_active_envs,3,1)).reshape(num_active_envs,2)
    #torque2_after = np.einsum("bij,bjk->bik", jac_inv2, F_after[:,1].reshape(num_active_envs,3,1)).reshape(num_active_envs,2)
    #torque3_after = np.einsum("bij,bjk->bik", jac_inv3, F_after[:,2].reshape(num_active_envs,3,1)).reshape(num_active_envs,2)

    # Check whether torque are applicable
    feasible_num += 1
    # What if ik failed, how to count success rate?
    # Solve force closure once.

    solution, total_wrench = solve_minimum_wrench(torch.tensor(tip_poses[0]), torch.tensor(tip_normals[0]), 0.5, min_force=min_force)
    #print("total_wrench:",total_wrench, solution)

    # Check whether the force is obtainable
    F_fc = solution.numpy()
    torque1 = jac_inv1[0] @ F_fc[0]
    torque2 = jac_inv2[0] @ F_fc[1]
    torque3 = jac_inv3[0] @ F_fc[2]

    flag = (np.abs(torque1) < torque_max).all() * (np.abs(torque2) < torque_max).all() * (np.abs(torque3) < torque_max).all()
    print(torque1, torque2, torque3)
    print("Feasibillity of fc solution:", flag, num_fk_feasible) # How many feasible solutions
    if flag:
        fc_num += 1

print("Feasibility rate:", fc_num/feasible_num, fc_num, feasible_num)

# mesh,o = construct_triangle(a,b,c)
# mesh.compute_vertex_normals()
# mesh.compute_triangle_normals()
# pb.resetJointState(r, 0, np.pi/3)
# pb.resetJointState(r, 1, np.pi/3)
# q = solve_arm_ik(np.array([0, 0, 0.0]), r)
# set_joint_angles(r, q)
# o3d.visualization.draw_geometries([mesh])
    
