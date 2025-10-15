import open3d as o3d
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from gpis.gpis import GPISUnified
from utils.create_arrow import create_direct_arrow
from spring_grasp_planner.optimizers import SimpleSpringGraspOptimizer

device = torch.device("cpu")

def vis_grasp(tip_pose, target_pose):
    tip_pose = tip_pose.cpu().detach().numpy().squeeze()
    target_pose = target_pose.cpu().detach().numpy().squeeze()
    tips = []
    targets = []
    arrows = []
    color_code = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0]])
    for i in range(4):
        tip = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        tip.paint_uniform_color(color_code[i])
        tip.translate(tip_pose[i])
        target = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
        target.paint_uniform_color(color_code[i] * 0.4)
        target.translate(target_pose[i])
        # create arrow point from tip to target
        arrow = create_direct_arrow(tip_pose[i], target_pose[i])
        arrow.paint_uniform_color(color_code[i])
        tips.append(tip)
        targets.append(target)
        arrows.append(arrow)
    return tips, targets, arrows


if __name__ == "__main__":
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--pcd_file", type=str, default=None)
    parser.add_argument("--mode", type=str, default="sp") # fc
    parser.add_argument("--hand", type=str, default="allegro")
    parser.add_argument("--mass", type=float, default=0.5) # Not use in the paper, may run into numerical issues.
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--vis_gpis", action="store_true", default=False)
    parser.add_argument("--fast_exp", action="store_true", default=False)
    parser.add_argument("--weight_config", type=str, default=None)
    args = parser.parse_args()

    if args.weight_config is not None:
        weight_config = json.load(open(f"weight_config/{args.weight_config}.json"))
    else:
        weight_config = None

    if args.pcd_file is not None:
        pcd = o3d.io.read_point_cloud(args.pcd_file)
    else:
        pcd = o3d.io.read_point_cloud("data/obj_cropped.ply")
    
    # GPIS formulation
    nominal_bound = 0.15
    scale = nominal_bound / max(pcd.get_axis_aligned_bounding_box().get_extent())
    center = pcd.get_axis_aligned_bounding_box().get_center()
    gpis_pcd = copy.deepcopy(pcd)
    gpis_pcd.scale(scale, [center[0], center[1], 0.0])
    gpis = GPISUnified(0.08, 1, pcd=gpis_pcd, device=device)
    
    if args.vis_gpis:
        test_mean, test_var, test_normal, lb, ub = gpis.get_visualization_data(steps=100)
        plt.imshow(test_mean[:,:,50], cmap="seismic", vmax=gpis.bound, vmin=-gpis.bound)
        plt.show()
        vis_points, vis_normals, vis_var = gpis.topcd(test_mean, test_normal,test_var=test_var,steps=100)
        vis_var = vis_var / vis_var.max()
        fitted_pcd = o3d.geometry.PointCloud()
        fitted_pcd.points = o3d.utility.Vector3dVector(vis_points)
        fitted_pcd.normals = o3d.utility.Vector3dVector(vis_normals)
        # Create color code from variance
        colors = np.zeros_like(vis_points)
        colors[:,0] = vis_var
        colors[:,2] = 1 - vis_var
        fitted_pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([fitted_pcd])
        np.savez(f"gpis_states/{args.exp_name}_gpis.npz", mean=test_mean, var=test_var, normal=test_normal, ub=ub, lb=lb)
    
    init_tip_pose = torch.tensor([[[0.05,0.05, 0.02],[0.06,-0.0, -0.01],[0.03,-0.04,0.0],[-0.07,-0.01, 0.02]]]).double().to(device)
    
    
    
    compliance = torch.tensor([[80.0,80.0,80.0,160.0]]).to(device)
    friction_mu = args.friction


    grasp_optimizer = SimpleSpringGraspOptimizer(
                                                optimize_target=True,
                                                num_iters=args.num_iters,       
                                                mass=args.mass, com=gpis.center[:3],
                                                gravity=False,
                                                weight_config=weight_config)
    num_guesses = 5
    init_tip_pose = init_tip_pose.repeat_interleave(num_guesses,dim=0)
    #target_pose = target_pose.repeat_interleave(num_guesses,dim=0)
    compliance = compliance.repeat_interleave(num_guesses,dim=0)
    
    target_pose = init_tip_pose.mean(dim=1, keepdim=True).repeat(1,4,1)
    target_pose = target_pose + (init_tip_pose - target_pose) * 0.3
    if args.vis_gpis:
        for i in range(init_tip_pose.shape[0]):
            tips, targets, arrows = vis_grasp(init_tip_pose[i], target_pose[i])
            o3d.visualization.draw_geometries([pcd, *tips, *targets, *arrows])
    
    opt_tip_pose, opt_stiffness, opt_target_pose, opt_margin, opt_R, opt_t = grasp_optimizer.optimize(init_tip_pose,target_pose, compliance, friction_mu, gpis, verbose=True)
    # Visualize target and tip pose

    # scale back
    opt_tip_pose = (opt_tip_pose - torch.tensor([gpis.center[0], gpis.center[1], 0.0]).to(device)) / scale + torch.tensor([center[0], center[1], 0.0]).to(device)
    opt_target_pose = (opt_target_pose - torch.tensor([gpis.center[0], gpis.center[1], 0.0]).to(device)) / scale + torch.tensor([center[0], center[1], 0.0]).to(device)

    pcd.colors = o3d.utility.Vector3dVector(np.array([0.0, 0.0, 1.0] * len(pcd.points)).reshape(-1,3))
    

    # Visualize grasp in pybullet
        
    # After transformation
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    floor = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.01).translate([-0.25,-0.25,-0.01])
    idx_list = []
    print("Optimal compliance:", opt_stiffness)
    for i in range(opt_tip_pose.shape[0]):
        if opt_margin[i].min() > 0.0:
            idx_list.append(i)
        else:
            continue
        if args.fast_exp:
            continue
        tips, targets, arrows = vis_grasp(opt_tip_pose[i], opt_target_pose[i])
        o3d.visualization.draw_geometries([pcd, *tips, *targets, *arrows])
    print("Feasible indices:",idx_list, "Feasible rate:", len(idx_list)/opt_tip_pose.shape[0])
    if len(idx_list) > 0:
        np.save(f"data/contact_{args.exp_name}.npy", opt_tip_pose.cpu().detach().numpy()[idx_list])
        np.save(f"data/target_{args.exp_name}.npy", opt_target_pose.cpu().detach().numpy()[idx_list])
        np.save(f"data/compliance_{args.exp_name}.npy", opt_stiffness.cpu().detach().numpy()[idx_list])
