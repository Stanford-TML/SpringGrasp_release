import numpy as np
import open3d as o3d
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--num_views", type=int, default=1)
args = parser.parse_args()

camera_list = ["415","435","455"] # remove 415 for experiment
if args.num_views == 1:
    active_cameras = [0]
elif args.num_views == 2:
    active_cameras = [1, 2]
elif args.num_views == 3:
    active_cameras = [0, 1, 2]

pcds = []
bottom = 0.013
bound = 0.2
for camera in camera_list:
    pcd = o3d.io.read_point_cloud(f"assets/saved_pcds/{args.exp_name}/obj_cropped_{camera}.ply")
    pcds.append(pcd)
pcds = [pcds[i] for i in active_cameras]
o3d.visualization.draw_geometries(pcds)

full_point_cloud = o3d.geometry.PointCloud()
points = []
colors = []
for i in range(len(pcds)):
    points.append(np.asarray(pcds[i].points))
    colors.append(np.asarray(pcds[i].colors))

points = np.concatenate(points, axis=0)
colors = np.concatenate(colors, axis=0)

# Squash the points as floor
bottom_points = points.copy()
bottom_points[:,2] = bottom
bottom_colors = colors.copy()

full_point_cloud.points = o3d.utility.Vector3dVector(np.vstack([points, bottom_points]))
full_point_cloud.colors = o3d.utility.Vector3dVector(np.vstack([colors, bottom_colors]))
sampled = full_point_cloud.farthest_point_down_sample(1000)
# remove outliers again
sampled = sampled.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)[0]
cvx_hull, _ = sampled.compute_convex_hull()
cvx_hull.compute_vertex_normals()
cvx_hull.compute_triangle_normals()

o3d.visualization.draw_geometries([sampled])
o3d.visualization.draw_geometries([cvx_hull])
o3d.io.write_point_cloud("data/obj_cropped.ply",sampled)
o3d.io.write_triangle_mesh("data/obj_cropped.obj",cvx_hull) # For arm motion planning collision detection
points = np.asarray(sampled.points)
np.random.shuffle(points)
np.save("data/obj_cropped.npy", points)


