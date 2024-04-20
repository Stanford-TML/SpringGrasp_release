import numpy as np
import open3d as o3d
import pybullet as pb
from argparse import ArgumentParser
c = pb.connect(pb.GUI)
camera_pos = np.array([-0.3, 0., 0.])
lookat_pos = np.array([0., 0., 0.])

IMAGE_SHAPE = [1000,1000]
FAR = 1.0
NEAR = 0.01

parser = ArgumentParser()
parser.add_argument("--exp_name", type=str, default="lego")
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--vis", action="store_true", default=False)
args = parser.parse_args()

def getIntrinsicParams(proj_mat, w,h):
    f_x = proj_mat[0] * w / 2
    f_y = proj_mat[5] * h / 2
    c_x = (-proj_mat[2] * w + w) / 2
    c_y = (proj_mat[6] * h + h) / 2
    return np.array([w, h, f_x, f_y, c_x, c_y])

def get_depth_image(projectionMat, viewMat):
    img = pb.getCameraImage(IMAGE_SHAPE[0], IMAGE_SHAPE[1], viewMatrix=viewMat, projectionMatrix=projectionMat)
    depth_image = NEAR * FAR /(FAR - (FAR - NEAR)*img[3])
    depth_image = (depth_image * 1000).astype(np.uint16)
    return depth_image

def create_partial_pointcloud(depth_image, intrinsic, extrinsic, noise=0.0):
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_image),
                                                          o3d.camera.PinholeCameraIntrinsic(int(intrinsic[0]), int(intrinsic[1]),*intrinsic[2:]),
                                                          depth_scale=1000.0, depth_trunc=9.9)
    points = np.asarray(pcd.points) * np.array([1,-1,-1]) + np.random.normal(0, noise, size=(len(pcd.points),3))
    mask = points[:,2] > -0.9
    pcd.points = o3d.utility.Vector3dVector(points[mask])
    pcd.transform(np.linalg.inv(extrinsic))
    simple_pcd = pcd.farthest_point_down_sample(2048)
    if args.vis:
        o3d.visualization.draw_geometries([simple_pcd])
    # generate random shuffle index
    shuffle_idx = np.arange(len(simple_pcd.points))
    np.random.shuffle(shuffle_idx)
    return np.asarray(simple_pcd.points)[shuffle_idx]

o_id = pb.loadURDF(f"assets/{args.exp_name}/{args.exp_name}.urdf", useFixedBase=True)

viewMat = pb.computeViewMatrix(cameraEyePosition=camera_pos.tolist(), 
                               cameraTargetPosition=lookat_pos.tolist(), 
                               cameraUpVector=[0,0,1])
projMat = pb.computeProjectionMatrixFOV(fov=70, aspect=1, nearVal=NEAR, farVal=FAR)
intrinsic = getIntrinsicParams(projMat, IMAGE_SHAPE[0], IMAGE_SHAPE[1])
vM = np.asarray(viewMat).reshape(4,4).T
depth_image = get_depth_image(projMat, viewMat)
partial_pcd = create_partial_pointcloud(depth_image, intrinsic, vM, noise=0.0)

np.save(f"./partial_pcd/{args.exp_name}.npy", partial_pcd)