import pybullet as pb
import pybullet_data
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import utils.rigidBodySento as rb

COLOR_CODE = [[1,0,0,1],
              [0,1,0,1],
              [0,0,1,1],
              [1,1,1,1]]

class GraspVisualizer:
    def __init__(self, hand_urdf, object_pcd):
        """
        hand_urdf: str
        object_pcd: o3d.geometry.PointCloud
        """
        self._client = pb.connect(pb.GUI)
        self.hand_id = pb.loadURDF(hand_urdf, flags=pb.URDF_MERGE_FIXED_LINKS)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.floor_id = pb.loadURDF("plane.urdf", useFixedBase=True)
        self.object_pcd = object_pcd
        pb.addUserDebugPoints(np.asarray(self.object_pcd.points),
                              np.asarray(self.object_pcd.colors),
                              pointSize=5)
        
    def visualize_grasp(self, joint_angles, wrist_pose, target_pose):
        """
        joint_angles: [16,] np.ndarray
        wrist_pose: [6,] np.ndarray
        target_pose: [4,3] np.ndarray
        """
        wrist_pos = wrist_pose[:3]
        wrist_ori = Rotation.from_euler("XYZ",wrist_pose[3:]).as_quat()
        target_vis = []
        for i in range(len(target_pose)):
            target_vis.append(rb.create_primitive_shape(pb, 0, pb.GEOM_SPHERE, [0.015], color=COLOR_CODE[i], collidable=False,init_xyz=target_pose[i]))
        
        pb.resetBasePositionAndOrientation(self.hand_id, wrist_pos, wrist_ori)
        for i in range(16):
            pb.resetJointState(self.hand_id, i, joint_angles[i])
        input("Press Enter to continue...")
        for vis in target_vis:
            pb.removeBody(vis)
