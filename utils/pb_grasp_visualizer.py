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

class HumanoidVisualizer:
    def __init__(self, robot_urdf, env_pcd=None, num_humanoid=1):
        self._client = pb.connect(pb.GUI)
        self.robot_ids = []
        for i in range(num_humanoid):
            self.robot_ids.append(pb.loadURDF(robot_urdf))
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.floor_id = pb.loadURDF("plane.urdf", useFixedBase=True)
        if env_pcd is not None:
            self.env_pcd = env_pcd
            pb.addUserDebugPoints(np.asarray(self.env_pcd.points),
                                  np.asarray(self.env_pcd.colors),
                                  pointSize=5)
        self.target_vis = []
            
    def visualize_robot(self, joint_angles, root_pose, target_pose, robot_id=0):
        """
        joint_angles: [29] np.ndarray
        root_pose: [6] np.ndarray
        target_pose: [2, 3] np.ndarray
        """
        root_pos = root_pose[:3]
        root_rot = Rotation.from_euler("XYZ", root_pose[3:]).as_quat()
        for i in range(len(target_pose)):
            self.target_vis.append(rb.create_primitive_shape(pb, 0, pb.GEOM_SPHERE, [0.015], color=COLOR_CODE[i], collidable=False,init_xyz=target_pose[i]))

        pb.resetBasePositionAndOrientation(self.robot_ids[robot_id], root_pos, root_rot)

        jid = 0
        for i in range(pb.getNumJoints(self.robot_ids[robot_id])):    
            if pb.getJointInfo(self.robot_ids[robot_id], i)[2] == pb.JOINT_REVOLUTE:
                pb.resetJointState(self.robot_ids[robot_id], i, joint_angles[jid])
                jid += 1
        

    def cleanup(self):
        input("Press Enter to continue...")
        for vis in self.target_vis:
            pb.removeBody(vis)
        self.target_vis = []
    