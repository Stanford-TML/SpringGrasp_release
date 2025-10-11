import os
import torch 
import numpy as np
from .rotation_conversions import quaternion_to_matrix, matrix_to_quaternion, axis_angle_to_quaternion, wxyz_to_xyzw, euler_angles_to_axis_angle
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters

script_path = os.path.realpath(__file__)
current_file_directory = os.path.dirname(script_path)

G1_ROTATION_AXIS = torch.tensor([[
    [0.0, 1.0, 0.0], # l_hip_pitch 1
    [1.0, 0.0, 0.0], # l_hip_roll 2
    [0.0, 0.0, 1.0], # l_hip_yaw 3
    [0.0, 1.0, 0.0], # l_knee 4
    [0.0, 1.0, 0.0], # l_ankle_pitch 5
    [1.0, 0.0, 0.0], # l_ankle_roll 6
    [0.0, 1.0, 0.0], # r_hip_pitch 7
    [1.0, 0.0, 0.0], # r_hip_roll 8
    [0.0, 0.0, 1.0], # r_hip_yaw 9
    [0.0, 1.0, 0.0], # r_knee 10
    [0.0, 1.0, 0.0], # r_ankle_pitch 11
    [1.0, 0.0, 0.0], # r_ankle_roll 12
    [0.0, 0.0, 1.0], # waist_yaw 13
    [1.0, 0.0, 0.0], # waist_roll 14
    [0.0, 1.0, 0.0], # torso 15
    [0.0, 1.0, 0.0], # l_shoulder_pitch 16
    [1.0, 0.0, 0.0], # l_shoulder_roll 17
    [0.0, 0.0, 1.0], # l_shoulder_yaw 18
    [0.0, 1.0, 0.0], # l_elbow 19
    [1.0, 0.0, 0.0], # l_wrist_roll 20
    [0.0, 1.0, 0.0], # l_wrist_pitch 21
    [0.0, 0.0, 1.0], # l_wrist_yaw 22
    [0.0, 1.0, 0.0], # r_shoulder_pitch 23
    [1.0, 0.0, 0.0], # r_shoulder_roll 24
    [0.0, 0.0, 1.0], # r_shoulder_yaw 25
    [0.0, 1.0, 0.0], # r_elbow 26
    [1.0, 0.0, 0.0], # r_wrist_roll 27
    [0.0, 1.0, 0.0], # r_wrist_pitch 28
    [0.0, 0.0, 1.0], # r_wrist_yaw 29
    ]]) # Need re indexing

JOINT_NAMES = ["left_hip_pitch",
               "left_hip_roll",
               "left_hip_yaw",
               "left_knee",
               "left_ankle_pitch",
               "left_ankle_roll",
               "right_hip_pitch",
               "right_hip_roll",
               "right_hip_yaw",
               "right_knee",
               "right_ankle_pitch",
               "right_ankle_roll",
               "waist_yaw",
               "waist_roll",
               "torso",
               "left_shoulder_pitch",
               "left_shoulder_roll",
               "left_shoulder_yaw",
               "left_elbow",
               "left_wrist_roll",
               "left_wrist_pitch",
               "left_wrist_yaw",
               "right_shoulder_pitch",
               "right_shoulder_roll",
               "right_shoulder_yaw",
               "right_elbow",
               "right_wrist_roll",
               "right_wrist_pitch",
               "right_wrist_yaw"]


class HumanoidModel:

    def __init__(self, mjcf_file = f"{current_file_directory}/../assets/g1/g1_29dof_rev_1_0.xml", 
                 extend_hand = False, extend_toe=False, extend_palm=False, extend_head = False,  device = torch.device("cpu")):
        self.mjcf_data = mjcf_data = self.from_mjcf(mjcf_file)
        self.extend_hand = extend_hand
        self.extend_head = extend_head
        self.extend_toe = extend_toe
        self.extend_palm = extend_palm
        self._remove_idx = 0
        if extend_hand:
            self.model_names = mjcf_data['node_names'] + ["left_hand_link", "right_hand_link"]
            self._parents = torch.cat((mjcf_data['parent_indices'], torch.tensor([15, 19]))).to(device) # Adding the hands joints
            arm_length = 0.3
            self._offsets = torch.cat((mjcf_data['local_translation'], torch.tensor([[arm_length, 0, 0], [arm_length, 0, 0]])), dim = 0)[None, ].to(device)
            self._local_rotation = torch.cat((mjcf_data['local_rotation'], torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])), dim = 0)[None, ].to(device)
            self._remove_idx += 2
        else:
            self._parents = mjcf_data['parent_indices']
            self.model_names = mjcf_data['node_names']
            self._offsets = mjcf_data['local_translation'][None, ].to(device)
            self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)
        if extend_toe:
            assert not extend_hand
            self.model_names = mjcf_data['node_names'] + ["left_toe_link", "right_toe_link"]
            self._parents = torch.cat((mjcf_data['parent_indices'], torch.tensor([6, 12]))).to(device) # Adding the hands joints
            foot_length = 0.08
            z_offset = -0.04
            self._offsets = torch.cat((mjcf_data['local_translation'], torch.tensor([[foot_length, 0, z_offset], [foot_length, 0, z_offset]])), dim = 0)[None, ].to(device)
            self._local_rotation = torch.cat((mjcf_data['local_rotation'], torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])), dim = 0)[None, ].to(device)
            self._remove_idx += 2
        if extend_palm:
            assert not extend_hand
            self.model_names = self.model_names + ["left_index_link", "left_little_link", "right_index_link", "right_little_link"]
            self._parents = torch.cat((self._parents, torch.tensor([22, 22, 29, 29]).to(device))).to(device) # Adding the hands joints
            palm_length = 0.06
            finger_offset = 0.02
            self._offsets = torch.cat((self._offsets, torch.tensor([[[palm_length,0, finger_offset], [palm_length, 0,-finger_offset], [palm_length, 0,finger_offset], [palm_length, 0,-finger_offset]]]).to(device)), dim = 1).to(device)
            self._local_rotation = torch.cat((self._local_rotation, torch.tensor([[[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]]).to(device)), dim = 1).to(device)
            self._remove_idx += 4
        if extend_head:
            self._remove_idx += 1
            self.model_names = self.model_names + ["head_link"]
            self._parents = torch.cat((self._parents, torch.tensor([0]).to(device))).to(device) # Adding the hands joints
            head_length = 0.4
            self._offsets = torch.cat((self._offsets, torch.tensor([[[0, 0, head_length]]]).to(device)), dim = 1).to(device)
            self._local_rotation = torch.cat((self._local_rotation, torch.tensor([[[1, 0, 0, 0]]]).to(device)), dim = 1).to(device)
            
        
        self.joints_range = mjcf_data['joints_range'].to(device)
        self._local_rotation_mat = quaternion_to_matrix(self._local_rotation).float() # w, x, y ,z
        
    @staticmethod
    def names_to_indices(names):
        return [JOINT_NAMES.index(name) for name in names]
    

    def from_mjcf(self, path):
        # function from Poselib: 
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
            
        xml_joint_root = xml_body_root.find("joint")
        
        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint")
            for joint in all_joints:
                if not joint.attrib.get("range") is None: 
                    joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)
            return node_index
        
        _add_xml_node(xml_body_root, -1, 0)
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "joints_range": torch.from_numpy(np.array(joints_range))
        }
        
    def fk_batch(self, pose, trans, convert_to_mat=True, return_full = False, dt=1/30):
        device, dtype = pose.device, pose.dtype
        pose_input = pose.clone()
        B, seq_len = pose.shape[:2]
        pose = pose[..., :len(self._parents), :] # H1 fitted joints might have extra joints
        if self.extend_hand and self.extend_head and pose.shape[-2] == 22:
            pose = torch.cat([pose, torch.zeros(B, seq_len, 1, 3).to(device).type(dtype)], dim = -2) # adding hand and head joints ???

        if convert_to_mat:
            pose_quat = axis_angle_to_quaternion(pose)
            pose_mat = quaternion_to_matrix(pose_quat)
        else:
            pose_mat = pose
        if pose_mat.shape != 5:
            pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        J = pose_mat.shape[2] - 1  # Exclude root
        
        wbody_pos, wbody_mat = self._compute_kinematics(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)
        
        return_dict = EasyDict()
        
        
        wbody_rot = wxyz_to_xyzw(matrix_to_quaternion(wbody_mat))
        if self.extend_hand:
            return_dict.global_translation_extend = wbody_pos.clone()
            return_dict.global_rotation_mat_extend = wbody_mat.clone()
            return_dict.global_rotation_extend = wbody_rot
            
            # wbody_pos = wbody_pos[..., :-self._remove_idx, :]
            # wbody_mat = wbody_mat[..., :-self._remove_idx, :, :]
            # wbody_rot = wbody_rot[..., :-self._remove_idx, :]
        
        return_dict.global_translation = wbody_pos
        return_dict.global_rotation_mat = wbody_mat
        return_dict.global_rotation = wbody_rot
        
        return return_dict
    
    def _compute_kinematics(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """
        
        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []

        expanded_offsets = (self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype))
        # print(expanded_offsets.shape, J)

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                jpos = (torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1) + positions_world[self._parents[i]])
                rot_mat = torch.matmul(rotations_world[self._parents[i]], torch.matmul(self._local_rotation_mat[:,  (i):(i + 1)], rotations[:, :, (i - 1):i, :]))
                # rot_mat = torch.matmul(rotations_world[self._parents[i]], rotations[:, :, (i - 1):i, :])
                # print(rotations[:, :, (i - 1):i, :].shape, self._local_rotation_mat.shape)
                
                positions_world.append(jpos)
                rotations_world.append(rot_mat)
        
        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world
    
    def compute_forward_kinematics(self, q, root_pos, root_rot):
        """
        q: in axis-angle representation [B, J]
        root_pos: [B, 3]
        root_rot: in euler angles [B,3]
        output: link positions [B, J, 3]
        """
        root_aa = euler_angles_to_axis_angle(root_rot, convention='XYZ')
        q = q[:,:, None] * G1_ROTATION_AXIS.to(q.device)
        pose_aa = torch.cat([root_aa[:, None], q], dim=1)
        return self.fk_batch(pose_aa[:,None,:,:], root_pos[:, None,:])["global_translation"].squeeze(1)

        

    
      
    

if __name__ == "__main__":
    humanoid = HumanoidModel()
    #pose_aa = torch.zeros(1, 1, 30, 3)
    root_pos = torch.zeros(1, 3).requires_grad_(True)
    root_rot = torch.zeros(1, 3).requires_grad_(True)
    q = torch.zeros(1, 29).requires_grad_(True)

    out = humanoid.compute_forward_kinematics(q, root_pos, root_rot)
    loss = out.sum()
    loss.backward()
    print(out)
    print(q.grad)

