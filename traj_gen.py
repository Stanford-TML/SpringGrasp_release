from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
import pybullet as pb
from argparse import ArgumentParser
import torch
import numpy as np
import time
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
    )
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from scipy.spatial.transform import Rotation


parser = ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--grasp_idx", type=int, default=0)
parser.add_argument("--traj_len", type=int, default=6)
parser.add_argument("--pause", action="store_true", default=False)
args = parser.parse_args()


tensor_args = TensorDeviceType()
world_file = "my_scene.yml"
robot_file = "iiwa14.yml"
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_file,
    world_file,
    tensor_args,
    trajopt_tsteps=args.traj_len+3,
    collision_checker_type=CollisionCheckerType.MESH,
    use_cuda_graph=True,
)
motion_gen = MotionGen(motion_gen_config)
robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
retract_cfg = motion_gen.get_retract_config()

joint_pose = torch.tensor([-0.7153848657390822, 0.23627692865494376, -0.06146527133579401, -1.2628601611175012, 0.01487889923612773, 1.6417360407890011, -2.344269879142319]).cuda()
# #Should be forward kinematics
# state = motion_gen.rollout_fn.compute_kinematics(
#     JointState.from_position(retract_cfg.view(1, -1))
# )

pb.connect(pb.GUI)

r = pb.loadURDF("assets/kuka_allegro/model_left.urdf", useFixedBase=True, basePosition=[-0.14134081,  0.50142033, -0.15], baseOrientation=[0, 0, -0.3826834, 0.9238795])
scene = pb.loadURDF("curobo_ws/curobo/src/curobo/content/assets/scene/nvblox/scene_updated.urdf", useFixedBase=True)
for j in range(7):
        pb.resetJointState(r, j, joint_pose[j])

default_hand_q = np.array([0, 0,
                          -0.12999636, 0.46788138, 0.438807, 0.48968481, 0.0,
                          -0.02283426, 0.26199859, 0.73503519, 0.45897687, 0.0,
                          0.12591666, 0.27977724, 0.65864516, 0.69471026, 0.0,
                          1.55866289, 0.16972215, -0.15359271,  1.68753028, 0.0])

for i in range(len(default_hand_q)):
    pb.resetJointState(r, i+7, default_hand_q[i])

wrist_poses = np.load(f"data/wrist_{args.exp_name}.npy") 
trajs = []
successes = []
for i in range(wrist_poses.shape[0]):
    wrist_pose = wrist_poses[i]
    wrist_ori = Rotation.from_euler("XYZ",wrist_pose[3:]).as_quat()[[3,0,1,2]]
    target_pose = Pose(torch.from_numpy(wrist_pose[:3]+np.array([0.0, 0.0, 0.0])).float().cuda(), quaternion=torch.from_numpy(wrist_ori).float().cuda())
    start_state = JointState.from_position(joint_pose.view(1, -1))
    t_start = time.time()
    result = motion_gen.plan(
            start_state,
            target_pose,
            enable_graph=True,
            enable_opt=False,
            max_attempts=10,
            num_trajopt_seeds=10,
            num_graph_seeds=10)
    if result is None:
        print("IK Failed!")
        successes.append(False)
        trajs.append(joint_pose.cpu().numpy().reshape(1, -1).repeat(args.traj_len, axis=0))
        continue
    print("Time taken: ", time.time()-t_start)
    print("Trajectory Generated: ", result.success)
    if not result.success:
        successes.append(False)
        trajs.append(joint_pose.cpu().numpy().reshape(1, -1).repeat(args.traj_len, axis=0))
        continue
    traj = result.get_interpolated_plan()

    if args.pause:
        input()
    traj = result.interpolated_plan
    for t in range(len(traj.position)):
        position = traj.position[t].cpu().numpy()
        #print(position)
        for j in range(7):
            pb.resetJointState(r, j, position[j])
        if args.pause: 
            input()
        time.sleep(0.1)
    position = result.debug_info["ik_solution"]
    for j in range(7):
        pb.resetJointState(r, j, position[j])
    if args.pause:
        input()
    successes.append(True)
    trajs.append(np.vstack([traj.position.cpu().numpy(), position.cpu().numpy()]))

trajs = np.stack(trajs)
print("Number of feasible trajectories:", len(trajs))
np.savez("traj.npz", trajs = trajs, successes = successes)