python optimize_pregrasp.py --exp_name realsense --fast_exp --weight_config 16
python traj_gen.py --exp_name realsense
python verify_grasp_robot.py --exp_name realsense --traj traj