import yaml
import pybullet as pb
import rigidBodySento as rb
import numpy as np

c = pb.connect(pb.GUI)

configs = yaml.safe_load(open("curobo_ws/curobo/src/curobo/content/configs/robot/spheres/iiwa14.yml","r"))

robot = pb.loadURDF("assets/kuka_allegro/model_left.urdf", useFixedBase=True)

default_hand_q = np.array([0, 0,
                          -0.12999636, 0.46788138, 0.438807, 0.48968481, 0.0,
                          -0.02283426, 0.26199859, 0.73503519, 0.75897687, 0.0,
                          0.12591666, 0.27977724, 0.65864516, 0.69471026, 0.0,
                          1.55866289, 0.16972215, -0.15359271,  1.68753028, 0.0])

for i in range(len(default_hand_q)):
    pb.resetJointState(robot, i+7, default_hand_q[i])

link_names = {"lbr_iiwa_link_0": -1}
for i in range(8):
    link_names[pb.getJointInfo(robot, i)[12].decode("utf-8")] = i

print(link_names)

color_codes = [[1,0,0,0.7],[0,1,0,0.7]]

for i, link in enumerate(configs["collision_spheres"].keys()):
    if link not in link_names:
        continue
    link_id = link_names[link]
    link_pos, link_ori = rb.get_link_com_xyz_orn(pb, robot, link_id)
    for sphere in configs["collision_spheres"][link]:
        s = rb.create_primitive_shape(pb, 0.0, shape=pb.GEOM_SPHERE, dim=(sphere["radius"],), collidable=False, color=color_codes[i%2])
        # Place the sphere relative to the link
        world_coord = list(pb.multiplyTransforms(link_pos, link_ori, sphere["center"], [0,0,0,1])[0])
        world_coord[1] += 0.0
        pb.resetBasePositionAndOrientation(s, world_coord, [0,0,0,1])


input()
