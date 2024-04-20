import pybullet as pb
import time
import numpy as np
c = pb.connect(pb.GUI)

arm = pb.loadURDF("model.urdf",useFixedBase=True,flags=pb.URDF_MERGE_FIXED_LINKS)

joint_angles = [0.0,np.pi/6,0.0,-2/3*np.pi,0.0, np.pi/6, 0.0] + [0.0, np.pi/9, np.pi/9, np.pi/6, 
                0.0, np.pi/9, np.pi/9, np.pi/6, 
                0.0, np.pi/9, np.pi/9, np.pi/6, 
                2 * np.pi/6, np.pi/9, np.pi/6, np.pi/6]
print(pb.getNumJoints(arm))

for i in range(pb.getNumJoints(arm)):
    pb.resetJointState(arm,i,joint_angles[i])

while True:
    time.sleep(0.01)
