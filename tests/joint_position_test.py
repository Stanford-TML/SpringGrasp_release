import rospy
import numpy as np
from allegro_hand_kdl.srv import PoseGoalRequest, PoseGoal
from sensor_msgs.msg import JointState



finger_pose = np.array([-0.12999636, 0.46788138, 0.438807, 0.48968481,
                        -0.02283426, 0.26199859, 0.73503519, 0.45897687,
                        0.12591666, 0.27977724, 0.65864516, 0.69471026,
                        1.55866289, 0.16972215, -0.15359271,  1.68753028])
current_q = np.array([0.0] * 16)

def callback(msg):
    current_q[:] = np.array(msg.position)

rospy.init_node("joint_position_test")
client = rospy.ServiceProxy("desired_pose", PoseGoal)
sub = rospy.Subscriber("/allegroHand_0/joint_states", JointState, callback)


# req = PoseGoalRequest()
#     # may be some sign need to invert...
# req.pose = finger_pose.tolist()
# res = client(req)
# print(res.success)
rate = rospy.Rate(10)
rate.sleep()
print(current_q)