import time

import numpy as np

import skrobot
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.planner import tinyfk_sqp_plan_trajectory
from skrobot.planner import tinyfk_sqp_inverse_kinematics
from skrobot.planner import TinyfkSweptSphereSdfCollisionChecker
from skrobot.planner import ConstraintManager
from skrobot.planner import ConstraintViewer
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config
from skrobot.planner.utils import update_fksolver

robot_model = skrobot.models.PR2()
robot_model.reset_manip_pose()
update_fksolver(robot_model)

link_list = [
    robot_model.r_shoulder_pan_link, robot_model.r_shoulder_lift_link,
    robot_model.r_upper_arm_roll_link, robot_model.r_elbow_flex_link,
    robot_model.r_forearm_roll_link, robot_model.r_wrist_flex_link,
    robot_model.r_wrist_roll_link,
    robot_model.l_shoulder_pan_link, robot_model.l_shoulder_lift_link,
    robot_model.l_upper_arm_roll_link, robot_model.l_elbow_flex_link,
    robot_model.l_forearm_roll_link, robot_model.l_wrist_flex_link,
    robot_model.l_wrist_roll_link]

coll_link_list = [
    robot_model.r_upper_arm_link, robot_model.r_forearm_link,
    robot_model.r_gripper_palm_link, robot_model.r_gripper_r_finger_link,
    robot_model.r_gripper_l_finger_link,
    robot_model.l_upper_arm_link, robot_model.l_forearm_link,
    robot_model.l_gripper_palm_link, robot_model.l_gripper_r_finger_link,
    robot_model.l_gripper_l_finger_link
    ]

joint_list = [link.joint for link in link_list]

box_center = np.array([1.8, 0.5, 0.9])
box = Box(extents=[0.7, 0.5, 0.6], with_sdf=True)
box.translate(box_center)

sscc = TinyfkSweptSphereSdfCollisionChecker(lambda X: box.sdf(X), robot_model)
for link in coll_link_list:
    sscc.add_collision_link(link)

target_pose_rarm = np.array([1.8, 0.6, 1.5])
target_pose_larm = np.array([1.4, 0.6, 1.5, 0, 0, 0])
target_pose_list = [target_pose_rarm, target_pose_larm]
coords_list = ["r_gripper_tool_frame", "l_gripper_tool_frame"]

av_init = get_robot_config(robot_model, joint_list, with_base=True)
ts = time.time()
av_sol = tinyfk_sqp_inverse_kinematics(coords_list, target_pose_list, av_init, joint_list, sscc, with_base=True)
print(time.time() - ts)

set_robot_config(robot_model, joint_list, av_sol, with_base=True)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
viewer.add(robot_model)
viewer.add(box)
for target_pose in target_pose_list:
    pos = target_pose[:3]
    viewer.add(Axis(pos=pos))
viewer.show()
