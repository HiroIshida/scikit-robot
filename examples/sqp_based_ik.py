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
    robot_model.r_wrist_roll_link]

coll_link_list = [
    robot_model.r_upper_arm_link, robot_model.r_forearm_link,
    robot_model.r_gripper_palm_link, robot_model.r_gripper_r_finger_link,
    robot_model.r_gripper_l_finger_link]

joint_list = [link.joint for link in link_list]

box_center = np.array([0.9, -0.2, 0.9])
box = Box(extents=[0.7, 0.5, 0.6], with_sdf=True)
box.translate(box_center)

sscc = TinyfkSweptSphereSdfCollisionChecker(lambda X: box.sdf(X), robot_model)
for link in coll_link_list:
    sscc.add_collision_link(link)

target_pos = [1.8, 0.6, 1.5]
target_rpy = [0, 0, 0]
target_pose = np.hstack([target_pos, target_rpy])
av_init = get_robot_config(robot_model, joint_list, with_base=True)
av_sol = tinyfk_sqp_inverse_kinematics("r_gripper_tool_frame", target_pose, av_init, joint_list, sscc, with_base=True)

set_robot_config(robot_model, joint_list, av_sol, with_base=True)

viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(641, 480))
viewer.add(robot_model)
viewer.add(Axis(pos=target_pos))
viewer.show()

