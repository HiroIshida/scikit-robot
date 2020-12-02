#!/usr/bin/env python

import skrobot
import numpy as np
import time
from skrobot.model import Box, MeshLink, Axis
from skrobot.planner.utils import get_robot_state, set_robot_state
from skrobot.planner.swept_sphere import assoc_swept_sphere
from skrobot.planner import CollisionChecker

try:
    robot_model
except:
    robot_model = skrobot.models.urdf.RobotModelFromURDF(
            urdf_file=skrobot.data.pr2_urdfpath())
    robot_model.init_pose()

    table = Box(extents=[0.7, 1.0, 0.05], with_sdf=True) 
    table.translate([0.8, 0.0, 0.65])

    viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))

    link_idx_table = {}
    for link_idx in range(len(robot_model.link_list)):
        name = robot_model.link_list[link_idx].name
        link_idx_table[name] = link_idx

    coll_link_names = ["r_upper_arm_link", "r_forearm_link", "r_gripper_palm_link", "r_gripper_r_finger_link", "r_gripper_l_finger_link"]

    coll_link_list = [robot_model.link_list[link_idx_table[lname]]
                 for lname in coll_link_names]

    link_names = ["r_shoulder_pan_link", "r_shoulder_lift_link",
                  "r_upper_arm_roll_link", "r_elbow_flex_link",
                  "r_forearm_roll_link", "r_wrist_flex_link",
                  "r_wrist_roll_link"]

    link_list = [robot_model.link_list[link_idx_table[lname]]
                 for lname in link_names]
    joint_list = [link.joint for link in link_list]

set_robot_state(robot_model, joint_list, [0.4, 0.6] + [-0.7]*5)

cc = CollisionChecker(table.sdf, robot_model)
[cc.add_collision_link(l) for l in coll_link_list]
cc.add_coll_spheres_to_viewer(viewer)
cc.update_color()
viewer.add(table)
viewer.add(robot_model)
viewer.show()
