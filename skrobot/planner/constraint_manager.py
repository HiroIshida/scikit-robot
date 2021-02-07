import uuid
import numpy as np
from skrobot.planner import tinyfk_sqp_inverse_kinematics

# TODO check added eq_configuration is valid by pre-solving collision checking

class EqualityConstraint(object):
    def __init__(self, n_wp, n_dof, idx_wp, name):
        assert (idx_wp in range(n_wp)), "index {0} is out fo range".format(idx_wp)
        self.n_wp = n_wp
        self.n_dof = n_dof
        self.idx_wp = idx_wp
        self.name = name

    def _check_func(self, func):
        # TODO insert jacobian test utils here
        error_report_prefix = "ill-formed function {} is detected. ".format(self.name)
        av_seq_dummy = np.zeros((self.n_wp, self.n_dof))
        try:
            f, jac = func(av_seq_dummy)
        except:
            raise Exception(
                    error_report_prefix + 
                    "check input dimension of the function")
        assert f.ndim == 1, "f must be one dim"
        dim_constraint = len(f)
        dof_all = self.n_wp * self.n_dof
        assert jac.shape == (dim_constraint, dof_all), error_report_prefix \
                + "shape of jac is strainge. Desired: {0}, Copmuted {1}".format((dim_constraint, dof_all), jac.shape)

class ConfigurationConstraint(EqualityConstraint):
    def __init__(self, n_wp, n_dof, idx_wp, av_desired, name=None):
        if name is None:
            name = 'eq_config_const_{}'.format(str(uuid.uuid1()).replace('-', '_'))
        super(ConfigurationConstraint, self).__init__(n_wp, n_dof, idx_wp, name)
        self.av_desired = av_desired
        self.rank = n_dof

    def gen_func(self):
        n_dof, n_wp = self.n_dof, self.n_wp
        n_dof_all = n_dof * n_wp
        def func(av_seq):
            f = av_seq[self.idx_wp] - self.av_desired
            grad = np.zeros((self.rank, n_dof_all)) 
            grad[:, n_dof*self.idx_wp:n_dof*(self.idx_wp+1)] = np.eye(self.rank)
            return f, grad
        self._check_func(func)
        return func

    def satisfying_angle_vector(self, **kwargs):
        return self.av_desired

class PoseConstraint(EqualityConstraint):
    def __init__(self, n_wp, n_dof, idx_wp, coords_name_list, pose_desired_list, 
            fksolver, joint_list, with_base,
            name=None):
        # here pose order is [x, y, z, r, p, y]
        if name is None:
            name = 'eq_pose_const_{}'.format(str(uuid.uuid1()).replace('-', '_'))
        super(PoseConstraint, self).__init__(n_wp, n_dof, idx_wp, name)

        self.coords_name_list = coords_name_list
        self.pose_desired_list = pose_desired_list

        self.joint_list = joint_list
        self.joint_ids = fksolver.get_joint_ids([j.name for j in joint_list])
        self.fksolver = fksolver
        self.with_base = with_base

        self.with_rot_list = []
        rank = 0
        for pose_desired in self.pose_desired_list:
            with_rot = (len(pose_desired) == 6)
            self.with_rot_list.append(with_rot)
            rank += len(pose_desired)
        self.rank = rank

    def gen_subfunc(self):
        # gen_func returns whole trajectory jacobian. but this function returns only sub jacobian
        coords_ids = self.fksolver.get_link_ids(self.coords_name_list)
        pose_vector_desired = np.hstack(self.pose_desired_list)

        def func(av):
            P_list = []
            J_list = []
            for coords_id, with_rot in zip(coords_ids, self.with_rot_list):
                self.fksolver.clear_cache() # because we set use_cache=True
                P, J = self.fksolver.solve_forward_kinematics(
                        [av], [coords_id], self.joint_ids,
                        with_rot=with_rot, with_base=self.with_base, with_jacobian=True, use_cache=True) 
                P_list.append(P[0])
                J_list.append(J)
            pose_vector_now = np.hstack(P_list)
            J = np.vstack(J_list)
            return (pose_vector_now - pose_vector_desired).flatten(), J
        return func

    def gen_func(self):
        n_dof_all = self.n_dof * self.n_wp
        coords_ids = self.fksolver.get_link_ids(self.coords_name_list)
        pose_vector_desired = np.hstack(self.pose_desired_list)

        def func(av_seq):
            J_whole = np.zeros((self.rank, n_dof_all))
            P_list = []
            J_list = []

            for coords_id, with_rot in zip(coords_ids, self.with_rot_list):
                self.fksolver.clear_cache() # because we set use_cache=True
                P, J = self.fksolver.solve_forward_kinematics(
                        [av_seq[self.idx_wp]], [coords_id], self.joint_ids,
                        with_rot=with_rot, with_base=self.with_base, with_jacobian=True, use_cache=True) 
                P_list.append(P[0])
                J_list.append(J)
            pose_vector_now = np.hstack(P_list)
            J_part = np.vstack(J_list)
            J_whole[:, self.n_dof*self.idx_wp:self.n_dof*(self.idx_wp+1)] = J_part
            return (pose_vector_now - pose_vector_desired).flatten(), J_whole
        self._check_func(func)
        return func

    def satisfying_angle_vector(self, **kwargs):
        if "option" in kwargs:
            option = kwargs["option"]
        else:
            option = {"maxitr": 200, "ftol": 1e-4, "sr_weight":1.0}

        if "av_init" in kwargs:
            av_init = kwargs["av_init"]
        else:
            n_dof = len(self.joint_list) + (3 if self.with_base else 0)
            av_init = np.zeros(n_dof)

        if "collision_checker" in kwargs:
            collision_checker = kwargs["collision_checker"]
        else:
            collision_checker = None

        coords_ids = self.fksolver.get_link_ids(self.coords_name_list)
        av_solved = tinyfk_sqp_inverse_kinematics(
            self.coords_name_list, self.pose_desired_list, av_init, self.joint_list,
            self.fksolver, collision_checker, with_base=self.with_base)
        return av_solved

def listify_if_not_list(something):
    if isinstance(something, list):
        return something
    return [something]

# give a problem specification
class ConstraintManager(object):
    def __init__(self, n_wp, joint_list, fksolver, with_base): 
        # must be with_base=True now
        self.n_wp = n_wp
        n_dof = len(joint_list) + (3 if with_base else 0)
        self.n_dof = n_dof
        self.constraint_table = {}
        self.joint_list = joint_list
        self.fksolver = fksolver
        self.with_base = with_base

    def add_eq_configuration(self, idx_wp, av_desired, force=False):
        constraint = ConfigurationConstraint(self.n_wp, self.n_dof, idx_wp, av_desired)
        self._add_constraint(idx_wp, constraint, force)

    def add_multi_pose_constraint(self, idx_wp, coords_name_list, pose_desired_list, force=False):
        constraint = PoseConstraint(self.n_wp, self.n_dof, idx_wp,
                coords_name_list, pose_desired_list,
                self.fksolver, self.joint_list, self.with_base)
        self._add_constraint(idx_wp, constraint, force)

    def add_pose_constraint(self, idx_wp, coords_name, pose_desired, force=False):
        constraint = PoseConstraint(self.n_wp, self.n_dof, idx_wp,
                [coords_name], [pose_desired],
                self.fksolver, self.joint_list, self.with_base)
        self._add_constraint(idx_wp, constraint, force)

    def _add_constraint(self, idx_wp, constraint, force):
        is_already_exist = idx_wp in self.constraint_table.keys()
        if is_already_exist and (not force):
            raise Exception("to overwrite the constraint, please set force=True")
        self.constraint_table[idx_wp] = constraint

    def gen_combined_constraint_func(self):
        has_initial_and_terminal_const = 0 in self.constraint_table.keys() and (self.n_wp-1) in self.constraint_table.keys(), "please set initial and terminal constraint"
        assert has_initial_and_terminal_const
        # correct all funcs
        func_list = []
        for constraint in self.constraint_table.values():
            func_list.append(constraint.gen_func())

        def func_combined(xi):
            # xi is the flattened angle vector
            av_seq = xi.reshape(self.n_wp, self.n_dof)
            f_list, jac_list = zip(*[fun(av_seq) for fun in func_list])
            return np.hstack(f_list), np.vstack(jac_list)
        return func_combined

    def gen_initial_trajectory(self, **kwargs):
        av_start = self.constraint_table[0].satisfying_angle_vector(**kwargs)
        av_goal = self.constraint_table[self.n_wp-1].satisfying_angle_vector(**kwargs)

        regular_interval = (av_goal - av_start) / (self.n_wp - 1)
        initial_trajectory = np.array(
            [av_start + i * regular_interval for i in range(self.n_wp)])
        return initial_trajectory

    def check_eqconst_validity(self, **kwargs):
        # As for the pose constraint, the validity is clear when generating the 
        # initial angle vector. So we are gonna check only terminal eq consts.
        assert ("collision_checker" in kwargs), "to check validity you must specify a collision checker"
        sscc = kwargs["collision_checker"]

        joint_ids = self.fksolver.get_joint_ids([j.name for j in self.joint_list])
        const_start = self.constraint_table[0]
        const_end = self.constraint_table[self.n_wp - 1]
        for const_config in [const_start, const_end]:
            if isinstance(const_config, ConfigurationConstraint):
                sd_vals, _ = sscc._compute_batch_sd_vals(
                    joint_ids, np.array([const_config.av_desired]), self.with_base)
                assert np.all(sd_vals > 0.0), "invalid eq-config constraint"
            if isinstance(const_config, PoseConstraint):
                msg = "invalid pose constraint"
                for pose in const_config.pose_desired_list:
                    position = pose[:3]
                    if isinstance(sscc.sdf, list):
                        for sdf in sscc.sdf:
                            assert sdf(position.reshape(1, 3)) > 3e-3, msg
                    else:
                        assert sscc.sdf(position.reshape(1, 3)) > 3e-3, msg

    def clear_constraint(self):
        self.constraint_table = {}
