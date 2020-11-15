import scipy
import numpy as np
import copy

def inverse_kinematics_slsqp(self, 
                             target_coords,
                             link_list,
                             end_effector_cascaded_coords,
                             ):
    joint_list = [link.joint for link in link_list]
    joint_limits = [[j.min_angle, j.max_angle] for j in joint_list]

    def set_joint_angles(av):
        return [j.joint_angle(a) for j, a in zip(joint_list, av)]

    def get_joint_angles():
        return np.array([j.joint_angle() for j in joint_list])

    def compute_jacobian_wrt_baselink(av0, move_target, rot_also=False):
        set_joint_angles(av0)
        base_link = self.link_list[0]
        J = self.calc_jacobian_from_link_list([move_target], link_list,
                                              transform_coords=base_link,
                                              rotation_axis=rot_also)
        return J

    def endcoord_forward_kinematics(av, rotalso=True):
        def quaternion_kinematic_matrix(q):
            # dq/dt = 0.5 * mat * omega 
            q1, q2, q3, q4 = q
            mat = np.array([
                [-q2, -q3, -q4], [q1, q4, -q3], [-q4, q1, q2], [q3, -q2, q1],
                ])
            return mat * 0.5

        J_geometric = compute_jacobian_wrt_baselink(av, end_effector_cascaded_coords, rot_also=rotalso)
        J_geo_pos = J_geometric[:3]

        pos = end_effector_cascaded_coords.worldpos()
        if rotalso:
            rot = end_effector_cascaded_coords.worldcoords().quaternion
            pose = np.hstack((pos, rot))
            kine_mat = quaternion_kinematic_matrix(rot)
            J_geo_rot = J_geometric[3:]
            J_geo_quat = kine_mat.dot(J_geo_rot)
            J = np.vstack((J_geo_pos, J_geo_quat))
        else:
            pose = pos
            J = J_geo_pos
        return pose, J

    av_solved = inverse_kinematics_slsqp_common(get_joint_angles(), endcoord_forward_kinematics, joint_limits,
            target_coords.worldpos(), target_coords.worldcoords().quaternion)
    set_joint_angles(av_solved)
    return av_solved

def plan_trajectory(self,
                    target_coords,
                    n_wp,
                    link_list,
                    end_effector_cascaded_coords,
                    coll_cascaded_coords_list,
                    signed_distance_function,
                    rot_also=True,
                    weights=None,
                    initial_trajectory=None):
    """Gradient based trajectory optimization using scipy's SLSQP. 
    Collision constraint is considered in an inequality constraits. 
    Terminal constraint (start and end) is considered as an 
    equality constraint.

    Parameters
    ----------
    target_coords : skrobot.coordinates.base.Coordinates
        target coordinate of the end effector 
        at the final step of the trajectory
    n_wp : int 
        number of waypoints
    link_list : skrobot.model.Link
        link list to be controlled (similar to inverse_kinematics function)
    end_effector_cascaded_coords : skrobot.coordinates.base.CascadedCoords
        cascaded coords of the end-effector 
    coll_cascaded_coords_list :  list[skrobot.coordinates.base.CascadedCoords]
        list of collision cascaded coords
    signed_distance_function : function object 
    [2d numpy.ndarray (n_point x 3)] -> [1d numpy.ndarray (n_point)]
    rot_also : bool
        if enabled, rotation of the target coords is considered
    weights : 1d numpy.ndarray 
        cost to move of each joint. For example, 
        if you set weights=numpy.array([1.0, 0.1, 0.1]) for a 
        3 DOF manipulator, moving the first joint is with 
        high cost compared to others.
    initial_trajectory : 2d numpy.ndarray (n_wp, n_dof)
        If None, initial trajectory is automatically generated. 
        If set, target_coords and n_wp are ignored. 
        If the considered geometry is complex, you should 
        set a feasible path as an initial solution. 

    Returns
    ------------
    planned_trajectory : 2d numpy.ndarray (n_wp, n_dof)
    """

    # common stuff
    joint_list = [link.joint for link in link_list]
    joint_limits = [[j.min_angle, j.max_angle] for j in joint_list]

    def set_joint_angles(av):
        return [j.joint_angle(a) for j, a in zip(joint_list, av)]

    def get_joint_angles():
        return np.array([j.joint_angle() for j in joint_list])

    # create initial solution for the optimization problem
    if initial_trajectory is None:
        av_init = get_joint_angles()
        res = self.inverse_kinematics(
            target_coords,
            link_list=link_list,
            move_target=end_effector_cascaded_coords,
            rotation_axis=rot_also)
        print("target inverse kinematics solved")
        av_target = get_joint_angles()

        assert (res is not False), "IK to the target coords isn't solvable. \
                You can directry pass initial_trajectory instead."

        regular_interval = (av_target - av_init) / (n_wp - 1)
        initial_trajectory = np.array(
            [av_init + i * regular_interval for i in range(n_wp)])

    # define forward kinematics functions
    def compute_jacobian_wrt_baselink(av0, move_target, rot_also=False):
        set_joint_angles(av0)
        base_link = self.link_list[0]
        J = self.calc_jacobian_from_link_list([move_target], link_list,
                                              transform_coords=base_link,
                                              rotation_axis=rot_also)
        return J

    def collision_forward_kinematics(av_seq):
        points = []
        jacobs = []
        for av in av_seq:
            for collision_coords in coll_cascaded_coords_list:
                J = compute_jacobian_wrt_baselink(av, collision_coords)
                pos = collision_coords.worldpos()
                points.append(pos)
                jacobs.append(J)
        return np.vstack(points), np.vstack(jacobs)

    # solve!
    n_features = len(coll_cascaded_coords_list)
    opt = GradBasedPlannerCommon(initial_trajectory,
                                 n_features,
                                 collision_forward_kinematics,
                                 joint_limits,
                                 signed_distance_function,
                                 weights=weights,
                                 )
    optimal_trajectory = opt.solve()

    return optimal_trajectory


def construct_smoothcost_fullmat(n_dof, n_wp, weights=None):

    def construct_smoothcost_mat(n_wp):
        acc_block = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])
        vel_block = np.array([[1, -1], [-1, 1]])
        A = np.zeros((n_wp, n_wp))
        for i in [1 + i for i in range(n_wp - 2)]:
            A[i - 1:i + 2, i - 1:i + 2] += acc_block
            A[i - 1:i + 1, i - 1:i + 1] += vel_block * 0.0  # do nothing
        return A

    w_mat = np.eye(n_dof) if weights is None else np.diag(weights)
    Amat = construct_smoothcost_mat(n_wp)
    Afullmat = np.kron(Amat, w_mat**2)
    return Afullmat


def scipinize(fun):
    closure_member = {'jac_cache': None}

    def fun_scipinized(x):
        f, jac = fun(x)
        closure_member['jac_cache'] = jac
        return f

    def fun_scipinized_jac(x):
        return closure_member['jac_cache']
    return fun_scipinized, fun_scipinized_jac

def inverse_kinematics_slsqp_common(av_init, 
        endeffector_fk, 
        joint_limits, 
        pos_target, 
        rot_target=None):

        def fun_objective(av):
            if rot_target is None:
                position, jac = endeffector_fk(av, rotalso=False)
                diff = position - pos_target
                cost = np.linalg.norm(diff) ** 2
                cost_grad = 2 * diff.dot(jac)
            else:
                #https://math.stackexchange.com/questions/90081/quaternion-distance
                pose, jac = endeffector_fk(av, rotalso=True)
                position, rot = pose[:3], pose[3:]
                pos_diff = position - pos_target
                cost_position = np.linalg.norm(position - pos_target) ** 2
                cost_position_grad = 2 * pos_diff.dot(jac[:3, :])

                inpro = np.sum(rot * rot_target)
                cost_rotation = 1 - inpro ** 2
                cost_rotation_grad = - 2 * inpro * rot_target.dot(jac[3:, :])

                cost = cost_position + cost_rotation
                cost_grad = cost_position_grad + cost_rotation_grad
            return cost, cost_grad

        f, jac = scipinize(fun_objective)
        res = scipy.optimize.minimize(
                f, av_init, method='SLSQP', jac=jac, bounds=joint_limits,
                options={'ftol': 1e-4, 'disp': False})
        return res.x

class GradBasedPlannerCommon:
    def __init__(self, av_seq_init, n_features,
                 collision_fk, joint_limit, sdf, weights=None):
        self.av_seq_init = av_seq_init
        self.n_features = n_features
        self.collision_fk = collision_fk
        self.sdf = sdf
        self.n_wp, self.n_dof = av_seq_init.shape
        self.joint_limit = joint_limit
        self.A = construct_smoothcost_fullmat(
            self.n_dof, self.n_wp, weights=weights)

    def fun_objective(self, x):
        f = (0.5 * self.A.dot(x).dot(x)).item() / self.n_wp
        grad = self.A.dot(x) / self.n_wp
        return f, grad

    def fun_ineq(self, xi):
        av_seq = xi.reshape(self.n_wp, self.n_dof)
        P_link, J_link = self.collision_fk(av_seq)

        sdf_grads = np.zeros(P_link.shape)
        F_link_cost0 = self.sdf(np.array(P_link))
        eps = 1e-7
        for i in range(3):
            P_link_ = copy.copy(P_link)
            P_link_[:, i] += eps
            F_link_cost1 = self.sdf(np.array(P_link_))
            sdf_grads[:, i] = (F_link_cost1 - F_link_cost0) / eps

        sdf_grads = sdf_grads.reshape(self.n_wp * self.n_features, 1, 3)
        J_link = J_link.reshape(self.n_wp * self.n_features, 3, self.n_dof)
        J_link_list = np.matmul(sdf_grads, J_link)
        J_link_block = J_link_list.reshape(
            self.n_wp, self.n_features, self.n_dof)
        J_link_full = scipy.linalg.block_diag(*list(J_link_block))
        F_cost_full, J_cost_full = F_link_cost0, J_link_full
        return F_cost_full, J_cost_full

    def fun_eq(self, xi):
        # terminal constraint
        Q = xi.reshape(self.n_wp, self.n_dof)
        q_start = self.av_seq_init[0]
        q_end = self.av_seq_init[-1]
        f = np.hstack((q_start - Q[0], q_end - Q[-1]))
        grad_ = np.zeros((self.n_dof * 2, self.n_dof * self.n_wp))
        grad_[:self.n_dof, :self.n_dof] = - np.eye(self.n_dof)
        grad_[-self.n_dof:, -self.n_dof:] = - np.eye(self.n_dof)
        return f, grad_

    def solve(self):
        eq_const_scipy, eq_const_jac_scipy = scipinize(self.fun_eq)
        eq_dict = {'type': 'eq', 'fun': eq_const_scipy,
                   'jac': eq_const_jac_scipy}
        ineq_const_scipy, ineq_const_jac_scipy = scipinize(self.fun_ineq)
        ineq_dict = {'type': 'ineq', 'fun': ineq_const_scipy,
                     'jac': ineq_const_jac_scipy}
        f, jac = scipinize(self.fun_objective)

        tmp = np.array(self.joint_limit)
        lower_limit = tmp[:, 0]
        uppre_limit = tmp[:, 1]

        bounds = list(zip(lower_limit, uppre_limit)) * self.n_wp

        xi_init = self.av_seq_init.reshape((self.n_dof * self.n_wp, ))
        res = scipy.optimize.minimize(f, xi_init, method='SLSQP', jac=jac,
                                      bounds=bounds,
                                      constraints=[eq_dict, ineq_dict],
                                      options={'ftol': 1e-4, 'disp': False})
        traj_opt = res.x.reshape(self.n_wp, self.n_dof)
        return traj_opt
