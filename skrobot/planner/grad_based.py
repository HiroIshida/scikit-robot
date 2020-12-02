import time
import scipy
import copy
import numpy as np
from . import utils
from .collision import CollisionChecker

def plan_trajectory(self,
                    av_start,
                    av_goal,
                    link_list,
                    coll_link_list,
                    signed_distance_function,
                    n_wp,
                    base_also=False,
                    weights=None,
                    initial_trajectory=None,
                    use_cpp=True
                    ):
    """Gradient based trajectory optimization using scipy's SLSQP. 
    Collision constraint is considered in an inequality constraits. 
    Terminal constraint (start and end) is considered as an 
    equality constraint.

    Parameters
    ----------
    av_start : numpy.ndarray(n_control_dof)
        joint angle vector at start point
    av_start : numpy.ndarray(n_control_dof)
        joint angle vector at goal point
    link_list : skrobot.model.Link
        link list to be controlled (similar to inverse_kinematics function)
    coll_link_list :  list[skrobot.coordinates.base.CascadedCoords]
        list of link that collision is checked
    signed_distance_function : function object 
    [2d numpy.ndarray (n_point x 3)] -> [1d numpy.ndarray (n_point)]
    n_wp : int 
        number of waypoints
    weights : 1d numpy.ndarray 
        cost to move of each joint. For example, 
        if you set weights=numpy.array([1.0, 0.1, 0.1]) for a 
        3 DOF manipulator, moving the first joint is with 
        high cost compared to others.
    initial_trajectory : 2d numpy.ndarray (n_wp, n_dof)
        If None, initial trajectory is automatically generated. 

    Returns
    ------------
    planned_trajectory : 2d numpy.ndarray (n_wp, n_dof)
    """

    # common stuff
    joint_list = [link.joint for link in link_list]
    joint_limits = [[j.min_angle, j.max_angle] for j in joint_list]
    if base_also:
        joint_limits += [[-np.inf, np.inf]]*3

    n_feature = len(coll_link_list)

    # create initial solution for the optimization problem
    if initial_trajectory is None:
        regular_interval = (av_goal - av_start) / (n_wp - 1)
        initial_trajectory = np.array(
            [av_start + i * regular_interval for i in range(n_wp)])

    cc = CollisionChecker(signed_distance_function, self)
    for link in coll_link_list:
        cc.add_collision_link(link)
    tinyfk_joint_ids = self.fksolver.get_joint_ids([j.name for j in joint_list])

    def collision_ineq_fun(av_seq):
        F, J = cc._collision_dists(tinyfk_joint_ids, av_seq, 
                base_also=base_also,
                with_jacobian=True)
        return F, J

    opt = GradBasedPlannerCommon(initial_trajectory,
                                 collision_ineq_fun,
                                 joint_limits,
                                 weights=weights,
                                 )
    ts = time.time()
    optimal_trajectory = opt.solve()
    print("solving time : {0}".format(time.time() - ts))
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

class GradBasedPlannerCommon:
    def __init__(self, av_seq_init,
                 collision_ineq_fun, joint_limit, weights=None):
        self.av_seq_init = av_seq_init
        self.collision_ineq_fun = collision_ineq_fun
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
        return self.collision_ineq_fun(av_seq)

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
        eq_const_scipy, eq_const_jac_scipy = utils.scipinize(self.fun_eq)
        eq_dict = {'type': 'eq', 'fun': eq_const_scipy,
                   'jac': eq_const_jac_scipy}
        ineq_const_scipy, ineq_const_jac_scipy = utils.scipinize(self.fun_ineq)
        ineq_dict = {'type': 'ineq', 'fun': ineq_const_scipy,
                     'jac': ineq_const_jac_scipy}
        f, jac = utils.scipinize(self.fun_objective)

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
