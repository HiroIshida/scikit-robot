import numpy as np
import scipy

from skrobot.planner.sqp_based import _sqp_based_trajectory_optimization
from skrobot.planner.utils import scipinize
from skrobot.planner.utils import update_fksolver

def tinyfk_sqp_inverse_kinematics(
        coords_name,
        target_pose,
        av_guess,     
        joint_list,
        collision_checker,
        safety_margin=1e-2,
        with_base=False):

    with_rot = (len(target_pose)==6)

    joint_limit_list = [[j.min_angle, j.max_angle] for j in joint_list]
    if with_base:
        joint_limit_list += [[-np.inf, np.inf]] * 3
    n_dof = len(joint_list) + (3 if with_base else 0)

    joint_name_list = [j.name for j in joint_list]
    joint_ids = collision_checker.fksolver.get_joint_ids(joint_name_list)

    def collision_ineq_fun(av):
        with_jacobian = True
        sd_vals, sd_val_jac = collision_checker._compute_batch_sd_vals(
            joint_ids, av.reshape(1, -1),
            with_base=with_base, with_jacobian=with_jacobian)
        sd_vals_margined = sd_vals - safety_margin
        return sd_vals_margined, sd_val_jac

    end_link_ids = collision_checker.fksolver.get_link_ids([coords_name])

    def fun_objective(av):
        P, J = collision_checker.fksolver.solve_forward_kinematics(
                [av], end_link_ids, joint_ids, with_rot, with_base, True)
        diff = (target_pose - P[0])
        cost = np.sum(diff**2)
        grad = -2 * diff.dot(J)
        return cost, grad

    f, jac = scipinize(fun_objective)
    ineq_const_scipy, ineq_const_jac_scipy = scipinize(collision_ineq_fun)
    ineq_dict = {'type': 'ineq', 'fun': ineq_const_scipy,
                 'jac': ineq_const_jac_scipy}

    tmp = np.array(joint_limit_list)
    lower_limit, uppre_limit = tmp[:, 0], tmp[:, 1]
    bounds = list(zip(lower_limit, uppre_limit))
    slsqp_option = {'ftol': 1e-5, 'disp': True, 'maxiter': 100}
    res = scipy.optimize.minimize(
        f, av_guess, method='SLSQP',
        jac=jac, bounds=bounds, options=slsqp_option, constraints=[ineq_dict])
    return res.x


def tinyfk_sqp_plan_trajectory(collision_checker,
                               constraint_manager,
                               initial_trajectory,
                               joint_list,
                               n_wp,
                               safety_margin=1e-2,
                               with_base=False,
                               weights=None,
                               slsqp_option=None
                               ):
    # common stuff
    joint_limit_list = [[j.min_angle, j.max_angle] for j in joint_list]
    if with_base:
        joint_limit_list += [[-np.inf, np.inf]] * 3
    n_dof = len(joint_list) + (3 if with_base else 0)

    # determine default weight
    if weights is None:
        weights = [1.0] * len(joint_list)
        if with_base:
            weights += [3.0] * 3  # base should be difficult to move
    weights = tuple(weights)  # to use cache

    joint_name_list = [j.name for j in joint_list]
    joint_ids = collision_checker.fksolver.get_joint_ids(joint_name_list)

    def collision_ineq_fun(av_seq):
        with_jacobian = True
        sd_vals, sd_val_jac = collision_checker._compute_batch_sd_vals(
            joint_ids, av_seq,
            with_base=with_base, with_jacobian=with_jacobian)
        sd_vals_margined = sd_vals - safety_margin
        return sd_vals_margined, sd_val_jac

    optimal_trajectory = _sqp_based_trajectory_optimization(
        initial_trajectory,
        collision_ineq_fun,
        constraint_manager.gen_combined_constraint_func(),
        joint_limit_list,
        weights,
        slsqp_option)
    return optimal_trajectory
