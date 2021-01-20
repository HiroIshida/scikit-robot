import numpy as np
import scipy

from skrobot.planner.sqp_based import _sqp_based_trajectory_optimization
from skrobot.planner.utils import scipinize
from skrobot.planner.utils import update_fksolver
from skrobot.planner.utils import compute_joint_weights
from skrobot.utils.listify import listify


class InvalidInitConfigException(Exception):
    pass

def tinyfk_sqp_inverse_kinematics(
        coords_name_list,
        target_pose_list,
        av_guess,     
        joint_list,
        fksolver,
        collision_checker=None,
        safety_margin=1e-2,
        strategy="multi",
        with_base=False):

    coords_name_list = listify(coords_name_list)
    target_pose_list = listify(target_pose_list)
    assert len(coords_name_list) == len(target_pose_list)


    fix_negative_inf = lambda x: -6.28 if x == -np.inf else x
    fix_positive_inf = lambda x: 6.28 if x == np.inf else x
    with_rot_list = [len(tp)==6 for tp in target_pose_list]

    joint_limit_list = [[fix_negative_inf(j.min_angle), fix_positive_inf(j.max_angle)]
            for j in joint_list]
    if with_base:
        joint_limit_list += [[-np.inf, np.inf]] * 3
    n_dof = len(joint_list) + (3 if with_base else 0)

    joint_name_list = [j.name for j in joint_list]
    joint_ids = fksolver.get_joint_ids(joint_name_list)

    # Construct constraints
    if collision_checker is not None:
        def collision_ineq_fun(av):
            with_jacobian = True
            sd_vals, sd_val_jac = collision_checker._compute_batch_sd_vals(
                joint_ids, av.reshape(1, -1),
                with_base=with_base, with_jacobian=with_jacobian)
            sd_vals_margined = sd_vals - safety_margin
            return sd_vals_margined, sd_val_jac
        ineq_const_scipy, ineq_const_jac_scipy = scipinize(collision_ineq_fun)
        ineq_dict = {'type': 'ineq', 'fun': ineq_const_scipy,
                     'jac': ineq_const_jac_scipy}
        constraints = [ineq_dict]
    else:
        constraints = []

    # construct objective function
    elink_ids = fksolver.get_link_ids(coords_name_list)
    def fun_objective(av):
        cost_whole = 0.0
        grad_whole = np.zeros(n_dof)
        fksolver.clear_cache() # because we set use_cache=True
        for target_pose, elink_id, with_rot in zip(target_pose_list, elink_ids, with_rot_list):
            P, J = fksolver.solve_forward_kinematics(
                    [av], [elink_id], joint_ids, with_rot, with_base, True, use_cache=True)
            pose_diff = (target_pose - P[0])
            cost = np.sum(pose_diff**2)
            grad = -2 * pose_diff.dot(J)

            cost_whole += cost
            grad_whole += grad
        return cost_whole, grad_whole
    f, jac = scipinize(fun_objective)

    tmp = np.array(joint_limit_list)
    lower_limit, uppre_limit = tmp[:, 0], tmp[:, 1]
    bounds = list(zip(lower_limit, uppre_limit))
    slsqp_option = {'ftol': 1e-5, 'disp': True, 'maxiter': 100}
    if strategy=="multi":
        assert with_base
        pose = target_pose_list[0][:3]
        base_pose_nominal = np.array([pose[0], pose[1], 0.0])
        while True:
            av_guess[-3:] = base_pose_nominal + np.random.randn(3)
            res = scipy.optimize.minimize(
                f, av_guess, method='SLSQP',
                jac=jac, bounds=bounds, options=slsqp_option, constraints=constraints)
            if res.fun < 0.001:
                print("solved")
                break
            print("solve failed... retry")
        return res.x

    else:
        res = scipy.optimize.minimize(
            f, av_guess, method='SLSQP',
            jac=jac, bounds=bounds, options=slsqp_option, constraints=constraints)
        return res.x


def tinyfk_sqp_plan_trajectory(collision_checker,
                               constraint_manager,
                               initial_trajectory,
                               joint_list,
                               n_wp,
                               safety_margin=1e-2,
                               with_base=False,
                               weights=None,
                               slsqp_option=None,
                               callback=None
                               ):
    # common stuff
    joint_limit_list = [[j.min_angle, j.max_angle] for j in joint_list]
    if with_base:
        joint_limit_list += [[-np.inf, np.inf]] * 3
    n_dof = len(joint_list) + (3 if with_base else 0)

    # determine default weight
    if weights is None:
        weights = compute_joint_weights(joint_list, with_base)
    assert len(weights) == n_dof
    weights = tuple(weights)  # to use cache

    joint_name_list = [j.name for j in joint_list]
    joint_ids = collision_checker.fksolver.get_joint_ids(joint_name_list)

    ## check validity of the start angle vector
    sd_vals_start, _ = collision_checker._compute_batch_sd_vals(
            joint_ids, np.array([initial_trajectory[0]]), with_base=with_base)
    if not np.all(sd_vals_start > 1e-3):
        raise InvalidInitConfigException

    def collision_ineq_fun(av_seq):
        with_jacobian = True
        sd_vals, sd_val_jac = collision_checker._compute_batch_sd_vals(
            joint_ids, av_seq,
            with_base=with_base, with_jacobian=with_jacobian)
        sd_vals_margined = sd_vals - safety_margin
        return sd_vals_margined, sd_val_jac

    res = _sqp_based_trajectory_optimization(
        initial_trajectory,
        collision_ineq_fun,
        constraint_manager.gen_combined_constraint_func(),
        joint_limit_list,
        weights,
        slsqp_option,
        callback)
    return res
