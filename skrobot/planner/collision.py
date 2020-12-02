import numpy as np
import scipy
import copy
from sklearn.covariance import EmpiricalCovariance
from skrobot.model.primitives import Sphere
from skrobot.coordinates import CascadedCoords
from .utils import set_robot_state
from .utils import get_robot_state
import trimesh

class CollisionChecker(object):

    def __init__(self, sdf, robot_model):
        self.sdf = sdf
        self.robot_model = robot_model
        self.fksolver = robot_model.fksolver

        self.coll_link_name_list = []
        self.coll_sphere_list = []
        self.coll_radius_list = []
        self.coll_sphere_id_list = []

        self.color_normal_sphere = [250, 250, 10, 200]
        self.color_collision_sphere = [255, 0, 0, 200]

    def add_coll_spheres_to_viewer(self, viewer):
        for s in self.coll_sphere_list:
            viewer.add(s)

    def remove_collision_link(self):
        raise NotImplementedError

    def add_collision_link(self, coll_link):

        if coll_link.name in self.coll_link_name_list:
            return

        col_mesh = coll_link.collision_mesh
        assert type(col_mesh) == trimesh.base.Trimesh

        centers, R = compute_swept_sphere(col_mesh)
        sphere_list = []
        for center in centers:
            tmp = coll_link.copy_worldcoords()
            co_s = CascadedCoords(pos=tmp.worldpos(), rot=tmp.worldrot())
            co_s.translate(center)

            sp = Sphere(radius=R, pos=co_s.worldpos(), color=self.color_normal_sphere)
            coll_link.assoc(sp)
            sphere_list.append(sp)

        sphere_names = [sphere.name for sphere in sphere_list]

        self.coll_sphere_list.extend(sphere_list)
        self.coll_radius_list.extend([R]*len(sphere_list))

        # fksolver specific procedure
        coll_link_id = self.fksolver.get_link_ids([coll_link.name])[0]

        for sphere, c in zip(sphere_list, centers):
            self.fksolver.add_new_link(sphere.name, coll_link_id, c)
        self.coll_sphere_id_list.extend(
                self.fksolver.get_link_ids(sphere_names))

    def update_color(self, base_also=False):
        if base_also:
            raise NotImplementedError

        joint_list = [j for j in self.robot_model.joint_list]
        angle_vector = np.array([j.joint_angle() for j in joint_list])
        angle_vector_seq = angle_vector.reshape(1, -1)
        dists, _ = self.collision_dists(joint_list, angle_vector_seq, base_also=base_also, with_jacobian=False)
        print(dists)
        idxes_collide = np.where(dists < 0)[0]
        print(idxes_collide)

        n_feature = len(self.coll_sphere_list)
        for idx in range(n_feature):
            sphere = self.coll_sphere_list[idx]
            n_facet = len(sphere._visual_mesh.visual.face_colors)

            color = self.color_collision_sphere if idx in idxes_collide \
                    else self.color_normal_sphere
            print(color)
            sphere._visual_mesh.visual.face_colors = np.array([color]*n_facet)
        return dists

    def collision_dists(self, 
            joint_list, angle_vector_seq, **kwargs):
        """
        This method is CRITICAL for faster motion plan. 
        """

        joint_name_list = [j.name for j in joint_list]
        joint_id_list = self.fksolver.get_joint_ids(joint_name_list)
        return self._collision_dists(joint_id_list, angle_vector_seq, **kwargs)

    def _collision_dists(self, 
            joint_id_list, angle_vector_seq, base_also=False, with_jacobian=False):

        """
        This method is CRITICAL for faster motion plan.
        """
        rot_also = False
        n_wp, n_dof = angle_vector_seq.shape
        n_feature = len(self.coll_sphere_list)
        n_total_feature = n_wp * n_feature

        P_fk, dP_fk_dq = self.fksolver.solve_forward_kinematics(
                angle_vector_seq,
                self.coll_sphere_id_list,
                joint_id_list,
                rot_also, 
                base_also,
                with_jacobian)
        traj_coll_radius_list = self.coll_radius_list * n_wp
        cost_coll = self.sdf(P_fk) 

        if with_jacobian:
            # now compute grad of cost_coll
            dP_fk_dq = dP_fk_dq.reshape(n_total_feature, 3, n_dof)
            dcost_dP = np.zeros(P_fk.shape)
            eps = 1e-7
            for i in range(3):
                P_fk_plus = copy.copy(P_fk)
                P_fk_plus[:, i] += eps
                cost_coll_plus = self.sdf(P_fk_plus)
                dcost_dP[:, i] = (cost_coll_plus - cost_coll)/eps

            dcost_dP = dcost_dP.reshape(n_total_feature, 1, 3)

            dcost_dq = np.matmul(dcost_dP, dP_fk_dq)
            dcost_dq_block = dcost_dq.reshape(
                    n_wp, n_feature, n_dof)
            dcost_dq_full = scipy.linalg.block_diag(*list(dcost_dq_block))
        else:
            dcost_dq_full = None

        cost_coll_sphere = cost_coll - np.array(traj_coll_radius_list)
        return cost_coll_sphere, dcost_dq_full

def compute_swept_sphere(visual_mesh, 
        n_sphere=-1, 
        tol=0.1, 
        margin_factor=1.01):
    """
    n_sphere : if set -1, number of sphere is automatically determined.
    tol : tolerance
    """
    verts = visual_mesh.vertices
    mean = np.mean(verts, axis=0)
    verts_slided = verts - mean[None, :]
    cov = EmpiricalCovariance().fit(verts_slided)
    eig_vals, basis_tf_mat = np.linalg.eig(cov.covariance_)
    verts_mapped = verts_slided.dot(basis_tf_mat)

    def inverse_map(verts): # use this in the end of this function
        return verts.dot(basis_tf_mat.T) + mean[None, :]

    principle_axis = np.argmax(eig_vals)
    h_vert_max = np.max(verts_mapped[:, principle_axis])
    h_vert_min = np.min(verts_mapped[:, principle_axis])

    ## compute radius
    if principle_axis == 0:
        plane_axes = [1, 2]
    elif principle_axis == 1:
        plane_axes = [2, 0]
    else:
        plane_axes = [0, 1]

    def determine_radius(verts_2d_projected):
        X, Y = verts_2d_projected.T
        radius_vec = np.sqrt(X**2 + Y**2)
        R = np.max(radius_vec)
        return R

    margin_factor = 1.01
    R = determine_radius(verts_mapped[:, plane_axes]) * margin_factor
    sqraidus_vec = np.sum(verts_mapped[:, plane_axes] ** 2, axis=1)
    h_vec = verts_mapped[:, principle_axis]

    def get_h_center_max():
        def cond_all_inside_positive(h_center_max):
            sphere_heights = h_center_max + np.sqrt(R**2 - sqraidus_vec)
            return np.all(sphere_heights > h_vec)
        # get first index that satisfies the condition
        h_cand_list = np.linspace(0, h_vert_max, 30)
        idx = np.where([cond_all_inside_positive(h) for h in h_cand_list])[0][0]
        h_center_max = h_cand_list[idx]
        return h_center_max

    def get_h_center_min():
        def cond_all_inside_negative(h_center_min):
            sphere_heights = h_center_min - np.sqrt(R**2 - sqraidus_vec)
            return np.all(h_vec > sphere_heights)
        # get first index that satisfies the condition
        h_cand_list = np.linspace(0, h_vert_min, 30)
        idx = np.where([cond_all_inside_negative(h) for h in h_cand_list])[0][0]
        h_center_min = h_cand_list[idx]
        return h_center_min

    h_center_max = get_h_center_max()
    h_center_min = get_h_center_min()

    def create_centers_feature_space(n_sphere):
        h_centers = np.linspace(h_center_min, h_center_max, n_sphere)
        centers = np.zeros((n_sphere, 3))
        centers[:, principle_axis] = h_centers
        return centers

    if n_sphere == -1: # n_sphere is automatically determined
        n_sphere = 1
        while True:
            centers_feature_space = create_centers_feature_space(n_sphere)
            dists_foreach_sphere = np.array([np.sqrt(np.sum((verts_mapped - c[None, :])**2, axis=1)) for c in centers_feature_space])
            sdfs = np.min(dists_foreach_sphere, axis=0) - R
            maxsdf = np.max(sdfs)
            err_ratio = maxsdf/R
            if err_ratio < tol:
                break
            n_sphere+=1
    else:
        centers_feature_space = create_centers_feature_space(n_sphere)
    centers_original_space = inverse_map(centers_feature_space)

    return centers_original_space, R
