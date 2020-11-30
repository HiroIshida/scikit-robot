import numpy as np
from sklearn.covariance import EmpiricalCovariance
from skrobot.model.primitives import Sphere
from skrobot.coordinates import CascadedCoords
import trimesh

def assoc_swept_sphere(robot_model, link):
    col_mesh = link.collision_mesh
    assert type(col_mesh) == trimesh.base.Trimesh

    centers, R = compute_swept_sphere(col_mesh)
    spheres = []
    for center in centers:
        tmp = link.copy_worldcoords()
        co_s = CascadedCoords(pos=tmp.worldpos(), rot=tmp.worldrot())
        co_s.translate(center)

        sp = Sphere(radius=R, pos = co_s.worldpos())
        link.assoc(sp)
        spheres.append(sp)
    feature_radius_list = [R] * len(spheres)

    return spheres, feature_radius_list

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
