from numbers import Number
from logging import getLogger
import hashlib
import os
import numpy as np
import pysdfgen
from math import floor
from skrobot.coordinates.math import normalize_vector
from skrobot.coordinates import CascadedCoords
from scipy.interpolate import RegularGridInterpolator

logger = getLogger(__name__)

class SignedDistanceFunction(CascadedCoords):
    def __init__(self, origin, use_abs=False, *args, **kwargs):
        super(SignedDistanceFunction, self).__init__(*args, **kwargs)

        self.sdf_to_obj_transform = CascadedCoords(
            pos=origin)
        self._origin = np.array(origin)
        self.use_abs = use_abs

    def __call__(self, points_obj):
        """ compute signed distance
        Parameters
        -------
        points_obj : 2d numpy.ndarray (n_point x 3)
            input points w.r.t sdf coordinate system
        Returns
        ------
        singed distances : 1d numpy.ndarray (n_point)
        """
        points_sdf = self.transform_pts_obj_to_sdf(points_obj)
        sd = self._signed_distance(points_sdf)
        if self.use_abs:
            return np.abs(sd)
        return sd

    def on_surface(self, points_obj):
        """Determines whether or not a point is on the object surface.

        Parameters
        ----------
        points_obj : :obj:`numpy.ndarray` 
            Nx3 ndarray w.r.t obj

        Returns
        -------
        :obj:`tuple` of numpy.ndarray[bool], ndarray[float]
        """
        sd_vals = self.__call__(points_obj)
        logicals = np.abs(sd_vals) < self._surface_threshold
        return logicals, sd_vals

    def surface_points(self, n_sample=1000):
        points_, dists = self._surface_points(n_sample=n_sample) 
        points = self.transform_pts_sdf_to_obj(points_)
        return points, dists

    @property
    def origin(self):
        """Return the location of the origin w.r.t grid basis.

        Returns
        -------
        self._origin : numpy.ndarray
            The 3-ndarray that contains the location of
            the origin of the mesh grid in real space.
        """
        return self._origin

    def transform_pts_obj_to_sdf(self, points_obj):
        """Converts a point w.r.t. sdf basis to the grid basis.

        Parameters
        ----------
        points_obj : numpy Nx3 ndarray or numeric scalar
            points to transform from sdf basis in meters to grid basis

        Returns
        -------
        points_sdf : numpy Nx3 ndarray or scalar
            points in grid basis
        """
        points_sdf = self.copy_worldcoords().transform(
            self.sdf_to_obj_transform).inverse_transform_vector(points_obj)
        return points_sdf

    def transform_pts_sdf_to_obj(self, points_sdf):
        """Converts a point w.r.t. grid basis to the obj basis.

        Parameters
        ----------
        points_sdf : numpy.ndarray 
            Nx3 ndarray
            points to transform from grid basis to sdf basis in meters

        Returns
        -------
        points_obj : numpy.ndarray
            Nx3 ndarray. points in sdf basis (meters)
        """
        points_obj = self.copy_worldcoords().transform(
            self.sdf_to_obj_transform).transform_vector(
                points_sdf.astype(np.float32))
        return points_obj


class UnionSDF(SignedDistanceFunction):

    def __init__(self, sdf_list, *args, **kwargs):
        origin = np.zeros(3)
        use_abs = False 
        super(UnionSDF, self).__init__(origin, use_abs, *args, **kwargs)

        use_abs_list = [sdf.use_abs for sdf in sdf_list]
        all_false = np.all(np.array(use_abs_list) == False)
        assert all_false, "use_abs for each sdf must be consistent"

        self.sdf_list = sdf_list

        threshold_list = [sdf._surface_threshold for sdf in sdf_list]
        self._surface_threshold = max(threshold_list)

    def __call__(self, points_obj):
        sd_vals_list = np.array([sdf(points_obj) for sdf in self.sdf_list])
        sd_vals_union = np.min(sd_vals_list, axis=0)
        return sd_vals_union

    def surface_points(self, n_sample=1000):
        # equaly asign sample number to each sdf.surface_points()
        n_list = len(self.sdf_list)
        n_sample_each = int(floor(n_sample/n_list))
        n_sample_last = n_sample - n_sample_each * (n_list - 1)
        num_list = [n_sample_each]*(n_list - 1) + [n_sample_last]

        points = np.vstack([sdf.surface_points(n_sample=n_sample_)[0] 
            for sdf, n_sample_ in zip(self.sdf_list, num_list)])
        logicals, sd_vals = self.on_surface(points)
        return points[logicals], sd_vals[logicals]


class BoxSDF(SignedDistanceFunction):

    def __init__(self, origin, width, use_abs=False,
                 *args, **kwargs):
        super(BoxSDF, self).__init__(origin, use_abs, *args, **kwargs)
        self._width = np.array(width)
        self._surface_threshold = np.min(self._width) * 1e-2

    def _signed_distance(self, points_sdf):
        """ compute signed distance
        Parameters
        -------
        points_sdf : 2d numpy.ndarray (n_point x 3)
            input points w.r.t box coordinates
        Returns
        ------
        singed distances : 1d numpy.ndarray (n_point)
        """
        n_pts, dim = points_sdf.shape
        assert dim == 3, "dim must be 3"

        b = self._width * 0.5
        c = self._origin 

        center = np.array(c).reshape(1, dim)
        center_copied = np.repeat(center, n_pts, axis=0)
        P = points_sdf - center_copied
        Q = np.abs(P) - np.repeat(np.array(b).reshape(1, dim), n_pts, axis=0)
        left__ = np.array([Q, np.zeros((n_pts, dim))])
        left_ = np.max(left__, axis=0)
        left = np.sqrt(np.sum(left_**2, axis=1))
        right_ = np.max(Q, axis=1)
        right = np.min(np.array([right_, np.zeros(n_pts)]), axis=0)
        sd = left + right
        return sd

    def _surface_points(self, n_sample=1000):
        # surface points by raymarching
        vecs = np.random.randn(n_sample, 3)
        norms = np.linalg.norm(vecs, axis=1).reshape(-1, 1)
        unit_vecs = vecs / np.repeat(norms, 3, axis=1)

        # start ray marching
        ray_directions = unit_vecs
        ray_tips = np.zeros((n_sample, 3))
        self._signed_distance(ray_tips)
        while True:
            sd = self._signed_distance(ray_tips).reshape(n_sample, -1)
            ray_tips += ray_directions * np.repeat(sd, 3, axis=1)
            if np.all(np.abs(sd) < self._surface_threshold):
                break
        sd_final = self._signed_distance(ray_tips)

        return ray_tips, sd_final

class GridSDF(SignedDistanceFunction):

    def __init__(self, sdf_data, origin, resolution, use_abs=False,
                 *args, **kwargs):
        super(GridSDF, self).__init__(origin, use_abs=use_abs, *args, **kwargs)
        # optionally use only the absolute values
        # (useful for non-closed meshes in 3D)
        self._data = np.abs(sdf_data) if use_abs else sdf_data
        self._dims = np.array(self._data.shape)
        self._resolution = resolution
        self._surface_threshold = resolution * np.sqrt(2) / 2.0

        # create regular grid interpolator
        xlin, ylin, zlin = [np.array(range(d)) * resolution for d in self._data.shape]
        self.itp = RegularGridInterpolator(
                (xlin, ylin, zlin), 
                self._data,
                bounds_error=False,
                fill_value=np.inf)

        spts, _ = self._surface_points()

        self.sdf_to_obj_transform = CascadedCoords(
            pos=self.origin)

    def is_out_of_bounds(self, points_sdf):
        """Returns True if points is an out of bounds access.

        Parameters
        ----------
        points_sdf : numpy.ndarray
            3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        is_out : bool
            If points is in grid, return True.
        """
        points_grid = np.array(points_sdf) / self._resolution
        return np.logical_or(
            (points_grid < 0).any(axis=1),
            (points_grid >= np.array(self._dims)).any(axis=1))

    def _signed_distance(self, points_sdf):
        """Returns the signed distance at the given coordinates

        Parameters
        ----------
        points_sdf : numpy.ndarray (Nx3) in sdf basis

        Returns
        -------
        numpy.ndarray
        """
        points_sdf = np.array(points_sdf)
        sd_vals = self.itp(points_sdf)
        return sd_vals

    def _surface_points(self, n_sample=None):
        """Returns the points on the surface.
        """
        surface_points = np.where(np.abs(self._data) < self._surface_threshold)
        x = surface_points[0]
        y = surface_points[1]
        z = surface_points[2]
        surface_points = np.c_[x, np.c_[y, z]]
        surface_values = self._data[surface_points[:, 0],
                                   surface_points[:, 1],
                                   surface_points[:, 2]]
        if n_sample is not None:
            # somple points WITHOUT duplication
            n_pts = len(surface_points)
            n_sample = min(n_sample, n_pts)
            idxes = np.random.permutation(n_pts)[:n_sample]
            surface_points, surface_values = surface_points[idxes], surface_values[idxes]

        return surface_points * self._resolution, surface_values

    @staticmethod
    def from_file(filepath):
        """Return GridSDF instance from .sdf file.

        Parameters
        ----------
        filepath : str or pathlib.Path
            path of .sdf file

        Returns
        -------
        sdf_instance : skrobot.exchange.sdf.GridSDF
            instance of sdf
        """
        with open(filepath, 'r') as f:
            # dimension of each axis should all be equal for LSH
            nx, ny, nz = [int(i) for i in f.readline().split()]
            ox, oy, oz = [float(i) for i in f.readline().split()]
            dims = np.array([nx, ny, nz])
            origin = np.array([ox, oy, oz])

            # resolution of the grid cells in original mesh coords
            resolution = float(f.readline())
            sdf_data = np.zeros(dims)

            # loop through file, getting each value
            count = 0
            for k in range(nz):
                for j in range(ny):
                    for i in range(nx):
                        sdf_data[i][j][k] = float(f.readline())
                        count += 1
        return GridSDF(sdf_data, origin, resolution)

    @staticmethod
    def from_objfile(obj_filepath, dim=100, padding=5):
        """Return GridSDF instance from .obj file.

        This file Internally create .sdf file from .obj file.
        Converting obj to GridSDF tooks a some time.

        Parameters
        ----------
        obj_filepath : str or pathlib.Path
            path of objfile
        dim : int
            dim of sdf
        padding : int
            number of padding

        Returns
        -------
        sdf_instance : skrobot.exchange.sdf.GridSDF
            instance of sdf
        """

        sdf_cache_dir = os.path.expanduser("~") + "/.skrobot/sdf/"
        if not os.path.exists(sdf_cache_dir):
            os.makedirs(sdf_cache_dir)

        filename, extension = os.path.splitext(str(obj_filepath))
        hashed_filename = hashlib.md5(filename.encode()).hexdigest()

        sdf_cache_path = sdf_cache_dir + hashed_filename + ".sdf"
        if not os.path.exists(sdf_cache_path):
            logger.info('pre-computing sdf and making a cache at {0}.'.format(sdf_cache_path))
            pysdfgen.obj2sdf(str(obj_filepath), dim, padding, output_filepath=sdf_cache_path)
            logger.info('finish pre-computation')
        return GridSDF.from_file(sdf_cache_path)
