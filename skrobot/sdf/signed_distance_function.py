from numbers import Number

import numpy as np
import pysdfgen

from skrobot.coordinates.math import normalize_vector
from skrobot.coordinates.similarity_transform import \
    SimilarityTransformCoordinates

class SignedDistanceFunction(SimilarityTransformCoordinates):
    def __init__(self, origin, scale, *args, **kwargs):
        super(SignedDistanceFunction, self).__init__(*args, **kwargs)

        self.sdf_to_obj_transform = SimilarityTransformCoordinates(
            pos=origin,
            scale=scale)
        self._origin = np.array(origin)

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
        return sd

    def surface_points(self, **kwargs):
        points_, dists = self._surface_points(**kwargs)
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

    @property
    def surface_threshold(self):
        """Threshold of surface value.

        Returns
        -------
        self._surface_threshold : float
            threshold
        """
        return self._surface_threshold


    def transform_pts_obj_to_sdf(self, points_obj, direction=False):
        """Converts a point w.r.t. sdf basis to the grid basis.

        If direction is True, don't translate.

        Parameters
        ----------
        points_obj : numpy Nx3 ndarray or numeric scalar
            points to transform from sdf basis in meters to grid basis

        Returns
        -------
        points_sdf : numpy Nx3 ndarray or scalar
            points in grid basis
        """
        if isinstance(points_obj, Number):
            return self.copy_worldcoords().transform(
                self.sdf_to_obj_transform
            ).inverse_transformation().scale * points_obj.T
        if direction is True:
            # 1 / s [R^T v - R^Tp] p == 0 case
            points_sdf = np.dot(points_obj.T, self.copy_worldcoords().transform(
                self.sdf_to_obj_transform).worldrot().T)
        else:
            print(points_obj)
            points_sdf = self.copy_worldcoords().transform(
                self.sdf_to_obj_transform).inverse_transform_vector(points_obj)
        return points_sdf

    def transform_pts_sdf_to_obj(self, points_sdf, direction=False):
        """Converts a point w.r.t. grid basis to the obj basis.

        If direction is True, then don't translate.

        Parameters
        ----------
        points_sdf : numpy.ndarray or numbers.Number
            Nx3 ndarray or numeric scalar
            points to transform from grid basis to sdf basis in meters
        direction : bool
            If this value is True, points_sdf treated as normal vectors.

        Returns
        -------
        points_obj : numpy.ndarray
            Nx3 ndarray. points in sdf basis (meters)
        """
        if isinstance(points_sdf, Number):
            return self.copy_worldcoords().transform(
                self.sdf_to_obj_transform).scale * points_sdf

        if direction:
            points_obj = np.dot(points_sdf.T, self.copy_worldcoords().transform(
                self.sdf_to_obj_transform).worldrot().T)
        else:
            points_obj = self.copy_worldcoords().transform(
                self.sdf_to_obj_transform).transform_vector(
                    points_sdf.astype(np.float32))
        return points_obj

class BoxSDF(SignedDistanceFunction):

    def __init__(self, origin, width,
                 *args, **kwargs):
        scale = 1.0
        super(BoxSDF, self).__init__(origin, scale, *args, **kwargs)
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

    def _surface_points(self, N=20):
        # surface points by raymarching
        vecs = np.random.randn(N, 3)
        norms = np.linalg.norm(vecs, axis=1).reshape(-1, 1)
        unit_vecs = vecs / np.repeat(norms, 3, axis=1)

        # start ray marching
        ray_directions = unit_vecs
        ray_tips = np.zeros((N, 3))
        self._signed_distance(ray_tips)
        while True:
            sd = self._signed_distance(ray_tips).reshape(N, -1)
            ray_tips += ray_directions * np.repeat(sd, 3, axis=1)
            if np.all(np.abs(sd) < self._surface_threshold):
                break
        sd_final = self._signed_distance(ray_tips)

        return ray_tips, sd_final

class GridSDF(SignedDistanceFunction):

    def __init__(self, sdf_data, origin, resolution,
                 use_abs=True,
                 *args, **kwargs):
        super(GridSDF, self).__init__(origin, resolution, *args, **kwargs)
        self.num_interpolants = 8
        self.min_point_x = [0, 2, 3, 5]
        self.max_point_x = [1, 4, 6, 7]
        self.min_point_y = [0, 1, 3, 6]
        self.max_point_y = [2, 4, 5, 7]
        self.min_point_z = [0, 1, 2, 4]
        self.max_point_z = [3, 5, 6, 7]

        self._data = sdf_data
        self._dims = self.data.shape
        self.resolution = resolution

        spts, _ = self.surface_points()
        self._center = 0.5 * (np.min(spts, axis=0) + np.max(spts, axis=0))

        self.sdf_to_obj_transform = SimilarityTransformCoordinates(
            pos=self.origin,
            scale=self.resolution)

        # buffer
        self._points_buf = np.zeros([self.num_interpolants, 3], dtype=np.int)

        # optionally use only the absolute values
        # (useful for non-closed meshes in 3D)
        self.use_abs = use_abs
        if use_abs:
            self._data = np.abs(self.data)

    @property
    def dimensions(self):
        """GridSDF dimension information.

        Returns
        -------
        self._dims : numpy.ndarray
            dimension of this sdf.
        """
        return self._dims

    @property
    def resolution(self):
        """The grid resolution (how wide each grid cell is).

        Resolution is max dist from a surface when the surface
        is orthogonal to diagonal grid cells

        Returns
        -------
        self._resolution : float
            The width of each grid cell.
        """
        return self._resolution

    @resolution.setter
    def resolution(self, res):
        """Setter of resolution.

        Parameters
        ----------
        res : float
            new resolution.
        """
        self._resolution = res
        self._surface_threshold = res * np.sqrt(2) / 2.0

    @property
    def center(self):
        """Center of grid.

        This basically transforms the world frame to grid center.

        Returns
        -------
        :obj:`numpy.ndarray`
        """
        return self._center

    def on_surface(self, points_sdf):
        """Determines whether or not a point is on the object surface.

        Parameters
        ----------
        points_sdf : :obj:`numpy.ndarray` of int
            A 2- or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        :obj:`tuple` of numpy.ndarray[bool], float
            If the points on th surface and the corresponding 
            signed distances
        """
        sdf_val = self[points_sdf]
        logicals = np.abs(sdf_val) < self.surface_threshold
        return logicals, sdf_val

    def is_out_of_bounds(self, points_sdf):
        """Returns True if points is an out of bounds access.

        Parameters
        ----------
        points_sdf : numpy.ndarray or list of int
            3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        is_out : bool
            If points is in grid, return True.
        """
        points = np.array(points_sdf)
        if points.ndim == 1:
            return np.array(points < 0).any() or \
                np.array(points >= self.dimensions).any()
        elif points.ndim == 2:
            return np.logical_or(
                (points < 0).any(axis=1),
                (points >= np.array(self.dimensions)).any(axis=1))
        else:
            raise ValueError

    @property
    def data(self):
        """The GridSDF data.

        Returns
        -------
        self._data : numpy.ndarray
            The 3-dimensional ndarray that holds the grid of signed distances.
        """
        return self._data

    def _signed_distance(self, points_sdf):
        """Returns the signed distance at the given coordinates

        Interpolating if necessary.

        Parameters
        ----------
        points_sdf : numpy.ndarray
            A 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        float or numpy.ndarray
            The signed distance at the given points (interpolated).
        """
        points_sdf = np.array(points_sdf)
        if points_sdf.ndim == 1:
            if len(points_sdf) != 3:
                raise IndexError('Indexing must be 3 dimensional')
            if self.is_out_of_bounds(points_sdf):
                # logging.debug('Out of bounds access. Snapping to GridSDF dims')
                pass

            # snap to grid dims
            point_buf = np.zeros(3)
            point_buf[0] = max(0, min(points_sdf[0], self.dimensions[0] - 1))
            point_buf[1] = max(0, min(points_sdf[1], self.dimensions[1] - 1))
            point_buf[2] = max(0, min(points_sdf[2], self.dimensions[2] - 1))

            # regular indexing if integers
            if type(points_sdf[0]) is int and \
                    type(points_sdf[1]) is int and \
                    type(points_sdf[2]) is int:
                point_buf = point_buf.astype(np.int)
                return self.data[point_buf[0], point_buf[1], point_buf[2]]

            # otherwise interpolate
            min_point = np.floor(point_buf)
            max_point = min_point + 1  # assumed to be on grid
            self._points_buf[self.min_point_x, 0] = min_point[0]
            self._points_buf[self.max_point_x, 0] = max_point[0]
            self._points_buf[self.min_point_y, 1] = min_point[1]
            self._points_buf[self.max_point_y, 1] = max_point[1]
            self._points_buf[self.min_point_z, 2] = min_point[2]
            self._points_buf[self.max_point_z, 2] = max_point[2]

            # bilinearly interpolate points
            sd = 0.0
            for i in range(self.num_interpolants):
                p = self._points_buf[i, :]
                if self.is_out_of_bounds(p):
                    v = 0.0
                else:
                    v = self.data[p[0], p[1], p[2]]
                w = np.prod(-np.abs(p - point_buf) + 1)
                sd = sd + w * v

            return sd
        elif points_sdf.ndim == 2:
            # for batch input
            point_buf = np.maximum(
                0, np.minimum(points_sdf, np.array(self.dimensions) - 1))
            sd = np.zeros(len(points_sdf), dtype=np.float64)
            no_interpolating = (
                point_buf == np.array(point_buf, dtype=np.int32)).all(axis=1)
            no_interpolating_point = np.array(
                point_buf[no_interpolating], dtype=np.int32)
            if len(no_interpolating_point) > 0:
                sd[no_interpolating] = self.data[
                    no_interpolating_point[:, 0],
                    no_interpolating_point[:, 1],
                    no_interpolating_point[:, 2],
                ]

            interpolating_point = point_buf[np.logical_not(no_interpolating)]
            if len(interpolating_point) == 0:
                return sd

            min_point = np.floor(interpolating_point)
            max_point = min_point + 1  # assumed to be on grid

            n = len(interpolating_point)
            points_buf = np.zeros([n, self.num_interpolants, 3], dtype=np.int)
            points_buf[:, self.min_point_x, 0] = np.repeat(
                min_point[:, 0][None, ], 4, axis=0).T
            points_buf[:, self.max_point_x, 0] = np.repeat(
                max_point[:, 0][None, ], 4, axis=0).T
            points_buf[:, self.min_point_y, 1] = np.repeat(
                min_point[:, 1][None, ], 4, axis=0).T
            points_buf[:, self.max_point_y, 1] = np.repeat(
                max_point[:, 1][None, ], 4, axis=0).T
            points_buf[:, self.min_point_z, 2] = np.repeat(
                min_point[:, 2][None, ], 4, axis=0).T
            points_buf[:, self.max_point_z, 2] = np.repeat(
                max_point[:, 2][None, ], 4, axis=0).T

            # bilinearly interpolate points
            interpolating_sd = sd[np.logical_not(no_interpolating)]
            for i in range(self.num_interpolants):
                p = points_buf[:, i, :]
                valid = np.logical_not(self.is_out_of_bounds(p))
                p = p[valid]
                v = self.data[p[:, 0], p[:, 1], p[:, 2]]
                w = np.prod(-np.abs(p - interpolating_point[valid]) + 1,
                            axis=1)
                interpolating_sd[valid] = interpolating_sd[valid] + w * v
            sd[np.logical_not(no_interpolating)] = interpolating_sd
            return sd
        else:
            raise ValueError

    def __getitem__(self, points_sdf):
        """Returns the signed distance at the given coordinates.

        Parameters
        ----------
        points_sdf : numpy.ndarray
            A or 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        Returns
        -------
        sd : float
            The signed distance at the given points_sdf (interpolated).
        """
        return self._signed_distance(points_sdf)

    def surface_normal(self, points_sdf, delta=1.5):
        """Returns the sdf surface normal at the given coordinates

        Returns the sdf surface normal at the given coordinates by
        computing the tangent plane using GridSDF interpolation.

        Parameters
        ----------
        points_sdf : numpy.ndarray
            A 3-dimensional ndarray that indicates the desired
            coordinates in the grid.

        delta : float
            A radius for collecting surface points near the target coords
            for calculating the surface normal.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            The 3-dimensional ndarray that represents the surface normal.

        Raises
        ------
        IndexError
            If the points does not have three entries.
        """
        points_sdf = np.array(points_sdf)
        if points_sdf.ndim == 1:
            if len(points_sdf) != 3:
                raise IndexError('Indexing must be 3 dimensional')

            # log warning if out of bounds access
            if self.is_out_of_bounds(points_sdf):
                # print('Out of bounds access. Snapping to GridSDF dims')
                pass

            # snap to grid dims
            points_sdf[0] = max(
                0, min(points_sdf[0], self.dimensions[0] - 1))
            points_sdf[1] = max(
                0, min(points_sdf[1], self.dimensions[1] - 1))
            points_sdf[2] = max(
                0, min(points_sdf[2], self.dimensions[2] - 1))
            index_point = np.zeros(3)

            # check points on surface
            sdf_val = self[points_sdf]
            if np.abs(sdf_val) >= self.surface_threshold:
                return None

            # collect all surface points within the delta sphere
            X = []
            d = np.zeros(3)
            dx = -delta
            while dx <= delta:
                dy = -delta
                while dy <= delta:
                    dz = -delta
                    while dz <= delta:
                        d = np.array([dx, dy, dz])
                        if dx != 0 or dy != 0 or dz != 0:
                            d = delta * normalize_vector(d)
                        index_point[0] = points_sdf[0] + d[0]
                        index_point[1] = points_sdf[1] + d[1]
                        index_point[2] = points_sdf[2] + d[2]
                        sdf_val = self[index_point]
                        if np.abs(sdf_val) < self.surface_threshold:
                            X.append([index_point[0], index_point[1],
                                      index_point[2], sdf_val])
                        dz += delta
                    dy += delta
                dx += delta

            # fit a plane to the surface points
            X.sort(key=lambda x: x[3])
            X = np.array(X)[:, :3]
            A = X - np.mean(X, axis=0)
            try:
                U, S, V = np.linalg.svd(A.T)
                n = U[:, 2]
            except np.linalg.LinAlgError:
                return None
            return n
        elif points_sdf.ndim == 2:
            invalid_normals = self.is_out_of_bounds(points_sdf)
            valid_normals = np.logical_not(invalid_normals)
            n = len(points_sdf)
            indices = np.arange(n)[valid_normals]
            normals = np.nan * np.ones((n, 3))
            points_sdf = points_sdf[valid_normals]

            if len(points_sdf) == 0:
                return normals
            points_sdf = np.maximum(
                0, np.minimum(points_sdf, np.array(self.dimensions) - 1))

            # check points on surface
            sdf_val = self[points_sdf]
            valid_surfaces = np.abs(sdf_val) < self.surface_threshold
            indices = indices[valid_surfaces]

            points_sdf = points_sdf[valid_surfaces]

            if len(points_sdf) == 0:
                return normals

            # collect all surface points within the delta sphere
            X = np.inf * np.ones((len(points_sdf), 27, 4), dtype=np.float64)
            dx = - delta
            for i in range(3):
                dy = - delta
                for j in range(3):
                    dz = - delta
                    for k in range(3):
                        d = np.array([dx, dy, dz])
                        if dx != 0 or dy != 0 or dz != 0:
                            d = delta * normalize_vector(d)
                        index_point = points_sdf + d
                        sdf_val = self[index_point]
                        flags = np.abs(sdf_val) < self.surface_threshold
                        X[flags, (i * 9) + (j * 3) + k, :3] = index_point[
                            flags]
                        X[flags, (i * 9) + (j * 3) + k, 3] = sdf_val[flags]
                        dz += delta
                    dy += delta
                dx += delta

            # fit a plane to the surface points
            for i, x in enumerate(X):
                x = x[~np.isinf(x[:, 3])]
                if len(x) != 0:
                    x = x[np.argsort(x[:, 3])]
                    x = x[:, :3]
                    A = x - np.mean(x, axis=0)
                    try:
                        U, S, V = np.linalg.svd(A.T)
                        normal = U[:, 2]
                    except np.linalg.LinAlgError:
                        normal = np.nan * np.ones(3)
                else:
                    normal = np.nan * np.ones(3)
                normals[indices[i]] = normal
            return normals
        else:
            raise ValueError

    def _surface_points(self):
        """Returns the points on the surface.

        Parameters
        ----------
        grid_basis : bool
            If False, the surface points are transformed to the world frame.
            If True (default), the surface points are left in grid coordinates.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of int, :obj:`numpy.ndarray` of
            float. The points on the surface and the signed distances at
            those points.
        """
        surface_points = np.where(np.abs(self.data) < self.surface_threshold)
        x = surface_points[0]
        y = surface_points[1]
        z = surface_points[2]
        surface_points = np.c_[x, np.c_[y, z]]
        surface_values = self.data[surface_points[:, 0],
                                   surface_points[:, 1],
                                   surface_points[:, 2]]
        return surface_points, surface_values

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
        sdf_filepath = pysdfgen.obj2sdf(str(obj_filepath), dim, padding)
        return GridSDF.from_file(sdf_filepath)
