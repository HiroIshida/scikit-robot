# The testing order is 
# BoxSDF -> SignedDistanceFunction -> GridSDF > UnionSDF

import unittest
import os
import numpy as np
from numpy import testing
import trimesh
import skrobot

class TestSDF(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        # prepare gridsdf
        objfile_path = skrobot.data.bunny_objpath()
        bunnymesh = trimesh.load_mesh(objfile_path)
        filename, extension = os.path.splitext(objfile_path)
        sdffile_path = filename + ".sdf"
        if os.path.exists(sdffile_path):
            gridsdf = skrobot.sdf.GridSDF.from_file(sdffile_path)
        else:
            gridsdf = skrobot.sdf.GridSDF.from_objfile(objfile_path)
        cls.gridsdf = gridsdf
        cls.bunnymesh = bunnymesh

        # prepare boxsdf and boxmodel
        box_withds = np.array([0.05, 0.1, 0.05])
        boxsdf = skrobot.sdf.BoxSDF([0, 0, 0], box_withds)
        boxtrans = np.array([0.0, 0.1, 0.0])
        boxsdf.translate(boxtrans)

        cls.box_withds = box_withds
        cls.boxsdf = boxsdf
        cls.boxtrans = boxtrans
        cls.points_box_edge_sdf = np.array(
               [-0.5 * box_withds, 0.5 * box_withds])

        # preapare UnionSDF
        unionsdf = skrobot.sdf.UnionSDF(sdf_list=[boxsdf, gridsdf])
        cls.unionsdf = unionsdf

    def test_box__signed_distance(self):
        sdf = self.boxsdf
        X_origin = np.zeros((1, 3))
        self.assertEqual(sdf._signed_distance(X_origin), -min(self.box_withds) * 0.5)
        testing.assert_array_equal(
                sdf._signed_distance(self.points_box_edge_sdf), [0, 0])

    def test_box__surface_points(self):
        sdf = self.boxsdf
        ray_tips, sd_vals = sdf._surface_points()
        testing.assert_array_equal(sdf._signed_distance(ray_tips), sd_vals)
        assert np.all(np.abs(sd_vals) < sdf._surface_threshold)

    def test__transform_pts_obj_to_sdf_and_sdf_to_obj(self):
        sdf, trans = self.boxsdf, self.boxtrans
        points_obj = np.random.randn(100, 3)
        points_sdf = sdf._transform_pts_obj_to_sdf(points_obj)

        # test transform_pts_obj_to_sdf
        points_sdf_should_be = points_obj - \
                np.repeat(trans.reshape((1, -1)), 100, axis=0)
        testing.assert_array_almost_equal(points_sdf, points_sdf_should_be)

        # test transform_pts_sdf_to_obj
        points_obj_recreated = sdf._transform_pts_sdf_to_obj(points_sdf)
        testing.assert_array_almost_equal(points_obj_recreated, points_obj)

    def test___call__(self):
        sdf, trans = self.boxsdf, self.boxtrans
        points_box_edge_obj = np.array(
                [x + trans for x in self.points_box_edge_sdf])
        testing.assert_array_almost_equal(
                sdf(points_box_edge_obj), [0, 0])

    def test_surface_points(self):
        sdf = self.boxsdf
        surface_points_obj, _ = sdf.surface_points(n_sample=20)
        sdf_vals = sdf(surface_points_obj)
        assert np.all(np.abs(sdf_vals) < sdf._surface_threshold)

    def test_on_surface(self):
        sdf = self.boxsdf
        points_box_edge_obj = sdf._transform_pts_sdf_to_obj(self.points_box_edge_sdf)
        logicals_positive, _ = sdf.on_surface(points_box_edge_obj)
        assert np.all(logicals_positive) 

        points_origin = np.zeros((1, 3))
        logicals_negative, _ = sdf.on_surface(points_origin)
        assert np.all(~logicals_negative)

    def test_gridsdf_is_out_of_bounds(self):
        sdf, mesh = self.gridsdf, self.bunnymesh
        vertices_obj = mesh.vertices
        b_min = np.min(vertices_obj, axis=0)
        b_max = np.max(vertices_obj, axis=0)
        center = 0.5 * (b_min + b_max)
        width = b_max - b_min
        points_outer_bbox = np.array([
            center + width,
            center - width
            ])
        # this condition maybe depends on the padding when creating sdf
        assert np.all(sdf.is_out_of_bounds(points_outer_bbox))
        assert np.all(~sdf.is_out_of_bounds(vertices_obj))

    def test_gridsdf__signed_distance(self):
        sdf, mesh = self.gridsdf, self.bunnymesh
        vertices_obj = mesh.vertices
        vertices_sdf = sdf._transform_pts_obj_to_sdf(vertices_obj)
        sd_vals = sdf._signed_distance(vertices_sdf)
        # all vertices of the mesh must be on the surface
        assert np.all(np.abs(sd_vals) < sdf._surface_threshold) 

        # sd of points outside of bounds must be np.inf
        point_outofbound = (sdf._dims + 1).reshape(1, 3)
        sd_vals = sdf._signed_distance(point_outofbound)
        assert np.all(np.isinf(sd_vals))

    def test_gridsdf_surface_points(self):
        sdf, mesh = self.gridsdf, self.bunnymesh
        surf_points_obj, _ = sdf.surface_points()
        logicals, _ = sdf.on_surface(surf_points_obj)
        assert np.all(logicals) 

    def test_unionsdf___call__(self):
        sdf = self.unionsdf
        pts_on_surface = np.array([
            [-0.07196818,  0.16532058, -0.04285806],
            [ 0.02802324,  0.11360088, -0.00837826],
            [-0.05472828,  0.03257335,  0.00886164],
            [ 0.0077233 ,  0.15      , -0.01742908],
            [ 0.02802324,  0.11360088, -0.00837826],
            [-0.07714015,  0.15152866,  0.0329975 ]
            ])
        assert np.all(abs(sdf(pts_on_surface)) < sdf._surface_threshold)

    def test_unionsdf_surface_points(self):
        sdf = self.unionsdf
        sdf.surface_points()
        pts, sd_vals = sdf.surface_points()
        assert np.all(np.abs(sd_vals) < sdf._surface_threshold)

        sub_sdf1, sub_sdf2 = self.unionsdf.sdf_list
        on_surface1 = sub_sdf1(pts) < sub_sdf1._surface_threshold
        on_surface2 = sub_sdf2(pts) < sub_sdf2._surface_threshold
        cond_or = np.logical_or(on_surface1, on_surface2) 
        assert np.all(cond_or) # at least on either of the surface

        cond_and = (sum(on_surface1) > 0) and (sum(on_surface2) > 0)
        assert cond_and # each surface has at least a single points
