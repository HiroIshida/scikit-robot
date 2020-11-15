# The testing order is 
# BoxSDF -> SignedDistanceFunction -> GridSDF

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
        boxsdf = skrobot.sdf.BoxSDF([0, 0, 0], [0.005, 0.01, 0.1])
        boxmodel = skrobot.model.Link()
        boxmodel.assoc(boxsdf)
        boxtrans = np.array([0.1, 0.1, 0.1])
        boxmodel.translate(boxtrans)

        cls.boxsdf = boxsdf
        cls.boxmodel = boxmodel
        cls.boxtrans = boxtrans
        cls.points_box_edge_sdf = np.array([
            [-0.0025, -0.005, -0.02],
            [-0.0025, -0.005, 0.02],
            ])

    def test_box__signed_distance(self):
        sdf, model = self.boxsdf, self.boxmodel
        X_origin = np.zeros((1, 3))
        self.assertEqual(sdf._signed_distance(X_origin), -0.0025)
        testing.assert_array_equal(
                sdf._signed_distance(self.points_box_edge_sdf), [0, 0])

    def test_box__surface_points(self):
        sdf, model = self.boxsdf, self.boxmodel
        ray_tips, sd_vals = sdf._surface_points()
        testing.assert_array_equal(sdf._signed_distance(ray_tips), sd_vals)
        assert np.all(np.abs(sd_vals) < sdf._surface_threshold)

    def test__transform_pts_obj_to_sdf_and_sdf_to_obj(self):
        sdf, model, trans = self.boxsdf, self.boxmodel, self.boxtrans
        points_obj = np.random.randn(100, 3)
        points_sdf = sdf.transform_pts_obj_to_sdf(points_obj)

        # test transform_pts_obj_to_sdf
        points_sdf_should_be = points_obj - \
                np.repeat(trans.reshape((1, -1)), 100, axis=0)
        testing.assert_array_almost_equal(points_sdf, points_sdf_should_be)

        # test transform_pts_sdf_to_obj
        points_obj_recreated = sdf.transform_pts_sdf_to_obj(points_sdf)
        testing.assert_array_almost_equal(points_obj_recreated, points_obj)

    def test___call__(self):
        sdf, model, trans = self.boxsdf, self.boxmodel, self.boxtrans
        points_box_edge_obj = np.array(
                [x + trans for x in self.points_box_edge_sdf])
        testing.assert_array_almost_equal(
                sdf(points_box_edge_obj), [0, 0])

    def test_surface_points(self):
        sdf, model = self.boxsdf, self.boxmodel
        surface_points_obj, _ = sdf.surface_points(N=20)
        sdf_vals = sdf(surface_points_obj)
        assert np.all(np.abs(sdf_vals) < sdf._surface_threshold)

    def test_gridsdf_is_out_of_bounds(self):
        sdf, mesh = self.gridsdf, self.bunnymesh
        vertices_obj = mesh.vertices
        vertices_sdf = sdf.transform_pts_obj_to_sdf(vertices_obj)
        b_min = np.min(vertices_sdf, axis=0)
        b_max = np.max(vertices_sdf, axis=0)
        center = 0.5 * (b_min + b_max)
        width = b_max - b_min
        points_outer_bbox = np.array([
            center + width,
            center - width
            ])
        # this condition maybe depends on the padding when creating sdf
        assert np.all(sdf.is_out_of_bounds(points_outer_bbox))
        assert np.all(~sdf.is_out_of_bounds(vertices_sdf))

    def test_gridsdf__signed_distance(self):
        sdf, mesh = self.gridsdf, self.bunnymesh
        vertices_obj = mesh.vertices
        vertices_sdf = sdf.transform_pts_obj_to_sdf(vertices_obj)
        sd_vals = sdf._signed_distance(vertices_sdf)
        assert np.all(np.abs(sd_vals) < sdf._surface_threshold) 

    def test_gridsdf_on_surface(self):
        # TODO this method must be w.r.t. the obj coordinate
        sdf, mesh = self.gridsdf, self.bunnymesh
        vertices_obj = mesh.vertices
        vertices_sdf = sdf.transform_pts_obj_to_sdf(vertices_obj)
        logicals_should_be_postive, _ = sdf.on_surface(vertices_sdf)
        # vertices must be on surface
        assert np.all(logicals_should_be_postive) 

        # compute bounding box
        b_min = np.min(vertices_sdf, axis=0)
        b_max = np.max(vertices_sdf, axis=0)
        center = 0.5 * (b_min + b_max)
        width = b_max - b_min
        eps = 1e-2
        points_outer_bbox = np.array([
            center + (0.5 + eps) * width,
            center - (0.5 + eps) * width
            ])
        logicals_should_be_negative, _ = sdf.on_surface(points_outer_bbox)
        # points in slightly outside of box must be out of the surface
        assert np.all(~logicals_should_be_negative) 

    def test_gridsdf_surface_points(self):
        sdf, mesh = self.gridsdf, self.bunnymesh
        surf_points_obj, _ = sdf.surface_points()
        surf_points_sdf = sdf.transform_pts_obj_to_sdf(surf_points_obj)
        logicals, _ = sdf.on_surface(surf_points_sdf)
        assert np.all(logicals) 
