import unittest
import os
import copy
import numpy as np
from numpy import testing
import trimesh
import skrobot

class TestSDF(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        objfile_path = skrobot.data.bunny_objpath()
        filename, extension = os.path.splitext(objfile_path)
        sdffile_path = filename + ".sdf"
        if os.path.exists(sdffile_path):
            gridsdf = skrobot.sdf.GridSDF.from_file(sdffile_path)
        else:
            gridsdf = skrobot.sdf.GridSDF.from_objfile(objfile_path)
        cls.gridsdf = gridsdf
        cls.boxsdf = skrobot.sdf.BoxSDF([0, 0, 0], [0.005, 0.01, 0.1])
        cls.points_box_edge_sdf = np.array([
            [-0.0025, -0.005, -0.02],
            [-0.0025, -0.005, 0.02],
            ])

    def test_box__signed_distance(self):
        sdf = copy.deepcopy(self.boxsdf)
        X_origin = np.zeros((1, 3))
        self.assertEqual(sdf._signed_distance(X_origin), -0.0025)
        testing.assert_array_equal(
                sdf._signed_distance(self.points_box_edge_sdf), [0, 0])

    def test_box__surface_points(self):
        sdf = copy.deepcopy(self.boxsdf)
        ray_tips, sd_vals = sdf._surface_points()
        testing.assert_array_equal(sdf._signed_distance(ray_tips), sd_vals)
        assert np.all(np.abs(sd_vals) < sdf._surface_threshold)

    def test__transform_pts_obj_to_sdf_and_sdf_to_obj(self):
        # transform_pts_obj_to_sdf and transform_pts_sdf_to_obj
        sdf = copy.deepcopy(self.boxsdf)
        boxmodel = skrobot.model.Link()
        boxmodel.assoc(sdf)
        translation = np.array([0.1, 0.1, 0.1])
        boxmodel.translate(translation)
        points_obj = np.random.randn(100, 3)
        points_sdf = sdf.transform_pts_obj_to_sdf(points_obj.T)

        # test transform_pts_obj_to_sdf
        points_sdf_should_be = points_obj - \
                np.repeat(translation.reshape((1, -1)), 100, axis=0)
        testing.assert_array_almost_equal(points_sdf, points_sdf_should_be)

        # test transform_pts_sdf_to_obj
        points_obj_recreated = sdf.transform_pts_sdf_to_obj(points_sdf.T)
        testing.assert_array_almost_equal(points_obj_recreated, points_obj)

    def test___call__(self):
        sdf = copy.deepcopy(self.boxsdf)
        boxmodel = skrobot.model.Link()
        boxmodel.assoc(sdf)
        trans = np.array([0.1, 0.1, 0.1])
        boxmodel.translate(trans)
        points_box_edge_obj = np.array(
                [x + trans for x in self.points_box_edge_sdf])
        testing.assert_array_almost_equal(
                sdf(points_box_edge_obj), [0, 0])

    def test_surface_points(self):
        sdf = copy.deepcopy(self.boxsdf)
        boxmodel = skrobot.model.Link()
        boxmodel.assoc(sdf)
        trans = np.array([0.1, 0.1, 0.1])
        boxmodel.translate(trans)

        surface_points_obj, _ = sdf.surface_points(N=20)
        sdf_vals = sdf(surface_points_obj.T)
        assert np.all(np.abs(sdf_vals) < sdf._surface_threshold)
