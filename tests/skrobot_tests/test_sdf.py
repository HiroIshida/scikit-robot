import unittest
import os
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

    def test_box__signed_distance(self):
        sdf = self.boxsdf 
        X_origin = np.zeros((1, 3))
        self.assertEqual(sdf._signed_distance(X_origin), -0.0025)

        X_edges = np.array([
            [-0.0025, -0.005, -0.02],
            [-0.0025, -0.005, 0.02],
            ])
        testing.assert_array_equal(sdf._signed_distance(X_edges), [0, 0])

    def test_box__surface_points(self):
        N = 20
        ray_tips, sd_vals = sdf._surface_points(N=N)
        testing.assert_array_equal(sdf._signed_distance(ray_tips), sd_vals)
        assert np.all(np.abs(sd_vals) < sdf._surface_threshold)

