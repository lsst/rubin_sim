import os
import unittest

import numpy as np

from rubin_sim.data import get_data_dir
from rubin_sim.utils import LsstCameraFootprint


class TestLsstcamerafootprint(unittest.TestCase):
    def setUp(self):
        self.obj_ra = np.array([10.0, 12.1], float)
        self.obj_dec = np.array([-30.0, -30.0], float)
        self.obs_ra = np.array([10.0, 10.0], float)
        self.obs_dec = np.array([-30.0, -30.0], float)
        self.obs_rot_sky_pos = np.zeros(2)

    def test_camera(self):
        camera = LsstCameraFootprint(
            units="degrees",
            footprint_file=os.path.join(get_data_dir(), "tests", "fov_map.npz"),
        )
        idx_obs = camera(self.obj_ra, self.obj_dec, self.obs_ra, self.obs_dec, self.obs_rot_sky_pos)
        # The first of these objects should be in the middle of the FOV, while the second is outside
        self.assertEqual(idx_obs, [0])


if __name__ == "__main__":
    unittest.main()
