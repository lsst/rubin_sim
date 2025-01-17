import unittest

import numpy as np

from rubin_sim.moving_objects import BaseObs


@unittest.skip("Temporary skip until ephemerides replaced")
class TestCamera(unittest.TestCase):
    def setUp(self):
        obj_ra = np.array([10.0, 12.1], float)
        obj_dec = np.array([-30.0, -30.0], float)
        obj_mjd = np.array([59580.16, 59580.16], float)
        self.ephems = np.array(
            list(zip(obj_ra, obj_dec, obj_mjd)),
            dtype=([("ra", float), ("dec", float), ("mjd", float)]),
        )
        obs_ra = np.array([10.0, 10.0], float)
        obs_dec = np.array([-30.0, -30.0], float)
        obs_mjd = np.array([59580.16, 59580.16], float)
        obs_rot_sky_pos = np.zeros(2)
        self.obs = np.array(
            list(zip(obs_ra, obs_dec, obs_rot_sky_pos, obs_mjd)),
            dtype=([("ra", float), ("dec", float), ("rotSkyPos", float), ("mjd", float)]),
        )

    def test_camera_fov(self):
        obs = BaseObs(
            obs_ra="ra",
            obs_dec="dec",
            obs_time_col="mjd",
            footprint="camera",
        )
        idx_obs = obs.sso_in_camera_fov(self.ephems, self.obs)
        self.assertEqual(idx_obs, [0])


if __name__ == "__main__":
    unittest.main()
