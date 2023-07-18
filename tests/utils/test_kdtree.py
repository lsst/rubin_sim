import unittest

import numpy as np

import rubin_sim.utils as utils


class KdTreeTestCase(unittest.TestCase):
    def test_kd_tree_api(self):
        """
        Make sure the API provided by scipy to the kdTree algorithm is functional.
        """
        _ra = np.linspace(0.0, 2.0 * np.pi)
        _dec = np.linspace(-np.pi, np.pi)

        ra, dec = np.meshgrid(_ra, _dec)
        tree = utils._build_tree(ra.flatten(), dec.flatten())

        x, y, z = utils._xyz_from_ra_dec(_ra, _dec)
        indx = tree.query_ball_point(list(zip(x, y, z)), utils.xyz_angular_radius())

        self.assertEqual(indx.shape, _ra.shape)


if __name__ == "__main__":
    unittest.main()
