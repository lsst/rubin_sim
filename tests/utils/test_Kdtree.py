import numpy as np
import unittest
import rubin_sim.utils as utils


class KdTreeTestCase(unittest.TestCase):
    def testKDTreeAPI(self):
        """
        Make sure the API provided by scipy to the kdTree algorithm is functional.
        """
        _ra = np.linspace(0.0, 2.0 * np.pi)
        _dec = np.linspace(-np.pi, np.pi)

        Ra, Dec = np.meshgrid(_ra, _dec)
        tree = utils._buildTree(Ra.flatten(), Dec.flatten())

        x, y, z = utils._xyz_from_ra_dec(_ra, _dec)
        indx = tree.query_ball_point(list(zip(x, y, z)), utils.xyz_angular_radius())

        self.assertEqual(indx.shape, _ra.shape)


if __name__ == "__main__":
    unittest.main()
