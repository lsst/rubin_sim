import unittest

import numpy as np

from rubin_sim.site_models import CloudModel


class TestCloudModel(unittest.TestCase):
    def test_call(self):
        cloud_model = CloudModel()
        in_cloud = 1.53
        efd_data = {"cloud": in_cloud}
        alt = np.zeros(50, float)
        az = np.zeros(50, float)
        target_dict = {"altitude": alt, "azimuth": az}
        out_cloud = cloud_model(efd_data, target_dict)["cloud"]
        # Test that we propagated cloud value over the whole sky.
        self.assertEqual(in_cloud, out_cloud.max())
        self.assertEqual(in_cloud, out_cloud.min())
        self.assertEqual(len(out_cloud), len(alt))


if __name__ == "__main__":
    unittest.main()
