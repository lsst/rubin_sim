import numpy as np
import unittest
from rubin_sim.site_models import CloudModel


class TestCloudModel(unittest.TestCase):
    def test_call(self):
        cloudModel = CloudModel()
        in_cloud = 1.53
        efdData = {"cloud": in_cloud}
        alt = np.zeros(50, float)
        az = np.zeros(50, float)
        targetDict = {"altitude": alt, "azimuth": az}
        out_cloud = cloudModel(efdData, targetDict)["cloud"]
        # Test that we propagated cloud value over the whole sky.
        self.assertEqual(in_cloud, out_cloud.max())
        self.assertEqual(in_cloud, out_cloud.min())
        self.assertEqual(len(out_cloud), len(alt))


if __name__ == "__main__":
    unittest.main()
