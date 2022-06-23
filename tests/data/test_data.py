import unittest
import os
from rubin_sim.data import get_baseline, get_data_dir, data_versions


class DataTest(unittest.TestCase):
    def testBaseline(self):
        """
        Get the baseline sim location
        """
        data_dir = get_data_dir()
        print(data_dir)
        print(os.listdir(os.path.join(data_dir, "sim_baseline")))
        baseline = get_baseline()
        versions = data_versions()


if __name__ == "__main__":
    unittest.main()
