import os
import unittest

from rubin_scheduler.data import data_versions, get_baseline, get_data_dir

from rubin_sim.data import get_data_dir as gdd


class DataTest(unittest.TestCase):
    def testBaseline(self):
        """
        Get the baseline sim location
        """
        data_dir = get_data_dir()
        dd2 = gdd()

        assert data_dir == dd2

        if "sim_baseline" in os.listdir(data_dir):
            baseline = get_baseline()
        versions = data_versions()


if __name__ == "__main__":
    unittest.main()
