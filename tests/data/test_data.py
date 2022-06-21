import unittest
from rubin_sim.data import get_baseline, get_data_dir, data_versions


class DataTest(unittest.TestCase):
    def testBaseline(self):
        """
        Get the baseline sim location
        """

        baseline = get_baseline()
        data_dir = get_data_dir()
        versions = data_versions()


if __name__ == "__main__":
    unittest.main()
