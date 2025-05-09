import unittest

import numpy as np

import rubin_sim.maf_proto as maf


class TestBatches(unittest.TestCase):

    def test_glance(self):
        """Test that the simple way to run MAF works"""
        ss = maf.glance(quick_test=True)
        assert np.size(ss) > 0

    def test_sne(self):
        ss = maf.sne_batch(quick_test=True)
        assert np.size(ss) > 0

    def test_kne(self):
        ss = maf.kne_batch(quick_test=True)
        assert np.size(ss) > 0


if __name__ == "__main__":
    unittest.main()
