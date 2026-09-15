import unittest

import numpy as np

import rubin_sim.splat as splat


class TestBatches(unittest.TestCase):

    def test_glance(self):
        """Test that the simple way to run splat works"""
        ss = splat.glance(quick_test=True)
        assert np.size(ss) > 0

    def test_agn(self):
        agn = splat.agn_batch(quick_test=True)
        assert agn is None

    def test_sne(self):
        ss = splat.sne_batch(quick_test=True)
        assert np.size(ss) > 0

    def test_kne(self):
        ss = splat.kne_batch(quick_test=True)
        assert np.size(ss) > 0

    def test_astrom(self):
        astrom = splat.astrometry_batch(quick_test=True)
        assert np.size(astrom) > 0

    def test_color_slope(self):
        color_s = splat.color_slope_batch(quick_test=True)
        assert np.size(color_s) > 0

    def test_microlensing(self):
        summary = splat.microlensing_batch(quick_test=True)
        assert np.size(summary) > 0

    def test_xrb(self):
        summary = splat.xrb_batch(quick_test=True)
        assert np.size(summary) > 0

    def test_bd(self):
        summary = splat.bd_batch(quick_test=True)
        assert np.size(summary) > 0

    def test_tde(self):
        summary = splat.tde_batch(quick_test=True)
        assert np.size(summary) > 0


if __name__ == "__main__":

    unittest.main()
