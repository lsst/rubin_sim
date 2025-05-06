import unittest

import numpy as np

import rubin_sim.maf_proto as maf


class TestSlicers(unittest.TestCase):

    def test_slicer(self):

        s1 = maf.Slicer(nside=128)
        s2 = maf.Slicer(nside=16)

        assert s1.shape > s2.shape

    def test_user_slicer(self):
        """Test we can set different points."""

        ra = np.arange(0, 360, 1)
        dec = ra * 0 - 20

        s1 = maf.Slicer()
        s1.setup_slice_points(ra_rad=np.radians(ra), dec_rad=np.radians(dec))

        assert s1.shape == ra.size


if __name__ == "__main__":
    unittest.main()
