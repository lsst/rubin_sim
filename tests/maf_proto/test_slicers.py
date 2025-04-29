import numpy as np
import unittest

import rubin_sim.maf_proto as maf


class TestSlicers(unittest.TestCase):

    def test_slicer(self):

        s1 = maf.Slicer(nside=128)
        s2 = maf.Slicer(nside=16)

        assert s1.shape > s2.shape

    def test_user_slicer(self):

        ra = np.arange(0, 360, 1)
        dec = ra * 0 - 20

        s1 = maf.UserSlicer(ra, dec)

        assert s1.shape == ra.size


if __name__ == "__main__":
    unittest.main()
