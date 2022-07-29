import unittest
import numpy as np
from rubin_sim.utils import survey_start_mjd
from rubin_sim.satellite_constellations import (
    starlink_constellation_v1,
    starlink_constellation_v2,
    oneweb_constellation,
    Constellation,
)


class TestSatellites(unittest.TestCase):
    def test_constellations(self):
        """Test instantiation of slicer sets slicer type as expected."""

        mjd0 = survey_start_mjd()
        sv1 = starlink_constellation_v1()
        sv2 = starlink_constellation_v2()
        ow = oneweb_constellation()

        const = Constellation(sv1)

        length, n_s = const.check_pointing(85., 0., mjd0+1, 30.)

        lengths, n_s = const.check_pointings(np.array([85., 82.]),
                                             np.array([0., 0.]),
                                             np.arange(2)+mjd0+1.5, 30.)

        assert(np.size(lengths) == 2)
        assert(np.size(n_s) == 2)


if __name__ == "__main__":
    unittest.main()
