import unittest

import numpy as np
from rubin_scheduler.utils import SURVEY_START_MJD

from rubin_sim.satellite_constellations import Constellation, oneweb_tles, starlink_tles_v1, starlink_tles_v2


class TestSatellites(unittest.TestCase):
    def test_constellations(self):
        """Test stellite constellations"""

        mjd0 = SURVEY_START_MJD
        sv1 = starlink_tles_v1()
        _ = starlink_tles_v2()
        _ = oneweb_tles()

        const = Constellation(sv1)

        lengths, n_s = const.check_pointings(
            np.array([85.0, 82.0]),
            np.array([0.0, 0.0]),
            np.arange(2) + mjd0 + 1.5,
            30.0,
        )

        assert np.size(lengths) == 2
        assert np.size(n_s) == 2


if __name__ == "__main__":
    unittest.main()
