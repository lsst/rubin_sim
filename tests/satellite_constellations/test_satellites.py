import unittest
from rubin_sim.satellite_constellations import (starlink_constellation_v1,
                                                starlink_constellation_v2,
                                                oneweb_constellation,
                                                Constellation)


class TestSatellites(unittest.TestCase):

    def test_constellations(self):
        """Test instantiation of slicer sets slicer type as expected."""
        sv1 = starlink_constellation_v1()
        sv2 = starlink_constellation_v2()
        ow = oneweb_constellation()

        const = Constellation(sv1)


if __name__ == "__main__":
    unittest.main()
