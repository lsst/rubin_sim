import numpy as np
import unittest
from rubin_sim.scheduler.utils import season_calc, create_season_offset
import healpy as hp


class TestFeatures(unittest.TestCase):

    def testSeason(self):
        """
        Test that the season utils work as intended
        """
        night = 365.25 * 3.5
        plain = season_calc(night)
        assert(plain == 3)

        mod2 = season_calc(night, modulo=2)
        assert(mod2 == 1)

        mod3 = season_calc(night, modulo=3)
        assert(mod3 == 0)

        mod3 = season_calc(night, modulo=3, max_season=2)
        assert(mod3 == -1)

        mod3 = season_calc(night, modulo=3, max_season=2, offset=-365.25*2)
        assert(mod3 == 1)

        mod3 = season_calc(night, modulo=3, max_season=2, offset=-365.25*10)
        assert(mod3 == -1)

        mod3 = season_calc(night, modulo=3, offset=-365.25*10)
        assert(mod3 == -1)


if __name__ == "__main__":
    unittest.main()
