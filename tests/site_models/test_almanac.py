import unittest

from rubin_sim.site_models import Almanac


class TestAlmanac(unittest.TestCase):
    def test_alm(self):
        alma = Almanac()

        mjd = 59853.35

        # Dead simple make sure the things load.
        planets = alma.get_planet_positions(mjd)
        sun = alma.get_sunset_info(mjd)
        moon = alma.get_sun_moon_positions(mjd)
        indx = alma.mjd_indx(mjd)


if __name__ == "__main__":
    unittest.main()
