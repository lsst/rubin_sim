# pylint: disable=too-many-arguments
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name

# imports
import unittest

import astropy
import numpy as np

from rubin_sim.utils import riseset

# constants

RANDOM_SEED = 6563
LOCATION = astropy.coordinates.EarthLocation.of_site("Cerro Pachon")


class TestRiseset(unittest.TestCase):
    num_times = 100
    tolerance = 1e-8
    alt_deg = -14.0

    def setUp(self):
        np.random.seed(RANDOM_SEED)
        self.request_mjds = 60000 + 365.24 * np.random.random(self.num_times)

    def test_riseset_times_down(self):
        for body in ("sun", "moon"):
            mjds = riseset.riseset_times(
                self.request_mjds,
                which_direction="down",
                which_night="nearest",
                alt=self.alt_deg,
                location=LOCATION,
                body=body,
                tolerance=self.tolerance,
            )
            self.assertTrue(np.allclose(mjds, self.request_mjds, atol=0.5))
            self.assertTrue(np.allclose(_body_alts(mjds, body), self.alt_deg, atol=self.tolerance))
            self.assertTrue(np.all(~_body_rising(mjds, body)))

    def test_riseset_times_up(self):
        for body in ("sun", "moon"):
            mjds = riseset.riseset_times(
                self.request_mjds,
                which_direction="up",
                which_night="nearest",
                alt=self.alt_deg,
                location=LOCATION,
                body=body,
                tolerance=self.tolerance,
            )
            self.assertTrue(np.allclose(mjds, self.request_mjds, atol=0.5))
            self.assertTrue(np.allclose(_body_alts(mjds, body), self.alt_deg, atol=self.tolerance))
            self.assertTrue(np.all(_body_rising(mjds, body)))

    def test_riseset_times_next_up(self):
        mjds = riseset.riseset_times(
            self.request_mjds,
            which_direction="up",
            which_night="next",
            alt=self.alt_deg,
            location=LOCATION,
            body="sun",
            tolerance=self.tolerance,
        )
        self.assertTrue(np.all(mjds >= self.request_mjds))
        self.assertTrue(np.allclose(_body_alts(mjds, "sun"), self.alt_deg, atol=self.tolerance))
        self.assertTrue(np.all(_body_rising(mjds, "sun")))

    def test_riseset_times_previous_up(self):
        mjds = riseset.riseset_times(
            self.request_mjds,
            which_direction="up",
            which_night="previous",
            alt=self.alt_deg,
            location=LOCATION,
            body="sun",
            tolerance=self.tolerance,
        )
        self.assertTrue(np.all(mjds <= self.request_mjds))
        self.assertTrue(np.allclose(_body_alts(mjds, "sun"), self.alt_deg, atol=self.tolerance))
        self.assertTrue(np.all(_body_rising(mjds, "sun")))

    def test_riseset_times_next_down(self):
        mjds = riseset.riseset_times(
            self.request_mjds,
            which_direction="down",
            which_night="next",
            alt=self.alt_deg,
            location=LOCATION,
            body="sun",
            tolerance=self.tolerance,
        )
        self.assertTrue(np.all(mjds >= self.request_mjds))
        self.assertTrue(np.allclose(_body_alts(mjds, "sun"), self.alt_deg, atol=self.tolerance))
        self.assertTrue(np.all(~_body_rising(mjds, "sun")))

    def test_riseset_times_previous_down(self):
        mjds = riseset.riseset_times(
            self.request_mjds,
            which_direction="down",
            which_night="previous",
            alt=self.alt_deg,
            location=LOCATION,
            body="sun",
            tolerance=self.tolerance,
        )
        self.assertTrue(np.all(mjds <= self.request_mjds))
        self.assertTrue(np.allclose(_body_alts(mjds, "sun"), self.alt_deg, atol=self.tolerance))
        self.assertTrue(np.all(~_body_rising(mjds, "sun")))


def _body_alts(mjds, body):
    times = astropy.time.Time(mjds, scale="utc", format="mjd", location=LOCATION)
    crds = astropy.coordinates.get_body(body, times, location=LOCATION)
    alt = crds.transform_to(astropy.coordinates.AltAz(obstime=times, location=LOCATION)).alt.deg
    return alt


def _body_rising(mjds, body):
    test_dt = 1 / (24 * 60.0)
    now_body_alts = _body_alts(mjds, body)
    future_body_alts = _body_alts(mjds + test_dt, body)
    rising = future_body_alts > now_body_alts
    return rising


run_tests_now = __name__ == "__main__"
if run_tests_now:
    unittest.main()
