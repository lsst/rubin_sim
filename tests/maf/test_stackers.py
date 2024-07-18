import os
import sys
import unittest
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from rubin_scheduler.utils import Site, _alt_az_pa_from_ra_dec, calc_lmst

import rubin_sim.maf.stackers as stackers
from rubin_sim.data import get_data_dir
from rubin_sim.maf import get_sim_data

try:
    import lsst.summit.utils.efdUtils
except ModuleNotFoundError:
    pass

TEST_DB = "example_v3.4_0yrs.db"


class TestStackerClasses(unittest.TestCase):
    def setUp(self):
        # get some of the test data
        test_db = os.path.join(get_data_dir(), "tests", TEST_DB)
        query = "select * from observations limit 1000"
        self.test_data = get_sim_data(test_db, None, [], full_sql_query=query)

    def test_stackers_run(self):
        """Just run all of the stackers with our example data."""
        for stacker_class in stackers.BaseStacker.registry.values():
            stacker = stacker_class()
            stacker_name = stacker.__class__.__name__.lower()
            if stacker_name.startswith("sdss"):
                continue
            if isinstance(stacker, stackers.BaseMoStacker):
                continue
            try:
                stacker.run(self.test_data)
            except NotImplementedError:
                pass
            except:
                print(f"Failed at stacker {stacker.__class__.__name__}")
                raise

    def test_add_cols(self):
        """Test that we can add columns as expected."""
        data = np.zeros(90, dtype=list(zip(["alt"], [float])))
        data["alt"] = np.arange(0, 90)
        stacker = stackers.ZenithDistStacker(alt_col="alt", degrees=True)
        newcol = stacker.cols_added[0]
        # First - are the columns added if they are not there.
        data, cols_present = stacker._add_stacker_cols(data)
        self.assertEqual(cols_present, False)
        self.assertIn(newcol, data.dtype.names)
        # Next - if they are present, is that information passed back?
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data, cols_present = stacker._add_stacker_cols(data)
            self.assertEqual(cols_present, True)

    def test_eq(self):
        """
        Test that stackers can be compared
        """
        s1 = stackers.ParallaxFactorStacker()
        s2 = stackers.ParallaxFactorStacker()
        assert s1 == s2

        # Test if they have numpy array atributes
        s1.ack = np.arange(10)
        s2.ack = np.arange(10)
        assert s1 == s2

        # Change the array and test
        s1.ack += 1
        assert s1 != s2

        s2 = stackers.NormAirmassStacker()
        assert s1 != s2

    def test_norm_airmass(self):
        """
        Test the normalized airmass stacker.
        """
        rng = np.random.RandomState(232)
        data = np.zeros(600, dtype=list(zip(["airmass", "fieldDec"], [float, float])))
        data["airmass"] = rng.random_sample(600)
        data["fieldDec"] = rng.random_sample(600) * np.pi - np.pi / 2.0
        data["fieldDec"] = np.degrees(data["fieldDec"])
        stacker = stackers.NormAirmassStacker(degrees=True)
        data = stacker.run(data)
        for i in np.arange(data.size):
            self.assertLessEqual(data["normairmass"][i], data["airmass"][i])
        self.assertLess(np.min(data["normairmass"] - data["airmass"]), 0)

    def test_parallax_factor(self):
        """
        Test the parallax factor.
        """

        data = np.zeros(
            600,
            dtype=list(
                zip(
                    ["fieldRA", "fieldDec", "observationStartMJD"],
                    [float, float, float],
                )
            ),
        )
        data["fieldRA"] = data["fieldRA"] + 0.1
        data["fieldDec"] = data["fieldDec"] - 0.1
        data["observationStartMJD"] = np.arange(data.size) + 49000.0
        stacker = stackers.ParallaxFactorStacker(degrees=True)
        data = stacker.run(data)
        self.assertLess(max(np.abs(data["ra_pi_amp"])), 1.1)
        self.assertLess(max(np.abs(data["dec_pi_amp"])), 1.1)
        self.assertLess(np.max(data["ra_pi_amp"] ** 2 + data["dec_pi_amp"] ** 2), 1.1)
        self.assertGreater(min(np.abs(data["ra_pi_amp"])), 0.0)
        self.assertGreater(min(np.abs(data["dec_pi_amp"])), 0.0)

    def _t_dither_range(self, diffsra, diffsdec, ra, dec, max_dither):
        self.assertLessEqual(np.abs(diffsra).max(), max_dither)
        self.assertLessEqual(np.abs(diffsdec).max(), max_dither)
        offsets = np.sqrt(diffsra**2 + diffsdec**2)
        self.assertLessEqual(offsets.max(), max_dither)
        self.assertGreater(diffsra.max(), 0)
        self.assertGreater(diffsdec.max(), 0)
        self.assertLess(diffsra.min(), 0)
        self.assertLess(diffsdec.min(), 0)

    def _t_dither_per_night(self, diffsra, diffsdec, ra, dec, nights):
        n = np.unique(nights)
        for ni in n:
            match = np.where(nights == ni)[0]
            dra_on_night = np.abs(np.diff(diffsra[match]))
            ddec_on_night = np.abs(np.diff(diffsdec[match]))
            if dra_on_night.max() > 0.0005:
                print(ni)
                m = np.where(dra_on_night > 0.0005)[0]
                print(diffsra[match][m])
                print(ra[match][m])
                print(dec[match][m])
                print(dra_on_night[m])
            self.assertAlmostEqual(dra_on_night.max(), 0)
            self.assertAlmostEqual(ddec_on_night.max(), 0)

    def test_ha_stacker(self):
        """Test the Hour Angle stacker"""
        data = np.zeros(100, dtype=list(zip(["observationStartLST", "fieldRA"], [float, float])))
        data["observationStartLST"] = np.arange(100) / 99.0 * np.pi * 2
        stacker = stackers.HourAngleStacker(degrees=True)
        data = stacker.run(data)
        # Check that data is always wrapped
        self.assertLess(np.max(data["HA"]), 12.0)
        self.assertGreater(np.min(data["HA"]), -12.0)
        # Check that HA is zero if lst == RA
        data = np.zeros(1, dtype=list(zip(["observationStartLST", "fieldRA"], [float, float])))
        data = stacker.run(data)
        self.assertEqual(data["HA"], 0.0)
        data = np.zeros(1, dtype=list(zip(["observationStartLST", "fieldRA"], [float, float])))
        data["observationStartLST"] = 20.0
        data["fieldRA"] = 20.0
        data = stacker.run(data)
        self.assertEqual(data["HA"], 0.0)
        # Check a value
        data = np.zeros(1, dtype=list(zip(["observationStartLST", "fieldRA"], [float, float])))
        data["observationStartLST"] = 0.0
        data["fieldRA"] = np.degrees(np.pi / 2.0)
        data = stacker.run(data)
        np.testing.assert_almost_equal(data["HA"], -6.0)

    def test_pa_stacker(self):
        """Test the parallacticAngleStacker"""
        data = np.zeros(
            100,
            dtype=list(
                zip(
                    [
                        "observationStartMJD",
                        "fieldDec",
                        "fieldRA",
                        "observationStartLST",
                    ],
                    [float] * 4,
                )
            ),
        )
        data["observationStartMJD"] = np.arange(100) * 0.2 + 50000
        site = Site(name="LSST")
        data["observationStartLST"] = calc_lmst(data["observationStartMJD"], site.longitude_rad)
        data["observationStartLST"] = data["observationStartLST"] * 180.0 / 12.0
        stacker = stackers.ParallacticAngleStacker(degrees=True)
        data = stacker.run(data)
        # Check values are in good range
        assert data["PA"].max() <= 180
        assert data["PA"].min() >= -180

        # Check compared to the util
        check_pa = []
        ras = np.radians(data["fieldRA"])
        decs = np.radians(data["fieldDec"])
        for ra, dec, mjd in zip(ras, decs, data["observationStartMJD"]):
            alt, az, pa = _alt_az_pa_from_ra_dec(ra, dec, mjd, site.longitude_rad, site.latitude_rad)

            check_pa.append(pa)
        check_pa = np.degrees(check_pa)

        np.testing.assert_array_almost_equal(data["PA"], check_pa, decimal=0)

    def test_galactic_stacker(self):
        """
        Test the galactic coordinate stacker
        """
        ra, dec = np.degrees(
            np.meshgrid(np.arange(0, 2.0 * np.pi, 0.1), np.arange(-np.pi / 2, np.pi / 2, 0.1))
        )
        ra = np.ravel(ra)
        dec = np.ravel(dec)
        data = np.zeros(ra.size, dtype=list(zip(["ra", "dec"], [float] * 2)))
        data["ra"] += ra
        data["dec"] += dec
        s = stackers.GalacticStacker(ra_col="ra", dec_col="dec")
        new_data = s.run(data)
        c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg).transform_to("galactic")
        expected_l, expected_b = c.l.rad, c.b.rad
        np.testing.assert_array_equal(new_data["gall"], expected_l)
        np.testing.assert_array_equal(new_data["galb"], expected_b)

        # Check that we have all the quadrants populated
        q1 = np.where((new_data["gall"] < np.pi) & (new_data["galb"] < 0.0))[0]
        q2 = np.where((new_data["gall"] < np.pi) & (new_data["galb"] > 0.0))[0]
        q3 = np.where((new_data["gall"] > np.pi) & (new_data["galb"] < 0.0))[0]
        q4 = np.where((new_data["gall"] > np.pi) & (new_data["galb"] > 0.0))[0]
        assert q1.size > 0
        assert q2.size > 0
        assert q3.size > 0
        assert q4.size > 0

    def test_ecliptic_stacker(self):
        ra, dec = np.degrees(
            np.meshgrid(np.arange(0, 2.0 * np.pi, 0.1), np.arange(-np.pi / 2, np.pi / 2, 0.1))
        )
        ra = np.ravel(ra)
        dec = np.ravel(dec)
        data = np.zeros(ra.size, dtype=list(zip(["ra", "dec"], [float] * 2)))
        data["ra"] += ra
        data["dec"] += dec
        s = stackers.EclipticStacker(ra_col="ra", dec_col="dec", degrees=True)
        _ = s.run(data)

        data = np.zeros(ra.size, dtype=list(zip(["ra", "dec"], [float] * 2)))
        data["ra"] += ra
        data["dec"] += dec
        s = stackers.EclipticStacker(ra_col="ra", dec_col="dec", degrees=True, subtract_sun_lon=False)
        _ = s.run(data)

    def test_teff_stacker(self):
        rng = np.random.default_rng(seed=6563)
        num_points = 5
        data = np.zeros(
            num_points,
            dtype=list(zip(["fiveSigmaDepth", "filter", "visitExposureTime"], [float, (np.str_, 1), float])),
        )
        data["fiveSigmaDepth"] = 23 + rng.random(num_points)
        data["filter"] = ["g"] * num_points
        data["visitExposureTime"] = [30] * num_points

        stacker = stackers.TeffStacker("fiveSigmaDepth", "filter", "visitExposureTime")
        value = stacker.run(data)
        np.testing.assert_array_equal(value["fiveSigmaDepth"], value["fiveSigmaDepth"])
        assert np.all(0.1 < value["t_eff"]) and np.all(value["t_eff"] < 10)

    def test_observation_start_datetime64_stacker(self):
        rng = np.random.default_rng(seed=6563)
        num_points = 5
        data = np.zeros(
            num_points,
            dtype=list(zip(["observationStartMJD"], [float])),
        )
        data["observationStartMJD"] = 61000 + 3000 * rng.random(num_points)

        stacker = stackers.ObservationStartDatetime64Stacker("observationStartMJD")
        value = stacker.run(data)
        recovered_mjd = Time(value["observationStartDatetime64"], format="datetime64").mjd
        assert np.allclose(recovered_mjd, data["observationStartMJD"])

    def test_day_obs_stackers(self):
        rng = np.random.default_rng(seed=6563)
        num_points = 5
        obs_start_mjds = 61000 + 3000 * rng.random(num_points)

        data = np.zeros(
            num_points,
            dtype=list(zip(["observationStartMJD"], [float])),
        )
        data["observationStartMJD"] = obs_start_mjds

        try:
            summit_get_day_obs = lsst.summit.utils.efdUtils.getDayObsForTime
        except NameError:
            summit_get_day_obs = None

        day_obs_int_stacker = stackers.DayObsStacker("observationStartMJD")
        day_obs_int = day_obs_int_stacker.run(data)
        assert np.all(day_obs_int["dayObs"] > 20200000)
        assert np.all(day_obs_int["dayObs"] < 20500000)
        if summit_get_day_obs is not None:
            summit_day_obs_int = np.array([summit_get_day_obs(Time(m, format="mjd")) for m in obs_start_mjds])
            np.testing.assert_array_equal(day_obs_int["dayObs"], summit_day_obs_int)

        day_obs_mjd_stacker = stackers.DayObsMJDStacker("observationStartMJD")
        new_data = day_obs_mjd_stacker.run(data)
        # If there were no offset, all values would be between 0 and 1.
        # With the 0.5 day offset specified in SITCOMTN-32, this becomes
        # 0.5 and 1.5.
        assert np.all((obs_start_mjds - new_data["day_obs_mjd"]) <= 1.5)
        assert np.all((obs_start_mjds - new_data["day_obs_mjd"]) >= 0.5)

        day_obs_iso_stacker = stackers.DayObsISOStacker("observationStartMJD")
        _ = day_obs_iso_stacker.run(data)

    def test_overhead_stacker(self):
        num_visits = 10
        num_first_night_visits = 5

        start_mjd = 61000.2
        exptime = 30.0

        exposure_times = np.full(num_visits, exptime, dtype=float)

        rng = np.random.default_rng(seed=6563)
        overheads = 2.0 + rng.random(num_visits)

        visit_times = overheads + exposure_times
        observation_end_mjds = start_mjd + np.cumsum(visit_times) / (24 * 60 * 60)
        mjds = observation_end_mjds - visit_times / (24 * 60 * 60)
        mjds[num_first_night_visits:] += 1

        data = np.core.records.fromarrays(
            (mjds, exposure_times, visit_times),
            dtype=np.dtype(
                [
                    ("observationStartMJD", mjds.dtype),
                    ("visitExposureTime", exposure_times.dtype),
                    ("visitTime", visit_times.dtype),
                ]
            ),
        )

        overhead_stacker = stackers.OverheadStacker()
        new_data = overhead_stacker.run(data)
        measured_overhead = new_data["overhead"]

        # There is no visit before the first, so the first overhead should be
        # nan.
        self.assertTrue(np.isnan(measured_overhead[0]))

        # Test that the gap between nights is nan.
        self.assertTrue(np.isnan(measured_overhead[num_first_night_visits]))

        # Make sure there are no unexpected nans
        self.assertEqual(np.count_nonzero(np.isnan(measured_overhead)), 2)

        # Make sure all measured values are correct
        these_gaps = ~np.isnan(measured_overhead)
        self.assertTrue(np.allclose(measured_overhead[these_gaps], overheads[these_gaps]))

    def testHelp(self):
        # essentially, test that every stacker has cols_req and cols_added
        with open(os.devnull, "w") as new_target:
            sys.stdout = new_target
            stackers.BaseStacker.help(doc=True)


if __name__ == "__main__":
    unittest.main()
