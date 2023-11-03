import os
import unittest
import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from rubin_scheduler.data import get_data_dir
from rubin_scheduler.utils import Site, _alt_az_pa_from_ra_dec, calc_lmst

import rubin_sim.maf.stackers as stackers
from rubin_sim.maf import get_sim_data


class TestStackerClasses(unittest.TestCase):
    def setUp(self):
        # get some of the test data
        test_db = os.path.join(get_data_dir(), "tests", "example_dbv1.7_0yrs.db")
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

    @unittest.skip("Dither Stackers deprecated")
    def test_setup_dither_stackers(self):
        # Test that we get no stacker when using default columns.
        ra_col = "fieldRA"
        dec_col = "fieldDec"
        degrees = True
        stackerlist = stackers.setup_dither_stackers(ra_col, dec_col, degrees)
        self.assertEqual(len(stackerlist), 0)
        # Test that we get one (and the right one)
        # when using particular columns.
        ra_col = "hexDitherFieldPerNightRa"
        dec_col = "hexDitherFieldPerNightDec"
        stackerlist = stackers.setup_dither_stackers(ra_col, dec_col, degrees)
        self.assertEqual(len(stackerlist), 1)
        self.assertEqual(stackerlist[0], stackers.HexDitherFieldPerNightStacker())
        # Test that kwargs are passed along.
        stackerlist = stackers.setup_dither_stackers(ra_col, dec_col, degrees, max_dither=0.5)
        self.assertEqual(stackerlist[0].max_dither, np.radians(0.5))

    @unittest.skip("Dither Stackers deprecated")
    def test_base_dither_stacker(self):
        # Test that the base dither stacker matches the type of a stacker.
        s = stackers.HexDitherFieldPerNightStacker()
        self.assertTrue(isinstance(s, stackers.BaseDitherStacker))
        s = stackers.ParallaxFactorStacker()
        self.assertFalse(isinstance(s, stackers.BaseDitherStacker))

    @unittest.skip("Dither Stackers deprecated")
    def test_random_dither(self):
        """
        Test the random dither pattern.
        """
        max_dither = 0.5
        data = np.zeros(600, dtype=list(zip(["fieldRA", "fieldDec"], [float, float])))
        # Set seed so the test is stable
        rng = np.random.RandomState(42)
        # Restrict dithers to area where wraparound is not a problem for
        # comparisons.
        data["fieldRA"] = np.degrees(rng.random_sample(600) * (np.pi) + np.pi / 2.0)
        data["fieldDec"] = np.degrees(rng.random_sample(600) * np.pi / 2.0 - np.pi / 4.0)
        stacker = stackers.RandomDitherFieldPerVisitStacker(max_dither=max_dither)
        data = stacker.run(data)
        diffsra = (data["fieldRA"] - data["randomDitherFieldPerVisitRa"]) * np.cos(
            np.radians(data["fieldDec"])
        )
        diffsdec = data["fieldDec"] - data["randomDitherFieldPerVisitDec"]
        # Check dithers within expected range.
        self._t_dither_range(diffsra, diffsdec, data["fieldRA"], data["fieldDec"], max_dither)

    @unittest.skip("Dither Stackers deprecated")
    def test_random_dither_per_night(self):
        """
        Test the per-night random dither pattern.
        """
        max_dither = 0.5
        ndata = 600
        # Set seed so the test is stable
        rng = np.random.RandomState(42)

        data = np.zeros(
            ndata,
            dtype=list(
                zip(
                    ["fieldRA", "fieldDec", "fieldId", "night"],
                    [float, float, int, int],
                )
            ),
        )
        data["fieldRA"] = rng.rand(ndata) * (np.pi) + np.pi / 2.0
        data["fieldDec"] = rng.rand(ndata) * np.pi / 2.0 - np.pi / 4.0
        data["fieldId"] = np.floor(rng.rand(ndata) * ndata)
        data["night"] = np.floor(rng.rand(ndata) * 10).astype("int")
        stacker = stackers.RandomDitherPerNightStacker(max_dither=max_dither)
        data = stacker.run(data)
        diffsra = (np.radians(data["fieldRA"]) - np.radians(data["randomDitherPerNightRa"])) * np.cos(
            np.radians(data["fieldDec"])
        )
        diffsdec = np.radians(data["fieldDec"]) - np.radians(data["randomDitherPerNightDec"])
        self._t_dither_range(diffsra, diffsdec, data["fieldRA"], data["fieldDec"], max_dither)
        # Check that dithers on the same night are the same.
        self._t_dither_per_night(diffsra, diffsdec, data["fieldRA"], data["fieldDec"], data["night"])

    @unittest.skip("Dither Stackers deprecated")
    def test_spiral_dither_per_night(self):
        """
        Test the per-night spiral dither pattern.
        """
        max_dither = 0.5
        ndata = 2000
        # Set seed so the test is stable
        rng = np.random.RandomState(42)

        data = np.zeros(
            ndata,
            dtype=list(
                zip(
                    ["fieldRA", "fieldDec", "fieldId", "night"],
                    [float, float, int, int],
                )
            ),
        )
        data["fieldRA"] = rng.rand(ndata) * (np.pi) + np.pi / 2.0
        data["fieldRA"] = np.zeros(ndata) + np.pi / 2.0
        data["fieldDec"] = rng.rand(ndata) * np.pi / 2.0 - np.pi / 4.0
        data["fieldDec"] = np.zeros(ndata)
        data["fieldId"] = np.floor(rng.rand(ndata) * ndata)
        data["night"] = np.floor(rng.rand(ndata) * 20).astype("int")
        stacker = stackers.SpiralDitherPerNightStacker(max_dither=max_dither)
        data = stacker.run(data)
        diffsra = (data["fieldRA"] - data["spiralDitherPerNightRa"]) * np.cos(np.radians(data["fieldDec"]))
        diffsdec = data["fieldDec"] - data["spiralDitherPerNightDec"]
        self._t_dither_range(diffsra, diffsdec, data["fieldRA"], data["fieldDec"], max_dither)
        # Check that dithers on the same night are the same.
        self._t_dither_per_night(diffsra, diffsdec, data["fieldRA"], data["fieldDec"], data["night"])

    @unittest.skip("Dither Stackers deprecated")
    def test_hex_dither_per_night(self):
        """
        Test the per-night hex dither pattern.
        """
        max_dither = 0.5
        ndata = 2000
        # Set seed so the test is stable
        rng = np.random.RandomState(42)

        data = np.zeros(
            ndata,
            dtype=list(
                zip(
                    ["fieldRA", "fieldDec", "fieldId", "night"],
                    [float, float, int, int],
                )
            ),
        )
        data["fieldRA"] = rng.rand(ndata) * (np.pi) + np.pi / 2.0
        data["fieldDec"] = rng.rand(ndata) * np.pi / 2.0 - np.pi / 4.0
        data["fieldId"] = np.floor(rng.rand(ndata) * ndata)
        data["night"] = np.floor(rng.rand(ndata) * 217).astype("int")
        stacker = stackers.HexDitherPerNightStacker(max_dither=max_dither)
        data = stacker.run(data)
        diffsra = (data["fieldRA"] - data["hexDitherPerNightRa"]) * np.cos(np.radians(data["fieldDec"]))
        diffsdec = data["fieldDec"] - data["hexDitherPerNightDec"]
        self._t_dither_range(diffsra, diffsdec, data["fieldRA"], data["fieldDec"], max_dither)
        # Check that dithers on the same night are the same.
        self._t_dither_per_night(diffsra, diffsdec, data["fieldRA"], data["fieldDec"], data["night"])

    @unittest.skip("Dither Stackers deprecated")
    def test_random_rot_dither_per_filter_change_stacker(self):
        """
        Test the rotational dither stacker.
        """
        max_dither = 90
        filt = np.array(["r", "r", "r", "g", "g", "g", "r", "r"])
        rot_tel_pos = np.array([0, 0, 0, 0, 0, 0, 0, 0], float)
        # Test that have a dither in rot offset for every filter change.
        odata = np.zeros(len(filt), dtype=list(zip(["filter", "rotTelPos"], [(np.str_, 1), float])))
        odata["filter"] = filt
        odata["rotTelPos"] = rot_tel_pos
        stacker = stackers.RandomRotDitherPerFilterChangeStacker(
            max_dither=max_dither, degrees=True, random_seed=99
        )
        data = stacker.run(odata)  # run the stacker
        random_dithers = data["randomDitherPerFilterChangeRotTelPos"]
        # Check that first three visits have the same rot_tel_pos, etc.
        rot_offsets = rot_tel_pos - random_dithers
        # no dither w/o a filter change
        self.assertEqual(rot_offsets[0], 0)
        offset_changes = np.where(rot_offsets[1:] != rot_offsets[:-1])[0]
        filt_changes = np.where(filt[1:] != filt[:-1])[0]
        # dither after every filter
        np.testing.assert_array_equal(offset_changes, filt_changes)

        # now test to ensure that user-defined max_rot_angle value works
        # and that visits in between filter changes for which no offset
        # can be found are left undithered
        # (g band visits span rotator range, so can't be dithered)
        gvisits = np.where(filt == "g")
        maxrot = 30
        rot_tel_pos[gvisits[0][0]] = -maxrot
        rot_tel_pos[gvisits[0][-1]] = maxrot
        odata["rotTelPos"] = rot_tel_pos
        stacker = stackers.RandomRotDitherPerFilterChangeStacker(
            max_dither=max_dither,
            degrees=True,
            min_rot_angle=-maxrot,
            max_rot_angle=maxrot,
            random_seed=19231,
        )
        data = stacker.run(odata)
        random_dithers = data["randomDitherPerFilterChangeRotTelPos"]
        # Check that we respected the range.
        self.assertEqual(random_dithers.max(), 30)
        self.assertEqual(random_dithers.min(), -30)
        # Check that g band visits weren't dithered.
        rot_offsets = rot_tel_pos - random_dithers
        self.assertEqual(rot_offsets[gvisits].all(), 0)

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


if __name__ == "__main__":
    unittest.main()
