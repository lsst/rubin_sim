import os
import unittest

import numpy as np
from rubin_scheduler.data import get_data_dir

import rubin_sim.skybrightness as sb
from rubin_sim.phot_utils import Bandpass


class TestSkyModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # initalize the class with empty models
        cls._sm_mags = None
        cls._sm_mags2 = None
        cls._sm_spec = None
        cls._sm_spec2 = None

    @classmethod
    def tearDownClass(cls):
        del cls._sm_mags
        del cls._sm_mags2
        del cls._sm_spec
        del cls._sm_spec2

    @property
    def sm_mags(self):
        cls = type(self)
        if cls._sm_mags is None:
            cls._sm_mags = sb.SkyModel(mags=True)
        return cls._sm_mags

    @property
    def sm_mags2(self):
        cls = type(self)
        if cls._sm_mags2 is None:
            cls._sm_mags2 = sb.SkyModel(mags=True)
        return cls._sm_mags2

    @property
    def sm_spec(self):
        cls = type(self)
        if cls._sm_spec is None:
            cls._sm_spec = sb.SkyModel(mags=False)
        return cls._sm_spec

    @property
    def sm_spec2(self):
        cls = type(self)
        if cls._sm_spec2 is None:
            cls._sm_spec2 = sb.SkyModel(mags=False)
        return cls._sm_spec2

    def testmerged_comp(self):
        """
        Test that the 3 components that have been merged return the
        same result if they are computed independently
        """

        sky1 = sb.SkyModel(
            twilight=False,
            zodiacal=False,
            moon=False,
            lower_atm=False,
            upper_atm=False,
            airglow=False,
            scattered_star=False,
            merged_spec=True,
        )
        sky1.set_ra_dec_mjd([36.0], [-68.0], 49353.18, degrees=True)

        sky2 = sb.SkyModel(
            twilight=False,
            zodiacal=False,
            moon=False,
            lower_atm=True,
            upper_atm=True,
            airglow=False,
            scattered_star=True,
            merged_spec=False,
        )
        sky2.set_ra_dec_mjd([36.0], [-68.0], 49353.18, degrees=True)

        dummy, spec1 = sky1.return_wave_spec()
        dummy, spec2 = sky2.return_wave_spec()

        np.testing.assert_almost_equal(spec1, spec2)

        # and then check for the mags
        sky1 = sb.SkyModel(
            twilight=False,
            zodiacal=False,
            moon=False,
            lower_atm=False,
            upper_atm=False,
            airglow=False,
            scattered_star=False,
            merged_spec=True,
            mags=True,
        )
        sky1.set_ra_dec_mjd([36.0], [-68.0], 49353.18, degrees=True)

        sky2 = sb.SkyModel(
            twilight=False,
            zodiacal=False,
            moon=False,
            lower_atm=True,
            upper_atm=True,
            airglow=False,
            scattered_star=True,
            merged_spec=False,
            mags=True,
        )
        sky2.set_ra_dec_mjd([36.0], [-68.0], 49353.18, degrees=True)

        m1 = sky1.return_mags()
        m2 = sky2.return_mags()
        for key in m1:
            np.testing.assert_almost_equal(m1[key], m2[key], decimal=2)

    def test_setups(self):
        """
        Check that things are the same if the model is set up with
        radecmjd or all the parameters independently
        """

        sm1 = self.sm_spec
        sm1.set_ra_dec_mjd([36.0], [-68.0], 49353.18, degrees=True)

        sm2 = self.sm_spec2
        sm2.set_params(
            azs=sm1.azs,
            alts=sm1.alts,
            moon_phase=sm1.moon_phase,
            moon_alt=sm1.moon_alt,
            moon_az=sm1.moon_az,
            sun_alt=sm1.sun_alt,
            sun_az=sm1.sun_az,
            sun_eclip_lon=sm1.sun_eclip_lon,
            eclip_lon=sm1.eclip_lon,
            eclip_lat=sm1.eclip_lat,
            solar_flux=sm1.solar_flux,
            degrees=False,
        )

        dummy, spec1 = sm1.return_wave_spec()
        dummy, spec2 = sm2.return_wave_spec()

        np.testing.assert_allclose(spec1, spec2, rtol=1e-13)

        # Check that the degrees kwarg works
        sm2.set_params(
            azs=np.degrees(sm1.azs),
            alts=np.degrees(sm1.alts),
            moon_phase=sm1.moon_phase,
            moon_alt=np.degrees(sm1.moon_alt),
            moon_az=np.degrees(sm1.moon_az),
            sun_alt=np.degrees(sm1.sun_alt),
            sun_az=np.degrees(sm1.sun_az),
            sun_eclip_lon=np.degrees(sm1.sun_eclip_lon),
            eclip_lon=np.degrees(sm1.eclip_lon),
            eclip_lat=np.degrees(sm1.eclip_lat),
            solar_flux=sm1.solar_flux,
            degrees=True,
        )

        at_list = [
            "azs",
            "alts",
            "moon_phase",
            "moon_alt",
            "moon_az",
            "sun_alt",
            "sun_az",
            "sun_eclip_lon",
            "eclip_lon",
            "eclip_lat",
            "solar_flux",
        ]

        # Check each attribute that should match
        for attr in at_list:
            np.testing.assert_allclose(getattr(sm1, attr), getattr(sm2, attr))

        # Check the interpolation points
        for name in sm1.points.dtype.names:
            np.testing.assert_allclose(sm1.points[name], sm2.points[name])

        # Check the final output spectra
        np.testing.assert_allclose(sm1.spec, sm2.spec)

    def test_mags(self):
        """
        Test that the interpolated mags are similar to mags computed
        from interpolated spectra
        """

        through_path = os.path.join(get_data_dir(), "throughputs", "baseline")
        filters = ["u", "g", "r", "i", "z", "y"]

        bps = {}
        for filter_name in filters:
            bp = np.loadtxt(
                os.path.join(through_path, "total_%s.dat" % filter_name),
                dtype=list(zip(["wave", "trans"], [float] * 2)),
            )
            lsst_bp = Bandpass()
            lsst_bp.set_bandpass(bp["wave"], bp["trans"])
            bps[filter_name] = lsst_bp

        sm1 = self.sm_spec
        sm1.set_ra_dec_mjd([36.0], [-68.0], 49353.18, degrees=True)
        mags1 = sm1.return_mags(bandpasses=bps)

        sm2 = self.sm_mags
        sm2.set_ra_dec_mjd([36.0], [-68.0], 49353.18, degrees=True)
        mag2 = sm2.return_mags()

        # Let's set the tolerance for matching the throughputs to be 0.001
        # This allows for small changes in the filter throughputs
        # without requiring recomputation of
        # sims_skybrighntess_pre, while still requiring a
        # reasonable match against the measured sky
        for i, filtername in enumerate(filters):
            np.testing.assert_allclose(mags1[filtername], mag2[filtername], rtol=5e-3)

    def test_get_computed(self):
        """
        Make sure we can recover computed values.
        """

        sm = self.sm_mags
        sm.set_ra_dec_mjd([36.0, 36.0], [-68.0, -70.0], 49353.18, degrees=True)
        val_dict = sm.get_computed_vals()

        attr_to_check = [
            "ra",
            "dec",
            "alts",
            "azs",
            "airmass",
            "solar_flux",
            "moon_phase",
            "moon_az",
            "moon_alt",
            "sun_alt",
            "sun_az",
            "az_rel_sun",
            "moon_sun_sep",
            "az_rel_moon",
            "eclip_lon",
            "eclip_lat",
            "moon_ra",
            "moon_dec",
            "sun_ra",
            "sun_dec",
            "sun_eclip_lon",
        ]

        for attr in attr_to_check:
            assert attr in val_dict
            if np.size(val_dict[attr]) > 1:
                np.testing.assert_array_equal(getattr(sm, attr), val_dict[attr])
            else:
                self.assertEqual(getattr(sm, attr), val_dict[attr])

        # Check that things that should be radians are in radian range
        rad_list = [
            "ra",
            "azs",
            "moon_az",
            "sun_az",
            "az_rel_sun",
            "az_rel_moon",
            "eclip_lon",
            "moon_ra",
            "sun_ra",
            "sun_eclip_lon",
        ]

        for attr in rad_list:
            if val_dict[attr] is not None:
                assert np.min(val_dict[attr]) >= 0
                assert np.max(val_dict[attr]) <= 2.0 * np.pi

        # Radians in negative to positive pi range
        rad_list = [
            "moon_alt",
            "sun_alt",
            "alts",
            "dec",
            "moon_dec",
            "sun_dec",
            "eclip_lat",
        ]
        for attr in rad_list:
            if val_dict[attr] is not None:
                assert np.min(val_dict[attr]) >= -np.pi
                assert np.max(val_dict[attr]) <= np.pi

    def test90_deg(self):
        """
        Make sure we can look all the way to 90 degree altitude.
        """
        mjd = 56973.268218
        sm = self.sm_mags
        sm.set_ra_dec_mjd(0.0, 90.0, mjd, degrees=True, az_alt=True)
        mags = sm.return_mags()
        for key in mags:
            assert True not in np.isnan(mags[key])
        assert True not in np.isnan(sm.spec)

    def test_fewer_mags(self):
        """
        Test that can call and only interpolate a few magnitudes.
        """
        mjd = 56973.268218
        sm = self.sm_mags
        sm.set_ra_dec_mjd(0.0, 90.0, mjd, degrees=True, az_alt=True)
        all_mags = sm.return_mags()

        filter_names = ["u", "g", "r", "i", "z", "y"]
        for filter_name in filter_names:
            sm.set_ra_dec_mjd(0.0, 90.0, mjd, degrees=True, az_alt=True, filter_names=[filter_name])
            one_mag = sm.return_mags()
            self.assertEqual(all_mags[filter_name], one_mag[filter_name])

        # Test that I can do subset of mags
        subset = ["u", "r", "y"]
        sm.set_ra_dec_mjd(0.0, 90.0, mjd, degrees=True, az_alt=True, filter_names=subset)
        sub_mags = sm.return_mags()
        for filter_name in subset:
            self.assertEqual(all_mags[filter_name], sub_mags[filter_name])

    def test_set_ra_dec_alt_az_mjd(self):
        """
        Make sure sending in self-computed alt, az works
        """
        sm1 = self.sm_mags
        sm2 = self.sm_mags2
        ra = np.array([0.0, 0.0, 0.0])
        dec = np.array([-0.1, -0.2, -0.3])
        mjd = 5900
        sm1.set_ra_dec_mjd(ra, dec, mjd)
        m1 = sm1.return_mags()
        sm2.set_ra_dec_alt_az_mjd(ra, dec, sm1.alts, sm1.azs, mjd)
        m2 = sm1.return_mags()

        attr_list = ["ra", "dec", "alts", "azs"]
        for attr in attr_list:
            np.testing.assert_equal(getattr(sm1, attr), getattr(sm2, attr))

        for key in m1:
            np.testing.assert_allclose(m1[key], m2[key], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
