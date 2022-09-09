import numpy as np
import rubin_sim.skybrightness as sb
import unittest
from rubin_sim.phot_utils import Bandpass
from rubin_sim.data import get_data_dir
import os


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

    def testmergedComp(self):
        """
        Test that the 3 components that have been merged return the
        same result if they are computed independently
        """

        sky1 = sb.SkyModel(
            twilight=False,
            zodiacal=False,
            moon=False,
            lowerAtm=False,
            upperAtm=False,
            airglow=False,
            scatteredStar=False,
            mergedSpec=True,
        )
        sky1.setRaDecMjd([36.0], [-68.0], 49353.18, degrees=True)

        sky2 = sb.SkyModel(
            twilight=False,
            zodiacal=False,
            moon=False,
            lowerAtm=True,
            upperAtm=True,
            airglow=False,
            scatteredStar=True,
            mergedSpec=False,
        )
        sky2.setRaDecMjd([36.0], [-68.0], 49353.18, degrees=True)

        dummy, spec1 = sky1.returnWaveSpec()
        dummy, spec2 = sky2.returnWaveSpec()

        np.testing.assert_almost_equal(spec1, spec2)

        # and then check for the mags
        sky1 = sb.SkyModel(
            twilight=False,
            zodiacal=False,
            moon=False,
            lowerAtm=False,
            upperAtm=False,
            airglow=False,
            scatteredStar=False,
            mergedSpec=True,
            mags=True,
        )
        sky1.setRaDecMjd([36.0], [-68.0], 49353.18, degrees=True)

        sky2 = sb.SkyModel(
            twilight=False,
            zodiacal=False,
            moon=False,
            lowerAtm=True,
            upperAtm=True,
            airglow=False,
            scatteredStar=True,
            mergedSpec=False,
            mags=True,
        )
        sky2.setRaDecMjd([36.0], [-68.0], 49353.18, degrees=True)

        m1 = sky1.returnMags()
        m2 = sky2.returnMags()
        for key in m1:
            np.testing.assert_almost_equal(m1[key], m2[key], decimal=2)

    def testSetups(self):
        """
        Check that things are the same if the model is set up with
        radecmjd or all the parameters independently
        """

        sm1 = self.sm_spec
        sm1.setRaDecMjd([36.0], [-68.0], 49353.18, degrees=True)

        sm2 = self.sm_spec2
        sm2.setParams(
            azs=sm1.azs,
            alts=sm1.alts,
            moonPhase=sm1.moonPhase,
            moonAlt=sm1.moonAlt,
            moonAz=sm1.moonAz,
            sunAlt=sm1.sunAlt,
            sunAz=sm1.sunAz,
            sunEclipLon=sm1.sunEclipLon,
            eclipLon=sm1.eclipLon,
            eclipLat=sm1.eclipLat,
            solarFlux=sm1.solarFlux,
            degrees=False,
        )

        dummy, spec1 = sm1.returnWaveSpec()
        dummy, spec2 = sm2.returnWaveSpec()

        np.testing.assert_array_equal(spec1, spec2)

        # Check that the degrees kwarg works
        sm2.setParams(
            azs=np.degrees(sm1.azs),
            alts=np.degrees(sm1.alts),
            moonPhase=sm1.moonPhase,
            moonAlt=np.degrees(sm1.moonAlt),
            moonAz=np.degrees(sm1.moonAz),
            sunAlt=np.degrees(sm1.sunAlt),
            sunAz=np.degrees(sm1.sunAz),
            sunEclipLon=np.degrees(sm1.sunEclipLon),
            eclipLon=np.degrees(sm1.eclipLon),
            eclipLat=np.degrees(sm1.eclipLat),
            solarFlux=sm1.solarFlux,
            degrees=True,
        )

        atList = [
            "azs",
            "alts",
            "moonPhase",
            "moonAlt",
            "moonAz",
            "sunAlt",
            "sunAz",
            "sunEclipLon",
            "eclipLon",
            "eclipLat",
            "solarFlux",
        ]

        # Check each attribute that should match
        for attr in atList:
            np.testing.assert_allclose(getattr(sm1, attr), getattr(sm2, attr))

        # Check the interpolation points
        for name in sm1.points.dtype.names:
            np.testing.assert_allclose(sm1.points[name], sm2.points[name])

        # Check the final output spectra
        np.testing.assert_allclose(sm1.spec, sm2.spec)

    def testMags(self):
        """
        Test that the interpolated mags are similar to mags computed from interpolated spectra
        """

        throughPath = os.path.join(get_data_dir(), "throughputs", "baseline")
        filters = ["u", "g", "r", "i", "z", "y"]

        bps = {}
        for filterName in filters:
            bp = np.loadtxt(
                os.path.join(throughPath, "total_%s.dat" % filterName),
                dtype=list(zip(["wave", "trans"], [float] * 2)),
            )
            lsst_bp = Bandpass()
            lsst_bp.setBandpass(bp["wave"], bp["trans"])
            bps[filterName] = lsst_bp

        sm1 = self.sm_spec
        sm1.setRaDecMjd([36.0], [-68.0], 49353.18, degrees=True)
        mags1 = sm1.returnMags(bandpasses=bps)

        sm2 = self.sm_mags
        sm2.setRaDecMjd([36.0], [-68.0], 49353.18, degrees=True)
        mag2 = sm2.returnMags()

        # Let's set the tolerance for matching the throughputs to be 0.001
        # This allows for small changes in the filter throughputs without requiring recomputation of
        # sims_skybrighntess_pre, while still requiring a reasonable match against the measured sky
        for i, filtername in enumerate(filters):
            np.testing.assert_allclose(mags1[filtername], mag2[filtername], rtol=5e-3)

    def testGetComputed(self):
        """
        Make sure we can recover computed values.
        """

        sm = self.sm_mags
        sm.setRaDecMjd([36.0, 36.0], [-68.0, -70.0], 49353.18, degrees=True)
        valDict = sm.getComputedVals()

        attrToCheck = [
            "ra",
            "dec",
            "alts",
            "azs",
            "airmass",
            "solarFlux",
            "moonPhase",
            "moonAz",
            "moonAlt",
            "sunAlt",
            "sunAz",
            "azRelSun",
            "moonSunSep",
            "azRelMoon",
            "eclipLon",
            "eclipLat",
            "moonRA",
            "moonDec",
            "sunRA",
            "sunDec",
            "sunEclipLon",
        ]

        for attr in attrToCheck:
            assert attr in valDict
            if np.size(valDict[attr]) > 1:
                np.testing.assert_array_equal(getattr(sm, attr), valDict[attr])
            else:
                self.assertEqual(getattr(sm, attr), valDict[attr])

        # Check that things that should be radians are in radian range
        radList = [
            "ra",
            "azs",
            "moonAz",
            "sunAz",
            "azRelSun",
            "azRelMoon",
            "eclipLon",
            "moonRA",
            "sunRA",
            "sunEclipLon",
        ]

        for attr in radList:
            if valDict[attr] is not None:
                assert np.min(valDict[attr]) >= 0
                assert np.max(valDict[attr]) <= 2.0 * np.pi

        # Radians in negative to positive pi range
        radList = ["moonAlt", "sunAlt", "alts", "dec", "moonDec", "sunDec", "eclipLat"]
        for attr in radList:
            if valDict[attr] is not None:
                assert np.min(valDict[attr]) >= -np.pi
                assert np.max(valDict[attr]) <= np.pi

    def test90Deg(self):
        """
        Make sure we can look all the way to 90 degree altitude.
        """
        mjd = 56973.268218
        sm = self.sm_mags
        sm.setRaDecMjd(0.0, 90.0, mjd, degrees=True, azAlt=True)
        mags = sm.returnMags()
        for key in mags:
            assert True not in np.isnan(mags[key])
        assert True not in np.isnan(sm.spec)

    def testFewerMags(self):
        """
        Test that can call and only interpolate a few magnitudes.
        """
        mjd = 56973.268218
        sm = self.sm_mags
        sm.setRaDecMjd(0.0, 90.0, mjd, degrees=True, azAlt=True)
        all_mags = sm.returnMags()

        filterNames = ["u", "g", "r", "i", "z", "y"]
        for filterName in filterNames:
            sm.setRaDecMjd(
                0.0, 90.0, mjd, degrees=True, azAlt=True, filterNames=[filterName]
            )
            one_mag = sm.returnMags()
            self.assertEqual(all_mags[filterName], one_mag[filterName])

        # Test that I can do subset of mags
        subset = ["u", "r", "y"]
        sm.setRaDecMjd(0.0, 90.0, mjd, degrees=True, azAlt=True, filterNames=subset)
        sub_mags = sm.returnMags()
        for filterName in subset:
            self.assertEqual(all_mags[filterName], sub_mags[filterName])

    def test_setRaDecAltAzMjd(self):
        """
        Make sure sending in self-computed alt, az works
        """
        sm1 = self.sm_mags
        sm2 = self.sm_mags2
        ra = np.array([0.0, 0.0, 0.0])
        dec = np.array([-0.1, -0.2, -0.3])
        mjd = 5900
        sm1.setRaDecMjd(ra, dec, mjd)
        m1 = sm1.returnMags()
        sm2.setRaDecAltAzMjd(ra, dec, sm1.alts, sm1.azs, mjd)
        m2 = sm1.returnMags()

        attrList = ["ra", "dec", "alts", "azs"]
        for attr in attrList:
            np.testing.assert_equal(getattr(sm1, attr), getattr(sm2, attr))

        for key in m1:
            np.testing.assert_allclose(m1[key], m2[key], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
