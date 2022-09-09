import unittest
import numpy as np
import rubin_sim.utils as utils
from rubin_sim.utils import ObservationMetaData, Site, ModifiedJulianDate


class testDegrees(unittest.TestCase):
    """
    Test that all the pairs of methods that deal in degrees versus
    radians agree with each other.
    """

    def setUp(self):
        self.rng = np.random.RandomState(87334)
        self.raList = self.rng.random_sample(100) * 2.0 * np.pi
        self.decList = (self.rng.random_sample(100) - 0.5) * np.pi
        self.lon = self.rng.random_sample(1)[0] * 360.0
        self.lat = (self.rng.random_sample(1)[0] - 0.5) * 180.0

    def testUnitConversion(self):
        """
        Test that arcsecFromRadians, arcsecFromDegrees,
        radiansFromArcsec, and degreesFromArcsec are all
        self-consistent
        """

        radList = self.rng.random_sample(100) * 2.0 * np.pi
        degList = np.degrees(radList)

        arcsecRadList = utils.arcsecFromRadians(radList)
        arcsecDegList = utils.arcsecFromDegrees(degList)

        np.testing.assert_array_equal(arcsecRadList, arcsecDegList)

        arcsecList = self.rng.random_sample(100) * 1.0
        radList = utils.radiansFromArcsec(arcsecList)
        degList = utils.degreesFromArcsec(arcsecList)
        np.testing.assert_array_equal(np.radians(degList), radList)

    def testGalacticFromEquatorial(self):
        raList = self.raList
        decList = self.decList

        lonRad, latRad = utils._galacticFromEquatorial(raList, decList)
        lonDeg, latDeg = utils.galacticFromEquatorial(
            np.degrees(raList), np.degrees(decList)
        )

        np.testing.assert_array_almost_equal(lonRad, np.radians(lonDeg), 10)
        np.testing.assert_array_almost_equal(latRad, np.radians(latDeg), 10)

        for ra, dec in zip(raList, decList):
            lonRad, latRad = utils._galacticFromEquatorial(ra, dec)
            lonDeg, latDeg = utils.galacticFromEquatorial(
                np.degrees(ra), np.degrees(dec)
            )
            self.assertAlmostEqual(lonRad, np.radians(lonDeg), 10)
            self.assertAlmostEqual(latRad, np.radians(latDeg), 10)

    def testEquaorialFromGalactic(self):
        lonList = self.raList
        latList = self.decList

        raRad, decRad = utils._equatorialFromGalactic(lonList, latList)
        raDeg, decDeg = utils.equatorialFromGalactic(
            np.degrees(lonList), np.degrees(latList)
        )

        np.testing.assert_array_almost_equal(raRad, np.radians(raDeg), 10)
        np.testing.assert_array_almost_equal(decRad, np.radians(decDeg), 10)

        for lon, lat in zip(lonList, latList):
            raRad, decRad = utils._equatorialFromGalactic(lon, lat)
            raDeg, decDeg = utils.equatorialFromGalactic(
                np.degrees(lon), np.degrees(lat)
            )
            self.assertAlmostEqual(raRad, np.radians(raDeg), 10)
            self.assertAlmostEqual(decRad, np.radians(decDeg), 10)

    def testAltAzPaFromRaDec(self):
        mjd = 57432.7
        obs = ObservationMetaData(
            mjd=mjd, site=Site(longitude=self.lon, latitude=self.lat, name="LSST")
        )

        altRad, azRad, paRad = utils._altAzPaFromRaDec(self.raList, self.decList, obs)

        altDeg, azDeg, paDeg = utils.altAzPaFromRaDec(
            np.degrees(self.raList), np.degrees(self.decList), obs
        )

        np.testing.assert_array_almost_equal(altRad, np.radians(altDeg), 10)
        np.testing.assert_array_almost_equal(azRad, np.radians(azDeg), 10)
        np.testing.assert_array_almost_equal(paRad, np.radians(paDeg), 10)

        altRad, azRad, paRad = utils._altAzPaFromRaDec(self.raList, self.decList, obs)

        altDeg, azDeg, paDeg = utils.altAzPaFromRaDec(
            np.degrees(self.raList), np.degrees(self.decList), obs
        )

        np.testing.assert_array_almost_equal(altRad, np.radians(altDeg), 10)
        np.testing.assert_array_almost_equal(azRad, np.radians(azDeg), 10)
        np.testing.assert_array_almost_equal(paRad, np.radians(paDeg), 10)

        for (
            ra,
            dec,
        ) in zip(self.raList, self.decList):
            altRad, azRad, paRad = utils._altAzPaFromRaDec(ra, dec, obs)
            altDeg, azDeg, paDeg = utils.altAzPaFromRaDec(
                np.degrees(ra), np.degrees(dec), obs
            )

            self.assertAlmostEqual(altRad, np.radians(altDeg), 10)
            self.assertAlmostEqual(azRad, np.radians(azDeg), 10)
            self.assertAlmostEqual(paRad, np.radians(paDeg), 10)

    def testRaDecFromAltAz(self):
        azList = self.raList
        altList = self.decList
        mjd = 47895.6
        obs = ObservationMetaData(
            mjd=mjd, site=Site(longitude=self.lon, latitude=self.lat, name="LSST")
        )

        raRad, decRad = utils._raDecFromAltAz(altList, azList, obs)

        raDeg, decDeg = utils.raDecFromAltAz(
            np.degrees(altList), np.degrees(azList), obs
        )

        np.testing.assert_array_almost_equal(raRad, np.radians(raDeg), 10)
        np.testing.assert_array_almost_equal(decRad, np.radians(decDeg), 10)

        raRad, decRad = utils._raDecFromAltAz(altList, azList, obs)

        raDeg, decDeg = utils.raDecFromAltAz(
            np.degrees(altList), np.degrees(azList), obs
        )

        np.testing.assert_array_almost_equal(raRad, np.radians(raDeg), 10)
        np.testing.assert_array_almost_equal(decRad, np.radians(decDeg), 10)

        for alt, az in zip(altList, azList):
            raRad, decRad = utils._raDecFromAltAz(alt, az, obs)
            raDeg, decDeg = utils.raDecFromAltAz(np.degrees(alt), np.degrees(az), obs)

            self.assertAlmostEqual(raRad, np.radians(raDeg), 10)
            self.assertAlmostEqual(decRad, np.radians(decDeg), 10)

    def testGetRotSkyPos(self):
        rotTelList = self.rng.random_sample(len(self.raList)) * 2.0 * np.pi
        mjd = 56321.8

        obsTemp = ObservationMetaData(
            mjd=mjd, site=Site(longitude=self.lon, latitude=self.lat, name="LSST")
        )

        rotSkyRad = utils._getRotSkyPos(self.raList, self.decList, obsTemp, rotTelList)

        rotSkyDeg = utils.getRotSkyPos(
            np.degrees(self.raList),
            np.degrees(self.decList),
            obsTemp,
            np.degrees(rotTelList),
        )

        np.testing.assert_array_almost_equal(rotSkyRad, np.radians(rotSkyDeg), 10)

        rotSkyRad = utils._getRotSkyPos(
            self.raList, self.decList, obsTemp, rotTelList[0]
        )

        rotSkyDeg = utils.getRotSkyPos(
            np.degrees(self.raList),
            np.degrees(self.decList),
            obsTemp,
            np.degrees(rotTelList[0]),
        )

        np.testing.assert_array_almost_equal(rotSkyRad, np.radians(rotSkyDeg), 10)

        for ra, dec, rotTel in zip(self.raList, self.decList, rotTelList):

            rotSkyRad = utils._getRotSkyPos(ra, dec, obsTemp, rotTel)

            rotSkyDeg = utils.getRotSkyPos(
                np.degrees(ra), np.degrees(dec), obsTemp, np.degrees(rotTel)
            )

            self.assertAlmostEqual(rotSkyRad, np.radians(rotSkyDeg), 10)

    def testGetRotTelPos(self):
        rotSkyList = self.rng.random_sample(len(self.raList)) * 2.0 * np.pi
        mjd = 56789.3
        obsTemp = ObservationMetaData(
            mjd=mjd, site=Site(longitude=self.lon, latitude=self.lat, name="LSST")
        )

        rotTelRad = utils._getRotTelPos(self.raList, self.decList, obsTemp, rotSkyList)

        rotTelDeg = utils.getRotTelPos(
            np.degrees(self.raList),
            np.degrees(self.decList),
            obsTemp,
            np.degrees(rotSkyList),
        )

        np.testing.assert_array_almost_equal(rotTelRad, np.radians(rotTelDeg), 10)

        rotTelRad = utils._getRotTelPos(
            self.raList, self.decList, obsTemp, rotSkyList[0]
        )

        rotTelDeg = utils.getRotTelPos(
            np.degrees(self.raList),
            np.degrees(self.decList),
            obsTemp,
            np.degrees(rotSkyList[0]),
        )

        np.testing.assert_array_almost_equal(rotTelRad, np.radians(rotTelDeg), 10)

        for ra, dec, rotSky in zip(self.raList, self.decList, rotSkyList):

            obsTemp = ObservationMetaData(
                mjd=mjd, site=Site(longitude=self.lon, latitude=self.lat, name="LSST")
            )

            rotTelRad = utils._getRotTelPos(ra, dec, obsTemp, rotSky)

            rotTelDeg = utils.getRotTelPos(
                np.degrees(ra), np.degrees(dec), obsTemp, np.degrees(rotSky)
            )

            self.assertAlmostEqual(rotTelRad, np.radians(rotTelDeg), 10)


class AstrometryDegreesTest(unittest.TestCase):
    def setUp(self):
        self.nStars = 10
        self.rng = np.random.RandomState(8273)
        self.raList = self.rng.random_sample(self.nStars) * 2.0 * np.pi
        self.decList = (self.rng.random_sample(self.nStars) - 0.5) * np.pi
        self.mjdList = self.rng.random_sample(10) * 5000.0 + 52000.0
        self.pm_raList = utils.radiansFromArcsec(
            self.rng.random_sample(self.nStars) * 10.0 - 5.0
        )
        self.pm_decList = utils.radiansFromArcsec(
            self.rng.random_sample(self.nStars) * 10.0 - 5.0
        )
        self.pxList = utils.radiansFromArcsec(self.rng.random_sample(self.nStars) * 2.0)
        self.v_radList = self.rng.random_sample(self.nStars) * 500.0 - 250.0

    def testApplyPrecession(self):
        for mjd in self.mjdList:
            raRad, decRad = utils._applyPrecession(
                self.raList, self.decList, mjd=ModifiedJulianDate(TAI=mjd)
            )

            raDeg, decDeg = utils.applyPrecession(
                np.degrees(self.raList),
                np.degrees(self.decList),
                mjd=ModifiedJulianDate(TAI=mjd),
            )

            dRa = utils.arcsecFromRadians(raRad - np.radians(raDeg))
            np.testing.assert_array_almost_equal(dRa, np.zeros(self.nStars), 9)

            dDec = utils.arcsecFromRadians(raRad - np.radians(raDeg))
            np.testing.assert_array_almost_equal(dDec, np.zeros(self.nStars), 9)

    def testApplyProperMotion(self):
        for mjd in self.mjdList:
            raRad, decRad = utils._applyProperMotion(
                self.raList,
                self.decList,
                self.pm_raList,
                self.pm_decList,
                self.pxList,
                self.v_radList,
                mjd=ModifiedJulianDate(TAI=mjd),
            )

            raDeg, decDeg = utils.applyProperMotion(
                np.degrees(self.raList),
                np.degrees(self.decList),
                utils.arcsecFromRadians(self.pm_raList),
                utils.arcsecFromRadians(self.pm_decList),
                utils.arcsecFromRadians(self.pxList),
                self.v_radList,
                mjd=ModifiedJulianDate(TAI=mjd),
            )

            dRa = utils.arcsecFromRadians(raRad - np.radians(raDeg))
            np.testing.assert_array_almost_equal(dRa, np.zeros(self.nStars), 9)

            dDec = utils.arcsecFromRadians(raRad - np.radians(raDeg))
            np.testing.assert_array_almost_equal(dDec, np.zeros(self.nStars), 9)

        for ra, dec, pm_ra, pm_dec, px, v_rad in zip(
            self.raList,
            self.decList,
            self.pm_raList,
            self.pm_decList,
            self.pxList,
            self.v_radList,
        ):

            raRad, decRad = utils._applyProperMotion(
                ra,
                dec,
                pm_ra,
                pm_dec,
                px,
                v_rad,
                mjd=ModifiedJulianDate(TAI=self.mjdList[0]),
            )

            raDeg, decDeg = utils.applyProperMotion(
                np.degrees(ra),
                np.degrees(dec),
                utils.arcsecFromRadians(pm_ra),
                utils.arcsecFromRadians(pm_dec),
                utils.arcsecFromRadians(px),
                v_rad,
                mjd=ModifiedJulianDate(TAI=self.mjdList[0]),
            )

            self.assertAlmostEqual(
                utils.arcsecFromRadians(raRad - np.radians(raDeg)), 0.0, 9
            )
            self.assertAlmostEqual(
                utils.arcsecFromRadians(decRad - np.radians(decDeg)), 0.0, 9
            )

    def testAppGeoFromICRS(self):
        mjd = 42350.0
        for pmRaList in [self.pm_raList, None]:
            for pmDecList in [self.pm_decList, None]:
                for pxList in [self.pxList, None]:
                    for vRadList in [self.v_radList, None]:
                        raRad, decRad = utils._appGeoFromICRS(
                            self.raList,
                            self.decList,
                            pmRaList,
                            pmDecList,
                            pxList,
                            vRadList,
                            mjd=ModifiedJulianDate(TAI=mjd),
                        )

                        raDeg, decDeg = utils.appGeoFromICRS(
                            np.degrees(self.raList),
                            np.degrees(self.decList),
                            utils.arcsecFromRadians(pmRaList),
                            utils.arcsecFromRadians(pmDecList),
                            utils.arcsecFromRadians(pxList),
                            vRadList,
                            mjd=ModifiedJulianDate(TAI=mjd),
                        )

                        dRa = utils.arcsecFromRadians(raRad - np.radians(raDeg))
                        np.testing.assert_array_almost_equal(
                            dRa, np.zeros(self.nStars), 9
                        )

                        dDec = utils.arcsecFromRadians(raRad - np.radians(raDeg))
                        np.testing.assert_array_almost_equal(
                            dDec, np.zeros(self.nStars), 9
                        )

    def testObservedFromAppGeo(self):
        obs = ObservationMetaData(pointingRA=35.0, pointingDec=-45.0, mjd=43572.0)

        for includeRefraction in [True, False]:
            raRad, decRad = utils._observedFromAppGeo(
                self.raList,
                self.decList,
                includeRefraction=includeRefraction,
                altAzHr=False,
                obs_metadata=obs,
            )

            raDeg, decDeg = utils.observedFromAppGeo(
                np.degrees(self.raList),
                np.degrees(self.decList),
                includeRefraction=includeRefraction,
                altAzHr=False,
                obs_metadata=obs,
            )

            dRa = utils.arcsecFromRadians(raRad - np.radians(raDeg))
            np.testing.assert_array_almost_equal(dRa, np.zeros(self.nStars), 9)

            dDec = utils.arcsecFromRadians(raRad - np.radians(raDeg))
            np.testing.assert_array_almost_equal(dDec, np.zeros(self.nStars), 9)

            raDec, altAz = utils._observedFromAppGeo(
                self.raList,
                self.decList,
                includeRefraction=includeRefraction,
                altAzHr=True,
                obs_metadata=obs,
            )

            raRad = raDec[0]
            altRad = altAz[0]
            azRad = altAz[1]

            raDec, altAz = utils.observedFromAppGeo(
                np.degrees(self.raList),
                np.degrees(self.decList),
                includeRefraction=includeRefraction,
                altAzHr=True,
                obs_metadata=obs,
            )

            raDeg = raDec[0]
            altDeg = altAz[0]
            azDeg = altAz[1]

            dRa = utils.arcsecFromRadians(raRad - np.radians(raDeg))
            np.testing.assert_array_almost_equal(dRa, np.zeros(self.nStars), 9)

            dDec = utils.arcsecFromRadians(raRad - np.radians(raDeg))
            np.testing.assert_array_almost_equal(dDec, np.zeros(self.nStars), 9)

            dAz = utils.arcsecFromRadians(azRad - np.radians(azDeg))
            np.testing.assert_array_almost_equal(dAz, np.zeros(self.nStars), 9)

            dAlt = utils.arcsecFromRadians(altRad - np.radians(altDeg))
            np.testing.assert_array_almost_equal(dAlt, np.zeros(self.nStars), 9)

    def testAppGeoFromObserved(self):
        obs = ObservationMetaData(pointingRA=35.0, pointingDec=-45.0, mjd=43572.0)

        for includeRefraction in (True, False):
            for wavelength in (0.5, 0.2, 0.3):

                raRad, decRad = utils._appGeoFromObserved(
                    self.raList,
                    self.decList,
                    includeRefraction=includeRefraction,
                    wavelength=wavelength,
                    obs_metadata=obs,
                )

                raDeg, decDeg = utils.appGeoFromObserved(
                    np.degrees(self.raList),
                    np.degrees(self.decList),
                    includeRefraction=includeRefraction,
                    wavelength=wavelength,
                    obs_metadata=obs,
                )

                dRa = utils.arcsecFromRadians(raRad - np.radians(raDeg))
                np.testing.assert_array_almost_equal(dRa, np.zeros(len(dRa)), 9)

                dDec = utils.arcsecFromRadians(decRad - np.radians(decDeg))
                np.testing.assert_array_almost_equal(dDec, np.zeros(len(dDec)), 9)

    def testIcrsFromAppGeo(self):

        for mjd in (53525.0, 54316.3, 58463.7):
            for epoch in (2000.0, 1950.0, 2010.0):

                raRad, decRad = utils._icrsFromAppGeo(
                    self.raList,
                    self.decList,
                    epoch=epoch,
                    mjd=ModifiedJulianDate(TAI=mjd),
                )

                raDeg, decDeg = utils.icrsFromAppGeo(
                    np.degrees(self.raList),
                    np.degrees(self.decList),
                    epoch=epoch,
                    mjd=ModifiedJulianDate(TAI=mjd),
                )

                dRa = utils.arcsecFromRadians(np.abs(raRad - np.radians(raDeg)))
                self.assertLess(dRa.max(), 1.0e-9)

                dDec = utils.arcsecFromRadians(np.abs(decRad - np.radians(decDeg)))
                self.assertLess(dDec.max(), 1.0e-9)

    def testObservedFromICRS(self):
        obs = ObservationMetaData(pointingRA=35.0, pointingDec=-45.0, mjd=43572.0)
        for pmRaList in [self.pm_raList, None]:
            for pmDecList in [self.pm_decList, None]:
                for pxList in [self.pxList, None]:
                    for vRadList in [self.v_radList, None]:
                        for includeRefraction in [True, False]:

                            raRad, decRad = utils._observedFromICRS(
                                self.raList,
                                self.decList,
                                pm_ra=pmRaList,
                                pm_dec=pmDecList,
                                parallax=pxList,
                                v_rad=vRadList,
                                obs_metadata=obs,
                                epoch=2000.0,
                                includeRefraction=includeRefraction,
                            )

                            raDeg, decDeg = utils.observedFromICRS(
                                np.degrees(self.raList),
                                np.degrees(self.decList),
                                pm_ra=utils.arcsecFromRadians(pmRaList),
                                pm_dec=utils.arcsecFromRadians(pmDecList),
                                parallax=utils.arcsecFromRadians(pxList),
                                v_rad=vRadList,
                                obs_metadata=obs,
                                epoch=2000.0,
                                includeRefraction=includeRefraction,
                            )

                            dRa = utils.arcsecFromRadians(raRad - np.radians(raDeg))
                            np.testing.assert_array_almost_equal(
                                dRa, np.zeros(self.nStars), 9
                            )

                            dDec = utils.arcsecFromRadians(decRad - np.radians(decDeg))
                            np.testing.assert_array_almost_equal(
                                dDec, np.zeros(self.nStars), 9
                            )

    def testIcrsFromObserved(self):
        obs = ObservationMetaData(pointingRA=35.0, pointingDec=-45.0, mjd=43572.0)

        for includeRefraction in [True, False]:

            raRad, decRad = utils._icrsFromObserved(
                self.raList,
                self.decList,
                obs_metadata=obs,
                epoch=2000.0,
                includeRefraction=includeRefraction,
            )

            raDeg, decDeg = utils.icrsFromObserved(
                np.degrees(self.raList),
                np.degrees(self.decList),
                obs_metadata=obs,
                epoch=2000.0,
                includeRefraction=includeRefraction,
            )

            dRa = utils.arcsecFromRadians(raRad - np.radians(raDeg))
            np.testing.assert_array_almost_equal(dRa, np.zeros(self.nStars), 9)

            dDec = utils.arcsecFromRadians(decRad - np.radians(decDeg))
            np.testing.assert_array_almost_equal(dDec, np.zeros(self.nStars), 9)

    def testraDecFromPupilCoords(self):
        obs = ObservationMetaData(
            pointingRA=23.5, pointingDec=-115.0, mjd=42351.0, rotSkyPos=127.0
        )

        xpList = self.rng.random_sample(100) * 0.25 * np.pi
        ypList = self.rng.random_sample(100) * 0.25 * np.pi

        raRad, decRad = utils._raDecFromPupilCoords(
            xpList, ypList, obs_metadata=obs, epoch=2000.0
        )
        raDeg, decDeg = utils.raDecFromPupilCoords(
            xpList, ypList, obs_metadata=obs, epoch=2000.0
        )

        dRa = utils.arcsecFromRadians(raRad - np.radians(raDeg))
        np.testing.assert_array_almost_equal(dRa, np.zeros(len(xpList)), 9)

        dDec = utils.arcsecFromRadians(decRad - np.radians(decDeg))
        np.testing.assert_array_almost_equal(dDec, np.zeros(len(xpList)), 9)

    def testpupilCoordsFromRaDec(self):
        obs = ObservationMetaData(
            pointingRA=23.5, pointingDec=-115.0, mjd=42351.0, rotSkyPos=127.0
        )

        # need to make sure the test points are tightly distributed around the bore site, or
        # PALPY will throw an error
        raList = self.rng.random_sample(self.nStars) * np.radians(1.0) + np.radians(
            23.5
        )
        decList = self.rng.random_sample(self.nStars) * np.radians(1.0) + np.radians(
            -115.0
        )

        xpControl, ypControl = utils._pupilCoordsFromRaDec(
            raList, decList, obs_metadata=obs, epoch=2000.0
        )

        xpTest, ypTest = utils.pupilCoordsFromRaDec(
            np.degrees(raList), np.degrees(decList), obs_metadata=obs, epoch=2000.0
        )

        dx = utils.arcsecFromRadians(xpControl - xpTest)
        np.testing.assert_array_almost_equal(dx, np.zeros(self.nStars), 9)

        dy = utils.arcsecFromRadians(ypControl - ypTest)
        np.testing.assert_array_almost_equal(dy, np.zeros(self.nStars), 9)


if __name__ == "__main__":
    unittest.main()
