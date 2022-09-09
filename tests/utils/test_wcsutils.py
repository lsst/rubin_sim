import unittest
import numpy as np

from rubin_sim.utils import raDecFromNativeLonLat, nativeLonLatFromRaDec
from rubin_sim.utils import _raDecFromNativeLonLat, _nativeLonLatFromRaDec
from rubin_sim.utils import observedFromICRS, icrsFromObserved
from rubin_sim.utils import ObservationMetaData, haversine
from rubin_sim.utils import arcsecFromRadians, raDecFromAltAz, Site


class NativeLonLatTest(unittest.TestCase):
    def testNativeLonLat(self):
        """
        Test that nativeLonLatFromRaDec works by considering stars and pointings
        at intuitive locations
        """

        mjd = 53855.0

        raList_obs = [0.0, 0.0, 0.0, 270.0]
        decList_obs = [90.0, 90.0, 0.0, 0.0]

        raPointList_obs = [0.0, 270.0, 270.0, 0.0]
        decPointList_obs = [0.0, 0.0, 0.0, 0.0]

        lonControlList = [180.0, 180.0, 90.0, 270.0]
        latControlList = [0.0, 0.0, 0.0, 0.0]

        for rr_obs, dd_obs, rp_obs, dp_obs, lonc, latc in zip(
            raList_obs,
            decList_obs,
            raPointList_obs,
            decPointList_obs,
            lonControlList,
            latControlList,
        ):

            obsTemp = ObservationMetaData(mjd=mjd)

            rr, dd = icrsFromObserved(
                np.array([rr_obs, rp_obs]),
                np.array([dd_obs, dp_obs]),
                obs_metadata=obsTemp,
                epoch=2000.0,
                includeRefraction=True,
            )

            obs = ObservationMetaData(pointingRA=rr[1], pointingDec=dd[1], mjd=mjd)
            lon, lat = nativeLonLatFromRaDec(rr[0], dd[0], obs)
            distance = arcsecFromRadians(haversine(lon, lat, lonc, latc))
            self.assertLess(distance, 1.0)

    def testNativeLongLatComplicated(self):
        """
        Test that nativeLongLatFromRaDec works by considering stars and pointings
        at non-intuitive locations.
        """

        rng = np.random.RandomState(42)
        nPointings = 10
        raPointingList_icrs = rng.random_sample(nPointings) * 360.0
        decPointingList_icrs = rng.random_sample(nPointings) * 180.0 - 90.0
        mjdList = rng.random_sample(nPointings) * 10000.0 + 43000.0

        nStars = 10
        for raPointing_icrs, decPointing_icrs, mjd in zip(
            raPointingList_icrs, decPointingList_icrs, mjdList
        ):

            obs = ObservationMetaData(
                pointingRA=raPointing_icrs, pointingDec=decPointing_icrs, mjd=mjd
            )
            raList_icrs = rng.random_sample(nStars) * 360.0
            decList_icrs = rng.random_sample(nStars) * 180.0 - 90.0
            raList_obs, decList_obs = observedFromICRS(
                raList_icrs,
                decList_icrs,
                obs_metadata=obs,
                epoch=2000.0,
                includeRefraction=True,
            )

            obsTemp = ObservationMetaData(mjd=mjd)
            raPointing_obs, decPointing_obs = observedFromICRS(
                raPointing_icrs,
                decPointing_icrs,
                obs_metadata=obsTemp,
                epoch=2000.0,
                includeRefraction=True,
            )

            for ra_obs, dec_obs, ra_icrs, dec_icrs in zip(
                raList_obs, decList_obs, raList_icrs, decList_icrs
            ):

                raRad = np.radians(ra_obs)
                decRad = np.radians(dec_obs)
                sinRa = np.sin(raRad)
                cosRa = np.cos(raRad)
                sinDec = np.sin(decRad)
                cosDec = np.cos(decRad)

                # the three dimensional position of the star
                controlPosition = np.array([-cosDec * sinRa, cosDec * cosRa, sinDec])

                # calculate the rotation matrices needed to transform the
                # x, y, and z axes into the local x, y, and z axes
                # (i.e. the axes with z lined up with raPointing_obs, decPointing_obs)
                alpha = 0.5 * np.pi - np.radians(decPointing_obs)
                ca = np.cos(alpha)
                sa = np.sin(alpha)
                rotX = np.array([[1.0, 0.0, 0.0], [0.0, ca, sa], [0.0, -sa, ca]])

                cb = np.cos(np.radians(raPointing_obs))
                sb = np.sin(np.radians(raPointing_obs))
                rotZ = np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])

                # rotate the coordinate axes into the local basis
                xAxis = np.dot(rotZ, np.dot(rotX, np.array([1.0, 0.0, 0.0])))
                yAxis = np.dot(rotZ, np.dot(rotX, np.array([0.0, 1.0, 0.0])))
                zAxis = np.dot(rotZ, np.dot(rotX, np.array([0.0, 0.0, 1.0])))

                # calculate the local longitude and latitude of the star
                lon, lat = nativeLonLatFromRaDec(ra_icrs, dec_icrs, obs)
                cosLon = np.cos(np.radians(lon))
                sinLon = np.sin(np.radians(lon))
                cosLat = np.cos(np.radians(lat))
                sinLat = np.sin(np.radians(lat))

                # the x, y, z position of the star in the local coordinate
                # basis
                transformedPosition = np.array(
                    [-cosLat * sinLon, cosLat * cosLon, sinLat]
                )

                # convert that position back into the un-rotated bases
                testPosition = (
                    transformedPosition[0] * xAxis
                    + transformedPosition[1] * yAxis
                    + transformedPosition[2] * zAxis
                )

                # assert that testPosition and controlPosition should be equal
                distance = np.sqrt(np.power(controlPosition - testPosition, 2).sum())
                self.assertLess(distance, 1.0e-12)

    def testNativeLonLatVector(self):
        """
        Test that nativeLonLatFromRaDec works in a vectorized way; we do this
        by performing a bunch of tansformations passing in ra and dec as numpy arrays
        and then comparing them to results computed in an element-wise way
        """

        obs = ObservationMetaData(pointingRA=123.0, pointingDec=43.0, mjd=53467.2)

        nSamples = 100
        rng = np.random.RandomState(42)
        raList = rng.random_sample(nSamples) * 360.0
        decList = rng.random_sample(nSamples) * 180.0 - 90.0

        lonList, latList = nativeLonLatFromRaDec(raList, decList, obs)

        for rr, dd, lon, lat in zip(raList, decList, lonList, latList):
            lonControl, latControl = nativeLonLatFromRaDec(rr, dd, obs)
            distance = arcsecFromRadians(
                haversine(
                    np.radians(lon),
                    np.radians(lat),
                    np.radians(lonControl),
                    np.radians(latControl),
                )
            )

            self.assertLess(distance, 0.0001)

    def testRaDec(self):
        """
        Test that raDecFromNativeLonLat does invert
        nativeLonLatFromRaDec
        """
        rng = np.random.RandomState(42)
        nSamples = 100
        # because raDecFromNativeLonLat is only good
        rrList = rng.random_sample(nSamples) * 50.0
        # out to a zenith distance of ~ 70 degrees

        thetaList = rng.random_sample(nSamples) * 2.0 * np.pi

        rrPointingList = rng.random_sample(10) * 50.0
        thetaPointingList = rng.random_sample(10) * 2.0 * np.pi
        mjdList = rng.random_sample(nSamples) * 10000.0 + 43000.0

        for rrp, thetap, mjd in zip(rrPointingList, thetaPointingList, mjdList):

            site = Site(name="LSST")
            raZenith, decZenith = raDecFromAltAz(
                180.0, 0.0, ObservationMetaData(mjd=mjd, site=site)
            )

            rp = raZenith + rrp * np.cos(thetap)
            dp = decZenith + rrp * np.sin(thetap)
            obs = ObservationMetaData(pointingRA=rp, pointingDec=dp, mjd=mjd, site=site)

            raList_icrs = (raZenith + rrList * np.cos(thetaList)) % 360.0
            decList_icrs = decZenith + rrList * np.sin(thetaList)

            raList_obs, decList_obs = observedFromICRS(
                raList_icrs,
                decList_icrs,
                obs_metadata=obs,
                epoch=2000.0,
                includeRefraction=True,
            )

            # calculate the distance between the ICRS position and the observed
            # geocentric position
            dd_icrs_obs_list = arcsecFromRadians(
                haversine(
                    np.radians(raList_icrs),
                    np.radians(decList_icrs),
                    np.radians(raList_obs),
                    np.radians(decList_obs),
                )
            )

            for rr, dd, dd_icrs_obs in zip(raList_icrs, decList_icrs, dd_icrs_obs_list):
                lon, lat = nativeLonLatFromRaDec(rr, dd, obs)
                r1, d1 = raDecFromNativeLonLat(lon, lat, obs)

                # the distance between the input RA, Dec and the round-trip output
                # RA, Dec
                distance = arcsecFromRadians(
                    haversine(
                        np.radians(r1), np.radians(d1), np.radians(rr), np.radians(dd)
                    )
                )

                rr_obs, dec_obs = observedFromICRS(
                    rr, dd, obs_metadata=obs, epoch=2000.0, includeRefraction=True
                )

                # verify that the round trip through nativeLonLat only changed
                # RA, Dec by less than an arcsecond
                self.assertLess(distance, 1.0)

                # verify that any difference in the round trip is much less
                # than the distance between the ICRS and the observed geocentric
                # RA, Dec
                self.assertLess(distance, dd_icrs_obs * 0.01)

    def testRaDecVector(self):
        """
        Test that raDecFromNativeLonLat does invert
        nativeLonLatFromRaDec (make sure it works in a vectorized way)
        """
        rng = np.random.RandomState(42)
        nSamples = 100
        latList = rng.random_sample(nSamples) * 360.0
        lonList = rng.random_sample(nSamples) * 180.0 - 90.0
        raPoint = 95.0
        decPoint = 75.0

        obs = ObservationMetaData(
            pointingRA=raPoint, pointingDec=decPoint, mjd=53467.89
        )

        raList, decList = raDecFromNativeLonLat(lonList, latList, obs)

        for lon, lat, ra0, dec0 in zip(lonList, latList, raList, decList):
            ra1, dec1 = raDecFromNativeLonLat(lon, lat, obs)
            distance = arcsecFromRadians(
                haversine(
                    np.radians(ra0), np.radians(dec0), np.radians(ra1), np.radians(dec1)
                )
            )
            self.assertLess(distance, 0.1)

    def testDegreesVersusRadians(self):
        """
        Test that the radian and degree versions of nativeLonLatFromRaDec
        and raDecFromNativeLonLat are consistent with each other
        """

        rng = np.random.RandomState(873)
        nSamples = 1000
        obs = ObservationMetaData(pointingRA=45.0, pointingDec=-34.5, mjd=54656.76)
        raList = rng.random_sample(nSamples) * 360.0
        decList = rng.random_sample(nSamples) * 180.0 - 90.0

        lonDeg, latDeg = nativeLonLatFromRaDec(raList, decList, obs)
        lonRad, latRad = _nativeLonLatFromRaDec(
            np.radians(raList), np.radians(decList), obs
        )
        np.testing.assert_array_almost_equal(np.radians(lonDeg), lonRad, 15)
        np.testing.assert_array_almost_equal(np.radians(latDeg), latRad, 15)

        raDeg, decDeg = raDecFromNativeLonLat(raList, decList, obs)
        raRad, decRad = _raDecFromNativeLonLat(
            np.radians(raList), np.radians(decList), obs
        )
        np.testing.assert_array_almost_equal(np.radians(raDeg), raRad, 15)
        np.testing.assert_array_almost_equal(np.radians(decDeg), decRad, 15)


if __name__ == "__main__":
    unittest.main()
