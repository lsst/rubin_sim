import numpy as np
import unittest
from rubin_sim.utils import ObservationMetaData, ModifiedJulianDate
from rubin_sim.utils import Site, BoxBounds, CircleBounds


class ObservationMetaDataTest(unittest.TestCase):
    """
    This class will test that ObservationMetaData correctly assigns
    and returns its class variables (pointingRA, pointingDec, etc.)

    It will also test the behavior of the m5 member variable.
    """

    def testM5(self):
        """
        Test behavior of ObservationMetaData's m5 member variable
        """

        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName='u', m5=[12.0, 13.0])
        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName=['u', 'g'], m5=15.0)
        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName=['u', 'g'], m5=[12.0, 13.0, 15.0])

        obsMD = ObservationMetaData()
        self.assertIsNone(obsMD.m5)

        obsMD = ObservationMetaData(bandpassName='g', m5=12.0)
        self.assertAlmostEqual(obsMD.m5['g'], 12.0, 10)

        obsMD = ObservationMetaData(bandpassName=['u', 'g', 'r'], m5=[10, 11, 12])
        self.assertEqual(obsMD.m5['u'], 10)
        self.assertEqual(obsMD.m5['g'], 11)
        self.assertEqual(obsMD.m5['r'], 12)

    def testSeeing(self):
        """
        Test behavior of ObservationMetaData's seeing member variable
        """

        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName='u', seeing=[0.7, 0.6])
        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName=['u', 'g'], seeing=0.7)
        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName=['u', 'g'], seeing=[0.8, 0.7, 0.6])

        obsMD = ObservationMetaData()
        self.assertIsNone(obsMD.seeing)

        obsMD = ObservationMetaData(bandpassName='g', seeing=0.7)
        self.assertAlmostEqual(obsMD.seeing['g'], 0.7, 10)

        obsMD = ObservationMetaData(bandpassName=['u', 'g', 'r'], seeing=[0.7, 0.6, 0.5])
        self.assertEqual(obsMD.seeing['u'], 0.7)
        self.assertEqual(obsMD.seeing['g'], 0.6)
        self.assertEqual(obsMD.seeing['r'], 0.5)

    def testM5andSeeingAssignment(self):
        """
        Test assignment of m5 and seeing seeing and bandpass in ObservationMetaData
        """
        obsMD = ObservationMetaData(bandpassName=['u', 'g'], m5=[15.0, 16.0], seeing=[0.7, 0.6])
        self.assertAlmostEqual(obsMD.m5['u'], 15.0, 10)
        self.assertAlmostEqual(obsMD.m5['g'], 16.0, 10)
        self.assertAlmostEqual(obsMD.seeing['u'], 0.7, 10)
        self.assertAlmostEqual(obsMD.seeing['g'], 0.6, 10)

        obsMD.setBandpassM5andSeeing(bandpassName=['i', 'z'], m5=[25.0, 22.0], seeing=[0.5, 0.4])
        self.assertAlmostEqual(obsMD.m5['i'], 25.0, 10)
        self.assertAlmostEqual(obsMD.m5['z'], 22.0, 10)
        self.assertAlmostEqual(obsMD.seeing['i'], 0.5, 10)
        self.assertAlmostEqual(obsMD.seeing['z'], 0.4, 10)

        with self.assertRaises(KeyError):
            obsMD.m5['u']

        with self.assertRaises(KeyError):
            obsMD.m5['g']

        obsMD.m5 = [13.0, 14.0]
        obsMD.seeing = [0.2, 0.3]
        self.assertAlmostEqual(obsMD.m5['i'], 13.0, 10)
        self.assertAlmostEqual(obsMD.m5['z'], 14.0, 10)
        self.assertAlmostEqual(obsMD.seeing['i'], 0.2, 10)
        self.assertAlmostEqual(obsMD.seeing['z'], 0.3, 10)

        obsMD.setBandpassM5andSeeing(bandpassName=['k', 'j'], m5=[21.0, 23.0])
        self.assertAlmostEqual(obsMD.m5['k'], 21.0, 10)
        self.assertAlmostEqual(obsMD.m5['j'], 23.0, 10)
        self.assertIsNone(obsMD.seeing)

        obsMD.setBandpassM5andSeeing(bandpassName=['w', 'x'], seeing=[0.9, 1.1])
        self.assertAlmostEqual(obsMD.seeing['w'], 0.9, 10)
        self.assertAlmostEqual(obsMD.seeing['x'], 1.1, 10)

    def testDefault(self):
        """
        Test that ObservationMetaData's default variables are properly set
        """

        testObsMD = ObservationMetaData()

        self.assertEqual(testObsMD.pointingRA, None)
        self.assertEqual(testObsMD.pointingDec, None)
        self.assertEqual(testObsMD.rotSkyPos, None)
        self.assertEqual(testObsMD.bandpass, None)
        self.assertEqual(testObsMD.m5, None)
        self.assertEqual(testObsMD.seeing, None)
        self.assertAlmostEqual(testObsMD.site.longitude, -70.7494, 10)
        self.assertAlmostEqual(testObsMD.site.latitude, -30.2444, 10)
        self.assertAlmostEqual(testObsMD.site.height, 2650, 10)
        self.assertAlmostEqual(testObsMD.site.temperature_kelvin, 284.65, 10)
        self.assertAlmostEqual(testObsMD.site.temperature, 11.5, 10)
        self.assertAlmostEqual(testObsMD.site.pressure, 750.0, 10)
        self.assertAlmostEqual(testObsMD.site.humidity, 0.4, 10)
        self.assertAlmostEqual(testObsMD.site.lapseRate, 0.0065, 10)

    def testSite(self):
        """
        Test that site data gets passed correctly when it is not default
        """
        testSite = Site(longitude=20.0, latitude=-71.0, height=4.0,
                        temperature=100.0, pressure=500.0, humidity=0.1,
                        lapseRate=0.1)

        testObsMD = ObservationMetaData(site=testSite)

        self.assertAlmostEqual(testObsMD.site.longitude, 20.0, 10)
        self.assertAlmostEqual(testObsMD.site.longitude_rad, np.radians(20.0), 10)
        self.assertAlmostEqual(testObsMD.site.latitude, -71.0, 10)
        self.assertAlmostEqual(testObsMD.site.latitude_rad, np.radians(-71.0), 10)
        self.assertAlmostEqual(testObsMD.site.height, 4.0, 10)
        self.assertAlmostEqual(testObsMD.site.temperature, 100.0, 10)
        self.assertAlmostEqual(testObsMD.site.temperature_kelvin, 373.15, 10)
        self.assertAlmostEqual(testObsMD.site.pressure, 500.0, 10)
        self.assertAlmostEqual(testObsMD.site.humidity, 0.1, 10)
        self.assertAlmostEqual(testObsMD.site.lapseRate, 0.1, 10)

    def testAssignment(self):
        """
        Test that ObservationMetaData member variables get passed correctly
        """

        mjd = 5120.0
        RA = 1.5
        Dec = -1.1
        rotSkyPos = -10.0
        skyBrightness = 25.0

        testObsMD = ObservationMetaData()
        testObsMD.pointingRA = RA
        testObsMD.pointingDec = Dec
        testObsMD.rotSkyPos = rotSkyPos
        testObsMD.skyBrightness = skyBrightness
        testObsMD.mjd = mjd
        testObsMD.boundType = 'box'
        testObsMD.boundLength = [1.2, 3.0]

        self.assertAlmostEqual(testObsMD.pointingRA, RA, 10)
        self.assertAlmostEqual(testObsMD.pointingDec, Dec, 10)
        self.assertAlmostEqual(testObsMD.rotSkyPos, rotSkyPos, 10)
        self.assertAlmostEqual(testObsMD.skyBrightness, skyBrightness, 10)
        self.assertEqual(testObsMD.boundType, 'box')
        self.assertAlmostEqual(testObsMD.boundLength[0], 1.2, 10)
        self.assertAlmostEqual(testObsMD.boundLength[1], 3.0, 10)
        self.assertAlmostEqual(testObsMD.mjd.TAI, mjd, 10)

        # test reassignment

        testObsMD.pointingRA = RA+1.0
        testObsMD.pointingDec = Dec+1.0
        testObsMD.rotSkyPos = rotSkyPos+1.0
        testObsMD.skyBrightness = skyBrightness+1.0
        testObsMD.boundLength = 2.2
        testObsMD.boundType = 'circle'
        testObsMD.mjd = mjd + 10.0

        self.assertAlmostEqual(testObsMD.pointingRA, RA+1.0, 10)
        self.assertAlmostEqual(testObsMD.pointingDec, Dec+1.0, 10)
        self.assertAlmostEqual(testObsMD.rotSkyPos, rotSkyPos+1.0, 10)
        self.assertAlmostEqual(testObsMD.skyBrightness, skyBrightness+1.0, 10)
        self.assertEqual(testObsMD.boundType, 'circle')
        self.assertAlmostEqual(testObsMD.boundLength, 2.2, 10)
        self.assertAlmostEqual(testObsMD.mjd.TAI, mjd+10.0, 10)

        testObsMD = ObservationMetaData(mjd=mjd, pointingRA=RA,
                                        pointingDec=Dec, rotSkyPos=rotSkyPos, bandpassName='z',
                                        skyBrightness=skyBrightness)

        self.assertAlmostEqual(testObsMD.mjd.TAI, 5120.0, 10)
        self.assertAlmostEqual(testObsMD.pointingRA, 1.5, 10)
        self.assertAlmostEqual(testObsMD.pointingDec, -1.1, 10)
        self.assertAlmostEqual(testObsMD.rotSkyPos, -10.0, 10)
        self.assertEqual(testObsMD.bandpass, 'z')
        self.assertAlmostEqual(testObsMD.skyBrightness, skyBrightness, 10)

        # test assigning ModifiedJulianDate
        obs = ObservationMetaData()
        mjd = ModifiedJulianDate(TAI=57388.0)
        obs.mjd = mjd
        self.assertEqual(obs.mjd, mjd)

        mjd2 = ModifiedJulianDate(TAI=45000.0)
        obs.mjd = mjd2
        self.assertEqual(obs.mjd, mjd2)
        self.assertNotEqual(obs.mjd, mjd)

    def testBoundBuilding(self):
        """
        Make sure ObservationMetaData can build bounds
        """
        boxBounds = [0.1, 0.3]
        circObs = ObservationMetaData(boundType='circle', pointingRA=0.0, pointingDec=0.0,
                                      boundLength=1.0, mjd=53580.0)
        boundControl = CircleBounds(0.0, 0.0, np.radians(1.0))
        self.assertEqual(circObs.bounds, boundControl)

        squareObs = ObservationMetaData(boundType = 'box', pointingRA=0.0, pointingDec=0.0,
                                        boundLength=1.0, mjd=53580.0)
        boundControl = BoxBounds(0.0, 0.0, np.radians(1.0))
        self.assertEqual(squareObs.bounds, boundControl)

        boxObs = ObservationMetaData(boundType = 'box', pointingRA=0.0, pointingDec=0.0,
                                     boundLength=boxBounds, mjd=53580.0)
        boundControl = BoxBounds(0.0, 0.0, np.radians([0.1, 0.3]))
        self.assertEqual(boxObs.bounds, boundControl)

    def testBounds(self):
        """
        Test if ObservationMetaData correctly assigns the pointing[RA,Dec]
        when circle and box bounds are specified
        """

        circRA = 25.0
        circDec = 50.0
        radius = 5.0

        boxRA = 15.0
        boxDec = 0.0
        boxLength = np.array([5.0, 10.0])

        testObsMD = ObservationMetaData(boundType='circle',
                                        pointingRA = circRA, pointingDec=circDec,
                                        boundLength = radius, mjd=53580.0)
        self.assertAlmostEqual(testObsMD.pointingRA, 25.0, 10)
        self.assertAlmostEqual(testObsMD.pointingDec, 50.0, 10)

        testObsMD = ObservationMetaData(boundType='box',
                                        pointingRA=boxRA, pointingDec=boxDec, boundLength=boxLength,
                                        mjd=53580.0)
        self.assertAlmostEqual(testObsMD.pointingRA, 15.0, 10)
        self.assertAlmostEqual(testObsMD.pointingDec, 0.0, 10)

    def testSummary(self):
        """
        Make sure summary is safe even when no parameters have been set
        """
        obs = ObservationMetaData()
        obs.summary

    def testOpsimMetaData(self):
        """
        Make sure that an exception is raised if you pass a non-dict
        object in as OpsimMetaData
        """
        obs = ObservationMetaData(pointingRA=23.0, pointingDec=-11.0)

        with self.assertRaises(RuntimeError) as ee:
            obs.OpsimMetaData = 5.0
        self.assertIn("must be a dict", ee.exception.args[0])

        with self.assertRaises(RuntimeError) as ee:
            obs.OpsimMetaData = 5
        self.assertIn("must be a dict", ee.exception.args[0])

        with self.assertRaises(RuntimeError) as ee:
            obs.OpsimMetaData = [5.0, 3.0]
        self.assertIn("must be a dict", ee.exception.args[0])

        with self.assertRaises(RuntimeError) as ee:
            obs.OpsimMetaData = (5.0, 3.0)
        self.assertIn("must be a dict", ee.exception.args[0])

        obs.OpsimMetaData = {'a': 1, 'b': 2}

    def test_eq(self):
        """
        Test that we implemented __eq__ and __ne__ correctly
        """
        empty_obs = ObservationMetaData()
        other_empty_obs = ObservationMetaData()
        self.assertEqual(empty_obs, other_empty_obs)
        self.assertTrue(empty_obs == other_empty_obs)
        self.assertFalse(empty_obs != other_empty_obs)

        dummy_site = Site(longitude=23.1, latitude=-11.1, temperature=11.0,
                          height=8921.01, pressure=734.1, humidity=0.1,
                          lapseRate=0.006)

        ref_obs = ObservationMetaData(pointingRA=23.44, pointingDec=-19.1,
                                      mjd=59580.1, rotSkyPos=91.2,
                                      bandpassName = 'u', m5=24.3,
                                      skyBrightness=22.1, seeing=0.8,
                                      site=dummy_site)

        other_obs = ObservationMetaData(pointingRA=23.44, pointingDec=-19.1,
                                        mjd=59580.1, rotSkyPos=91.2,
                                        bandpassName = 'u', m5=24.3,
                                        skyBrightness=22.1, seeing=0.8,
                                        site=dummy_site)

        self.assertEqual(ref_obs, other_obs)
        self.assertTrue(ref_obs == other_obs)
        self.assertFalse(ref_obs != other_obs)

        other_obs = ObservationMetaData(pointingRA=23.41, pointingDec=-19.1,
                                        mjd=59580.1, rotSkyPos=91.2,
                                        bandpassName = 'u', m5=24.3,
                                        skyBrightness=22.1, seeing=0.8,
                                        site=dummy_site)

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(pointingRA=23.44, pointingDec=-19.2,
                                        mjd=59580.1, rotSkyPos=91.2,
                                        bandpassName = 'u', m5=24.3,
                                        skyBrightness=22.1, seeing=0.8,
                                        site=dummy_site)

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(pointingRA=23.44, pointingDec=-19.1,
                                        mjd=59580.2, rotSkyPos=91.2,
                                        bandpassName = 'u', m5=24.3,
                                        skyBrightness=22.1, seeing=0.8,
                                        site=dummy_site)

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(pointingRA=23.44, pointingDec=-19.1,
                                        mjd=59580.1, rotSkyPos=91.1,
                                        bandpassName = 'u', m5=24.3,
                                        skyBrightness=22.1, seeing=0.8,
                                        site=dummy_site)

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(pointingRA=23.44, pointingDec=-19.1,
                                        mjd=59580.1, rotSkyPos=91.2,
                                        bandpassName = 'g', m5=24.3,
                                        skyBrightness=22.1, seeing=0.8,
                                        site=dummy_site)

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(pointingRA=23.44, pointingDec=-19.1,
                                        mjd=59580.1, rotSkyPos=91.2,
                                        bandpassName = 'u', m5=24.1,
                                        skyBrightness=22.1, seeing=0.8,
                                        site=dummy_site)

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(pointingRA=23.44, pointingDec=-19.1,
                                        mjd=59580.1, rotSkyPos=91.2,
                                        bandpassName = 'u', m5=24.3,
                                        skyBrightness=22.2, seeing=0.8,
                                        site=dummy_site)

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(pointingRA=23.44, pointingDec=-19.1,
                                        mjd=59580.1, rotSkyPos=91.2,
                                        bandpassName = 'u', m5=24.3,
                                        skyBrightness=22.1, seeing=0.81,
                                        site=dummy_site)

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(pointingRA=23.44, pointingDec=-19.1,
                                        mjd=59580.1, rotSkyPos=91.2,
                                        bandpassName = 'u', m5=24.3,
                                        skyBrightness=22.1, seeing=0.8)

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        # use assignment to bring other_obs back into agreement with
        # ref_obs
        other_obs.site = dummy_site
        self.assertEqual(ref_obs, other_obs)
        self.assertTrue(ref_obs == other_obs)
        self.assertFalse(ref_obs != other_obs)

        # now try cases of m5, bandpass, and seeing being lists
        ref_obs.setBandpassM5andSeeing(bandpassName=['u', 'r', 'z'],
                                       m5=[22.1, 23.5, 24.2],
                                       seeing=[0.6, 0.7, 0.8])

        other_obs.setBandpassM5andSeeing(bandpassName=['u', 'r', 'z'],
                                         m5=[22.1, 23.5, 24.2],
                                         seeing=[0.6, 0.7, 0.8])

        self.assertEqual(ref_obs, other_obs)
        self.assertTrue(ref_obs == other_obs)
        self.assertFalse(ref_obs != other_obs)

        other_obs.setBandpassM5andSeeing(bandpassName=['u', 'i', 'z'],
                                         m5=[22.1, 23.5, 24.2],
                                         seeing=[0.6, 0.7, 0.8])

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs.setBandpassM5andSeeing(bandpassName=['u', 'r', 'z'],
                                         m5=[22.1, 23.4, 24.2],
                                         seeing=[0.6, 0.7, 0.8])

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs.setBandpassM5andSeeing(bandpassName=['u', 'r', 'z'],
                                         m5=[22.1, 23.5, 24.2],
                                         seeing=[0.2, 0.7, 0.8])

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs.setBandpassM5andSeeing(bandpassName=['u', 'z'],
                                         m5=[22.1, 24.2],
                                         seeing=[0.2, 0.8])

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)


if __name__ == "__main__":
    unittest.main()
