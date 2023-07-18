import unittest
import warnings

import numpy as np

from rubin_sim.utils import Site


class SiteTest(unittest.TestCase):
    def setUp(self):
        # LSST default values taken from LSE-30
        self.height = 2650.0
        self.longitude = -70.7494
        self.latitude = -30.2444
        self.temperature = 11.5
        self.humidity = 0.4
        self.pressure = 750.0
        self.lapse_rate = 0.0065

    def test_lsst_values(self):
        """
        Test that LSST values are set correctly
        """
        site = Site(name="LSST")
        self.assertEqual(site.name, "LSST")
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapse_rate, self.lapse_rate)
        self.assertEqual(site.height, self.height)

    def test_no_defaults(self):
        """
        Test that, if name is not 'LSST', values are set to None
        """
        with warnings.catch_warnings(record=True) as ww:
            site = Site(name="bob")

        msg = str(ww[0].message)

        self.assertIn("longitude", msg)
        self.assertIn("latitude", msg)
        self.assertIn("temperature", msg)
        self.assertIn("pressure", msg)
        self.assertIn("height", msg)
        self.assertIn("lapse_rate", msg)
        self.assertIn("humidity", msg)

        self.assertEqual(site.name, "bob")
        self.assertIsNone(site.longitude)
        self.assertIsNone(site.longitude_rad)
        self.assertIsNone(site.latitude)
        self.assertIsNone(site.latitude_rad)
        self.assertIsNone(site.temperature)
        self.assertIsNone(site.temperature_kelvin)
        self.assertIsNone(site.pressure)
        self.assertIsNone(site.humidity)
        self.assertIsNone(site.lapse_rate)
        self.assertIsNone(site.height)

    def test_override_lss_tdefaults(self):
        """
        Test that, even if LSST is specified, we are capable of overriding
        defaults
        """
        site = Site(name="LSST", longitude=26.0)
        self.assertEqual(site.name, "LSST")
        self.assertEqual(site.longitude, 26.0)
        self.assertEqual(site.longitude_rad, np.radians(26.0))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapse_rate, self.lapse_rate)
        self.assertEqual(site.height, self.height)

        site = Site(name="LSST", latitude=88.0)
        self.assertEqual(site.name, "LSST")
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, 88.0)
        self.assertEqual(site.latitude_rad, np.radians(88.0))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapse_rate, self.lapse_rate)
        self.assertEqual(site.height, self.height)

        site = Site(name="LSST", height=4.0)
        self.assertEqual(site.name, "LSST")
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapse_rate, self.lapse_rate)
        self.assertEqual(site.height, 4.0)

        site = Site(name="LSST", temperature=7.0)
        self.assertEqual(site.name, "LSST")
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, 7.0)
        self.assertEqual(site.temperature_kelvin, 280.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapse_rate, self.lapse_rate)
        self.assertEqual(site.height, self.height)

        site = Site(name="LSST", pressure=14.0)
        self.assertEqual(site.name, "LSST")
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, 14.0)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapse_rate, self.lapse_rate)
        self.assertEqual(site.height, self.height)

        site = Site(name="LSST", humidity=2.1)
        self.assertEqual(site.name, "LSST")
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, 2.1)
        self.assertEqual(site.lapse_rate, self.lapse_rate)
        self.assertEqual(site.height, self.height)

        site = Site(name="LSST", lapse_rate=3.2)
        self.assertEqual(site.name, "LSST")
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapse_rate, 3.2)
        self.assertEqual(site.height, self.height)

    def test_partial_params(self):
        """
        test that unspecified parameters get set to None
        """
        with warnings.catch_warnings(record=True) as ww:
            site = Site(longitude=45.0, temperature=20.0)

        msg = str(ww[0].message)
        self.assertIn("latitude", msg)
        self.assertIn("height", msg)
        self.assertIn("pressure", msg)
        self.assertIn("lapse_rate", msg)
        self.assertIn("humidity", msg)
        self.assertNotIn("longitue", msg)
        self.assertNotIn("temperature", msg)

        self.assertIsNone(site.name)
        self.assertIsNone(site.latitude)
        self.assertIsNone(site.latitude_rad)
        self.assertIsNone(site.height)
        self.assertIsNone(site.pressure)
        self.assertIsNone(site.humidity)
        self.assertIsNone(site.lapse_rate)
        self.assertEqual(site.longitude, 45.0)
        self.assertEqual(site.longitude_rad, np.pi / 4.0)
        self.assertEqual(site.temperature, 20.0)
        self.assertEqual(site.temperature_kelvin, 293.15)

    def test_eq(self):
        """
        Test that we have correctly implemented __eq__ in Site
        """
        reference_site = Site(
            name="ref",
            longitude=112.12,
            latitude=-83.121,
            temperature=112.1,
            height=3124.2,
            pressure=891.2,
            humidity=0.341,
            lapse_rate=0.008,
        )

        other_site = Site(
            name="ref",
            longitude=112.12,
            latitude=-83.121,
            temperature=112.1,
            height=3124.2,
            pressure=891.2,
            humidity=0.341,
            lapse_rate=0.008,
        )

        self.assertEqual(reference_site, other_site)
        self.assertFalse(reference_site != other_site)
        self.assertTrue(reference_site == other_site)

        # just in case we ever change the class to convert
        # to radians only on demand, call for latitude/longitude
        # in radians and then check that the two instances are
        # still equal (since __eq__ just loops over the contents
        # of self.__dict__, this could fail if other_site has not
        # yet assigned a value to longitude/latitude_rad
        reference_site.latitude_rad
        self.assertEqual(reference_site, other_site)
        self.assertFalse(reference_site != other_site)
        self.assertTrue(reference_site == other_site)

        reference_site.longitude_rad
        self.assertEqual(reference_site, other_site)
        self.assertFalse(reference_site != other_site)
        self.assertTrue(reference_site == other_site)

        reference_site.temperature_kelvin
        self.assertEqual(reference_site, other_site)
        self.assertFalse(reference_site != other_site)
        self.assertTrue(reference_site == other_site)

        # now test that __ne__ works correctly
        other_site = Site(
            name="other",
            longitude=112.12,
            latitude=-83.121,
            temperature=112.1,
            height=3124.2,
            pressure=891.2,
            humidity=0.341,
            lapse_rate=0.008,
        )

        self.assertNotEqual(reference_site, other_site)
        self.assertFalse(reference_site == other_site)
        self.assertTrue(reference_site != other_site)

        other_site = Site(
            name="ref",
            longitude=112.13,
            latitude=-83.121,
            temperature=112.1,
            height=3124.2,
            pressure=891.2,
            humidity=0.341,
            lapse_rate=0.008,
        )

        self.assertNotEqual(reference_site, other_site)
        self.assertFalse(reference_site == other_site)
        self.assertTrue(reference_site != other_site)

        other_site = Site(
            name="ref",
            longitude=112.12,
            latitude=-83.122,
            temperature=112.1,
            height=3124.2,
            pressure=891.2,
            humidity=0.341,
            lapse_rate=0.008,
        )

        self.assertNotEqual(reference_site, other_site)
        self.assertFalse(reference_site == other_site)
        self.assertTrue(reference_site != other_site)

        other_site = Site(
            name="ref",
            longitude=112.12,
            latitude=-83.121,
            temperature=112.2,
            height=3124.2,
            pressure=891.2,
            humidity=0.341,
            lapse_rate=0.008,
        )

        self.assertNotEqual(reference_site, other_site)
        self.assertFalse(reference_site == other_site)
        self.assertTrue(reference_site != other_site)

        other_site = Site(
            name="ref",
            longitude=112.12,
            latitude=-83.121,
            temperature=112.1,
            height=3124.3,
            pressure=891.2,
            humidity=0.341,
            lapse_rate=0.008,
        )

        self.assertNotEqual(reference_site, other_site)
        self.assertFalse(reference_site == other_site)
        self.assertTrue(reference_site != other_site)

        other_site = Site(
            name="ref",
            longitude=112.12,
            latitude=-83.121,
            temperature=112.1,
            height=3124.2,
            pressure=891.3,
            humidity=0.341,
            lapse_rate=0.008,
        )

        self.assertNotEqual(reference_site, other_site)
        self.assertFalse(reference_site == other_site)
        self.assertTrue(reference_site != other_site)

        other_site = Site(
            name="ref",
            longitude=112.12,
            latitude=-83.121,
            temperature=112.1,
            height=3124.2,
            pressure=891.2,
            humidity=0.342,
            lapse_rate=0.008,
        )

        self.assertNotEqual(reference_site, other_site)
        self.assertFalse(reference_site == other_site)
        self.assertTrue(reference_site != other_site)

        other_site = Site(
            name="ref",
            longitude=112.12,
            latitude=-83.121,
            temperature=112.1,
            height=3124.2,
            pressure=891.2,
            humidity=0.341,
            lapse_rate=0.009,
        )

        self.assertNotEqual(reference_site, other_site)
        self.assertFalse(reference_site == other_site)
        self.assertTrue(reference_site != other_site)

        # test blank Sites
        ref_site = Site()
        other_site = Site()
        self.assertEqual(ref_site, other_site)
        self.assertTrue(ref_site == other_site)
        self.assertFalse(ref_site != other_site)


if __name__ == "__main__":
    unittest.main()
