import unittest

import numpy as np

from rubin_sim.utils import BoxBounds, CircleBounds, ModifiedJulianDate, ObservationMetaData, Site


class ObservationMetaDataTest(unittest.TestCase):
    """
    This class will test that ObservationMetaData correctly assigns
    and returns its class variables (pointing_ra, pointing_dec, etc.)

    It will also test the behavior of the m5 member variable.
    """

    def test_m5(self):
        """
        Test behavior of ObservationMetaData's m5 member variable
        """

        self.assertRaises(RuntimeError, ObservationMetaData, bandpass_name="u", m5=[12.0, 13.0])
        self.assertRaises(RuntimeError, ObservationMetaData, bandpass_name=["u", "g"], m5=15.0)
        self.assertRaises(
            RuntimeError,
            ObservationMetaData,
            bandpass_name=["u", "g"],
            m5=[12.0, 13.0, 15.0],
        )

        obs_md = ObservationMetaData()
        self.assertIsNone(obs_md.m5)

        obs_md = ObservationMetaData(bandpass_name="g", m5=12.0)
        self.assertAlmostEqual(obs_md.m5["g"], 12.0, 10)

        obs_md = ObservationMetaData(bandpass_name=["u", "g", "r"], m5=[10, 11, 12])
        self.assertEqual(obs_md.m5["u"], 10)
        self.assertEqual(obs_md.m5["g"], 11)
        self.assertEqual(obs_md.m5["r"], 12)

    def test_seeing(self):
        """
        Test behavior of ObservationMetaData's seeing member variable
        """

        self.assertRaises(RuntimeError, ObservationMetaData, bandpass_name="u", seeing=[0.7, 0.6])
        self.assertRaises(RuntimeError, ObservationMetaData, bandpass_name=["u", "g"], seeing=0.7)
        self.assertRaises(
            RuntimeError,
            ObservationMetaData,
            bandpass_name=["u", "g"],
            seeing=[0.8, 0.7, 0.6],
        )

        obs_md = ObservationMetaData()
        self.assertIsNone(obs_md.seeing)

        obs_md = ObservationMetaData(bandpass_name="g", seeing=0.7)
        self.assertAlmostEqual(obs_md.seeing["g"], 0.7, 10)

        obs_md = ObservationMetaData(bandpass_name=["u", "g", "r"], seeing=[0.7, 0.6, 0.5])
        self.assertEqual(obs_md.seeing["u"], 0.7)
        self.assertEqual(obs_md.seeing["g"], 0.6)
        self.assertEqual(obs_md.seeing["r"], 0.5)

    def test_m5and_seeing_assignment(self):
        """
        Test assignment of m5 and seeing seeing and bandpass in ObservationMetaData
        """
        obs_md = ObservationMetaData(bandpass_name=["u", "g"], m5=[15.0, 16.0], seeing=[0.7, 0.6])
        self.assertAlmostEqual(obs_md.m5["u"], 15.0, 10)
        self.assertAlmostEqual(obs_md.m5["g"], 16.0, 10)
        self.assertAlmostEqual(obs_md.seeing["u"], 0.7, 10)
        self.assertAlmostEqual(obs_md.seeing["g"], 0.6, 10)

        obs_md.set_bandpass_m5and_seeing(bandpass_name=["i", "z"], m5=[25.0, 22.0], seeing=[0.5, 0.4])
        self.assertAlmostEqual(obs_md.m5["i"], 25.0, 10)
        self.assertAlmostEqual(obs_md.m5["z"], 22.0, 10)
        self.assertAlmostEqual(obs_md.seeing["i"], 0.5, 10)
        self.assertAlmostEqual(obs_md.seeing["z"], 0.4, 10)

        with self.assertRaises(KeyError):
            obs_md.m5["u"]

        with self.assertRaises(KeyError):
            obs_md.m5["g"]

        obs_md.m5 = [13.0, 14.0]
        obs_md.seeing = [0.2, 0.3]
        self.assertAlmostEqual(obs_md.m5["i"], 13.0, 10)
        self.assertAlmostEqual(obs_md.m5["z"], 14.0, 10)
        self.assertAlmostEqual(obs_md.seeing["i"], 0.2, 10)
        self.assertAlmostEqual(obs_md.seeing["z"], 0.3, 10)

        obs_md.set_bandpass_m5and_seeing(bandpass_name=["k", "j"], m5=[21.0, 23.0])
        self.assertAlmostEqual(obs_md.m5["k"], 21.0, 10)
        self.assertAlmostEqual(obs_md.m5["j"], 23.0, 10)
        self.assertIsNone(obs_md.seeing)

        obs_md.set_bandpass_m5and_seeing(bandpass_name=["w", "x"], seeing=[0.9, 1.1])
        self.assertAlmostEqual(obs_md.seeing["w"], 0.9, 10)
        self.assertAlmostEqual(obs_md.seeing["x"], 1.1, 10)

    def test_default(self):
        """
        Test that ObservationMetaData's default variables are properly set
        """

        test_obs_md = ObservationMetaData()

        self.assertEqual(test_obs_md.pointing_ra, None)
        self.assertEqual(test_obs_md.pointing_dec, None)
        self.assertEqual(test_obs_md.rot_sky_pos, None)
        self.assertEqual(test_obs_md.bandpass, None)
        self.assertEqual(test_obs_md.m5, None)
        self.assertEqual(test_obs_md.seeing, None)
        self.assertAlmostEqual(test_obs_md.site.longitude, -70.7494, 10)
        self.assertAlmostEqual(test_obs_md.site.latitude, -30.2444, 10)
        self.assertAlmostEqual(test_obs_md.site.height, 2650, 10)
        self.assertAlmostEqual(test_obs_md.site.temperature_kelvin, 284.65, 10)
        self.assertAlmostEqual(test_obs_md.site.temperature, 11.5, 10)
        self.assertAlmostEqual(test_obs_md.site.pressure, 750.0, 10)
        self.assertAlmostEqual(test_obs_md.site.humidity, 0.4, 10)
        self.assertAlmostEqual(test_obs_md.site.lapse_rate, 0.0065, 10)

    def test_site(self):
        """
        Test that site data gets passed correctly when it is not default
        """
        test_site = Site(
            longitude=20.0,
            latitude=-71.0,
            height=4.0,
            temperature=100.0,
            pressure=500.0,
            humidity=0.1,
            lapse_rate=0.1,
        )

        test_obs_md = ObservationMetaData(site=test_site)

        self.assertAlmostEqual(test_obs_md.site.longitude, 20.0, 10)
        self.assertAlmostEqual(test_obs_md.site.longitude_rad, np.radians(20.0), 10)
        self.assertAlmostEqual(test_obs_md.site.latitude, -71.0, 10)
        self.assertAlmostEqual(test_obs_md.site.latitude_rad, np.radians(-71.0), 10)
        self.assertAlmostEqual(test_obs_md.site.height, 4.0, 10)
        self.assertAlmostEqual(test_obs_md.site.temperature, 100.0, 10)
        self.assertAlmostEqual(test_obs_md.site.temperature_kelvin, 373.15, 10)
        self.assertAlmostEqual(test_obs_md.site.pressure, 500.0, 10)
        self.assertAlmostEqual(test_obs_md.site.humidity, 0.1, 10)
        self.assertAlmostEqual(test_obs_md.site.lapse_rate, 0.1, 10)

    def test_assignment(self):
        """
        Test that ObservationMetaData member variables get passed correctly
        """

        mjd = 5120.0
        RA = 1.5
        dec = -1.1
        rot_sky_pos = -10.0
        sky_brightness = 25.0

        test_obs_md = ObservationMetaData()
        test_obs_md.pointing_ra = RA
        test_obs_md.pointing_dec = dec
        test_obs_md.rot_sky_pos = rot_sky_pos
        test_obs_md.sky_brightness = sky_brightness
        test_obs_md.mjd = mjd
        test_obs_md.bound_type = "box"
        test_obs_md.bound_length = [1.2, 3.0]

        self.assertAlmostEqual(test_obs_md.pointing_ra, RA, 10)
        self.assertAlmostEqual(test_obs_md.pointing_dec, dec, 10)
        self.assertAlmostEqual(test_obs_md.rot_sky_pos, rot_sky_pos, 10)
        self.assertAlmostEqual(test_obs_md.sky_brightness, sky_brightness, 10)
        self.assertEqual(test_obs_md.bound_type, "box")
        self.assertAlmostEqual(test_obs_md.bound_length[0], 1.2, 10)
        self.assertAlmostEqual(test_obs_md.bound_length[1], 3.0, 10)
        self.assertAlmostEqual(test_obs_md.mjd.TAI, mjd, 10)

        # test reassignment

        test_obs_md.pointing_ra = RA + 1.0
        test_obs_md.pointing_dec = dec + 1.0
        test_obs_md.rot_sky_pos = rot_sky_pos + 1.0
        test_obs_md.sky_brightness = sky_brightness + 1.0
        test_obs_md.bound_length = 2.2
        test_obs_md.bound_type = "circle"
        test_obs_md.mjd = mjd + 10.0

        self.assertAlmostEqual(test_obs_md.pointing_ra, RA + 1.0, 10)
        self.assertAlmostEqual(test_obs_md.pointing_dec, dec + 1.0, 10)
        self.assertAlmostEqual(test_obs_md.rot_sky_pos, rot_sky_pos + 1.0, 10)
        self.assertAlmostEqual(test_obs_md.sky_brightness, sky_brightness + 1.0, 10)
        self.assertEqual(test_obs_md.bound_type, "circle")
        self.assertAlmostEqual(test_obs_md.bound_length, 2.2, 10)
        self.assertAlmostEqual(test_obs_md.mjd.TAI, mjd + 10.0, 10)

        test_obs_md = ObservationMetaData(
            mjd=mjd,
            pointing_ra=RA,
            pointing_dec=dec,
            rot_sky_pos=rot_sky_pos,
            bandpass_name="z",
            sky_brightness=sky_brightness,
        )

        self.assertAlmostEqual(test_obs_md.mjd.TAI, 5120.0, 10)
        self.assertAlmostEqual(test_obs_md.pointing_ra, 1.5, 10)
        self.assertAlmostEqual(test_obs_md.pointing_dec, -1.1, 10)
        self.assertAlmostEqual(test_obs_md.rot_sky_pos, -10.0, 10)
        self.assertEqual(test_obs_md.bandpass, "z")
        self.assertAlmostEqual(test_obs_md.sky_brightness, sky_brightness, 10)

        # test assigning ModifiedJulianDate
        obs = ObservationMetaData()
        mjd = ModifiedJulianDate(TAI=57388.0)
        obs.mjd = mjd
        self.assertEqual(obs.mjd, mjd)

        mjd2 = ModifiedJulianDate(TAI=45000.0)
        obs.mjd = mjd2
        self.assertEqual(obs.mjd, mjd2)
        self.assertNotEqual(obs.mjd, mjd)

    def test_bound_building(self):
        """
        Make sure ObservationMetaData can build bounds
        """
        box_bounds = [0.1, 0.3]
        circ_obs = ObservationMetaData(
            bound_type="circle",
            pointing_ra=0.0,
            pointing_dec=0.0,
            bound_length=1.0,
            mjd=53580.0,
        )
        bound_control = CircleBounds(0.0, 0.0, np.radians(1.0))
        self.assertEqual(circ_obs.bounds, bound_control)

        square_obs = ObservationMetaData(
            bound_type="box",
            pointing_ra=0.0,
            pointing_dec=0.0,
            bound_length=1.0,
            mjd=53580.0,
        )
        bound_control = BoxBounds(0.0, 0.0, np.radians(1.0))
        self.assertEqual(square_obs.bounds, bound_control)

        box_obs = ObservationMetaData(
            bound_type="box",
            pointing_ra=0.0,
            pointing_dec=0.0,
            bound_length=box_bounds,
            mjd=53580.0,
        )
        bound_control = BoxBounds(0.0, 0.0, np.radians([0.1, 0.3]))
        self.assertEqual(box_obs.bounds, bound_control)

    def test_bounds(self):
        """
        Test if ObservationMetaData correctly assigns the pointing[RA,Dec]
        when circle and box bounds are specified
        """

        circ_ra = 25.0
        circ_dec = 50.0
        radius = 5.0

        box_ra = 15.0
        box_dec = 0.0
        box_length = np.array([5.0, 10.0])

        test_obs_md = ObservationMetaData(
            bound_type="circle",
            pointing_ra=circ_ra,
            pointing_dec=circ_dec,
            bound_length=radius,
            mjd=53580.0,
        )
        self.assertAlmostEqual(test_obs_md.pointing_ra, 25.0, 10)
        self.assertAlmostEqual(test_obs_md.pointing_dec, 50.0, 10)

        test_obs_md = ObservationMetaData(
            bound_type="box",
            pointing_ra=box_ra,
            pointing_dec=box_dec,
            bound_length=box_length,
            mjd=53580.0,
        )
        self.assertAlmostEqual(test_obs_md.pointing_ra, 15.0, 10)
        self.assertAlmostEqual(test_obs_md.pointing_dec, 0.0, 10)

    def test_summary(self):
        """
        Make sure summary is safe even when no parameters have been set
        """
        obs = ObservationMetaData()
        obs.summary

    def test_sim_meta_data(self):
        """
        Make sure that an exception is raised if you pass a non-dict
        object in as simMetaData
        """
        obs = ObservationMetaData(pointing_ra=23.0, pointing_dec=-11.0)

        with self.assertRaises(RuntimeError) as ee:
            obs.sim_meta_data = 5.0
        self.assertIn("must be a dict", ee.exception.args[0])

        with self.assertRaises(RuntimeError) as ee:
            obs.sim_meta_data = 5
        self.assertIn("must be a dict", ee.exception.args[0])

        with self.assertRaises(RuntimeError) as ee:
            obs.sim_meta_data = [5.0, 3.0]
        self.assertIn("must be a dict", ee.exception.args[0])

        with self.assertRaises(RuntimeError) as ee:
            obs.sim_meta_data = (5.0, 3.0)
        self.assertIn("must be a dict", ee.exception.args[0])

        obs.sim_meta_data = {"a": 1, "b": 2}

    def test_eq(self):
        """
        Test that we implemented __eq__ and __ne__ correctly
        """
        empty_obs = ObservationMetaData()
        other_empty_obs = ObservationMetaData()
        self.assertEqual(empty_obs, other_empty_obs)
        self.assertTrue(empty_obs == other_empty_obs)
        self.assertFalse(empty_obs != other_empty_obs)

        dummy_site = Site(
            longitude=23.1,
            latitude=-11.1,
            temperature=11.0,
            height=8921.01,
            pressure=734.1,
            humidity=0.1,
            lapse_rate=0.006,
        )

        ref_obs = ObservationMetaData(
            pointing_ra=23.44,
            pointing_dec=-19.1,
            mjd=59580.1,
            rot_sky_pos=91.2,
            bandpass_name="u",
            m5=24.3,
            sky_brightness=22.1,
            seeing=0.8,
            site=dummy_site,
        )

        other_obs = ObservationMetaData(
            pointing_ra=23.44,
            pointing_dec=-19.1,
            mjd=59580.1,
            rot_sky_pos=91.2,
            bandpass_name="u",
            m5=24.3,
            sky_brightness=22.1,
            seeing=0.8,
            site=dummy_site,
        )

        self.assertEqual(ref_obs, other_obs)
        self.assertTrue(ref_obs == other_obs)
        self.assertFalse(ref_obs != other_obs)

        other_obs = ObservationMetaData(
            pointing_ra=23.41,
            pointing_dec=-19.1,
            mjd=59580.1,
            rot_sky_pos=91.2,
            bandpass_name="u",
            m5=24.3,
            sky_brightness=22.1,
            seeing=0.8,
            site=dummy_site,
        )

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(
            pointing_ra=23.44,
            pointing_dec=-19.2,
            mjd=59580.1,
            rot_sky_pos=91.2,
            bandpass_name="u",
            m5=24.3,
            sky_brightness=22.1,
            seeing=0.8,
            site=dummy_site,
        )

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(
            pointing_ra=23.44,
            pointing_dec=-19.1,
            mjd=59580.2,
            rot_sky_pos=91.2,
            bandpass_name="u",
            m5=24.3,
            sky_brightness=22.1,
            seeing=0.8,
            site=dummy_site,
        )

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(
            pointing_ra=23.44,
            pointing_dec=-19.1,
            mjd=59580.1,
            rot_sky_pos=91.1,
            bandpass_name="u",
            m5=24.3,
            sky_brightness=22.1,
            seeing=0.8,
            site=dummy_site,
        )

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(
            pointing_ra=23.44,
            pointing_dec=-19.1,
            mjd=59580.1,
            rot_sky_pos=91.2,
            bandpass_name="g",
            m5=24.3,
            sky_brightness=22.1,
            seeing=0.8,
            site=dummy_site,
        )

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(
            pointing_ra=23.44,
            pointing_dec=-19.1,
            mjd=59580.1,
            rot_sky_pos=91.2,
            bandpass_name="u",
            m5=24.1,
            sky_brightness=22.1,
            seeing=0.8,
            site=dummy_site,
        )

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(
            pointing_ra=23.44,
            pointing_dec=-19.1,
            mjd=59580.1,
            rot_sky_pos=91.2,
            bandpass_name="u",
            m5=24.3,
            sky_brightness=22.2,
            seeing=0.8,
            site=dummy_site,
        )

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(
            pointing_ra=23.44,
            pointing_dec=-19.1,
            mjd=59580.1,
            rot_sky_pos=91.2,
            bandpass_name="u",
            m5=24.3,
            sky_brightness=22.1,
            seeing=0.81,
            site=dummy_site,
        )

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs = ObservationMetaData(
            pointing_ra=23.44,
            pointing_dec=-19.1,
            mjd=59580.1,
            rot_sky_pos=91.2,
            bandpass_name="u",
            m5=24.3,
            sky_brightness=22.1,
            seeing=0.8,
        )

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
        ref_obs.set_bandpass_m5and_seeing(
            bandpass_name=["u", "r", "z"], m5=[22.1, 23.5, 24.2], seeing=[0.6, 0.7, 0.8]
        )

        other_obs.set_bandpass_m5and_seeing(
            bandpass_name=["u", "r", "z"], m5=[22.1, 23.5, 24.2], seeing=[0.6, 0.7, 0.8]
        )

        self.assertEqual(ref_obs, other_obs)
        self.assertTrue(ref_obs == other_obs)
        self.assertFalse(ref_obs != other_obs)

        other_obs.set_bandpass_m5and_seeing(
            bandpass_name=["u", "i", "z"], m5=[22.1, 23.5, 24.2], seeing=[0.6, 0.7, 0.8]
        )

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs.set_bandpass_m5and_seeing(
            bandpass_name=["u", "r", "z"], m5=[22.1, 23.4, 24.2], seeing=[0.6, 0.7, 0.8]
        )

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs.set_bandpass_m5and_seeing(
            bandpass_name=["u", "r", "z"], m5=[22.1, 23.5, 24.2], seeing=[0.2, 0.7, 0.8]
        )

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)

        other_obs.set_bandpass_m5and_seeing(bandpass_name=["u", "z"], m5=[22.1, 24.2], seeing=[0.2, 0.8])

        self.assertNotEqual(ref_obs, other_obs)
        self.assertFalse(ref_obs == other_obs)
        self.assertTrue(ref_obs != other_obs)


if __name__ == "__main__":
    unittest.main()
