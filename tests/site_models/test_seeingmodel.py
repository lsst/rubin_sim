import unittest

import numpy as np

from rubin_sim.site_models import SeeingModel


class TestSeeingModel(unittest.TestCase):
    def test_fwhm_system_zenith(self):
        # Check calculation is being done as expected.
        seeing_model = SeeingModel()
        self.assertAlmostEqual(seeing_model.fwhm_system_zenith, 0.39862262855989494, places=7)

    def test_fwhm_geom_eff(self):
        # Check that the translation between FWHM effective and geometric is done as expected.
        # (note that fwhmEff_tofwhmGeom & fwhmGeom_to_fwhmEff are static methods)
        # Document-20160 for reference.
        fwhm_eff = 1.23
        fwhm_geom = 0.822 * fwhm_eff + 0.052
        self.assertEqual(fwhm_geom, SeeingModel.fwhm_eff_to_fwhm_geom(fwhm_eff))
        self.assertEqual(fwhm_eff, SeeingModel.fwhm_geom_to_fwhm_eff(fwhm_geom))

    def test_call(self):
        # Check the calculation from fwhm_500 to fwhm_eff/fwhm_geom.
        # Use simple effective wavelengths and airmass values.
        filter_list = ["500", "1000"]
        effwavelens = np.array([500.0, 1000.0])
        seeing_model = SeeingModel(filter_list=filter_list, eff_wavelens=effwavelens)
        # Simple fwhm_500 input.
        fwhm_500 = 1.0
        # Single airmass.
        airmass = 1.0
        seeing = seeing_model(fwhm_500, airmass)
        fwhm_eff = seeing["fwhmEff"]
        fwhm_geom = seeing["fwhmGeom"]
        # Check shape of returned values.
        self.assertEqual(fwhm_eff.shape, (len(seeing_model.eff_wavelens),))
        # Check actual value of seeing in @ wavelen[0] @ zenith after addition of system.
        fwhm_system = seeing_model.fwhm_system_zenith
        expected_fwhm_eff = 1.16 * np.sqrt(fwhm_system**2 + 1.04 * fwhm_500**2)
        self.assertAlmostEqual(fwhm_eff[0], expected_fwhm_eff, 15)
        # Check expected value if we remove the system component.
        seeing_model.fwhm_system_zenith = 0
        seeing = seeing_model(fwhm_500, airmass)
        expected_fwhm_eff = 1.16 * np.sqrt(1.04) * fwhm_500
        self.assertAlmostEqual(seeing["fwhmEff"][0], expected_fwhm_eff, 15)
        # Check scaling with wavelength (remove system component).
        expected_fwhm_eff = 1.16 * np.sqrt(1.04) * fwhm_500 * np.power(500.0 / effwavelens[1], 0.3)
        self.assertAlmostEqual(seeing["fwhmEff"][1], expected_fwhm_eff, places=15)
        # Multiple airmasses.
        airmass = np.array([1.0, 1.5])
        seeing = seeing_model(fwhm_500, airmass)
        self.assertEqual(seeing["fwhmEff"].shape, (len(seeing_model.eff_wavelens), len(airmass)))
        expected_fwhm_eff = fwhm_500 * 1.16 * np.sqrt(1.04)
        self.assertEqual(seeing["fwhmEff"][0][0], expected_fwhm_eff)
        # Check scaling with airmass.
        expected_fwhm_eff = expected_fwhm_eff * np.power(airmass[1], 0.6)
        self.assertAlmostEqual(seeing["fwhmEff"][0][1], expected_fwhm_eff, places=15)


if __name__ == "__main__":
    unittest.main()
