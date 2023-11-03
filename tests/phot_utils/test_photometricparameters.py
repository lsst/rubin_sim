import os
import unittest

import numpy as np
from rubin_scheduler.data import get_data_dir

from rubin_sim.phot_utils import Bandpass, PhotometricParameters, PhysicalParameters, Sed


class PhotometricParametersUnitTest(unittest.TestCase):
    def test_init(self):
        """
        Test that the init and getters of PhotometricParameters work
        properly
        """
        defaults = PhotometricParameters()
        params = [
            "exptime",
            "nexp",
            "effarea",
            "gain",
            "readnoise",
            "darkcurrent",
            "othernoise",
            "platescale",
            "sigma_sys",
        ]

        for attribute in params:
            kwargs = {}
            kwargs[attribute] = -100.0
            test_case = PhotometricParameters(**kwargs)

            for pp in params:
                if pp != attribute:
                    self.assertEqual(defaults.__getattribute__(pp), test_case.__getattribute__(pp))
                else:
                    self.assertNotEqual(defaults.__getattribute__(pp), test_case.__getattribute__(pp))

                    self.assertEqual(test_case.__getattribute__(pp), -100.0)

    def test_exceptions(self):
        """
        Test that exceptions get raised when they ought to by the
        PhotometricParameters constructor

        We will instantiate PhotometricParameters with different incomplete
        lists of parameters set.  We will verify that the returned
        error messages correctly point out which parameters were ignored.
        """

        expected_message = {
            "exptime": "did not set exptime",
            "nexp": "did not set nexp",
            "effarea": "did not set effarea",
            "gain": "did not set gain",
            "platescale": "did not set platescale",
            "sigma_sys": "did not set sigma_sys",
            "readnoise": "did not set readnoise",
            "darkcurrent": "did not set darkcurrent",
            "othernoise": "did not set othernoise",
        }

        with self.assertRaises(RuntimeError) as context:
            PhotometricParameters(bandpass="x")

        for name in expected_message:
            self.assertIn(expected_message[name], context.exception.args[0])

        for name1 in expected_message:
            for name2 in expected_message:
                set_parameters = {name1: 2.0, name2: 2.0}
                with self.assertRaises(RuntimeError) as context:
                    PhotometricParameters(bandpass="x", **set_parameters)

                for name3 in expected_message:
                    if name3 not in set_parameters:
                        self.assertIn(expected_message[name3], context.exception.args[0])
                    else:
                        self.assertNotIn(expected_message[name3], context.exception.args[0])

    def test_defaults(self):
        """
        Test that PhotometricParameters are correctly assigned to defaults
        """
        bandpass_names = ["u", "g", "r", "i", "z", "y", None]
        for bp in bandpass_names:
            phot_params = PhotometricParameters(bandpass=bp)
            self.assertEqual(phot_params.bandpass, bp)
            self.assertAlmostEqual(phot_params.exptime, 15.0, 7)
            self.assertAlmostEqual(phot_params.nexp, 2, 7)
            self.assertAlmostEqual(phot_params.effarea / (np.pi * (6.423 * 100 / 2.0) ** 2), 1.0, 7)
            self.assertAlmostEqual(phot_params.gain, 2.3, 7)
            self.assertAlmostEqual(phot_params.darkcurrent, 0.2, 7)
            self.assertAlmostEqual(phot_params.readnoise, 8.8, 7)
            self.assertAlmostEqual(phot_params.othernoise, 0, 7)
            self.assertAlmostEqual(phot_params.platescale, 0.2, 7)
            if bp not in ["u", "z", "y"]:
                self.assertAlmostEqual(phot_params.sigma_sys, 0.005, 7)
            else:
                self.assertAlmostEqual(phot_params.sigma_sys, 0.0075, 7)

    def test_no_bandpass(self):
        """
        Test that if no bandpass is set, bandpass stays 'None' even after
        all other parameters are assigned.
        """
        phot_params = PhotometricParameters()
        self.assertEqual(phot_params.bandpass, None)
        self.assertAlmostEqual(phot_params.exptime, 15.0, 7)
        self.assertAlmostEqual(phot_params.nexp, 2, 7)
        self.assertAlmostEqual(phot_params.effarea / (np.pi * (6.423 * 100 / 2.0) ** 2), 1.0, 7)
        self.assertAlmostEqual(phot_params.gain, 2.3, 7)
        self.assertAlmostEqual(phot_params.darkcurrent, 0.2, 7)
        self.assertAlmostEqual(phot_params.readnoise, 8.8, 7)
        self.assertAlmostEqual(phot_params.othernoise, 0, 7)
        self.assertAlmostEqual(phot_params.platescale, 0.2, 7)
        self.assertAlmostEqual(phot_params.sigma_sys, 0.005, 7)

    def test_assignment(self):
        """
        Test that it is impossible to set PhotometricParameters on the fly
        """
        test_case = PhotometricParameters()
        control_case = PhotometricParameters()
        success = 0

        msg = ""
        try:
            test_case.exptime = -1.0
            success += 1
            msg += "was able to assign exptime; "
        except RuntimeError:
            self.assertEqual(test_case.exptime, control_case.exptime)

        try:
            test_case.nexp = -1.0
            success += 1
            msg += "was able to assign nexp; "
        except RuntimeError:
            self.assertEqual(test_case.nexp, control_case.nexp)

        try:
            test_case.effarea = -1.0
            success += 1
            msg += "was able to assign effarea; "
        except RuntimeError:
            self.assertEqual(test_case.effarea, control_case.effarea)

        try:
            test_case.gain = -1.0
            success += 1
            msg += "was able to assign gain; "
        except RuntimeError:
            self.assertEqual(test_case.gain, control_case.gain)

        try:
            test_case.readnoise = -1.0
            success += 1
            msg += "was able to assign readnoise; "
        except RuntimeError:
            self.assertEqual(test_case.readnoise, control_case.readnoise)

        try:
            test_case.darkcurrent = -1.0
            success += 1
            msg += "was able to assign darkcurrent; "
        except RuntimeError:
            self.assertEqual(test_case.darkcurrent, control_case.darkcurrent)

        try:
            test_case.othernoise = -1.0
            success += 1
            msg += "was able to assign othernoise; "
        except RuntimeError:
            self.assertEqual(test_case.othernoise, control_case.othernoise)

        try:
            test_case.platescale = -1.0
            success += 1
            msg += "was able to assign platescale; "
        except RuntimeError:
            self.assertEqual(test_case.platescale, control_case.platescale)

        try:
            test_case.sigma_sys = -1.0
            success += 1
            msg += "was able to assign sigma_sys; "
        except RuntimeError:
            self.assertEqual(test_case.sigma_sys, control_case.sigma_sys)

        try:
            test_case.bandpass = "z"
            success += 1
            msg += "was able to assign bandpass; "
        except RuntimeError:
            self.assertEqual(test_case.bandpass, control_case.bandpass)

        self.assertEqual(success, 0, msg=msg)

    def test_application(self):
        """
        Test that PhotometricParameters get properly propagated into
        Sed methods.  We will test this using Sed.calc_adu, since the ADU
        scale linearly with the appropriate parameter.
        """

        test_sed = Sed()
        test_sed.set_flat_sed()

        test_bandpass = Bandpass()
        test_bandpass.read_throughput(os.path.join(get_data_dir(), "throughputs", "baseline", "total_g.dat"))

        control = test_sed.calc_adu(test_bandpass, phot_params=PhotometricParameters())

        test_case = PhotometricParameters(exptime=30.0)

        test = test_sed.calc_adu(test_bandpass, phot_params=test_case)

        self.assertGreater(control, 0.0)
        self.assertEqual(control, 0.5 * test)


class PhysicalParametersUnitTest(unittest.TestCase):
    def test_assignment(self):
        """
        Make sure it is impossible to change the values stored in
        PhysicalParameters
        """

        pp = PhysicalParameters()
        control = PhysicalParameters()
        success = 0
        msg = ""

        try:
            pp.lightspeed = 2.0
            success += 1
            msg += "was able to assign lightspeed; "
        except RuntimeError:
            self.assertEqual(pp.lightspeed, control.lightspeed)

        try:
            pp.planck = 2.0
            success += 1
            msg += "was able to assign planck; "
        except RuntimeError:
            self.assertEqual(pp.planck, control.planck)

        try:
            pp.nm2m = 2.0
            success += 1
            msg += "was able to assign nm2m; "
        except RuntimeError:
            self.assertEqual(pp.nm2m, control.nm2m)

        try:
            pp.ergsetc2jansky = 2.0
            msg += "was able to assign ergsetc2jansky; "
            success += 1
        except RuntimeError:
            self.assertEqual(pp.ergsetc2jansky, control.ergsetc2jansky)

        self.assertEqual(success, 0, msg=msg)


if __name__ == "__main__":
    unittest.main()
