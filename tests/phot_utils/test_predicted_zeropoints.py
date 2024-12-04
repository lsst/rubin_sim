import unittest

import numpy as np
from rubin_scheduler.utils import SysEngVals

from rubin_sim.phot_utils import (
    predicted_zeropoint,
    predicted_zeropoint_e2v,
    predicted_zeropoint_hardware,
    predicted_zeropoint_hardware_e2v,
    predicted_zeropoint_hardware_itl,
    predicted_zeropoint_itl,
)


class PredictedZeropointsTst(unittest.TestCase):
    def test_predicted_zeropoints(self):
        bands = ["u", "g", "r", "i", "z", "y"]
        sev = SysEngVals()
        for b in bands:
            zp = predicted_zeropoint(band=b, airmass=1.0, exptime=1)
            self.assertAlmostEqual(zp, sev.zp_t[b], delta=0.005)
            zp_hardware = predicted_zeropoint_hardware(b, exptime=1)
            self.assertTrue(zp < zp_hardware)
            # Check the vendors
            zp_v = predicted_zeropoint_itl(band=b, airmass=1.0, exptime=1)
            self.assertAlmostEqual(zp, zp_v, delta=0.1)
            zp_v = predicted_zeropoint_e2v(band=b, airmass=1.0, exptime=1)
            self.assertAlmostEqual(zp, zp_v, delta=0.1)
            zp_v = predicted_zeropoint_hardware_itl(band=b, exptime=1)
            self.assertAlmostEqual(zp_hardware, zp_v, delta=0.1)
            zp_v = predicted_zeropoint_hardware_e2v(band=b, exptime=1)
            self.assertAlmostEqual(zp_hardware, zp_v, delta=0.1)
            # Check some of the scaling
            zp_test = predicted_zeropoint(band=b, airmass=1.5, exptime=1)
            self.assertTrue(zp > zp_test)
            zp_test = predicted_zeropoint(band=b, airmass=1.0, exptime=30)
            self.assertAlmostEqual(zp, zp_test - 2.5 * np.log10(30), places=7)

            funcs = [predicted_zeropoint, predicted_zeropoint_itl, predicted_zeropoint_e2v]
            for zpfunc in funcs:
                zp = []
                for x in np.arange(1.0, 2.5, 0.1):
                    for exptime in np.arange(1.0, 130, 30):
                        zp.append(zpfunc(b, x, exptime))
                zp = np.array(zp)
                self.assertTrue(zp.max() - zp.min() < 6)
                self.assertTrue(zp.max() < 35)
                self.assertTrue(zp.min() > 25)


if __name__ == "__main__":
    unittest.main()
