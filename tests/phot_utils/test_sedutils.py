import unittest
import numpy as np
from rubin_sim.phot_utils import Sed, Bandpass, get_imsim_flux_norm


class ImSimNormTestCase(unittest.TestCase):
    def test_norm(self):
        """
        Test that the special test case getImsimFluxNorm
        returns the same value as calling calc_flux_norm actually
        passing in the imsim Bandpass
        """

        bp = Bandpass()
        bp.imsim_bandpass()

        rng = np.random.RandomState(1123)
        wavelen = np.arange(300.0, 2000.0, 0.17)

        for ix in range(10):
            flux = rng.random_sample(len(wavelen)) * 100.0
            sed = Sed()
            sed.set_sed(wavelen=wavelen, flambda=flux)
            magmatch = rng.random_sample() * 5.0 + 10.0

            control = sed.calc_flux_norm(magmatch, bp)
            test = get_imsim_flux_norm(sed, magmatch)

            # something about how interpolation is done in Sed means
            # that the values don't come out exactly equal.  They come
            # out equal to 8 seignificant digits, though.
            self.assertEqual(control, test)


if __name__ == "__main__":
    unittest.main()
