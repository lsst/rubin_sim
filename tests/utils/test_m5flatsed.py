import unittest

from rubin_sim.utils import m5_flat_sed


class PhotoM5Test(unittest.TestCase):
    def testm5(self):
        filters = ["u", "g", "r", "i", "z", "y"]
        kwargs = {}
        # List all parameters to test, with better conditions first
        kwargs["musky"] = [23.0, 22.0]
        kwargs["fwhm_eff"] = [1.0, 1.5]
        kwargs["exp_time"] = [60.0, 30.0]
        kwargs["airmass"] = [1.0, 2.2]
        kwargs["tau_cloud"] = [0.0, 2.2]

        k_default = {}
        for key in kwargs:
            k_default[key] = kwargs[key][0]

        for filtername in filters:
            m5_baseline = m5_flat_sed(filtername, **k_default)
            for key in kwargs:
                k_new = k_default.copy()
                k_new[key] = kwargs[key][1]
                m5_new = m5_flat_sed(filtername, **k_new)
                assert m5_new < m5_baseline


if __name__ == "__main__":
    unittest.main()
