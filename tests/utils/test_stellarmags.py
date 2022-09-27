import unittest
import rubin_sim.utils as utils
import numpy as np


class StellarMagsTest(unittest.TestCase):
    """
    Test the example stellar colors code
    """

    def test_sm(self):
        keys = [
            "O",
            "B",
            "A",
            "F",
            "G",
            "K",
            "M",
            "HeWD_25200_80",
            "WD_11000_85",
            "WD_3000_85",
        ]
        filter_names = ["u", "g", "r", "i", "z", "y"]

        # Check each type returns the correct format
        for key in keys:
            result = utils.stellarMags(key)
            for fn in filter_names:
                self.assertIn(fn, result)
                self.assertTrue(
                    (isinstance(result[fn], float))
                    | (isinstance(result[fn], np.float64)),
                    msg="result is neither a float nor a numpy float64",
                )

        # Check the exception gets raised
        self.assertRaises(ValueError, utils.stellarMags, "ack")

        # Check the mags get fainter
        for st in keys:
            mags = utils.stellarMags(st)
            mags2 = utils.stellarMags(st, rmag=20.0)
        for key in mags:
            self.assertLess(mags[key], mags2[key])


if __name__ == "__main__":
    unittest.main()
