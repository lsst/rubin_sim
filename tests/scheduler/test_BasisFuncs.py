import numpy as np
import unittest
import rubin_sim.scheduler.basis_functions as basis_functions
from rubin_sim.scheduler.utils import empty_observation
from rubin_sim.scheduler.features import Conditions


class TestBasis(unittest.TestCase):
    def testVisit_repeat_basis_function(self):
        bf = basis_functions.Visit_repeat_basis_function()

        indx = np.array([1000])

        # 30 minute step
        delta = 30.0 / 60.0 / 24.0

        # Add 1st observation, should still be zero
        obs = empty_observation()
        obs["filter"] = "r"
        obs["mjd"] = 59000.0
        conditions = Conditions()
        conditions.mjd = np.max(obs["mjd"])
        bf.add_observation(obs, indx=indx)
        self.assertEqual(np.max(bf(conditions)), 0.0)

        # Advance time so now we want a pair
        conditions.mjd += delta
        self.assertEqual(np.max(bf(conditions)), 1.0)

        # Now complete the pair and it should go back to zero
        bf.add_observation(obs, indx=indx)

        conditions.mjd += delta
        self.assertEqual(np.max(bf(conditions)), 0.0)


if __name__ == "__main__":
    unittest.main()
