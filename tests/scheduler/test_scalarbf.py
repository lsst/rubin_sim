import unittest
import healpy as hp
import numpy as np
from rubin_sim.scheduler.features import Conditions
from rubin_sim.scheduler.utils import set_default_nside
from rubin_sim.scheduler.basis_functions import HealpixLimitedBasisFunctionMixin
from rubin_sim.scheduler.basis_functions import BaseBasisFunction


class SimpleArrayBasisFunction(BaseBasisFunction):
    def __init__(self, value, *args, **kwargs):
        self.assigned_value = value
        super().__init__(*args, **kwargs)

    def _calc_value(self, conditions, **kwargs):
        self.value = self.assigned_value
        return self.value


class SimpleArrayAtHpixBasisFunction(HealpixLimitedBasisFunctionMixin, SimpleArrayBasisFunction):
    pass


class TestHealpixLimitedBasisFunctionMixin(unittest.TestCase):
    def setUp(self):
        self.hpid = 2111
        random_seed = 6563
        self.nside = set_default_nside()
        self.npix = hp.nside2npix(self.nside)
        self.rng = np.random.default_rng(seed=random_seed)
        self.all_values = self.rng.random(self.npix)
        self.value = self.all_values[self.hpid]
        self.conditions = Conditions(nside=self.nside, mjd=60200.2)

    def test_array_data(self):
        basis_function = SimpleArrayBasisFunction(self.all_values)
        test_values = basis_function(self.conditions)
        self.assertTrue(np.array_equal(test_values, self.all_values))

    def test_scalar_data(self):
        basis_function = SimpleArrayAtHpixBasisFunction(self.hpid, self.all_values)
        test_value = basis_function(self.conditions)
        self.assertEqual(test_value, self.value)

        feasible = basis_function(self.conditions)
        self.assertTrue(feasible)

    def test_infeasible_at_hpix(self):
        values_invalid_at_hpix = self.all_values.copy()
        values_invalid_at_hpix[self.hpid] = -np.inf
        basis_function = SimpleArrayAtHpixBasisFunction(self.hpid, values_invalid_at_hpix)
        feasible = basis_function.check_feasibility(self.conditions)
        self.assertFalse(feasible)


if __name__ == "__main__":
    unittest.main()
