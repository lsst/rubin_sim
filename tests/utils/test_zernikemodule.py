import numbers
import unittest

import numpy as np

from rubin_sim.utils import ZernikePolynomialGenerator, _FactorialGenerator


class FactorialTestCase(unittest.TestCase):
    def test_factorial(self):
        gen = _FactorialGenerator()
        ii = gen.evaluate(9)
        ans = 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2
        self.assertEqual(ii, ans)

        ii = gen.evaluate(15)
        ans = 15 * 14 * 13 * 12 * 11 * 10 * 9 * 8 * 7 * 6 * 5 * 4 * 3 * 2
        self.assertEqual(ii, ans)

        ii = gen.evaluate(3)
        ans = 6
        self.assertEqual(ii, ans)

        self.assertEqual(gen.evaluate(0), 1)
        self.assertEqual(gen.evaluate(1), 1)


class ZernikeTestCase(unittest.TestCase):
    long_message = True

    @classmethod
    def setUpClass(cls):
        cls.d_r = 0.01
        cls.d_phi = 0.005 * np.pi
        r_grid = np.arange(0.0, 1.0, cls.d_r)
        phi_grid = np.arange(0.0, 2.0 * np.pi, cls.d_phi)
        mesh = np.meshgrid(r_grid, phi_grid)
        cls.r_grid = mesh[0].flatten()
        cls.phi_grid = mesh[1].flatten()

        r_grid = np.arange(0.0, 1.0, 0.1)
        phi_grid = np.arange(0.0, 2.0 * np.pi, 0.05 * np.pi)
        mesh = np.meshgrid(r_grid, phi_grid)
        cls.r_grid_small = mesh[0].flatten()
        cls.phi_grid_small = mesh[1].flatten()

    def test_orthogonality(self):
        """
        Test that ZernikePolynomialGenerator returns
        polynomials that are orthogonal on the unit disc
        """

        polynomials = {}
        z_gen = ZernikePolynomialGenerator()

        for n in range(3):
            for m in range(-n, n + 1, 2):
                vals = np.zeros(len(self.r_grid), dtype=float)
                for ii, (rr, pp) in enumerate(zip(self.r_grid, self.phi_grid)):
                    vals[ii] = z_gen.evaluate(rr, pp, n, m)
                nm_tuple = (n, m)
                polynomials[nm_tuple] = vals

        p_keys = list(polynomials.keys())
        for ii in range(len(p_keys)):
            p1_name = p_keys[ii]
            p1 = polynomials[p1_name]
            integral = (p1 * p1 * self.r_grid * self.d_r * self.d_phi).sum()
            normed_integral = integral / z_gen.norm(p1_name[0], p1_name[1])
            self.assertLess(np.abs(normed_integral - 1.0), 0.04)
            for jj in range(ii + 1, len(p_keys)):
                p2_name = p_keys[jj]
                p2 = polynomials[p2_name]
                dot = (p1 * p2 * self.r_grid * self.d_r * self.d_phi).sum()
                msg = "\n%s norm %e\n dot %e\n" % (p1_name, integral, dot)
                self.assertLess(np.abs(dot / integral), 0.01, msg=msg)

    def test_zeros(self):
        """
        Test that ZernikePolynomialGenerator returns zero
        when values of n and m require it.
        """
        rng = np.random.RandomState(88)
        z_gen = ZernikePolynomialGenerator()
        for n in range(4):
            for m in range(-(n - 1), n, 2):
                r = rng.random_sample()
                phi = rng.random_sample() * 2.0 * np.pi
                self.assertAlmostEqual(0.0, z_gen.evaluate(r, phi, n, m), 10)

    def test_r_greater_than_one(self):
        """
        Test that the expected error is raised if we try to evaluate
        the Zernike polynomial with r>1
        """
        z_gen = ZernikePolynomialGenerator()
        vv = z_gen.evaluate(1.2, 2.1, 2, 0)
        self.assertTrue(np.isnan(vv))
        vv = z_gen.evaluate(np.array([0.1, 0.5, 1.2]), np.array([0.1, 0.2, 0.3]), 2, -2)
        self.assertTrue(np.isnan(vv[2]))
        self.assertFalse(np.isnan(vv[0]))
        self.assertFalse(np.isnan(vv[1]))
        vv = z_gen.evaluate_xy(1.1, 1.2, 4, -2)
        self.assertTrue(np.isnan(vv))
        vv = z_gen.evaluate_xy(np.array([0.1, 0.2, 0.3]), np.array([0.1, 1.0, 0.1]), 4, 2)
        self.assertTrue(np.isnan(vv[1]))
        self.assertFalse(np.isnan(vv[0]))
        self.assertFalse(np.isnan(vv[2]))

    def test_array(self):
        """
        Test that ZernikePolynomialGenerator can handle arrays of inputs
        """
        z_gen = ZernikePolynomialGenerator()
        n = 2
        m = -2
        val_arr = z_gen.evaluate(self.r_grid_small, self.phi_grid_small, n, m)
        self.assertEqual(len(val_arr), len(self.r_grid_small))
        for ii, (rr, pp) in enumerate(zip(self.r_grid_small, self.phi_grid_small)):
            vv = z_gen.evaluate(rr, pp, n, m)
            self.assertAlmostEqual(vv, val_arr[ii], 14)

    def test_xy(self):
        """
        Test that ZernikePolynomialGenerator can handle Cartesian coordinates
        """
        n = 4
        m = 2
        z_gen = ZernikePolynomialGenerator()
        x = self.r_grid_small * np.cos(self.phi_grid_small)
        y = self.r_grid_small * np.sin(self.phi_grid_small)
        val_arr = z_gen.evaluate_xy(x, y, n, m)
        self.assertGreater(np.abs(val_arr).max(), 1.0e-6)
        for ii, (rr, pp) in enumerate(zip(self.r_grid_small, self.phi_grid_small)):
            vv = z_gen.evaluate(rr, pp, n, m)
            self.assertAlmostEqual(vv, val_arr[ii], 14)

    def test_xy_one_at_a_time(self):
        """
        Test that ZernikePolynomialGenerator can handle
        scalar Cartesian coordinates (as opposed to arrays
        of Cartesian coordinates)
        """
        n = 4
        m = 2
        z_gen = ZernikePolynomialGenerator()
        x = self.r_grid_small * np.cos(self.phi_grid_small)
        y = self.r_grid_small * np.sin(self.phi_grid_small)

        for ii in range(len(self.r_grid_small)):
            vv_r = z_gen.evaluate(self.r_grid_small[ii], self.phi_grid_small[ii], n, m)
            vv_xy = z_gen.evaluate_xy(x[ii], y[ii], n, m)
            self.assertAlmostEqual(vv_r, vv_xy, 14)
            self.assertIsInstance(vv_xy, numbers.Number)

    def test__zernike_origin(self):
        """
        Test that ZernikePolynomialGenerator is well-behaved
        at r=0
        """
        n = 4
        m = 2
        z_gen = ZernikePolynomialGenerator()
        ans = z_gen.evaluate(0.0, 1.2, n, m)
        self.assertEqual(ans, 0.0)
        ans = z_gen.evaluate(np.array([0.0, 0.0]), np.array([1.2, 2.1]), n, m)

        np.testing.assert_array_equal(ans, np.zeros(2, dtype=float))
        ans = z_gen.evaluate_xy(0.0, 0.0, n, m)
        self.assertEqual(ans, 0.0)
        ans = z_gen.evaluate_xy(np.zeros(2, dtype=float), np.zeros(2, dtype=float), n, m)
        np.testing.assert_array_equal(ans, np.zeros(2, dtype=float))

        n = 0
        m = 0
        ans = z_gen.evaluate(0.0, 1.2, n, m)
        self.assertEqual(ans, 1.0)
        ans = z_gen.evaluate(np.array([0.0, 0.0]), np.array([1.2, 2.1]), n, m)

        np.testing.assert_array_equal(ans, np.ones(2, dtype=float))
        ans = z_gen.evaluate_xy(0.0, 0.0, n, m)
        self.assertEqual(ans, 1.0)
        ans = z_gen.evaluate_xy(np.zeros(2, dtype=float), np.zeros(2, dtype=float), n, m)
        np.testing.assert_array_equal(ans, np.ones(2, dtype=float))


if __name__ == "__main__":
    unittest.main()
