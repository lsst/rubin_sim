import unittest

import numpy as np

from rubin_sim.moving_objects import chebeval, chebfit, make_cheb_matrix


class TestChebgrid(unittest.TestCase):
    def test_raise_error(self):
        x = np.linspace(-1, 1, 9)
        y = np.sin(x)
        dy = np.cos(x)
        p, resid, rms, maxresid = chebfit(x, y, dy, n_poly=4)
        with self.assertRaises(RuntimeError):
            chebeval(np.linspace(-1, 1, 17), p, interval=[1, 2, 3])

    def test_eval(self):
        x = np.linspace(-1, 1, 9)
        y = np.sin(x)
        dy = np.cos(x)
        p, resid, rms, maxresid = chebfit(x, y, dy, n_poly=4)
        yy_w_vel, vv = chebeval(np.linspace(-1, 1, 17), p)
        yy_wout_vel, vv = chebeval(np.linspace(-1, 1, 17), p, do_velocity=False)
        self.assertTrue(np.allclose(yy_wout_vel, yy_w_vel))
        # Test that we get a nan for a value outside the range of the
        # 'interval', if mask=True
        yy_w_vel, vv = chebeval(np.linspace(-2, 1, 17), p, mask=True)
        self.assertTrue(
            np.isnan(yy_w_vel[0]),
            msg="Expected NaN for masked/out of range value, but got %.2e" % (yy_w_vel[0]),
        )

    def test_ends_locked(self):
        x = np.linspace(-1, 1, 9)
        y = np.sin(x)
        dy = np.cos(x)
        for polynomial in range(4, 10):
            p, resid, rms, maxresid = chebfit(x, y, dy, n_poly=4)
            yy, vv = chebeval(np.linspace(-1, 1, 17), p)
            self.assertAlmostEqual(yy[0], y[0], places=13)
            self.assertAlmostEqual(yy[-1], y[-1], places=13)
            self.assertAlmostEqual(vv[0], dy[0], places=13)
            self.assertAlmostEqual(vv[-1], dy[-1], places=13)

    def test_accuracy(self):
        """If n_poly is  greater than number of values being fit,
        then fit should be exact."""
        x = np.linspace(0, np.pi, 9)
        y = np.sin(x)
        dy = np.cos(x)
        p, resid, rms, maxresid = chebfit(x, y, dy, n_poly=16)
        yy, vv = chebeval(x, p, interval=np.array([0, np.pi]))
        self.assertTrue(np.allclose(yy, y, rtol=1e-13))
        self.assertTrue(np.allclose(vv, dy, rtol=1e-13))
        self.assertLess(np.sum(resid), 1e-13)

    def test_accuracy_prefit_c1c2(self):
        """If n_poly is  greater than number of values being fit,
        then fit should be exact."""
        NPOINTS = 8
        NPOLY = 16
        x = np.linspace(0, np.pi, NPOINTS + 1)
        y = np.sin(x)
        dy = np.cos(x)
        xmatrix, dxmatrix = make_cheb_matrix(NPOINTS + 1, NPOLY)
        p, resid, rms, maxresid = chebfit(
            x, y, dy, x_multiplier=xmatrix, dx_multiplier=dxmatrix, n_poly=NPOLY
        )
        yy, vv = chebeval(x, p, interval=np.array([0, np.pi]))
        self.assertTrue(np.allclose(yy, y, rtol=1e-13))
        self.assertTrue(np.allclose(vv, dy, rtol=1e-13))
        self.assertLess(np.sum(resid), 1e-13)


if __name__ == "__main__":
    unittest.main()
