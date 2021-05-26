import numpy as np
import unittest

from rubin_sim.movingObjects import chebfit, makeChebMatrix, makeChebMatrixOnlyX, chebeval


class TestChebgrid(unittest.TestCase):

    def setUp(self):
        self.mbaDict = {}
        """
        # Skip because pickle doesn't work for py2 + py3 and don't have good solution yet.
        for day in range(2, 4):
            self.mbaDict[day] = pickle.load(open(os.path.join(getPackageDir('sims_movingObjects'),
                                                              "tests/cheb_testdata/day%d.pkl" % (day)),
                                                  "rb"))
        """

    def test_raise_error(self):
        x = np.linspace(-1, 1, 9)
        y = np.sin(x)
        dy = np.cos(x)
        p, resid, rms, maxresid = chebfit(x, y, dy, nPoly=4)
        with self.assertRaises(RuntimeError):
            chebeval(np.linspace(-1, 1, 17), p, interval=[1, 2, 3])

    def test_eval(self):
        x = np.linspace(-1, 1, 9)
        y = np.sin(x)
        dy = np.cos(x)
        p, resid, rms, maxresid = chebfit(x, y, dy, nPoly=4)
        yy_wVel, vv = chebeval(np.linspace(-1, 1, 17), p)
        yy_woutVel, vv = chebeval(np.linspace(-1, 1, 17), p, doVelocity=False)
        self.assertTrue(np.allclose(yy_woutVel, yy_wVel))
        # Test that we get a nan for a value outside the range of the 'interval', if mask=True
        yy_wVel, vv = chebeval(np.linspace(-2, 1, 17), p, mask=True)
        self.assertTrue(np.isnan(yy_wVel[0]),
                        msg='Expected NaN for masked/out of range value, but got %.2e' % (yy_wVel[0]))

    def test_ends_locked(self):
        x = np.linspace(-1, 1, 9)
        y = np.sin(x)
        dy = np.cos(x)
        for polynomial in range(4, 10):
            p, resid, rms, maxresid = chebfit(x, y, dy, nPoly=4)
            yy, vv = chebeval(np.linspace(-1, 1, 17), p)
            self.assertAlmostEqual(yy[0], y[0], places=13)
            self.assertAlmostEqual(yy[-1], y[-1], places=13)
            self.assertAlmostEqual(vv[0], dy[0], places=13)
            self.assertAlmostEqual(vv[-1], dy[-1], places=13)

    def test_accuracy(self):
        """If nPoly is  greater than number of values being fit, then fit should be exact.
        """
        x = np.linspace(0, np.pi, 9)
        y = np.sin(x)
        dy = np.cos(x)
        p, resid, rms, maxresid = chebfit(x, y, dy, nPoly=16)
        yy, vv = chebeval(x, p, interval=np.array([0, np.pi]))
        self.assertTrue(np.allclose(yy, y, rtol=1e-13))
        self.assertTrue(np.allclose(vv, dy, rtol=1e-13))
        self.assertLess(np.sum(resid), 1e-13)

    def test_accuracy_prefit_c1c2(self):
        """If nPoly is  greater than number of values being fit, then fit should be exact.
        """
        NPOINTS = 8
        NPOLY = 16
        x = np.linspace(0, np.pi, NPOINTS + 1)
        y = np.sin(x)
        dy = np.cos(x)
        xmatrix, dxmatrix = makeChebMatrix(NPOINTS + 1, NPOLY)
        p, resid, rms, maxresid = chebfit(x, y, dy, xMultiplier=xmatrix, dxMultiplier=dxmatrix, nPoly=NPOLY)
        yy, vv = chebeval(x, p, interval=np.array([0, np.pi]))
        self.assertTrue(np.allclose(yy, y, rtol=1e-13))
        self.assertTrue(np.allclose(vv, dy, rtol=1e-13))
        self.assertLess(np.sum(resid), 1e-13)

    """
    def test_real_mba(self):
        for day in range(2, 4):
            p, dec_resid, dec_rms, maxresid = chebfit(self.mbaDict[day]['t'],
                                                      self.mbaDict[day]['dec'],
                                                      self.mbaDict[day]['ddecdt'],
                                                      nPoly=self.mbaDict[day]['coeff'])
            rap, ra_resid, ra_rms, maxresid = chebfit(self.mbaDict[day]['t'],
                                                      self.mbaDict[day]['ra'],
                                                      (self.mbaDict[day]['dradt'] /
                                                       np.cos(np.pi * self.mbaDict[day]['dec'] / 180.)),
                                                      nPoly=self.mbaDict[day]['coeff'])
            self.assertTrue(np.allclose(self.mbaDict[day]['rap'], rap))
            self.assertTrue(np.allclose(self.mbaDict[day]['ra_resid'], ra_resid))
            self.assertTrue(np.allclose(self.mbaDict[day]['ra_rms'], ra_rms))
            self.assertTrue(np.allclose(self.mbaDict[day]['p'], p))
            self.assertTrue(np.allclose(self.mbaDict[day]['dec_resid'], dec_resid))
            self.assertTrue(np.allclose(self.mbaDict[day]['dec_rms'], dec_rms))
    """


if __name__ == '__main__':
    unittest.main()
