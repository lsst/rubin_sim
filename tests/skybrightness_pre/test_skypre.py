import unittest
import warnings

import healpy as hp
import numpy as np

import rubin_sim.skybrightness as sb
import rubin_sim.skybrightness_pre as sbp
import rubin_sim.utils as utils


class TestSkyPre(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.sm = sbp.SkyModelPre(init_load_length=3, load_length=3)
            mjd = cls.sm.mjds[1] + 4.0 / 60.0 / 24.0
            tmp = cls.sm.return_mags(mjd)
            cls.nside = hp.npix2nside(tmp["r"].size)
            cls.data_present = True
        except:
            cls.data_present = False
            warnings.warn("Data files not found, skipping tests. Check data/ for instructions to pull data.")

    def test_return_mags(self):
        """
        Test all the ways ReturnMags can be used
        """
        timestep_max = 15.0 / 60.0 / 24.0
        # Check both the healpix and opsim fields
        if self.data_present:
            sms = [self.sm]
            mjds = []
            for mjd in sms[0].mjds[100:102]:
                mjds.append(mjd)
                mjds.append(mjd + 0.0002)

            # Make sure there's an mjd that is between sunrise/set that gets tested
            diff = sms[0].mjds[1:] - sms[0].mjds[0:-1]
            between = np.where(diff >= timestep_max)[0][0]
            mjds.append(sms[0].mjds[between + 1] + timestep_max)

            indxes = [None, [100, 101]]
            filters = [["u", "g", "r", "i", "z", "y"], ["r"]]

            for sm in sms:
                for mjd in mjds:
                    for indx in indxes:
                        for filt in filters:
                            mags = sm.return_mags(mjd, indx=indx, filters=filt)
                            # Check the filters returned are correct
                            self.assertEqual(len(filt), len(list(mags.keys())))
                            self.assertEqual(set(filt), set(mags.keys()))
                            # Check the magnitudes are correct
                            if indx is not None:
                                self.assertEqual(
                                    np.size(mags[list(mags.keys())[0]]),
                                    np.size(indx),
                                )

    def test_sbp(self):
        """
        Check that values are similar enough
        """
        if self.data_present:
            original_model = sb.SkyModel(mags=True)
            pre_calc_model = self.sm

            hpindx = np.arange(hp.nside2npix(self.nside))
            ra, dec = utils.hpid2_ra_dec(self.nside, hpindx)

            # Run through a number of mjd values
            step = 30.0 / 60.0 / 24.0  # 30 minute timestep
            nmjd = 48
            mjds = np.arange(nmjd) * step + self.sm.mjds[10] + 0.1

            # Where to check the magnitudes match
            mag_am_limit = 1.5
            mag_tol = 0.27  # mags

            for mjd in mjds:
                original_model.set_ra_dec_mjd(ra, dec, mjd, degrees=True)
                if original_model.sun_alt < np.radians(-12.0):
                    sky1 = original_model.return_mags()
                    sky2 = pre_calc_model.return_mags(mjd)
                    am1 = original_model.airmass

                    for filtername in sky1:
                        good = np.where(
                            (am1 < mag_am_limit)
                            & (sky2[filtername] != hp.UNSEEN)
                            & (np.isfinite(sky1[filtername]))
                        )
                        diff = sky1[filtername][good] - sky2[filtername][good]
                        assert np.max(np.abs(diff)) <= mag_tol

    def test_various(self):
        """
        Test some various loading things
        """
        # check that the sims_data stuff loads
        sm = sbp.SkyModelPre(init_load_length=3)
        mjd = self.sm.mjds[10] + 0.1
        mags = sm.return_mags(mjd)


if __name__ == "__main__":
    unittest.main()
