import unittest
import rubin_sim.skybrightness as sb
import rubin_sim.skybrightness_pre as sbp
import healpy as hp
import numpy as np
import rubin_sim.utils as utils
import ephem
from rubin_sim.skybrightness.utils import mjd2djd
from rubin_sim.utils import _angularSeparation, raDec2Hpid, angularSeparation
import rubin_sim.site_models as sf
import warnings


class TestSkyPre(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        try:
            cls.sm = sbp.SkyModelPre(speedLoad=True)
            # cls.sm_fields = sbp.SkyModelPre(speedLoad=True, opsimFields=False, useSimsData=False)
            mjd = cls.sm.info['mjds'][1]+4./60./24.
            tmp = cls.sm.returnMags(mjd)
            cls.data_present = True
        except:
            cls.data_present = False
            warnings.warn('Data files not found, skipping tests. Check data/ for instructions to pull data.')

    def testReturnMags(self):
        """
        Test all the ways ReturnMags can be used
        """
        # Check both the healpix and opsim fields
        if self.data_present:
            sms = [self.sm]
            mjds = []
            for mjd in sms[0].info['mjds'][100:102]:
                mjds.append(mjd)
                mjds.append(mjd+.0002)

            # Make sure there's an mjd that is between sunrise/set that gets tested
            diff = sms[0].info['mjds'][1:] - sms[0].info['mjds'][0:-1]
            between = np.where(diff >= sms[0].header['timestep_max'])[0][0]
            mjds.append(sms[0].info['mjds'][between+1] + sms[0].header['timestep_max'])

            indxes = [None, [100, 101]]
            apply_masks = [True, False]
            apply_planets = [True, False]
            filters = [['u', 'g', 'r', 'i', 'z', 'y'], ['r']]

            for sm in sms:
                for mjd in mjds:
                    for indx in indxes:
                        for am in apply_masks:
                            for planet in apply_planets:
                                for filt in filters:
                                    mags = sm.returnMags(mjd, indx=indx, airmass_mask=am, filters=filt,
                                                         planet_mask=planet)
                                    # Check the filters returned are correct
                                    self.assertEqual(len(filt), len(list(mags.keys())))
                                    self.assertEqual(set(filt), set(mags.keys()))
                                    airmasses = sm.returnAirmass(mjd, indx=indx)
                                    # Check the magnitudes are correct
                                    if indx is not None:
                                        self.assertEqual(np.size(mags[list(mags.keys())[0]]), np.size(indx))
                                        self.assertEqual(np.size(airmasses), np.size(indx))

    @unittest.skip("13 March 2017--Takes a long time to load the data")
    def testCrazyDate(self):
        """
        Test date that falls at akward time
        """
        if self.data_present:
            mjd = 60291.35423611111
            sm = self.sm
            mags = sm.returnMags(mjd)
            sunmoon = sm.returnSunMoon(mjd)
            airmass = sm.returnAirmass(mjd)

            goodVals = np.where(mags['g'] != hp.UNSEEN)[0]
            assert(len(goodVals) > 0)
            for key in sunmoon:
                goodVals = np.where(sunmoon[key] != hp.UNSEEN)[0]
                assert(len(goodVals) > 0)
            goodVals = np.where(airmass != hp.UNSEEN)[0]
            assert(len(goodVals) > 0)

    def testSunMoon(self):
        """
        Test that the sun moon interpolation is good enough
        """
        if self.data_present:
            sm = self.sm
            telescope = utils.Site('LSST')
            Observatory = ephem.Observer()
            Observatory.lat = telescope.latitude_rad
            Observatory.lon = telescope.longitude_rad
            Observatory.elevation = telescope.height

            sun = ephem.Sun()
            moon = ephem.Moon()

            mjd1 = sm.info['mjds'][0]
            mjd2 = sm.info['mjds'][3]

            mjds = np.linspace(mjd1, mjd2, 20)

            # Demand Moon and Sun Positions match to within 3 arcmin
            arcmin_places = np.abs(np.floor(np.log10(3/60./180.*np.pi))).astype(int)

            for mjd in mjds:
                Observatory.date = mjd2djd(mjd)
                sun.compute(Observatory)
                moon.compute(Observatory)
                pre_calced = sm.returnSunMoon(mjd)

                self.assertLess(np.abs(pre_calced['sunAlt']-sun.alt), arcmin_places)
                sun_dist = _angularSeparation(sun.ra, sun.dec, pre_calced['sunRA'], pre_calced['sunDec'])
                self.assertAlmostEqual(sun_dist, 0., places=arcmin_places)

                self.assertLess(np.abs(pre_calced['moonAlt']-moon.alt), arcmin_places)
                moon_dist = _angularSeparation(moon.ra, moon.dec, pre_calced['moonRA'], pre_calced['moonDec'])
                self.assertAlmostEqual(moon_dist, 0., places=arcmin_places)

                self.assertAlmostEqual(np.radians(pre_calced['moonSunSep']),
                                       np.radians(moon.phase/100.*180.), places=arcmin_places)

    def testSBP(self):
        """
        Check that values are similar enough
        """
        if self.data_present:
            original_model = sb.SkyModel(mags=True)
            pre_calc_model = self.sm

            hpindx = np.arange(hp.nside2npix(pre_calc_model.header['nside']))
            ra, dec = utils.hpid2RaDec(pre_calc_model.header['nside'], hpindx)

            # Run through a number of mjd values
            step = 30. / 60. / 24.  # 30 minute timestep
            nmjd = 48
            mjds = np.arange(nmjd)*step + self.sm.info['mjds'][10]+0.1

            # Tolerance for difference between true and interpolated airmass
            am_tol = 0.05
            am_limit = 3.

            # Where to check the magnitudes match
            mag_am_limit = 1.5
            mag_tol = 0.27  # mags

            for mjd in mjds:
                original_model.setRaDecMjd(ra, dec, mjd, degrees=True)
                if original_model.sunAlt < np.radians(-12.):
                    sky1 = original_model.returnMags()
                    sky2 = pre_calc_model.returnMags(mjd)
                    am1 = original_model.airmass
                    am2 = pre_calc_model.returnAirmass(mjd)
                    good_am = np.where((am1 >= 1.) & (am1 <= am_limit))
                    diff = am1[good_am] - am2[good_am]
                    # Check that the interpolated airmass is close
                    assert(np.max(np.abs(diff)) < am_tol)

                    for filtername in sky1:
                        good = np.where((am1 < mag_am_limit) & (sky2[filtername] != hp.UNSEEN) &
                                        (np.isfinite(sky1[filtername])))
                        diff = sky1[filtername][good] - sky2[filtername][good]
                        assert(np.max(np.abs(diff)) <= mag_tol)

   @unittest.skip("Don't want to add sims_data as dependency, and this does a large file load too")
    def test_various(self):
        """
        Test some various loading things
        """
        # check that the sims_data stuff loads
        sm = sbp.SkyModelPre(speedLoad=True)
        mjd = self.sm.info['mjds'][10]+0.1
        mags = sm.returnMags(mjd)
        # check that it'll load up something later properly
        mags = sm.returnMags(60000)


if __name__ == "__main__":
    unittest.main()
