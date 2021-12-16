import numpy as np
from astropy.io import ascii, fits
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u
import rubin_sim.maf as maf
from .baseMetric import BaseMetric
from .simpleMetrics import Coaddm5Metric
from .starDensity import StarDensityMetric
from maf.mafContrib.LSSObsStrategy.galaxyCountsMetric_extended import GalaxyCountsMetric_extended \
    as GalaxyCountsMetric

# make fake LF for old galaxy of given integrated B, distance modulus mu, in any of filters ugrizY
def makeFakeLF(intB, mu, filtername):
    if (filtername == 'y'):
        filtername == 'Y'
    modelBmag = 6.856379  # integrated B mag of the model LF being read
    LF = ascii.read('LF_-1.5_10Gyr.dat', header_start=12)
    mags = LF['magbinc']
    counts = LF[filtername+'mag']
    # shift model LF to requested distance and dim it
    mags = mags + mu
    modelBmag = modelBmag + mu
    # scale model counts up/down to reach the requested intB
    factor = np.power(10.0, -0.4*(intB-modelBmag))
    counts = factor * counts
    return mags, counts


def make_LF_dicts():
    lf_dict_i = {}
    lf_dict_g = {}
    tmp_MB = -10.0

    for i in range(101):
        mbkey = f'MB{tmp_MB:.2f}'
        iLFmags, iLFcounts = makeFakeLF(tmp_MB, 0.0, 'i')
        lf_dict_i[mbkey] = (np.array(iLFmags), np.array(iLFcounts))
        gLFmags, gLFcounts = makeFakeLF(tmp_MB, 0.0, 'g')
        lf_dict_g[mbkey] = (np.array(gLFmags), np.array(gLFcounts))
        tmp_MB += 0.1

    return lf_dict_g, lf_dict_i


def sum_luminosity(LFmags, LFcounts):
    magref = LFmags[0]
    totlum = 0.0

    for mag, count in zip(LFmags, LFcounts):
        tmpmags = np.repeat(mag, count)
        totlum += np.sum(10.0**((magref - tmpmags)/2.5))

    mtot = magref-2.5*np.log10(totlum)
    return mtot


def sblimit(glim, ilim, nstars, distlim):
    distance_limit = distlim*1e6  # distance limit in parsecs
    distmod_lim = 5.0*np.log10(distance_limit) - 5.0

    if (glim > 15) and (ilim > 15):
        # print(glim, ilim, nstars)
        fake_MB = -10.0
        ng = 1e6
        ni = 1e6

        while (ng > nstars) and (ni > nstars) and fake_MB < -2.0:
            # B_fake = distmod_limit+fake_MB
            mbkey = f'MB{fake_MB:.2f}'
            iLFmags0, iLFcounts0 = lf_dict_i[mbkey]
            gLFmags0, gLFcounts0 = lf_dict_g[mbkey]
            iLFcounts = np.random.poisson(iLFcounts0)
            gLFcounts = np.random.poisson(gLFcounts0)
            iLFmags = iLFmags0+distmod_lim  # Add the distance modulus to make it apparent mags
            gLFmags = gLFmags0+distmod_lim  # Add the distance modulus to make it apparent mags
            # print(iLFcounts0-iLFcounts)
            gsel = (gLFmags <= glim)
            isel = (iLFmags <= ilim)
            ng = np.sum(gLFcounts[gsel])
            ni = np.sum(iLFcounts[isel])
            # print('fake_MB: ',fake_MB, ' ng: ',ng, ' ni: ', ni, ' nstars: ', nstars)
            fake_MB += 0.1

        if fake_MB > -9.9:
            gmag_tot = sum_luminosity(gLFmags[gsel], gLFcounts[gsel]) - distmod_lim
            imag_tot = sum_luminosity(iLFmags[isel], iLFcounts[isel]) - distmod_lim
            # S = m + 2.5logA, where in this case things are in sq. arcmin, so A = 1 arcmin^2 = 3600 arcsec^2
            sbtot_g = distmod_lim + gmag_tot + 2.5*np.log10(3600.0)
            sbtot_i = distmod_lim + imag_tot + 2.5*np.log10(3600.0)
            mg_lim = gmag_tot
            mi_lim = imag_tot
            sbg_lim = sbtot_g
            sbi_lim = sbtot_i
            if (ng < ni):
                flag_lim = 'g'
            else:
                flag_lim = 'i'
        else:
            mg_lim = 999.9
            mi_lim = 999.9
            sbg_lim = 999.9
            sbi_lim = 999.9
            flag_lim = 'none'
    else:
        mg_lim = 999.9
        mi_lim = 999.9
        sbg_lim = -999.9
        sbi_lim = -999.9
        flag_lim = 'none'

    return mg_lim, mi_lim, sbg_lim, sbi_lim, flag_lim


class MyMetricInProgress(maf.BaseMetric):
    """Documentation please. Numpy style docstrings.

    This example metric just finds the time of first observation of a particular part of the sky.

    Parameters
    ----------
    specificColumns : `str`, opt
        It's nice to be flexible about what the relevant columns are called, so specify them here.
        seeingCol = FWHMeff, etc.
    kwargs : `float`, ?
        Probably there are other things you need to set?
    """
    # def __init__(self, nside=16, cmd_frac=0.1, stargal_contamination=0.40, nsigma=10.0, **kwargs):
    def __init__(self, radius=2.45, distlim=None, cmd_frac=0.1, stargal_contamination=0.40, nsigma=10.0, **kwargs):
        # maps = ["CoaddM5"]
        # self.mjdCol = mjdCol
        # self.nside = nside
        self.radius = radius
        self.filterCol = "filter"
        self.m5Col = "fiveSigmaDepth"
        self.cmd_frac = cmd_frac
        self.stargal_contamination = stargal_contamination
        self.nsigma = nsigma

        if distlim is not None:
            self.distlim = distlim
        else:
            self.distlim = None
            lv_dat0 = fits.getdata('lsst_galaxies_1p25to9Mpc_table.fits')
            # Keep only galaxies at dec < 35 deg., and with stellar masses > 10^7 M_Sun.
            lv_dat_cuts = (lv_dat0['dec'] < 35.0) & (lv_dat0['MStars'] > 1e7) & (lv_dat0['MStars'] < 1e14)
            lv_dat = lv_dat0[lv_dat_cuts]
            sc_dat = SkyCoord(ra=lv_dat['ra']*u.deg, dec=lv_dat['dec']*u.deg, distance=lv_dat['dist_Mpc']*u.Mpc)
            self.sc_dat = sc_dat

        self.Coaddm5Metric = maf.simpleMetrics.Coaddm5Metric(m5Col=self.m5Col)
        self.StarDensityMetric24 = maf.starDensity.StarDensityMetric(rmagLimit=24)
        self.StarDensityMetric24p5 = maf.starDensity.StarDensityMetric(rmagLimit=24.5)
        self.StarDensityMetric25 = maf.starDensity.StarDensityMetric(rmagLimit=25)
        self.StarDensityMetric25p5 = maf.starDensity.StarDensityMetric(rmagLimit=25.5)
        self.StarDensityMetric26 = maf.starDensity.StarDensityMetric(rmagLimit=26)
        self.StarDensityMetric26p5 = maf.starDensity.StarDensityMetric(rmagLimit=26.5)
        self.StarDensityMetric27 = maf.starDensity.StarDensityMetric(rmagLimit=27)
        self.GalaxyCountsMetric = maf.mafContrib.LSSObsStrategy.galaxyCountsMetric_extended.GalaxyCountsMetric_extended(m5Col=self.m5Col)
        # self.GalaxyCountsMetric = maf.mafContrib.LSSObsStrategy.galaxyCountsMetric_extended.GalaxyCountsMetric_extended(m5Col=self.m5Col, nside=self.nside)
        cols = [self.m5Col, self.filterCol] # Add any columns that your metric needs to run -- mjdCol is just an example
        maps = ['DustMap', 'StellarDensityMap']
        super().__init__(col=cols, maps=maps, units='#', **kwargs)

    def run(self, dataSlice, slicePoint=None):
        # This is where you write what your metric does.
        # dataSlice == the numpy recarray containing the pointing information,
        # with the columns that you said you needed in 'cols'
        # slicePoint == the information about where you're evaluating this on the sky -- ra/dec,
        # and if you specified that you need a dustmap or stellar density map, etc., those values will also
        # be defined here

        # here's a super simple example .. replace with your own code to calculate your metric values
        # tMin = dataSlice[self.mjdCol].min()
        rband = (dataSlice[self.filterCol] == 'r')
        gband = (dataSlice[self.filterCol] == 'g')
        iband = (dataSlice[self.filterCol] == 'i')
        r5 = self.Coaddm5Metric.run(dataSlice[rband])
        g5 = self.Coaddm5Metric.run(dataSlice[gband])
        i5 = self.Coaddm5Metric.run(dataSlice[iband])
        nstar24 = self.StarDensityMetric24.run(dataSlice, slicePoint)
        nstar24p5 = self.StarDensityMetric24p5.run(dataSlice, slicePoint)
        nstar25 = self.StarDensityMetric25.run(dataSlice, slicePoint)
        nstar25p5 = self.StarDensityMetric25p5.run(dataSlice, slicePoint)
        nstar26 = self.StarDensityMetric26.run(dataSlice, slicePoint)
        nstar26p5 = self.StarDensityMetric26p5.run(dataSlice, slicePoint)
        nstar27 = self.StarDensityMetric27.run(dataSlice, slicePoint)
        nstar = nstar27

        # import pdb; pdb.set_trace()

        if 'nside' in slicePoint.keys():
            nside = slicePoint['nside']
            try:
                ngal = self.GalaxyCountsMetric.run(dataSlice, slicePoint, nside=nside)
            except:
                ngal = 1e7
                # print('healpix ',slicePoint['sid'], 'failed in GalaxyCountsMetric')
        else:
            ngal = self.GalaxyCountsMetric.run(dataSlice, slicePoint)


        # print(dataSlice)
        nstar_all = nstar*0.0
        rbinvals = np.arange(24.0, 27.5, 0.5)
        rbinlimits = [nstar24, nstar24p5, nstar25, nstar25p5, nstar26, nstar26p5, nstar27]

        # Star density is number of stars per square arcsec.
        # Convert to a total number per healpix, then number per sq. arcmin:
        # First, get the slicer pixel area.
        # import pdb; pdb.set_trace()

        '''
        # Is this correct? How do I ensure that the metrics are using this healpix scale?
        nside = self.nside

        # Calculate the factor to go from number per healpix to number per square arcminute or per square arcsec
        pixarea_deg = hp.nside2pixarea(nside, degrees=True)*(u.degree**2)
        pixarea_arcmin = pixarea_deg.to(u.arcmin**2)
        pixarea_arcsec = pixarea_deg.to(u.arcsec**2)

        nstar_all_per_healpix = nstar_all*pixarea_arcsec.value
        nstar_all_per_arcmin = nstar_all_per_healpix/pixarea_arcmin.value

        # Number of galaxies is the total in each healpix. Convert to number per sq. arcmin:
        ngal_per_arcmin = ngal/pixarea_arcmin.value

        # Star density is number of stars per square arcsec.
        # Convert to a total number per healpix, then number per sq. arcmin:
        nstar_per_healpix = nstar*pixarea_arcsec.value
        nstar_per_arcmin = nstar_per_healpix/pixarea_arcmin.value
        '''

        nstar0 = rbinlimits[np.argmin(np.abs(rbinvals-r5))]

        if 'nside' in slicePoint.keys():
            nside = slicePoint['nside']
            # Calculate the factor to go from number per healpix to number per square arcminute or per square arcsec
            area_deg = hp.nside2pixarea(nside, degrees=True)*(u.degree**2)
            area_arcmin = area_deg.to(u.arcmin**2)
            area_arcsec = area_deg.to(u.arcsec**2)
        else:
            area_arcsec = np.pi*((self.radius*u.deg).to(u.arcsec)**2)
            area_arcmin = np.pi*((self.radius*u.deg).to(u.arcmin)**2)

        nstar_all = nstar0*area_arcsec.value

        ngal_per_arcmin = ngal/area_arcmin.value
        nstar_all_per_arcmin = nstar_all/area_arcmin.value

        nstars_required = self.nsigma*np.sqrt(ngal_per_arcmin*(self.cmd_frac*self.stargal_contamination)+(nstar_all_per_arcmin*self.cmd_frac))

        # Add a check so that if healpix slicer is used, distlim is also ***required***
        if self.distlim is not None:
            distlim = self.distlim
        else:
            sc_slice = SkyCoord(ra=slicePoint['ra']*u.rad, dec=slicePoint['dec']*u.rad)
            seps = sc_slice.separation(self.sc_dat)
            distlim = self.sc_dat[seps.argmin()].distance

        # How do I use the distances from the lv_dat catalog here?
        mg_lim, mi_lim, sb_g_lim, sb_i_lim, flag_lim = sblimit(g5, i5, nstars_required, distlim=distlim.value)
        # mg_lim, mi_lim, sb_g_lim, sb_i_lim, flag_lim = sblimit(g5, i5, nstars_required, distlim=lv_dat['dist_Mpc'])

        # Use the conversion from Appendix A of Komiyama+2018, ApJ, 853, 29:
        # V = g_hsc - 0.371*(gi_hsc)-0.068
        mv = mg_lim - 0.371 * (mg_lim - mi_lim) - 0.068
        # sbv = sb_g_lim - 0.371 * (sb_g_lim - sb_i_lim) - 0.068

        # import pdb; pdb.set_trace()

        return mv

####################

######




lf_dict_g, lf_dict_i = make_LF_dicts()

lv_dat0 = fits.getdata('lsst_galaxies_1p25to9Mpc_table.fits')

# Keep only galaxies at dec < 35 deg., and with stellar masses > 10^7 M_Sun.

lv_dat_cuts = (lv_dat0['dec'] < 35.0) & (lv_dat0['MStars'] > 1e7) & (lv_dat0['MStars'] < 1e14)
lv_dat = lv_dat0[lv_dat_cuts]

slicer = maf.UserPointsSlicer(lv_dat['ra'], lv_dat['dec'])
