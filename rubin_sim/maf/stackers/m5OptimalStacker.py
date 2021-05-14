from __future__ import print_function
import numpy as np
from .baseStacker import BaseStacker
from rubin_sim.utils import Site
from .generalStackers import FiveSigmaStacker

__all__ = ['M5OptimalStacker', 'generate_sky_slopes']


def generate_sky_slopes():
    """
    Fit a line to how the sky brightness changes with airmass.
    """
    import rubin_sim.skybrightness as sb
    import healpy as hp
    sm = sb.SkyModel(mags=True, moon=False, twilight=False, zodiacal=False)
    mjd = 57000
    nside = 32
    airmass_limit = 2.0
    dec, ra = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    dec = np.pi/2 - dec
    sm.setRaDecMjd(ra, dec, mjd)
    mags = sm.returnMags()
    filters = mags.dtype.names
    filter_slopes = {}
    for filterName in filters:
        good = np.where((~np.isnan(mags[filterName])) & (sm.airmass < airmass_limit))
        pf = np.polyfit(sm.airmass[good], mags[filterName][good], 1)
        filter_slopes[filterName] = pf[0]
    print(filter_slopes)


class M5OptimalStacker(BaseStacker):
    """
    Make a new m5 column as if observations were taken on the meridian.
    If the moon is up, assume sky brightness stays the same.

    Assumes seeing scales as airmass^0.6. Uses linear fits for sky and airmass relation.

    Parameters
    ----------
    airmassCol : str ('airmass')
        Column name for the airmass per pointing.
    decCol : str ('dec_rad')
        Column name for the pointing declination.
    skyBrightCol: str ('filtSkyBrightness')
        Column name for the sky brighntess per pointing.
    filterCol : str ('filter')
        Column name for the filter name.
    m5Col : str ('fiveSigmaDepth')
        Colum name for the five sigma limiting depth per pointing.
    moonAltCol : str ('moonAlt')
        Column name for the moon altitude per pointing.
    sunAltCol : str ('sunAltCol')
        Column name for the sun altitude column.
    site : str ('LSST')
        Name of the site.

    Returns
    -------
    numpy.array
        Adds a column to that is approximately what the five-sigma depth would have
        been if the observation had been taken on the meridian.
    """
    colsAdded = ['m5Optimal']

    def __init__(self, airmassCol='airmass', decCol='fieldDec',
                 skyBrightCol='skyBrightness', seeingCol='seeingFwhmEff',
                 filterCol='filter',
                 moonAltCol='moonAlt', sunAltCol='sunAlt',
                 site='LSST'):

        self.site = Site(site)
        self.units = ['mags']
        self.airmassCol = airmassCol
        self.decCol = decCol
        self.skyBrightCol = skyBrightCol
        self.seeingCol = seeingCol
        self.filterCol = filterCol
        self.moonAltCol = moonAltCol
        self.sunAltCol = sunAltCol
        self.m5_stacker = FiveSigmaStacker()
        self.m5Col = self.m5_stacker.colsAdded[0]
        self.colsReq = [airmassCol, decCol, skyBrightCol,
                        seeingCol, filterCol, moonAltCol, sunAltCol]
        self.colsReq.extend(self.m5_stacker.colsReq)
        self.colsReq = list(set(self.colsReq))

    def _run(self, simData, cols_present=False):
        simData, m5col_present = self.m5_stacker._addStackerCols(simData)
        simData = self.m5_stacker._run(simData, m5col_present)
        # kAtm values from rubin_sim.operations gen_output.py
        kAtm = {'u': 0.50, 'g': 0.21, 'r': 0.13, 'i': 0.10,
                'z': 0.07, 'y': 0.18}
        # Linear fits to sky brightness change, no moon, twilight, or zodiacal components
        # Use generate_sky_slopes to regenerate if needed.
        skySlopes = {'g': -0.52611780327408397, 'i': -0.67898669252082422,
                     'r': -0.61378749766766827, 'u': -0.27840980367303503,
                     'y': -0.69635091524779691, 'z': -0.69652846002009128}
        min_z_possible = np.abs(np.radians(simData[self.decCol]) - self.site.latitude_rad)
        min_airmass_possible = 1./np.cos(min_z_possible)
        for filterName in np.unique(simData[self.filterCol]):
            deltaSky = skySlopes[filterName]*(simData[self.airmassCol] - min_airmass_possible)
            deltaSky[np.where((simData[self.moonAltCol] > 0) |
                              (simData[self.sunAltCol] > np.radians(-18.)))] = 0
            # Using Approximation that FWHM~X^0.6. So seeing term in m5 of: 0.25 * log (7.0/FWHMeff )
            # Goes to 0.15 log(FWHM_min / FWHM_eff) in the difference
            m5Optimal = (simData[self.m5Col] - 0.5*deltaSky -
                         0.15*np.log10(min_airmass_possible / simData[self.airmassCol]) -
                         kAtm[filterName]*(min_airmass_possible - simData[self.airmassCol]))
            good = np.where(simData[self.filterCol] == filterName)
            simData['m5Optimal'][good] = m5Optimal[good]
        return simData
