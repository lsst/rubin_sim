import warnings
import numpy as np
import palpy
from rubin_sim.utils import Site, m5_flat_sed
from .baseStacker import BaseStacker

__all__ = ['NormAirmassStacker', 'ParallaxFactorStacker', 'HourAngleStacker',
           'FilterColorStacker', 'ZenithDistStacker', 'ParallacticAngleStacker',
           'DcrStacker', 'FiveSigmaStacker', 'SaturationStacker']


class SaturationStacker(BaseStacker):
    """Calculate the saturation limit of a point source. Assumes Guassian PSF.
    
    Parameters
    ----------
    pixscale : float, optional (0.2)
        Arcsec per pixel
    gain : float, optional (2.3)
        electrons per adu
    saturation_e : float, optional (150e3)
        The saturation level in electrons
    zeropoints : dict-like, optional (None)
        The zeropoints for the telescope. Keys should be str with filter names, values in mags.
        If None, will use Rubin-like zeropoints.
    km : dict-like, optional (None)
        Atmospheric extinction values.  Keys should be str with filter names.
        If None, will use Rubin-like zeropoints.
    """
    colsAdded = ['saturation_mag']

    def __init__(self, seeingCol='seeingFwhmEff', skybrightnessCol='skyBrightness',
                 exptimeCol='visitExposureTime', nexpCol='numExposures',
                 filterCol='filter', airmassCol='airmass',
                 saturation_e=150e3, zeropoints=None, km=None, pixscale=0.2, gain=1.0):
        self.units = ['mag']
        self.colsReq = [seeingCol, skybrightnessCol, exptimeCol, nexpCol, filterCol, airmassCol]
        self.seeingCol = seeingCol
        self.skybrightnessCol = skybrightnessCol
        self.exptimeCol = exptimeCol
        self.nexpCol = nexpCol
        self.filterCol = filterCol
        self.airmassCol = airmassCol
        self.saturation_adu = saturation_e/gain
        self.pixscale = 0.2
        names = ['u', 'g', 'r', 'i', 'z', 'y']
        types = [float]*6
        if zeropoints is None:
            # Note these zeropoints are calculating the number of *electrons* per second (thus gain=1)
            # https://github.com/lsst-pst/syseng_throughputs/blob/master/notebooks/Syseng%20Throughputs%20Repo%20Demo.ipynb
            self.zeropoints = np.array([27.03, 28.38, 28.15, 27.86, 27.46, 26.68]).view(list(zip(names, types)))
            self.saturation_adu = saturation_e 
        else:
            self.zeropoints = zeropoints

        if km is None:
            # Also from notebook above
            self.km = np.array([0.491, 0.213, 0.126, 0.096, 0.069, 0.170]).view(list(zip(names, types)))
        else:
            self.km = km

    def _run(self, simData, cols_present=False):
        for filtername in np.unique(simData[self.filterCol]):
            in_filt = np.where(simData[self.filterCol] == filtername)[0]
            # Calculate the length of the on-sky time per EXPOSURE
            exptime = simData[self.exptimeCol][in_filt] / simData[self.nexpCol][in_filt]
            # Calculate sky counts per pixel per second from skybrightness + zeropoint (e/1s)
            sky_counts = 10.**(0.4*(self.zeropoints[filtername]
                                    - simData[self.skybrightnessCol][in_filt])) * self.pixscale**2
            # Total sky counts in each exposure 
            sky_counts = sky_counts * exptime
            # The counts available to the source (at peak) in each exposure is the
            # difference between saturation and sky
            remaining_counts_peak = (self.saturation_adu - sky_counts)
            # Now to figure out how many counts there would be total, if there are that many in the peak
            sigma = simData[self.seeingCol][in_filt]/2.354
            source_counts = remaining_counts_peak * 2.*np.pi*(sigma/self.pixscale)**2
            # source counts = counts per exposure (expTimeCol / nexp)
            # Translate to counts per second, to apply zeropoint 
            count_rate = source_counts / exptime
            simData['saturation_mag'][in_filt] = -2.5*np.log10(count_rate) + self.zeropoints[filtername]
            # Airmass correction
            simData['saturation_mag'][in_filt] -= self.km[filtername]*(simData[self.airmassCol][in_filt] - 1.)

        return simData


class FiveSigmaStacker(BaseStacker):
    """
    Calculate the 5-sigma limiting depth for a point source in the given conditions.

    This is generally not needed, unless the m5 parameters have been updated
    or m5 was not previously calculated.
    """
    colsAdded = ['m5_simsUtils']

    def __init__(self, airmassCol='airmass', seeingCol='seeingFwhmEff', skybrightnessCol='skyBrightness',
                 filterCol='filter', exptimeCol='visitExposureTime'):
        self.units = ['mag']
        self.colsReq = [airmassCol, seeingCol, skybrightnessCol, filterCol, exptimeCol]
        self.airmassCol = airmassCol
        self.seeingCol = seeingCol
        self.skybrightnessCol = skybrightnessCol
        self.filterCol = filterCol
        self.exptimeCol = exptimeCol

    def _run(self, simData, cols_present=False):
        if cols_present:
            # Column already present in data; assume it needs updating and recalculate.
            return simData
        filts = np.unique(simData[self.filterCol])
        for filtername in filts:
            infilt = np.where(simData[self.filterCol] == filtername)
            simData['m5_simsUtils'][infilt] = m5_flat_sed(filtername,
                                                          simData[infilt][self.skybrightnessCol],
                                                          simData[infilt][self.seeingCol],
                                                          simData[infilt][self.exptimeCol],
                                                          simData[infilt][self.airmassCol])
        return simData


class NormAirmassStacker(BaseStacker):
    """Calculate the normalized airmass for each opsim pointing.
    """
    colsAdded = ['normairmass']

    def __init__(self, airmassCol='airmass', decCol='fieldDec',
                 degrees=True, telescope_lat = -30.2446388):
        self.units = ['X / Xmin']
        self.colsReq = [airmassCol, decCol]
        self.airmassCol = airmassCol
        self.decCol = decCol
        self.telescope_lat = telescope_lat
        self.degrees = degrees

    def _run(self, simData, cols_present=False):
        """Calculate new column for normalized airmass."""
        # Run method is required to calculate column.
        # Driver runs getColInfo to know what columns are needed from db & which are calculated,
        #  then gets data from db and then calculates additional columns (via run methods here).
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        dec = simData[self.decCol]
        if self.degrees:
            dec = np.radians(dec)
        min_z_possible = np.abs(dec - np.radians(self.telescope_lat))
        min_airmass_possible = 1./np.cos(min_z_possible)
        simData['normairmass'] = simData[self.airmassCol] / min_airmass_possible
        return simData


class ZenithDistStacker(BaseStacker):
    """Calculate the zenith distance for each pointing.
    If 'degrees' is True, then assumes altCol is in degrees and returns degrees.
    If 'degrees' is False, assumes altCol is in radians and returns radians.
    """
    colsAdded = ['zenithDistance']

    def __init__(self, altCol='altitude', degrees=True):
        self.altCol = altCol
        self.degrees = degrees
        if self.degrees:
            self.units = ['degrees']
        else:
            self.unit = ['radians']
        self.colsReq = [self.altCol]

    def _run(self, simData, cols_present=False):
        """Calculate new column for zenith distance."""
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        if self.degrees:
            simData['zenithDistance'] = 90.0 - simData[self.altCol]
        else:
            simData['zenithDistance'] = np.pi/2.0 - simData[self.altCol]
        return simData


class ParallaxFactorStacker(BaseStacker):
    """Calculate the parallax factors for each opsim pointing.  Output parallax factor in arcseconds.
    """
    colsAdded = ['ra_pi_amp', 'dec_pi_amp']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', dateCol='observationStartMJD', degrees=True):
        self.raCol = raCol
        self.decCol = decCol
        self.dateCol = dateCol
        self.units = ['arcsec', 'arcsec']
        self.colsReq = [raCol, decCol, dateCol]
        self.degrees = degrees

    def _gnomonic_project_toxy(self, RA1, Dec1, RAcen, Deccen):
        """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccenp.
        Input radians.
        """
        # also used in Global Telescope Network website
        cosc = np.sin(Deccen) * np.sin(Dec1) + np.cos(Deccen) * np.cos(Dec1) * np.cos(RA1-RAcen)
        x = np.cos(Dec1) * np.sin(RA1-RAcen) / cosc
        y = (np.cos(Deccen)*np.sin(Dec1) - np.sin(Deccen)*np.cos(Dec1)*np.cos(RA1-RAcen)) / cosc
        return x, y

    def _run(self, simData, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        ra_pi_amp = np.zeros(np.size(simData), dtype=[('ra_pi_amp', 'float')])
        dec_pi_amp = np.zeros(np.size(simData), dtype=[('dec_pi_amp', 'float')])
        ra_geo1 = np.zeros(np.size(simData), dtype='float')
        dec_geo1 = np.zeros(np.size(simData), dtype='float')
        ra_geo = np.zeros(np.size(simData), dtype='float')
        dec_geo = np.zeros(np.size(simData), dtype='float')
        ra = simData[self.raCol]
        dec = simData[self.decCol]
        if self.degrees:
            ra = np.radians(ra)
            dec = np.radians(dec)

        for i, ack in enumerate(simData):
            mtoa_params = palpy.mappa(2000., simData[self.dateCol][i])
            # Object with a 1 arcsec parallax
            ra_geo1[i], dec_geo1[i] = palpy.mapqk(ra[i], dec[i],
                                                  0., 0., 1., 0., mtoa_params)
            # Object with no parallax
            ra_geo[i], dec_geo[i] = palpy.mapqk(ra[i], dec[i],
                                                0., 0., 0., 0., mtoa_params)
        x_geo1, y_geo1 = self._gnomonic_project_toxy(ra_geo1, dec_geo1,
                                                     ra, dec)
        x_geo, y_geo = self._gnomonic_project_toxy(ra_geo, dec_geo, ra, dec)
        # Return ra_pi_amp and dec_pi_amp in arcseconds.
        ra_pi_amp[:] = np.degrees(x_geo1-x_geo)*3600.
        dec_pi_amp[:] = np.degrees(y_geo1-y_geo)*3600.
        simData['ra_pi_amp'] = ra_pi_amp
        simData['dec_pi_amp'] = dec_pi_amp
        return simData


class DcrStacker(BaseStacker):
    """Calculate the RA,Dec offset expected for an object due to differential chromatic refraction.

    For DCR calculation, we also need zenithDistance, HA, and PA -- but these will be explicitly
    handled within this stacker so that setup is consistent and they run in order. If those values
    have already been calculated elsewhere, they will not be overwritten.

    Parameters
    ----------
    filterCol : str
        The name of the column with filter names. Default 'fitler'.
    altCol : str
        Name of the column with altitude info. Default 'altitude'.
    raCol : str
        Name of the column with RA. Default 'fieldRA'.
    decCol : str
        Name of the column with Dec. Default 'fieldDec'.
    lstCol : str
        Name of the column with local sidereal time. Default 'observationStartLST'.
    site : str or rubin_sim.utils.Site
        Name of the observory or a rubin_sim.utils.Site object. Default 'LSST'.
    mjdCol : str
        Name of column with modified julian date. Default 'observationStartMJD'
    dcr_magnitudes : dict
        Magitude of the DCR offset for each filter at altitude/zenith distance of 45 degrees.
        Defaults u=0.07, g=0.07, r=0.50, i=0.045, z=0.042, y=0.04 (all arcseconds).

    Returns
    -------
    numpy.array
        Returns array with additional columns 'ra_dcr_amp' and 'dec_dcr_amp' with the DCR offsets
        for each observation.  Also runs ZenithDistStacker and ParallacticAngleStacker.
    """
    colsAdded = ['ra_dcr_amp', 'dec_dcr_amp']  # zenithDist, HA, PA

    def __init__(self, filterCol='filter', altCol='altitude', degrees=True,
                 raCol='fieldRA', decCol='fieldDec', lstCol='observationStartLST',
                 site='LSST', mjdCol='observationStartMJD',
                 dcr_magnitudes=None):
        self.units = ['arcsec', 'arcsec']
        if dcr_magnitudes is None:
            # DCR amplitudes are in arcseconds.
            self.dcr_magnitudes = {'u': 0.07, 'g': 0.07, 'r': 0.050, 'i': 0.045, 'z': 0.042, 'y': 0.04}
        else:
            self.dcr_magnitudes = dcr_magnitudes
        self.zdCol = 'zenithDistance'
        self.paCol = 'PA'
        self.filterCol = filterCol
        self.raCol = raCol
        self.decCol = decCol
        self.degrees = degrees
        self.colsReq = [filterCol, raCol, decCol, altCol, lstCol]
        #  'zenithDist', 'PA', 'HA' are additional columns required, coming from other stackers which must
        #  also be configured -- so we handle this explicitly here.
        self.zstacker = ZenithDistStacker(altCol=altCol, degrees=self.degrees)
        self.pastacker = ParallacticAngleStacker(raCol=raCol, decCol=decCol, mjdCol=mjdCol,
                                                 degrees=self.degrees,
                                                 lstCol=lstCol, site=site)
        # Note that RA/Dec could be coming from a dither stacker!
        # But we will assume that coord stackers will be handled separately.


    def _run(self, simData, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        # Need to make sure the Zenith stacker gets run first
        # Call _run method because already added these columns due to 'colsAdded' line.
        simData = self.zstacker.run(simData)
        simData = self.pastacker.run(simData)
        if self.degrees:
            zenithTan = np.tan(np.radians(simData[self.zdCol]))
            parallacticAngle = np.radians(simData[self.paCol])
        else:
            zenithTan = np.tan(simData[self.zdCol])
            parallacticAngle = simData[self.paCol]
        dcr_in_ra = zenithTan * np.sin(parallacticAngle)
        dcr_in_dec = zenithTan * np.cos(parallacticAngle)
        for filtername in np.unique(simData[self.filterCol]):
            fmatch = np.where(simData[self.filterCol] == filtername)
            dcr_in_ra[fmatch] = self.dcr_magnitudes[filtername] * dcr_in_ra[fmatch]
            dcr_in_dec[fmatch] = self.dcr_magnitudes[filtername] * dcr_in_dec[fmatch]
        simData['ra_dcr_amp'] = dcr_in_ra
        simData['dec_dcr_amp'] = dcr_in_dec
        return simData


class HourAngleStacker(BaseStacker):
    """Add the Hour Angle for each observation.
    Always in HOURS.
    """
    colsAdded = ['HA']

    def __init__(self, lstCol='observationStartLST', raCol='fieldRA', degrees=True):
        self.units = ['Hours']
        self.colsReq = [lstCol, raCol]
        self.lstCol = lstCol
        self.raCol = raCol
        self.degrees = degrees

    def _run(self, simData, cols_present=False):
        """HA = LST - RA """
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        if len(simData) == 0:
            return simData
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            lst = np.radians(simData[self.lstCol])
        else:
            ra = simData[self.raCol]
            lst = simData[self.lstCol]
        # Check that LST is reasonable
        if (np.min(lst) < 0) | (np.max(lst) > 2.*np.pi):
            warnings.warn('LST values are not between 0 and 2 pi')
        # Check that RA is reasonable
        if (np.min(ra) < 0) | (np.max(ra) > 2.*np.pi):
            warnings.warn('RA values are not between 0 and 2 pi')
        ha = lst - ra
        # Wrap the results so HA between -pi and pi
        ha = np.where(ha < -np.pi, ha + 2. * np.pi, ha)
        ha = np.where(ha > np.pi, ha - 2. * np.pi, ha)
        # Convert radians to hours
        simData['HA'] = ha*12/np.pi
        return simData


class ParallacticAngleStacker(BaseStacker):
    """Add the parallactic angle to each visit.
    If 'degrees' is True, this will be in degrees (as are all other angles). If False, then in radians.
    """
    colsAdded = ['PA']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, mjdCol='observationStartMJD',
                 lstCol='observationStartLST', site='LSST'):

        self.lstCol = lstCol
        self.raCol = raCol
        self.decCol = decCol
        self.degrees = degrees
        self.mjdCol = mjdCol
        self.site = Site(name=site)
        self.units = ['radians']
        self.colsReq = [self.raCol, self.decCol, self.mjdCol, self.lstCol]
        self.haStacker = HourAngleStacker(lstCol=lstCol, raCol=raCol, degrees=self.degrees)

    def _run(self, simData, cols_present=False):
        # Equation from:
        # http://www.gb.nrao.edu/~rcreager/GBTMetrology/140ft/l0058/gbtmemo52/memo52.html
        # or
        # http://www.gb.nrao.edu/GBT/DA/gbtidl/release2pt9/contrib/contrib/parangle.pro
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        # Using the run method (not _run) means that if HA is present, it will not be recalculated.
        simData = self.haStacker.run(simData)
        if self.degrees:
            dec = np.radians(simData[self.decCol])
        else:
            dec = simData[self.decCol]
        simData['PA'] = np.arctan2(np.sin(simData['HA']*np.pi/12.), (np.cos(dec) *
                                   np.tan(self.site.latitude_rad) - np.sin(dec) *
                                   np.cos(simData['HA']*np.pi/12.)))
        if self.degrees:
            simData['PA'] = np.degrees(simData['PA'])
        return simData


class FilterColorStacker(BaseStacker):
    """Translate filters ('u', 'g', 'r' ..) into RGB tuples.

    This is useful for making movies if you want to make the pointing have a related color-tuple for a plot.
    """
    colsAdded = ['rRGB', 'gRGB', 'bRGB']

    def __init__(self, filterCol='filter'):
        self.filter_rgb_map = {'u': (0, 0, 1),   # dark blue
                               'g': (0, 1, 1),  # cyan
                               'r': (0, 1, 0),    # green
                               'i': (1, 0.5, 0.3),  # orange
                               'z': (1, 0, 0),    # red
                               'y': (1, 0, 1)}  # magenta
        self.filterCol = filterCol
        # self.units used for plot labels
        self.units = ['rChan', 'gChan', 'bChan']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.filterCol]

    def _run(self, simData, cols_present=False):
        # Translate filter names into numbers.
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        filtersUsed = np.unique(simData[self.filterCol])
        for f in filtersUsed:
            if f not in self.filter_rgb_map:
                raise IndexError('Filter %s not in filter_rgb_map' % (f))
            match = np.where(simData[self.filterCol] == f)[0]
            simData['rRGB'][match] = self.filter_rgb_map[f][0]
            simData['gRGB'][match] = self.filter_rgb_map[f][1]
            simData['bRGB'][match] = self.filter_rgb_map[f][2]
        return simData
