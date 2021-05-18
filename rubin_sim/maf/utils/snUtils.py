import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import numpy.lib.recfunctions as rf


class Lims:
    """
    class to handle light curve of SN

    Parameters
    ---------------
    Li_files : str
       light curve reference file
    mag_to_flux_files : str
       files of magnitude to flux
    band : str
        band considered
    SNR : float
        Signal-To-Noise Ratio cut
    mag_range : pair(float),opt
        mag range considered
        Default : (23., 27.5)
    dt_range : pair(float)
        difference time range considered (cadence)
        Default : (0.5, 25.)
    """

    def __init__(self, Li_files, mag_to_flux_files, band, SNR,
                 mag_range=(23., 27.5), dt_range=(0.5, 25.)):

        self.band = band
        self.SNR = SNR
        self.lims = []
        self.mag_to_flux = []
        self.mag_range = mag_range
        self.dt_range = dt_range

        for val in Li_files:
            self.lims.append(self.get_lims(self.band, np.load(val), SNR))
        for val in mag_to_flux_files:
            self.mag_to_flux.append(np.load(val))
        self.interp()

    def get_lims(self, band, tab, SNR):
        """
        Estimations of the limits

        Parameters
        ---------------
        band : str
          band to consider
        tab : numpy array
          table of data
        SNR : float
           Signal-to-Noise Ratio cut

        Returns:
        -----------
        dict of limits with redshift and band as keys.

        """

        lims = {}

        for z in np.unique(tab['z']):

            idx = (tab['z'] == z) & (tab['band'] == 'LSST::'+band)
            idx &= (tab['flux_e'] > 0.)
            sel = tab[idx]

            if len(sel) > 0:
                li2 = np.sqrt(np.sum(sel['flux_e']**2))
                lim = 5. * li2 / SNR
                if z not in lims.keys():
                    lims[z] = {}
                lims[z][band] = lim

        return lims

    def mesh(self, mag_to_flux):
        """
        Mesh grid to estimate five-sigma depth values (m5) from mags input.

        Parameters
        ---------------
        mag_to_flux : magnitude to flux values

        Returns
        -----------
        m5 values
        time difference dt (cadence)
        metric=sqrt(dt)*F5 where F5 is the 5-sigma flux

        """
        dt = np.linspace(self.dt_range[0], self.dt_range[1], 100)
        m5 = np.linspace(self.mag_range[0], self.mag_range[1], 50)
        ida = mag_to_flux['band'] == self.band
        fa = interpolate.interp1d(
            mag_to_flux[ida]['m5'], mag_to_flux[ida]['flux_e'])
        f5 = fa(m5)
        F5, DT = np.meshgrid(f5, dt)
        M5, DT = np.meshgrid(m5, dt)
        metric = np.sqrt(DT) * F5

        return M5, DT, metric

    def interp(self):
        """
        Estimate a grid of interpolated values
        in the plane (m5, cadence, metric)

        Parameters
        ---------------
        None

        """

        M5_all = []
        DT_all = []
        metric_all = []

        for val in self.mag_to_flux:
            M5, DT, metric = self.mesh(val)
            M5_all.append(M5)
            DT_all.append(DT)
            metric_all.append(metric)

        sorted_keys = []
        for i in range(len(self.lims)):
            sorted_keys.append(np.sort([k for k in self.lims[i].keys()])[::-1])
        figa, axa = plt.subplots()

        for kk, lim in enumerate(self.lims):
            fmt = {}
            ll = [lim[zz][self.band] for zz in sorted_keys[kk]]
            cs = axa.contour(M5_all[kk], DT_all[kk], metric_all[kk], ll)

            points_values = None
            for io, col in enumerate(cs.collections):
                if col.get_segments():

                    myarray = col.get_segments()[0]
                    res = np.array(myarray[:, 0], dtype=[('m5', 'f8')])
                    res = rf.append_fields(res, 'cadence', myarray[:, 1])
                    res = rf.append_fields(
                        res, 'z', [sorted_keys[kk][io]]*len(res))
                    if points_values is None:
                        points_values = res
                    else:
                        points_values = np.concatenate((points_values, res))
            self.points_ref = points_values

        plt.close(figa)  # do not display

    def interp_griddata(self, data):
        """
        Estimate metric interpolation for data (m5,cadence)

        Parameters
        ---------------
        data : data where interpolation has to be done (m5,cadence)

        Returns
        -----------
        griddata interpolation (m5,cadence,metric)

        """

        ref_points = self.points_ref
        res = interpolate.griddata((ref_points['m5'], ref_points['cadence']), ref_points['z'], (
            data['m5_mean'], data['cadence_mean']), method='cubic')
        return res


class GenerateFakeObservations:
    """ Class to generate Fake observations

    Parameters
    ---------
    config: yaml-like
       configuration file (parameter choice: filter, cadence, m5,Nseasons, ...)
    list : str,opt
        Name of the columns used.
        Default : 'observationStartMJD', 'fieldRA', 'fieldDec','filter','fiveSigmaDepth','visitExposureTime','numExposures','visitTime','season'

    Returns
    ---------
    recordarray of observations with the fields:
    MJD, Ra, Dec, band,m5,Nexp, ExpTime, Season
    """

    def __init__(self, config,
                 mjdCol='observationStartMJD', RaCol='fieldRA',
                 DecCol='fieldDec', filterCol='filter', m5Col='fiveSigmaDepth',
                 exptimeCol='visitExposureTime', nexpCol='numExposures', seasonCol='season'):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = seasonCol
        self.nexpCol = nexpCol

        # now make fake obs
        self.make_fake(config)

    def make_fake(self, config):
        """ Generate Fake observations

        Parameters
        ---------
        config: yaml-like
          configuration file (parameter choice: filter, cadence, m5,Nseasons, ...)


        """
        bands = config['bands']
        cadence = dict(zip(bands, config['Cadence']))
        shift_days = dict(
            zip(bands, [config['shift_days']*io for io in range(len(bands))]))
        m5 = dict(zip(bands, config['m5']))
        Nvisits = dict(zip(bands, config['Nvisits']))
        Exposure_Time = dict(zip(bands, config['Exposure_Time']))
        inter_season_gap = 300.

        Ra = config['Ra']
        Dec = config['Dec']
        rtot = []
        # for season in range(1, config['nseasons']+1):
        for il, season in enumerate(config['seasons']):
            # mjd_min = config['MJD_min'] + float(season-1)*inter_season_gap
            mjd_min = config['MJD_min'][il]
            mjd_max = mjd_min+config['season_length']

            for i, band in enumerate(bands):
                mjd = np.arange(mjd_min, mjd_max+cadence[band], cadence[band])
                mjd += shift_days[band]
                m5_coadded = self.m5_coadd(m5[band],
                                           Nvisits[band],
                                           Exposure_Time[band])
                myarr = np.array(mjd, dtype=[(self.mjdCol, 'f8')])
                myarr = rf.append_fields(myarr, [self.RaCol, self.DecCol, self.filterCol], [
                                         [Ra]*len(myarr), [Dec]*len(myarr), [band]*len(myarr)])
                myarr = rf.append_fields(myarr, [self.m5Col, self.nexpCol, self.exptimeCol, self.seasonCol], [
                                         [m5_coadded]*len(myarr), [Nvisits[band]]*len(myarr), [Nvisits[band]*Exposure_Time[band]]*len(myarr), [season]*len(myarr)])
                rtot.append(myarr)

        res = np.copy(np.concatenate(rtot))
        res.sort(order=self.mjdCol)

        self.Observations = res

    def m5_coadd(self, m5, Nvisits, Tvisit):
        """ Coadded m5 estimation

        Parameters
        ---------
        m5 : list(float)
           list of five-sigma depth values
         Nvisits : list(float)
           list of the number of visits
          Tvisit : list(float)
           list of the visit times

       Returns
        ---------
       m5_coadd : list(float)
          list of m5 coadded values

        """
        m5_coadd = m5+1.25*np.log10(float(Nvisits)*Tvisit/30.)
        return m5_coadd


class ReferenceData:
    """
    class to handle light curve of SN

    Parameters
    ---------------
    Li_files : str
      light curve reference file
    mag_to_flux_files : str
      files of magnitude to flux
    band : str
      band considered
    z : float
      redshift considered
    """

    def __init__(self, Li_files, mag_to_flux_files, band, z):

        self.band = band
        self.z = z
        self.fluxes = []
        self.mag_to_flux = []

        for val in Li_files:
            self.fluxes.append(self.interp_fluxes(
                self.band, np.load(val), self.z))
        for val in mag_to_flux_files:
            self.mag_to_flux.append(
                self.interp_mag(self.band, np.load(val)))

    def interp_fluxes(self, band, tab, z):
        """
        Flux interpolator

        Parameters
        ---------------
        band : str
           band considered
        tab : array
           reference data with (at least) fields z,band,time,DayMax
        z : float
         redshift considered

        Returns
        -----
        list (float) of interpolated fluxes (in e/sec)
        """
        lims = {}
        idx = (np.abs(tab['z'] - z) < 1.e-5) & (tab['band'] == 'LSST::'+band)
        sel = tab[idx]
        selc = np.copy(sel)
        difftime = (sel['time']-sel['DayMax'])
        selc = rf.append_fields(selc, 'deltaT', difftime)
        return interpolate.interp1d(selc['deltaT'], selc['flux_e'], bounds_error=False, fill_value=0.)

    def interp_mag(self, band, tab):
        """
        magnitude (m5) to flux (e/sec) interpolator

        Parameters
        ---------------
        band : str
           band considered
        tab : array
           reference data with (at least) fields band,m5,flux_e,
        z : float
         redshift considered

        Returns
        -----
        list (float) of interpolated fluxes (in e/sec)
        """
        idx = tab['band'] == band
        sel = tab[idx]
        return interpolate.interp1d(sel['m5'], sel['flux_e'], bounds_error=False, fill_value=0.)
