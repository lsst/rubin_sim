import numpy as np
import matplotlib.pylab as plt
import numpy.lib.recfunctions as rf
from scipy import interpolate
import rubin_sim.maf.metrics as metrics
from rubin_sim.maf.utils.snUtils import GenerateFakeObservations
from collections.abc import Iterable
import time

__all__ = ['SNSNRMetric']

class SNSNRMetric(metrics.BaseMetric):
    """
    Metric to estimate the detection rate for faint supernovae (x1,color) = (-2.0,0.2)

    Parameters
    ----------
    list : str, optional
        Name of the columns used to estimate the metric
        Default : 'observationStartMJD', 'fieldRA', 'fieldDec','filter','fiveSigmaDepth',
        'visitExposureTime','night','observationId', 'numExposures','visitTime'
    coadd :  bool, optional
        to make "coaddition" per night (uses snStacker)
        Default : True
    lim_sn : class, optional
        Reference data used to simulate LC points (interpolation)
    names_ref : str, optional
        names of the simulator used to produce reference data
    season : flota, optional
        season num
        Default : 1.
    z : float, optional
        redshift for this study
        Default : 0.01
    """

    def __init__(self, metricName='SNSNRMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', coadd=True, lim_sn=None, names_ref=None, season=1, z=0.01, **kwargs):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = 'season'
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol]
        if coadd:
            cols += ['coadd']
        super(SNSNRMetric, self).__init__(
            col=cols, metricName=metricName, **kwargs)

        self.filterNames = np.array(['u', 'g', 'r', 'i', 'z', 'y'])
        self.blue_cutoff = 300.
        self.red_cutoff = 800.
        self.min_rf_phase = -20.
        self.max_rf_phase = 40.
        self.z = z
        self.names_ref = names_ref
        self.season = season

        # SN DayMax: current date - shift days
        self.shift = 10.

        # These are reference LC
        self.lim_sn = lim_sn

        self.display = False

    def run(self, dataSlice, slicePoint=None):
        """
        run the metric

        Parameters
        ----------
        dataSlice : array
          simulation data under study

        Returns
        -------
        detection rate : float

        """
        time_ref = time.time()
        goodFilters = np.in1d(dataSlice['filter'], self.filterNames)
        dataSlice = dataSlice[goodFilters]
        if dataSlice.size == 0:
            return None
        dataSlice.sort(order=self.mjdCol)

        if self.season != -1:
            seasons = self.season
        else:
            seasons = np.unique(dataSlice['season'])

        if not isinstance(seasons, Iterable):
            seasons = [seasons]

        self.info_season = None
        for seas in seasons:
            info = self.season_info(dataSlice, seas)
            if info is not None and info['season_length'] >= self.shift:
                if self.info_season is None:
                    self.info_season = info
                else:
                    self.info_season = np.concatenate((self.info_season, info))

        self.info_season = self.check_seasons(self.info_season)
        if self.info_season is None:
            return 0.

        sel = dataSlice[np.in1d(dataSlice['season'], np.array(seasons))]

        detect_frac = None
        if len(sel) >= 5:
            detect_frac = self.process(sel)

        if detect_frac is not None:
            return np.median(detect_frac['frac_obs_{}'.format(self.names_ref[0])])
        else:
            return 0.

    def process(self, sel):
        """Process one season

        Parameters
        -----------
        sel : array
          array of observations
        season : int
          season number

        Returns
        --------
        record array with the following fields:
          fieldRA (float)
          fieldDec (float)
          season (float)
         band (str)
         frac_obs_name_ref (float)

        """

        self.band = np.unique(sel[self.filterCol])[0]
        time_ref = time.time()
        snr_obs = self.snr_slice(sel)  # SNR for observations
        snr_fakes = self.snr_fakes(sel)  # SNR for fakes
        detect_frac = self.detection_rate(
            snr_obs, snr_fakes)  # Detection rate
        snr_obs = np.asarray(snr_obs)
        snr_fakes = np.asarray(snr_fakes)
        #self.plot(snr_obs, snr_fakes)
        # plt.show()
        detect_frac = np.asarray(detect_frac)

        return detect_frac

    def snr_slice(self, dataSlice, j=-1, output_q=None):
        """
        Estimate SNR for a given dataSlice

        Parameters
        -----------
        dataSlice : `np.recarray`
        j : `int`, optional
        output_q : `int`, optional

        Returns
        --------
        array with the following fields (all are of f8 type, except band which is of U1)

        SNR_name_ref:  Signal-To-Noise Ratio estimator
        season : season
        cadence: cadence of the season
        season_length: length of the season
        MJD_min: min MJD of the season
        DayMax: SN max luminosity MJD (aka T0)
        MJD:
        m5_eff: mean m5 of obs passing the min_phase, max_phase cut
        fieldRA: mean field RA
        fieldDec: mean field Dec
        band:  band
        m5: mean m5 (over the season)
        Nvisits: median number of visits (per observation) (over the season)
        ExposureTime: median exposure time (per observation) (over the season)

        """

        # Get few infos: RA, Dec, Nvisits, m5, exptime
        fieldRA = np.mean(dataSlice[self.RaCol])
        fieldDec = np.mean(dataSlice[self.DecCol])
        # one visit = 2 exposures
        Nvisits = np.median(dataSlice[self.nexpCol]/2.)
        m5 = np.mean(dataSlice[self.m5Col])
        exptime = np.median(dataSlice[self.exptimeCol])
        dataSlice.sort(order=self.mjdCol)
        mjds = dataSlice[self.mjdCol]
        band = np.unique(dataSlice[self.filterCol])[0]

        # Define MJDs to consider for metric estimation
        # basically: step of one day between MJDmin and MJDmax
        dates = None

        for val in self.info_season:
            if dates is None:
                dates = np.arange(
                    val['MJD_min']+self.shift, val['MJD_max']+1., 1.)
            else:
                dates = np.concatenate(
                    (dates, np.arange(val['MJD_min']+self.shift, val['MJD_max']+1., 1.)))

        # SN  DayMax: dates-shift where shift is chosen in the input yaml file
        T0_lc = dates-self.shift

        # for these DayMax, estimate the phases of LC points corresponding to the current dataSlice MJDs

        time_for_lc = -T0_lc[:, None]+mjds

        phase = time_for_lc/(1.+self.z)  # phases of LC points
        # flag: select LC points only in between min_rf_phase and max_phase
        phase_max = self.shift/(1.+self.z)
        flag = (phase >= self.min_rf_phase) & (phase <= phase_max)

        # tile m5, MJDs, and seasons to estimate all fluxes and SNR at once
        m5_vals = np.tile(dataSlice[self.m5Col], (len(time_for_lc), 1))
        season_vals = np.tile(dataSlice[self.seasonCol], (len(time_for_lc), 1))

        # estimate fluxes and snr in SNR function
        fluxes_tot, snr = self.snr(
            time_for_lc, m5_vals, flag, season_vals, T0_lc)

        # now save the results in a record array
        _, idx = np.unique(snr['season'], return_inverse=True)
        infos = self.info_season[idx]

        vars_info = ['cadence', 'season_length', 'MJD_min']
        snr = rf.append_fields(
            snr, vars_info, [infos[name] for name in vars_info])
        snr = rf.append_fields(snr, 'DayMax', T0_lc)
        snr = rf.append_fields(snr, 'MJD', dates)
        snr = rf.append_fields(snr, 'm5_eff', np.mean(
            np.ma.array(m5_vals, mask=~flag), axis=1))
        global_info = [(fieldRA, fieldDec, band, m5,
                        Nvisits, exptime)]*len(snr)
        names = ['fieldRA', 'fieldDec', 'band',
                 'm5', 'Nvisits', 'ExposureTime']
        global_info = np.rec.fromrecords(global_info, names=names)
        snr = rf.append_fields(
            snr, names, [global_info[name] for name in names])

        if output_q is not None:
            output_q.put({j: snr})
        else:
            return snr

    def season_info(self, dataSlice, season):
        """
        Get info on seasons for each dataSlice

        Parameters
        ----------
        dataSlice : array
            array of observations

        Returns
        -------
        recordarray with the following fields:
        season, cadence, season_length, MJDmin, MJDmax
        """

        rv = []

        idx = (dataSlice[self.seasonCol] == season)
        slice_sel = dataSlice[idx]
        if len(slice_sel) < 5:
            return None
        slice_sel.sort(order=self.mjdCol)
        mjds_season = slice_sel[self.mjdCol]
        cadence = np.mean(mjds_season[1:]-mjds_season[:-1])
        mjd_min = np.min(mjds_season)
        mjd_max = np.max(mjds_season)
        season_length = mjd_max-mjd_min
        Nvisits = np.median(slice_sel[self.nexpCol])
        m5 = np.median(slice_sel[self.m5Col])
        rv.append((float(season), cadence,
                   season_length, mjd_min, mjd_max, Nvisits, m5))

        info_season = np.rec.fromrecords(
            rv, names=['season', 'cadence', 'season_length', 'MJD_min', 'MJD_max', 'Nvisits', 'm5'])

        return info_season

    def snr(self, time_lc, m5_vals, flag, season_vals, T0_lc):
        """
        Estimate SNR vs time

        Parameters
        -----------
        time_lc :
        m5_vals : list(float)
            five-sigme depth values
        flag : array(bool)
            flag to be applied (example: selection from phase cut)
        season_vals : array(float)
            season values
        T0_lc : array(float)
            array of T0 for supernovae

        Returns
        -------
        fluxes_tot : list(float)
            list of (interpolated) fluxes
        snr_tab : array with the following fields:
            snr_name_ref (float) : Signal-to-Noise values
            season (float) : season num.
        """

        seasons = np.ma.array(season_vals, mask=~flag)

        fluxes_tot = {}
        snr_tab = None

        for ib, name in enumerate(self.names_ref):
            fluxes = self.lim_sn.fluxes[ib](time_lc)
            if name not in fluxes_tot.keys():
                fluxes_tot[name] = fluxes
            else:
                fluxes_tot[name] = np.concatenate((fluxes_tot[name], fluxes))

            flux_5sigma = self.lim_sn.mag_to_flux[ib](m5_vals)
            snr = fluxes**2/flux_5sigma**2
            snr_season = 5.*np.sqrt(np.sum(snr*flag, axis=1))

            if snr_tab is None:
                snr_tab = np.asarray(np.copy(snr_season), dtype=[
                    ('SNR_'+name, 'f8')])
            else:
                snr_tab = rf.append_fields(
                    snr_tab, 'SNR_'+name, np.copy(snr_season))
            """    
            snr_tab = rf.append_fields(
                snr_tab, 'season', np.mean(seasons, axis=1))
            """
            snr_tab = rf.append_fields(
                snr_tab, 'season', self.get_season(T0_lc))

        # check if any masked value remaining
        # this would correspond to case where no obs point has been selected
        # ie no points with phase in [phase_min,phase_max]
        # this happens when internight gaps are large (typically larger than shift)
        idmask = np.where(snr_tab.mask)
        if len(idmask) > 0:
            tofill = np.copy(snr_tab['season'])
            season_recover = self.get_season(
                T0_lc[np.where(snr_tab.mask)])
            tofill[idmask] = season_recover
            snr_tab = np.ma.filled(snr_tab, fill_value=tofill)

        return fluxes_tot, snr_tab

    def get_season(self, T0):
        """
        Estimate the seasons corresponding to T0 values

        Parameters
        ----------
        T0 : list(float)
            set of T0 values

        Returns
        --------
        list (float) of corresponding seasons
        """

        diff_min = T0[:, None]-self.info_season['MJD_min']
        diff_max = -T0[:, None]+self.info_season['MJD_max']
        seasons = np.tile(self.info_season['season'], (len(diff_min), 1))
        flag = (diff_min >= 0) & (diff_max >= 0)
        seasons = np.ma.array(seasons, mask=~flag)

        return np.mean(seasons, axis=1)

    def snr_fakes(self, dataSlice):
        """
        Estimate SNR for fake observations
        in the same way as for observations (using SNR_Season)

        Parameters
        -----------
        dataSlice : array
            array of observations

        Returns
        --------
        snr_tab : array with the following fields:
            snr_name_ref (float) : Signal-to-Noise values
            season (float) : season num.

        """

        # generate fake observations
        fake_obs = None

        # idx = (dataSlice[self.seasonCol] == season)
        band = np.unique(dataSlice[self.filterCol])[0]
        fake_obs = self.gen_fakes(dataSlice, band)

        # estimate SNR vs MJD

        snr_fakes = self.snr_slice(
            fake_obs[fake_obs['filter'] == band])

        return snr_fakes

    def gen_fakes(self, slice_sel, band):
        """
        Generate fake observations
        according to observing values extracted from simulations

        Parameters
        ----------
        slice_sel : array
            array of observations
        band : str
            band to consider

        Returns
        --------
        fake_obs_season : array
            array of observations with the following fields
            observationStartMJD (float)
            fieldRA (float)
            fieldDec (float)
            filter (U1)
            fiveSigmaDepth (float)
            numExposures (float)
            visitExposureTime (float)
            season (int)
        """
        fieldRA = np.mean(slice_sel[self.RaCol])
        fieldDec = np.mean(slice_sel[self.DecCol])
        Tvisit = 30.

        fake_obs = None
        for val in self.info_season:
            cadence = val['cadence']
            mjd_min = val['MJD_min']
            mjd_max = val['MJD_max']
            season_length = val['season_length']
            Nvisits = val['Nvisits']
            m5 = val['m5']

            # build the configuration file

            config_fake = {}
            config_fake['Ra'] = fieldRA
            config_fake['Dec'] = fieldDec
            config_fake['bands'] = [band]
            config_fake['Cadence'] = [cadence]
            config_fake['MJD_min'] = [mjd_min]
            config_fake['season_length'] = season_length
            config_fake['Nvisits'] = [Nvisits]
            m5_nocoadd = m5-1.25*np.log10(float(Nvisits)*Tvisit/30.)
            config_fake['m5'] = [m5_nocoadd]
            config_fake['seasons'] = [val['season']]
            config_fake['Exposure_Time'] = [30.]
            config_fake['shift_days'] = 0.
            fake_obs_season = GenerateFakeObservations(
                config_fake).Observations
            if fake_obs is None:
                fake_obs = fake_obs_season
            else:
                fake_obs = np.concatenate((fake_obs, fake_obs_season))
        return fake_obs

    def plot(self, snr_obs, snr_fakes):
        """ Plot SNR vs time

        Parameters
        ----------
        snr_obs : array
            array estimated using snr_slice(observations)
        snr_obs : array
            array estimated using snr_slice(fakes)
        """

        fig, ax = plt.subplots(figsize=(10, 7))

        title = 'season {} - {} band - z={}'.format(
            self.season, self.band, self.z)
        fig.suptitle(title)
        ax.plot(snr_obs['MJD'], snr_obs['SNR_{}'.format(
            self.names_ref[0])], label='Simulation')
        ax.plot(snr_fakes['MJD'], snr_fakes['SNR_{}'.format(
            self.names_ref[0])], ls='--', label='Fakes')

    def PlotHistory(self, fluxes, mjd, flag, snr, T0_lc, dates):
        """ Plot history of Plot
        For each MJD, fluxes and snr are plotted
        Each plot may be saved as a png to make a video afterwards

        Parameters
        ----------
        fluxes : list(float)
            LC fluxes
        mjd : list(float)
            mjds of the fluxes
        flag : array
            flag for selection of fluxes
        snr : list
            signal-to-noise ratio
        T0_lc : list(float)
            list of T0 supernovae
        dates : list(float)
            date of the display (mjd)
        """

        dir_save = '/home/philippe/LSST/sn_metric_new/Plots'
        import pylab as plt
        plt.ion()
        fig, ax = plt.subplots(ncols=1, nrows=2)
        fig.canvas.draw()

        colors = ['b', 'r']
        myls = ['-', '--']
        mfc = ['b', 'None']
        tot_label = []
        fontsize = 12
        mjd_ma = np.ma.array(mjd, mask=~flag)
        fluxes_ma = {}
        for key, val in fluxes.items():
            fluxes_ma[key] = np.ma.array(val, mask=~flag)
        key = list(fluxes.keys())[0]
        jmax = len(fluxes_ma[key])
        tot_label = []
        tot_label_snr = []
        min_flux = []
        max_flux = []

        for j in range(jmax):

            for ib, name in enumerate(fluxes_ma.keys()):
                tot_label.append(ax[0].errorbar(
                    mjd_ma[j], fluxes_ma[name][j], marker='s', color=colors[ib], ls=myls[ib], label=name))

                tot_label_snr.append(ax[1].errorbar(
                    snr['MJD'][:j], snr['SNR_'+name][:j], color=colors[ib], label=name))
                fluxx = fluxes_ma[name][j]
                fluxx = fluxx[~fluxx.mask]
                if len(fluxx) >= 2:
                    min_flux.append(np.min(fluxx))
                    max_flux.append(np.max(fluxx))
                else:
                    min_flux.append(0.)
                    max_flux.append(200.)

            min_fluxes = np.min(min_flux)
            max_fluxes = np.max(max_flux)

            tot_label.append(ax[0].errorbar([T0_lc[j], T0_lc[j]], [
                             min_fluxes, max_fluxes], color='k', label='DayMax'))
            tot_label.append(ax[0].errorbar([dates[j], dates[j]], [
                             min_fluxes, max_fluxes], color='k', ls='--', label='Current MJD'))
            fig.canvas.flush_events()
            # plt.savefig('{}/{}_{}.png'.format(dir_save, 'snr', 1000 + j))
            if j != jmax-1:
                ax[0].clear()
                tot_label = []
                tot_label_snr = []

        labs = [l.get_label() for l in tot_label]
        ax[0].legend(tot_label, labs, ncol=1, loc='best',
                     prop={'size': fontsize}, frameon=False)
        ax[0].set_ylabel('Flux [e.sec$^{-1}$]', fontsize=fontsize)

        ax[1].set_xlabel('MJD', fontsize=fontsize)
        ax[1].set_ylabel('SNR', fontsize=fontsize)
        ax[1].legend()
        labs = [l.get_label() for l in tot_label_snr]
        ax[1].legend(tot_label_snr, labs, ncol=1, loc='best',
                     prop={'size': fontsize}, frameon=False)
        for i in range(2):
            ax[i].tick_params(axis='x', labelsize=fontsize)
            ax[i].tick_params(axis='y', labelsize=fontsize)

    def detection_rate(self, snr_obs, snr_fakes):
        """
        Estimate the time fraction(per season) for which
        snr_obs > snr_fakes = detection rate
        For regular cadences one should get a result close to 1

        Parameters
        ----------
        snr_obs : array
            array estimated using snr_slice(observations)

        snr_fakes: array
            array estimated using snr_slice(fakes)

        Returns
        -------
        record array with the following fields:
            fieldRA (float)
            fieldDec (float)
            season (float)
            band (str)
            frac_obs_name_ref (float)
        """

        ra = np.mean(snr_obs['fieldRA'])
        dec = np.mean(snr_obs['fieldDec'])
        band = np.unique(snr_obs['band'])[0]

        rtot = []

        for season in np.unique(snr_obs['season']):
            idx = snr_obs['season'] == season
            sel_obs = snr_obs[idx]
            idxb = snr_fakes['season'] == season
            sel_fakes = snr_fakes[idxb]

            sel_obs.sort(order='MJD')
            sel_fakes.sort(order='MJD')
            r = [ra, dec, season, band]
            names = [self.RaCol, self.DecCol, 'season', 'band']
            for sim in self.names_ref:
                fakes = interpolate.interp1d(
                    sel_fakes['MJD'], sel_fakes['SNR_'+sim])
                obs = interpolate.interp1d(sel_obs['MJD'], sel_obs['SNR_'+sim])
                mjd_min = np.max(
                    [np.min(sel_obs['MJD']), np.min(sel_fakes['MJD'])])
                mjd_max = np.min(
                    [np.max(sel_obs['MJD']), np.max(sel_fakes['MJD'])])
                mjd = np.arange(mjd_min, mjd_max, 1.)

                diff_res = obs(mjd)-fakes(mjd)

                idx = diff_res >= 0
                r += [len(diff_res[idx])/len(diff_res)]
                names += ['frac_obs_'+sim]
            rtot.append(tuple(r))

        return np.rec.fromrecords(rtot, names=names)

    def check_seasons(self, tab):
        """ Check wether seasons have no overlap
        if it is the case: modify MJD_min and season length of the corresponding season
        return only seasons with season_length > 30 days

        Parameters
        -----------
        tab : array with the following fields:

        Returns
        -------
        tab : array with the following fields:
        """
        if tab is None or len(tab) == 1:
            return tab

        if len(tab) > 1:
            diff = tab['MJD_min'][1:]-tab['MJD_max'][:-1]
            idb = np.argwhere(diff < 20.)
            if len(idb) >= 1:
                tab['MJD_min'][idb+1] = tab['MJD_max'][idb]+20.
                tab['season_length'][idb+1] = tab['MJD_max'][idb+1] - \
                    tab['MJD_min'][idb+1]

            return tab[tab['season_length'] > 30.]
