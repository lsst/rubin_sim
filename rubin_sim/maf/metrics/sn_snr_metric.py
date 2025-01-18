__all__ = ("SNSNRMetric",)

from collections.abc import Iterable

import matplotlib.pylab as plt
import numpy as np
import numpy.lib.recfunctions as rf
from scipy import interpolate

import rubin_sim.maf.metrics as metrics
from rubin_sim.maf.utils.sn_utils import GenerateFakeObservations


class SNSNRMetric(metrics.BaseMetric):
    """
    Metric to estimate the detection rate for faint supernovae
    (x1,color) = (-2.0,0.2)

    Parameters
    ----------
    coadd :  `bool`, optional
        to make "coaddition" per night (uses snStacker)
    lim_sn : class, optional
        Reference data used to simulate LC points (interpolation)
    names_ref : `str`, optional
        names of the simulator used to produce reference data
    season : `float`, optional
        season num
    z : `float`, optional
        redshift for this study
    """

    def __init__(
        self,
        metric_name="SNSNRMetric",
        mjd_col="observationStartMJD",
        ra_col="fieldRA",
        dec_col="fieldDec",
        filter_col="filter",
        m5_col="fiveSigmaDepth",
        exptime_col="visitExposureTime",
        night_col="night",
        obsid_col="observationId",
        nexp_col="numExposures",
        vistime_col="visitTime",
        coadd=True,
        lim_sn=None,
        names_ref=None,
        season=1,
        z=0.01,
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.exptime_col = exptime_col
        self.season_col = "season"
        self.night_col = night_col
        self.obsid_col = obsid_col
        self.nexp_col = nexp_col
        self.vistime_col = vistime_col

        cols = [
            self.night_col,
            self.m5_col,
            self.filter_col,
            self.mjd_col,
            self.obsid_col,
            self.nexp_col,
            self.vistime_col,
            self.exptime_col,
            self.season_col,
        ]
        if coadd:
            cols += ["coadd"]
        super(SNSNRMetric, self).__init__(col=cols, metric_name=metric_name, **kwargs)

        self.filter_names = np.array(["u", "g", "r", "i", "z", "y"])
        self.blue_cutoff = 300.0
        self.red_cutoff = 800.0
        self.min_rf_phase = -20.0
        self.max_rf_phase = 40.0
        self.z = z
        self.names_ref = names_ref
        self.season = season

        # SN DayMax: current date - shift days
        self.shift = 10.0

        # These are reference LC
        self.lim_sn = lim_sn

        self.display = False

    def run(self, data_slice, slice_point=None):
        """
        run the metric

        Parameters
        ----------
        data_slice : `np.ndarray`, (N,)
          simulation data under study

        Returns
        -------
        detection rate : `float`
        """
        good_filters = np.isin(data_slice["filter"], self.filter_names)
        data_slice = data_slice[good_filters]
        if data_slice.size == 0:
            return None
        data_slice.sort(order=self.mjd_col)

        if self.season != -1:
            seasons = self.season
        else:
            seasons = np.unique(data_slice["season"])

        if not isinstance(seasons, Iterable):
            seasons = [seasons]

        self.info_season = None
        for seas in seasons:
            info = self.season_info(data_slice, seas)
            if info is not None and info["season_length"] >= self.shift:
                if self.info_season is None:
                    self.info_season = info
                else:
                    self.info_season = np.concatenate((self.info_season, info))

        self.info_season = self.check_seasons(self.info_season)
        if self.info_season is None:
            return 0.0

        sel = data_slice[np.isin(data_slice["season"], np.array(seasons))]

        detect_frac = None
        if len(sel) >= 5:
            detect_frac = self.process(sel)

        if detect_frac is not None:
            return np.median(detect_frac["frac_obs_{}".format(self.names_ref[0])])
        else:
            return 0.0

    def process(self, sel):
        """Process one season

        Parameters
        -----------
        sel : `np.narray`
          array of observations
        season : `int`
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

        self.band = np.unique(sel[self.filter_col])[0]
        snr_obs = self.snr_slice(sel)  # SNR for observations
        snr_fakes = self.snr_fakes(sel)  # SNR for fakes
        detect_frac = self.detection_rate(snr_obs, snr_fakes)  # Detection rate
        # snr_obs = np.asarray(snr_obs)
        # snr_fakes = np.asarray(snr_fakes)
        # self.plot(snr_obs, snr_fakes)
        # plt.show()
        detect_frac = np.asarray(detect_frac)

        return detect_frac

    def snr_slice(self, data_slice, j=-1, output_q=None):
        """Estimate SNR for a given data_slice

        Parameters
        -----------
        data_slice : `np.ndarray`
        j : `int`, optional
        output_q : `int`, optional

        Returns
        --------
        snr : `np.ndarray` containing
        SNR_name_ref:  Signal-To-Noise Ratio estimator
        season : season
        cadence: cadence of the season
        season_length: length of the season
        MJD_min: min MJD of the season
        DayMax: SN max luminosity MJD (aka T0)
        MJD:
        m5_eff: mean m5 of obs passing the min_phase, max_phase cut
        field_ra: mean field RA
        field_dec: mean field Dec
        band:  band
        m5: mean m5 (over the season)
        nvisits: median number of visits (per observation) (over the season)
        ExposureTime: median exposure time (per observation) (over the season)
        """

        # Get few infos: RA, Dec, nvisits, m5, exptime
        field_ra = np.mean(data_slice[self.ra_col])
        field_dec = np.mean(data_slice[self.dec_col])
        # one visit = 2 exposures
        nvisits = np.median(data_slice[self.nexp_col] / 2.0)
        m5 = np.mean(data_slice[self.m5_col])
        exptime = np.median(data_slice[self.exptime_col])
        data_slice.sort(order=self.mjd_col)
        mjds = data_slice[self.mjd_col]
        band = np.unique(data_slice[self.filter_col])[0]

        # Define MJDs to consider for metric estimation
        # basically: step of one day between MJDmin and MJDmax
        dates = None

        for val in self.info_season:
            if dates is None:
                dates = np.arange(val["MJD_min"] + self.shift, val["MJD_max"] + 1.0, 1.0)
            else:
                dates = np.concatenate(
                    (
                        dates,
                        np.arange(val["MJD_min"] + self.shift, val["MJD_max"] + 1.0, 1.0),
                    )
                )

        # SN  DayMax: dates-shift where shift is chosen in the input yaml file
        t0_lc = dates - self.shift

        # for these DayMax, estimate the phases of LC points
        # corresponding to the current data_slice MJDs

        time_for_lc = -t0_lc[:, None] + mjds

        phase = time_for_lc / (1.0 + self.z)  # phases of LC points
        # flag: select LC points only in between min_rf_phase and max_phase
        phase_max = self.shift / (1.0 + self.z)
        flag = (phase >= self.min_rf_phase) & (phase <= phase_max)

        # tile m5, MJDs, and seasons to estimate all fluxes and SNR at once
        m5_vals = np.tile(data_slice[self.m5_col], (len(time_for_lc), 1))

        # estimate fluxes and snr in SNR function
        fluxes_tot, snr = self.snr(time_for_lc, m5_vals, flag, t0_lc)

        # now save the results in a record array
        _, idx = np.unique(snr["season"], return_inverse=True)
        infos = self.info_season[idx]

        vars_info = ["cadence", "season_length", "MJD_min"]
        snr = rf.append_fields(snr, vars_info, [infos[name] for name in vars_info])
        snr = rf.append_fields(snr, "DayMax", t0_lc)
        snr = rf.append_fields(snr, "MJD", dates)
        snr = rf.append_fields(snr, "m5_eff", np.mean(np.ma.array(m5_vals, mask=~flag), axis=1))
        global_info = [(field_ra, field_dec, band, m5, nvisits, exptime)] * len(snr)
        names = ["field_ra", "field_dec", "band", "m5", "nvisits", "ExposureTime"]
        global_info = np.rec.fromrecords(global_info, names=names)
        snr = rf.append_fields(snr, names, [global_info[name] for name in names])

        if output_q is not None:
            output_q.put({j: snr})
        else:
            return snr

    def season_info(self, data_slice, season):
        """Get info on seasons for each data_slice

        Parameters
        ----------
        data_slice : `np.ndarray`, (N,)
            array of observations

        Returns
        -------
        info_season : `np.ndarray`
        season, cadence, season_length, MJDmin, MJDmax
        """

        rv = []

        idx = data_slice[self.season_col] == season
        slice_sel = data_slice[idx]
        if len(slice_sel) < 5:
            return None
        slice_sel.sort(order=self.mjd_col)
        mjds_season = slice_sel[self.mjd_col]
        cadence = np.mean(mjds_season[1:] - mjds_season[:-1])
        mjd_min = np.min(mjds_season)
        mjd_max = np.max(mjds_season)
        season_length = mjd_max - mjd_min
        nvisits = np.median(slice_sel[self.nexp_col])
        m5 = np.median(slice_sel[self.m5_col])
        rv.append((float(season), cadence, season_length, mjd_min, mjd_max, nvisits, m5))

        info_season = np.rec.fromrecords(
            rv,
            names=[
                "season",
                "cadence",
                "season_length",
                "MJD_min",
                "MJD_max",
                "nvisits",
                "m5",
            ],
        )

        return info_season

    def snr(self, time_lc, m5_vals, flag, t0_lc):
        """Estimate SNR vs time

        Parameters
        -----------
        time_lc : `np.ndarray`, (N,)
        m5_vals : `list` [`float`]
            five-sigme depth values
        flag : `np.ndarray`, (N,)
            flag to be applied (example: selection from phase cut)
        season_vals : `np.ndarray`, (N,)
            season values
        t0_lc : `np.ndarray`, (N,)
            array of T0 for supernovae

        Returns
        -------
        fluxes_tot : `list` [`float`]
            list of (interpolated) fluxes
        snr_tab : `np.ndarray`, (N,)
            snr_name_ref (float) : Signal-to-Noise values
            season (float) : season num.
        """
        fluxes_tot = {}
        snr_tab = None

        for ib, name in enumerate(self.names_ref):
            fluxes = self.lim_sn.fluxes[ib](time_lc)
            if name not in fluxes_tot.keys():
                fluxes_tot[name] = fluxes
            else:
                fluxes_tot[name] = np.concatenate((fluxes_tot[name], fluxes))

            flux_5sigma = self.lim_sn.mag_to_flux[ib](m5_vals)
            snr = fluxes**2 / flux_5sigma**2
            snr_season = 5.0 * np.sqrt(np.sum(snr * flag, axis=1))

            if snr_tab is None:
                snr_tab = np.asarray(np.copy(snr_season), dtype=[("SNR_" + name, "f8")])
            else:
                snr_tab = rf.append_fields(snr_tab, "SNR_" + name, np.copy(snr_season))
            """
            snr_tab = rf.append_fields(
                snr_tab, 'season', np.mean(seasons, axis=1))
            """
            snr_tab = rf.append_fields(snr_tab, "season", self.get_season(t0_lc))

        # check if any masked value remaining
        # this would correspond to case where no obs point has been selected
        # ie no points with phase in [phase_min,phase_max]
        # this happens when internight gaps are large
        # (typically larger than shift)
        idmask = np.where(snr_tab.mask)
        if len(idmask) > 0:
            tofill = np.copy(snr_tab["season"])
            season_recover = self.get_season(t0_lc[np.where(snr_tab.mask)])
            tofill[idmask] = season_recover
            snr_tab = np.ma.filled(snr_tab, fill_value=tofill)

        return fluxes_tot, snr_tab

    def get_season(self, t0):
        """
        Estimate the seasons corresponding to t0 values

        Parameters
        ----------
        t0 : `list` [`float`]
            set of t0 values

        Returns
        --------
        mean_seasons : `list` [`float`]
            list (float) of corresponding seasons
        """

        diff_min = t0[:, None] - self.info_season["MJD_min"]
        diff_max = -t0[:, None] + self.info_season["MJD_max"]
        seasons = np.tile(self.info_season["season"], (len(diff_min), 1))
        flag = (diff_min >= 0) & (diff_max >= 0)
        seasons = np.ma.array(seasons, mask=~flag)

        return np.mean(seasons, axis=1)

    def snr_fakes(self, data_slice):
        """Estimate SNR for fake observations
        in the same way as for observations (using SNR_Season)

        Parameters
        -----------
        data_slice : `np.ndarray`, (N,)
            array of observations

        Returns
        --------
        snr_tab : `np.ndarray`
            snr_name_ref (float) : Signal-to-Noise values
            season (float) : season num.

        """

        # generate fake observations
        fake_obs = None

        # idx = (data_slice[self.season_col] == season)
        band = np.unique(data_slice[self.filter_col])[0]
        fake_obs = self.gen_fakes(data_slice, band)

        # estimate SNR vs MJD

        snr_fakes = self.snr_slice(fake_obs[fake_obs["filter"] == band])

        return snr_fakes

    def gen_fakes(self, slice_sel, band):
        """Generate fake observations
        according to observing values extracted from simulations

        Parameters
        ----------
        slice_sel : `np.ndarray`, (N,)
            array of observations
        band : `str`
            band to consider

        Returns
        --------
        fake_obs_season : `np.ndarray`
            array of observations with the following fields
            observationStartMJD (float)
            field_ra (float)
            field_dec (float)
            filter (U1)
            fiveSigmaDepth (float)
            numExposures (float)
            visitExposureTime (float)
            season (int)
        """
        field_ra = np.mean(slice_sel[self.ra_col])
        field_dec = np.mean(slice_sel[self.dec_col])
        tvisit = 30.0

        fake_obs = None
        for val in self.info_season:
            cadence = val["cadence"]
            mjd_min = val["MJD_min"]
            season_length = val["season_length"]
            nvisits = val["nvisits"]
            m5 = val["m5"]

            # build the configuration file

            config_fake = {}
            config_fake["Ra"] = field_ra
            config_fake["Dec"] = field_dec
            config_fake["bands"] = [band]
            config_fake["Cadence"] = [cadence]
            config_fake["MJD_min"] = [mjd_min]
            config_fake["season_length"] = season_length
            config_fake["nvisits"] = [nvisits]
            m5_nocoadd = m5 - 1.25 * np.log10(float(nvisits) * tvisit / 30.0)
            config_fake["m5"] = [m5_nocoadd]
            config_fake["seasons"] = [val["season"]]
            config_fake["Exposure_Time"] = [30.0]
            config_fake["shift_days"] = 0.0
            fake_obs_season = GenerateFakeObservations(config_fake).Observations
            if fake_obs is None:
                fake_obs = fake_obs_season
            else:
                fake_obs = np.concatenate((fake_obs, fake_obs_season))
        return fake_obs

    def plot(self, snr_obs, snr_fakes):
        """Plot SNR vs time

        Parameters
        ----------
        snr_obs : `np.ndarray`, (N,)
            array estimated using snr_slice(observations)
        snr_obs : `np.ndarray`, (N,)
            array estimated using snr_slice(fakes)
        """

        fig, ax = plt.subplots(figsize=(10, 7))

        title = "season {} - {} band - z={}".format(self.season, self.band, self.z)
        fig.suptitle(title)
        ax.plot(
            snr_obs["MJD"],
            snr_obs["SNR_{}".format(self.names_ref[0])],
            label="Simulation",
        )
        ax.plot(
            snr_fakes["MJD"],
            snr_fakes["SNR_{}".format(self.names_ref[0])],
            ls="--",
            label="Fakes",
        )

    def plot_history(self, fluxes, mjd, flag, snr, t0_lc, dates):
        """Plot history of lightcurve
        For each MJD, fluxes and snr are plotted
        Each plot may be saved as a png to make a video afterwards

        Parameters
        ----------
        fluxes : `list` [`float`]
            LC fluxes
        mjd : list(float)
            mjds of the fluxes
        flag : array
            flag for selection of fluxes
        snr : list
            signal-to-noise ratio
        t0_lc : list(float)
            list of T0 supernovae
        dates : list(float)
            date of the display (mjd)
        """

        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots(ncols=1, nrows=2)
        fig.canvas.draw()

        colors = ["b", "r"]
        myls = ["-", "--"]
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
                tot_label.append(
                    ax[0].errorbar(
                        mjd_ma[j],
                        fluxes_ma[name][j],
                        marker="s",
                        color=colors[ib],
                        ls=myls[ib],
                        label=name,
                    )
                )

                tot_label_snr.append(
                    ax[1].errorbar(
                        snr["MJD"][:j],
                        snr["SNR_" + name][:j],
                        color=colors[ib],
                        label=name,
                    )
                )
                fluxx = fluxes_ma[name][j]
                fluxx = fluxx[~fluxx.mask]
                if len(fluxx) >= 2:
                    min_flux.append(np.min(fluxx))
                    max_flux.append(np.max(fluxx))
                else:
                    min_flux.append(0.0)
                    max_flux.append(200.0)

            min_fluxes = np.min(min_flux)
            max_fluxes = np.max(max_flux)

            tot_label.append(
                ax[0].errorbar(
                    [t0_lc[j], t0_lc[j]],
                    [min_fluxes, max_fluxes],
                    color="k",
                    label="DayMax",
                )
            )
            tot_label.append(
                ax[0].errorbar(
                    [dates[j], dates[j]],
                    [min_fluxes, max_fluxes],
                    color="k",
                    ls="--",
                    label="Current MJD",
                )
            )
            fig.canvas.flush_events()
            # plt.savefig('{}/{}_{}.png'.format(dir_save, 'snr', 1000 + j))
            if j != jmax - 1:
                ax[0].clear()
                tot_label = []
                tot_label_snr = []

        labs = [ll.get_label() for ll in tot_label]
        ax[0].legend(tot_label, labs, ncol=1, loc="best", prop={"size": fontsize}, frameon=False)
        ax[0].set_ylabel("Flux [e.sec$^{-1}$]", fontsize=fontsize)

        ax[1].set_xlabel("MJD", fontsize=fontsize)
        ax[1].set_ylabel("SNR", fontsize=fontsize)
        ax[1].legend()
        labs = [ll.get_label() for ll in tot_label_snr]
        ax[1].legend(
            tot_label_snr,
            labs,
            ncol=1,
            loc="best",
            prop={"size": fontsize},
            frameon=False,
        )
        for i in range(2):
            ax[i].tick_params(axis="x", labelsize=fontsize)
            ax[i].tick_params(axis="y", labelsize=fontsize)

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

        ra = np.mean(snr_obs["fieldRA"])
        dec = np.mean(snr_obs["fieldDec"])
        band = np.unique(snr_obs["band"])[0]

        rtot = []

        for season in np.unique(snr_obs["season"]):
            idx = snr_obs["season"] == season
            sel_obs = snr_obs[idx]
            idxb = snr_fakes["season"] == season
            sel_fakes = snr_fakes[idxb]

            sel_obs.sort(order="MJD")
            sel_fakes.sort(order="MJD")
            r = [ra, dec, season, band]
            names = [self.ra_col, self.dec_col, "season", "band"]
            for sim in self.names_ref:
                fakes = interpolate.interp1d(sel_fakes["MJD"], sel_fakes["SNR_" + sim])
                obs = interpolate.interp1d(sel_obs["MJD"], sel_obs["SNR_" + sim])
                mjd_min = np.max([np.min(sel_obs["MJD"]), np.min(sel_fakes["MJD"])])
                mjd_max = np.min([np.max(sel_obs["MJD"]), np.max(sel_fakes["MJD"])])
                mjd = np.arange(mjd_min, mjd_max, 1.0)

                diff_res = obs(mjd) - fakes(mjd)

                idx = diff_res >= 0
                r += [len(diff_res[idx]) / len(diff_res)]
                names += ["frac_obs_" + sim]
            rtot.append(tuple(r))

        return np.rec.fromrecords(rtot, names=names)

    def check_seasons(self, tab):
        """Check whether seasons have no overlap
        if it is the case: modify MJD_min and season length of
        the corresponding season
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
            diff = tab["MJD_min"][1:] - tab["MJD_max"][:-1]
            idb = np.argwhere(diff < 20.0)
            if len(idb) >= 1:
                tab["MJD_min"][idb + 1] = tab["MJD_max"][idb] + 20.0
                tab["season_length"][idb + 1] = tab["MJD_max"][idb + 1] - tab["MJD_min"][idb + 1]

            return tab[tab["season_length"] > 30.0]
