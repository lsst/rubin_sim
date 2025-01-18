__all__ = ("SNNSNMetric",)


import healpy as hp
import numpy as np
import numpy.lib.recfunctions as nlr
import pandas as pd
from scipy.interpolate import interp1d

from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.utils.sn_n_sn_utils import LcfastNew, SnRate, load_sne_cached
from rubin_sim.phot_utils import DustValues


class SNNSNMetric(BaseMetric):
    """
    Measure zlim of type Ia supernovae.

    Parameters
    --------------
    metricName : `str`, opt
        metric name (default : SNSNRMetric)
    mjd_col : `str`, opt
        mjd column name (default : observationStartMJD)
    filter_col : `str`, opt
        filter column name (default: filter)
    m5_col : `str`, opt
        five-sigma depth column name (default : fiveSigmaDepth)
    exptime_col : `str`, opt
        exposure time column name (default : visitExposureTime)
    night_col : `str`, opt
        night column name (default : night)
    obsid_col : `str`, opt
        observation id column name (default : observationId)
    nexp_col : `str`, opt
        number of exposure column name (default : numExposures)
    vistime_col : `str`, opt
        visit time column name (default : visitTime)
    seeing_col : `str`, opt
        seeing column name (default: seeingFwhmEff)
    note_col : `str`, opt
        note column name (default: note)
    season : `list`, opt
        list of seasons to process (float)(default: -1 = all seasons)
    coadd : `bool`, opt
        coaddition per night (and per band) (default : True)
    zmin : `float`, opt
        min redshift for the study (default: 0.0)
    zmax : `float`, opt
        max redshift for the study (default: 1.2)
    verbose : `bool`, opt
        verbose mode (default: False)
    n_bef : `int`, opt
        number of LC points LC before T0 (default:5)
    n_aft : `int`, opt
        number of LC points after T0 (default: 10)
    snr_min : `float`, opt
        minimal SNR of LC points (default: 5.0)
    n_phase_min : `int`, opt
        number of LC points with phase<= -5(default:1)
    n_phase_max : `int`, opt
        number of LC points with phase>= 20 (default: 1)
    zlim_coeff: float, opt
        corresponds to the zlim_coeff fraction of SN with z<zlim
    bands : `str`, opt
        bands to consider (default: grizy)
    gammaName: `str`, opt
        name of the gamma ref file to load (default: gamma_WFD.hdf5)
    dust : `bool`, opt
        Apply dust extinction to visit depth values (default False)
    hard_dust_cut : `float`, opt
      If set, cut any point on the sky that has an ebv extinction
      higher than the hard_dust_cut value.
      Default 0.25
    """

    def __init__(
        self,
        metric_name="SNNSNMetric",
        mjd_col="observationStartMJD",
        filter_col="filter",
        m5_col="fiveSigmaDepth",
        exptime_col="visitExposureTime",
        night_col="night",
        obsid_col="observationId",
        nexp_col="numExposures",
        vistime_col="visitTime",
        seeing_col="seeingFwhmEff",
        note_col="scheduler_note",
        season=[-1],
        coadd_night=True,
        zmin=0.1,
        zmax=0.5,
        z_step=0.03,
        daymax_step=3.0,
        verbose=False,
        ploteffi=False,
        n_bef=3,
        n_aft=8,
        snr_min=1.0,
        n_phase_min=1,
        n_phase_max=1,
        sigma_c=0.04,
        zlim_coeff=0.95,
        bands="grizy",
        add_dust=False,
        hard_dust_cut=0.25,
        gamma_name="gamma_WFD.hdf5",
        **kwargs,
    ):
        # n_bef / n_aft = 3/8 for WFD, 4/10 for DDF

        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.exptime_col = exptime_col
        self.season_col = "season"
        self.night_col = night_col
        self.obsid_col = obsid_col
        self.nexp_col = nexp_col
        self.vistime_col = vistime_col
        self.seeing_col = seeing_col
        self.note_col = note_col

        self.ploteffi = ploteffi
        self.t0s = "all"
        self.zlim_coeff = zlim_coeff
        self.bands = bands
        self.coadd_night = coadd_night
        self.add_dust = add_dust
        self.hard_dust_cut = hard_dust_cut

        maps = ["DustMap"]
        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1

        cols = [
            self.night_col,
            self.m5_col,
            self.filter_col,
            self.mjd_col,
            self.obsid_col,
            self.nexp_col,
            self.vistime_col,
            self.exptime_col,
            self.note_col,
        ]

        super(SNNSNMetric, self).__init__(
            col=cols,
            metric_dtype="object",
            metric_name=metric_name,
            maps=maps,
            **kwargs,
        )

        self.season = season

        # LC selection parameters
        self.n_bef = n_bef  # nb points before peak
        self.n_aft = n_aft  # nb points after peak
        self.snr_min = snr_min  # SNR cut for points before/after peak
        self.n_phase_min = n_phase_min  # nb of point with phase <=-5
        self.n_phase_max = n_phase_max  # nb of points with phase >=20
        self.sigma_c = sigma_c

        # loading reference LC files
        lc_reference = load_sne_cached(gamma_name)
        # loading reference LC files

        self.lc_fast = {}
        for key, vals in lc_reference.items():
            self.lc_fast[key] = LcfastNew(
                vals,
                key[0],
                key[1],
                self.mjd_col,
                self.filter_col,
                self.exptime_col,
                self.m5_col,
                self.season_col,
                self.nexp_col,
                self.seeing_col,
                self.snr_min,
                light_output=False,
            )
        # loading parameters
        self.zmin = zmin  # zmin for the study
        self.zmax = zmax  # zmax for the study
        self.zstep = z_step  # zstep
        # get redshift range for processing
        zrange = list(np.arange(self.zmin, self.zmax, self.zstep))
        if zrange[0] < 1.0e-6:
            zrange[0] = 0.01

        self.zrange = np.unique(zrange)

        self.daymax_step = daymax_step  # daymax step
        self.min_rf_phase = -20.0  # min ref phase for LC points selection
        self.max_rf_phase = 60.0  # max ref phase for LC points selection

        self.min_rf_phase_qual = -15.0  # min ref phase for bounds effects
        self.max_rf_phase_qual = 30.0  # max ref phase for bounds effects

        # snrate
        self.rate_sn = SnRate(
            h0=70.0,
            om0=0.3,
            min_rf_phase=self.min_rf_phase_qual,
            max_rf_phase=self.max_rf_phase_qual,
        )

        # verbose mode - useful for debug and code performance estimation
        self.verbose = verbose

        # supernovae parameters for fisher estimation
        self.params = ["x0", "x1", "daymax", "color"]

    def run(self, data_slice, slice_point):
        """
        Run method of the metric

        Parameters
        --------------
        data_slice : `np.ndarray`
            Observations to process (scheduler simulations)
        slice_point : `bool`, opt
            Information about the location on the sky from the slicer

        Returns
        -------
        metricVal : `np.ndarray`
            ['n_sn', 'zlim'] at this point on the sky
        """
        # Hard dust cut
        if self.hard_dust_cut is not None:
            ebvof_mw = slice_point["ebv"]
            if ebvof_mw > self.hard_dust_cut:
                return self.badval

        # get area on-sky in this slice_point
        if "nside" in slice_point:
            self.pix_area = hp.nside2pixarea(slice_point["nside"], degrees=True)
        else:
            self.pix_area = 9.6

        # If we want to apply dust extinction.
        if self.add_dust:
            new_m5 = data_slice[self.m5_col] * 0
            for filtername in np.unique(data_slice[self.filter_col]):
                in_filt = np.where(data_slice[self.filter_col] == filtername)[0]
                a_x = self.ax1[filtername] * slice_point["ebv"]
                new_m5[in_filt] = data_slice[self.m5_col][in_filt] - a_x
            data_slice[self.m5_col] = new_m5

        # select observations filter
        good_filters = np.isin(data_slice[self.filter_col], list(self.bands))
        data_slice = data_slice[good_filters]

        # coaddition per night and per band (if requested by the user)
        if self.coadd_night:
            data_slice = self.coadd(data_slice)

        # This seems unlikely, but is a possible bailout point
        if len(data_slice) <= self.n_aft + self.n_bef:
            return self.badval

        # get season information (seasons calculated by gaps,
        # not by place on sky)
        data_slice = self.getseason(data_slice, mjd_col=self.mjd_col)

        # get redshift values per season
        zseason = self.z_season(self.season, data_slice)
        zseason_allz = self.z_season_allz(zseason)

        # estimate redshift completeness
        metric_values = self.metric(data_slice, zseason_allz, x1=-2.0, color=0.2, zlim=-1, metric="zlim")

        if metric_values is None:
            return self.badval

        if np.max(metric_values["zcomp"]) < 0:
            return self.badval

        # get redshift values per season up to zcomp
        zseason = pd.DataFrame(metric_values[["season", "zcomp"]])
        zseason.loc[:, "zmin"] = 0.01
        zseason.loc[:, "zstep"] = self.zstep
        zseason = zseason.rename(columns={"zcomp": "zmax"})
        zseason["zmax"] += self.zstep
        zseason_allz = self.z_season_allz(zseason)

        # get the total number of well-sampled SN up to zcomp
        nsn_zcomp = self.metric(
            data_slice,
            zseason_allz,
            x1=0.0,
            color=0.0,
            zlim=metric_values[["season", "zcomp"]],
            metric="nsn",
        )

        # final results
        if nsn_zcomp is None:
            return self.badval
        metric_values = metric_values.merge(nsn_zcomp, left_on=["season"], right_on=["season"])

        if self.verbose:
            print("metric_values", metric_values[["season", "zcomp", "nsn"]])

        idx = metric_values["zcomp"] > 0.0
        selmet = metric_values[idx]

        if len(selmet) > 0:
            zcomp = selmet["zcomp"].median()
            n_sn = selmet["nsn"].sum()
            res = np.rec.fromrecords([(n_sn, zcomp)], names=["n_sn", "zlim"])
        else:
            res = self.badval

        if self.verbose:
            print("final result", res)

        return res

    def season_length(self, seasons, data_slice, zseason):
        """
        Method to estimate season lengths vs z

        Parameters
        -----------
        seasons : `list` [`int`]
            list of seasons to process
        data_slice : `np.ndarray`, (N,)`
            array of observations
        zseason : `pd.DataFrame`
            redshift infos per season

        Returns
        --------
        seasons : `list` [`int`]
            list of seasons to process
        dur_z : `pd.DataFrame`
            season lengths vs z
        """
        # if seasons = -1: process the seasons seen in data
        if seasons == [-1]:
            seasons = np.unique(data_slice[self.season_col])

        # season infos
        dfa = pd.DataFrame(np.copy(data_slice))
        dfa = pd.DataFrame(dfa[dfa["season"].isin(seasons)])

        season_info = self.get_season_info(dfa, zseason)

        if self.verbose:
            print("season_info", season_info)

        if season_info.empty:
            return [], pd.DataFrame()

        season_info["season_copy"] = season_info["season"].values.copy()
        dur_z = (
            season_info.groupby("season_copy", group_keys=False)
            .apply(lambda x: self.nsn_expected_z(x), include_groups=False)
            .reset_index(drop=True)
        )
        return season_info["season"].to_list(), dur_z

    def get_season_info(self, dfa, zseason, min_duration=60.0):
        """
        method to get season infos vs z

        Parameters
        --------------
        dfa : `pd.DataFrame`
            data to process
        zseason : `pd.DataFrame`
            redshift infos per season
        min_duration : `float`, opt
            min season length to be accepted (default: 60 days)

        Returns
        --------
        season_info : `pd.DataFrame` with season length infos

        """
        dfa["season_copy"] = dfa["season"].values.copy()
        season_info = (
            dfa.groupby("season")
            .apply(lambda x: self.season_info(x, min_duration=min_duration), include_groups=False)
            .reset_index()
        )

        # season_info.index = season_info.index.droplevel()
        season_info = season_info.drop(columns=["level_1"])

        season_info = season_info.merge(zseason, left_on=["season"], right_on=["season"])

        season_info["T0_min"] = season_info["MJD_min"] - (1.0 + season_info["z"]) * self.min_rf_phase_qual
        season_info["T0_max"] = season_info["MJD_max"] - (1.0 + season_info["z"]) * self.max_rf_phase_qual
        season_info["season_length"] = season_info["T0_max"] - season_info["T0_min"]

        idx = season_info["season_length"] >= min_duration

        return pd.DataFrame(season_info[idx])

    def step_lc(self, obs, gen_par, x1=-2.0, color=0.2):
        """
        Method to generate lc

        Parameters
        ---------------
        obs : array
            observations
        gen_par : array
            simulation parameters
        x1 : `float`, opt
            stretch value (default: -2.0)
        color : `float`, opt
            color value (default: 0.2)

        Returns
        ----------
        SN light curves (astropy table)

        """
        obs["season_copy"] = obs["season"].values.copy()
        lc = obs.groupby(["season_copy"]).apply(
            lambda x: self.gen_lc(x, gen_par, x1, color), include_groups=False
        )
        return lc

    def step_efficiencies(self, lc):
        """
        Method to estimate observing efficiencies

        Parameter
        -------------
        lc: `pd.DataFrame`
           light curves

        Returns
        -----------
        `pd.DataFrame` with efficiencies

        """
        cols = ["season", "z", "x1", "color", "sntype"]
        for col in cols:
            lc[col + "_copy"] = lc[col].values.copy()

        sn_effis = lc.groupby(cols).apply(lambda x: self.sn_effi(x), include_groups=False).reset_index()

        # estimate efficiencies
        sn_effis["season"] = sn_effis["season"].astype(int)
        sn_effis["effi"] = sn_effis["nsel"] / sn_effis["ntot"]
        sn_effis["effi_err"] = np.sqrt(sn_effis["nsel"] * (1.0 - sn_effis["effi"])) / sn_effis["ntot"]

        # prevent NaNs, set effi to 0 where there is 0 ntot
        zero = np.where(sn_effis["ntot"] == 0)
        sn_effis["effi"].values[zero] = 0
        sn_effis["effi_err"].values[zero] = 0

        if self.verbose:
            for season in sn_effis["season"].unique():
                idx = sn_effis["season"] == season
                print("effis", sn_effis[idx])

        if self.ploteffi:
            from sn_metrics.sn_plot_live import plotNSN_effi

            for season in sn_effis["season"].unique():
                idx = sn_effis["season"] == season
                print("effis", sn_effis[idx])
                plotNSN_effi(sn_effis[idx], "effi", "effi_err", "Observing Efficiencies", ls="-")

        return sn_effis

    def step_nsn(self, sn_effis, dur_z):
        """
        Method to estimate the number of supernovae from efficiencies

        Parameters
        ----------
        sn_effis : `pd.DataFrame`
          data with efficiencies of observation
        dur_z :  array
          array of season length

        Returns
        -------
        initial sn_effis appended with a set of infos (duration, nsn)

        """
        # add season length here
        sn_effis = sn_effis.merge(dur_z, left_on=["season", "z"], right_on=["season", "z"])

        # estimate the number of supernovae
        sn_effis["nsn"] = sn_effis["effi"] * sn_effis["nsn_expected"]

        return sn_effis

    def season_info(self, grp, min_duration):
        """
        Method to estimate seasonal info (cadence, season length, ...)

        Parameters
        --------------
        grp : `pd.DataFrame` group
        min_duration : `float`
          minimal duration for a season to be considered

        Returns
        ---------
        `pd.DataFrame` with the following cols:
        - Nvisits: number of visits for this group
        - N_xx:  number of visits in xx where xx is defined in self.bandstat

        """
        df = pd.DataFrame([len(grp)], columns=["Nvisits"])
        df["MJD_min"] = grp[self.mjd_col].min()
        df["MJD_max"] = grp[self.mjd_col].max()
        df["season_length"] = df["MJD_max"] - df["MJD_min"]
        df["cadence"] = 0.0

        if len(grp) > 5:
            # to = grp.groupby(['night'])[self.mjd_col].median().sort_values()
            # df['cadence'] = np.mean(to.diff())
            nights = np.sort(grp["night"].unique())
            diff = np.asarray(nights[1:] - nights[:-1])
            df["cadence"] = np.median(diff).item()

            # select seasons of at least 30 days
        idx = df["season_length"] >= min_duration

        return df[idx]

    def duration_z(self, grp, min_duration=60.0):
        """
        Method to estimate the season length vs redshift
        This is necessary to take into account boundary effects
        when estimating the number of SN that can be detected

        daymin, daymax = min and max MJD of a season
        T0_min(z) =  daymin-(1+z)*min_rf_phase_qual
        T0_max(z) =  daymax-(1+z)*max_rf_phase_qual
        season_length(z) = T0_max(z)-T0_min(z)

        Parameters
        --------------
        grp : `pd.DataFrame` group
          data to process: season infos
        min_duration : `float`, opt
          min season length for a season to be processed (deafult: 60 days)

        Returns
        ----------
        `pd.DataFrame` with season_length, z, T0_min and T0_max cols

        """
        ## IS THIS CALLED FROM ANYWHERE?
        daymin = grp["MJD_min"].values
        daymax = grp["MJD_max"].values
        dur_z = pd.DataFrame(self.zrange, columns=["z"])
        dur_z["T0_min"] = daymin - (1.0 + dur_z["z"]) * self.min_rf_phase_qual
        dur_z["T0_max"] = daymax - (1.0 + dur_z["z"]) * self.max_rf_phase_qual
        dur_z["season_length"] = dur_z["T0_max"] - dur_z["T0_min"]
        # dur_z['season_length_orig'] = daymax-daymin
        # dur_z['season_length_orig'] = [daymax-daymin]*len(self.zrange)
        nsn = self.nsn_from_rate(dur_z)
        if self.verbose:
            print("dur_z", dur_z)
            print("nsn expected", nsn)
        dur_z = dur_z.merge(nsn, left_on=["z"], right_on=["z"])

        idx = dur_z["season_length"] > min_duration
        sel = dur_z[idx]
        if len(sel) < 2:
            return pd.DataFrame()
        return dur_z

    def calc_daymax(self, grp, daymax_step):
        """
        Method to estimate T0 (daymax) values for simulation.

        Parameters
        --------------
        grp: group (`pd.DataFrame` sense)
            group of data to process with the following cols:
            t0_min: T0 min value (per season)
            t0_max: T0 max value (per season)
        daymax_step: `float`
            step for T0 simulation

        Returns
        ----------
        `pd.DataFrame` with daymax, min_rf_phase, max_rf_phase values

        """

        if self.t0s == "all":
            t0_max = grp["T0_max"].values
            t0_min = grp["T0_min"].values
            num = (t0_max - t0_min) / daymax_step
            if t0_max - t0_min > 10:
                df = pd.DataFrame(np.linspace(t0_min, t0_max, int(np.max(num))), columns=["daymax"])
            else:
                df = pd.DataFrame([-1], columns=["daymax"])
        else:
            df = pd.DataFrame([0.0], columns=["daymax"])

        df["minRFphase"] = self.min_rf_phase
        df["maxRFphase"] = self.max_rf_phase

        return df

    def gen_lc(self, grp, gen_par_orig, x1, color):
        """
        Method to generate light curves from observations

        Parameters
        ---------------
        grp : pd group
          observations to process
        gen_par_orig : `pd.DataFrame`
          simulation parameters
        x1 : `float`
          SN stretch
        color : `float`
          SN color

        Returns
        ----------
        light curves as `pd.DataFrame`

        """
        season = grp.name
        idx = gen_par_orig["season"] == season
        gen_par = gen_par_orig[idx].to_records(index=False)

        sntype = dict(zip([(-2.0, 0.2), (0.0, 0.0)], ["faint", "medium"]))
        res = pd.DataFrame()
        key = (np.round(x1, 1), np.round(color, 1))
        vals = self.lc_fast[key]

        gen_par_cp = gen_par.copy()
        if key == (-2.0, 0.2):
            idx = gen_par_cp["z"] < 0.9
            gen_par_cp = gen_par_cp[idx]
        lc = vals(grp.to_records(index=False), 0.0, gen_par_cp, bands="grizy")
        lc["x1"] = key[0]
        lc["color"] = key[1]
        lc["sntype"] = sntype[key]
        res = pd.concat((res, lc))

        return res

    def sn_effi(self, lc):
        """
        Method to transform LCs to supernovae

        Parameters
        ---------------
        lc : pd grp
          light curve

        Returns
        ----------
        `pd.DataFrame` of sn efficiencies vs z
        """

        if self.verbose:
            print("effi for", lc.name)

        lcarr = lc.to_records(index=False)

        idx = lcarr["snr_m5"] >= self.snr_min

        lcarr = np.copy(lcarr[idx])

        t0s = np.unique(lcarr["daymax"])
        t0s.sort()

        delta_t = lcarr["daymax"] - t0s[:, np.newaxis]

        flag = np.abs(delta_t) < 1.0e-5

        resdf = pd.DataFrame(t0s, columns=["daymax"])

        # get n_phase_min, n_phase_max
        for vv in [
            "n_phmin",
            "n_phmax",
            "F_x0x0",
            "F_x0x1",
            "F_x0daymax",
            "F_x0color",
            "F_x1x1",
            "F_x1daymax",
            "F_x1color",
            "F_daymaxdaymax",
            "F_daymaxcolor",
            "F_colorcolor",
        ]:
            resdf[vv] = self.get_sum(lcarr, vv, len(delta_t), flag)

        nights = np.tile(lcarr["night"], (len(delta_t), 1))
        phases = np.tile(lcarr["phase"], (len(delta_t), 1))

        flagph = phases >= 0.0
        resdf["nepochs_aft"] = self.get_epochs(nights, flag, flagph)
        flagph = phases <= 0.0
        resdf["nepochs_bef"] = self.get_epochs(nights, flag, flagph)

        # replace NaN by 0
        # solution from: https://stackoverflow.com/
        # questions/77900971/
        # pandas-futurewarning-downcasting-
        # object-dtype-arrays-on-fillna-ffill-bfill
        with pd.option_context("future.no_silent_downcasting", True):
            resdf = resdf.fillna(0).infer_objects(copy=False)

        # get selection efficiencies
        effis = self.efficiencies(resdf)

        return effis

    def get_sum(self, lcarr, varname, nvals, flag):
        """
        Method to get the sum of variables using broadcasting

        Parameters
        --------------
        lcarr : numpy array
          data to process
        varname : `str`
          col to process in lcarr
        nvals : `int`
          dimension for tiling
        flag : array(bool)
          flag to apply

        Returns
        ----------
        array: the sum of the corresponding variable

        """

        phmin = np.tile(lcarr[varname], (nvals, 1))
        n_phmin = np.ma.array(phmin, mask=~flag)
        n_phmin = n_phmin.sum(axis=1)

        return n_phmin

    def get_epochs(self, nights, flag, flagph):
        """
        Method to get the number of epochs

        Parameters
        ---------------
        nights : array
          night number array
        flag : array(bool)
          flag to apply
        flagph : array(bool)
          flag to apply

        Returns
        -----------
        array with the number of epochs

        """
        nights_cp = np.copy(nights)
        B = np.ma.array(nights_cp, mask=~(flag & flagph))
        B.sort(axis=1)
        C = np.diff(B, axis=1) > 0
        D = C.sum(axis=1) + 1
        return D

    def sigma_s_nparams(self, grp):
        """
        Method to estimate variances of SN parameters
        from inversion of the Fisher matrix

        Parameters
        ---------------
        grp: `pd.DataFrame` of flux derivatives wrt SN parameters
        Returns
        ----------
        Diagonal elements of the inverted matrix (as `pd.DataFrame`)
        """

        # params = ['x0', 'x1', 'daymax', 'color']
        parts = {}
        for ia, vala in enumerate(self.params):
            for jb, valb in enumerate(self.params):
                if jb >= ia:
                    parts[ia, jb] = grp["F_" + vala + valb]

        # print(parts)
        size = len(grp)
        npar = len(self.params)
        fisher__big = np.zeros((npar * size, npar * size))
        big__diag = np.zeros((npar * size, npar * size))
        big__diag = []

        for iv in range(size):
            for ia, vala in enumerate(self.params):
                for jb, valb in enumerate(self.params):
                    if jb >= ia:
                        fisher__big[ia + npar * iv][jb + npar * iv] = parts[ia, jb][iv]

        # pprint.pprint(fisher__big)

        fisher__big = fisher__big + np.triu(fisher__big, 1).T
        big__diag = np.diag(np.linalg.inv(fisher__big))

        res = pd.DataFrame()
        for ia, vala in enumerate(self.params):
            indices = range(ia, len(big__diag), npar)
            res["Cov_{}{}".format(vala, vala)] = np.take(big__diag, indices)

        return res

    def efficiencies(self, dfo):
        """ "
        Method to estimate selection efficiencies

        Parameters
        ---------------
        df: `pd.DataFrame`
          data to process

        """

        if self.verbose:
            print(
                "selection params",
                self.n_phase_min,
                self.n_phase_max,
                self.n_bef,
                self.n_aft,
            )
            print(dfo)
        df = pd.DataFrame(dfo)
        df["select"] = df["n_phmin"] >= self.n_phase_min
        df["select"] &= df["n_phmax"] >= self.n_phase_max
        df["select"] &= df["nepochs_bef"] >= self.n_bef
        df["select"] &= df["nepochs_aft"] >= self.n_aft
        df["select"] = df["select"].astype(int)
        df["Cov_colorcolor"] = 100.0

        idx = df["select"] == 1

        bad_sn = pd.DataFrame(df.loc[~idx])
        good_sn = pd.DataFrame()
        if len(df[idx]) > 0:
            good_sn = pd.DataFrame(df.loc[idx].reset_index())
            sigma__fisher = self.sigma_s_nparams(good_sn)
            good_sn["Cov_colorcolor"] = sigma__fisher["Cov_colorcolor"]

        all_sn = pd.concat((good_sn, bad_sn))
        all_sn["select"] &= all_sn["Cov_colorcolor"] <= self.sigma_c**2
        idx = all_sn["select"] == 1

        if self.verbose:
            if len(good_sn) > 0:
                print("good SN", len(good_sn), good_sn[["daymax", "Cov_colorcolor"]])
            else:
                print("no good SN")
        return pd.DataFrame({"ntot": [len(all_sn)], "nsel": [len(all_sn[idx])]})

    def metric(self, data_slice, zseason, x1=-2.0, color=0.2, zlim=-1, metric="zlim"):
        """
        Method to run the metric

        Parameters
        ---------------
        data_slice: array
          observations to use for processing
        zseason: array
          season infos (season length vs z)
        x1: float, opt
          SN stretch (default: -2.0)
        color: float, opt
          SN color (default: -0.2)
        zlim: float, opt
          redshift limit used to estimate NSN (default: -1)
        metric: str, opt
          metric to estimate [zlim or nsn] (default: zlim)


        """
        """
        snType = "medium"
        if np.abs(x1 + 2.0) <= 1.0e-5:
            snType = "faint"
        """

        # get the season durations
        seasons, dur_z = self.season_length(self.season, data_slice, zseason)
        if not seasons or dur_z.empty:
            return None

        # chek the redshift range per season
        dur_z = self.check_dur_z(dur_z)
        if dur_z.empty:
            return None

        # Stick a season column on in case it got killed by a groupby
        if "season" not in list(dur_z.columns):
            dur_z["season"] = dur_z["index"] * 0 + np.unique(seasons)

        # get simulation parameters
        if np.size(np.unique(seasons)) > 1:
            dur_z["z_copy"] = dur_z["z"].values.copy()
            dur_z["season_copy"] = dur_z["season"].values.copy()
            gen_par = (
                dur_z.groupby(["z", "season"])
                .apply(lambda x: self.calc_daymax(x, self.daymax_step), include_groups=False)
                .reset_index()
            )
        else:
            dur_z["z_copy"] = dur_z["z"].values.copy()
            gen_par = (
                dur_z.groupby(["z"])
                .apply(lambda x: self.calc_daymax(x, self.daymax_step), include_groups=False)
                .reset_index()
            )
            gen_par["season"] = gen_par["level_1"] * 0 + np.unique(seasons)

        if gen_par.empty:
            return None

        # select observations corresponding to seasons
        obs = pd.DataFrame(np.copy(data_slice))
        obs = obs[obs["season"].isin(seasons)]

        # metric values in a DataFrame
        metric_values = pd.DataFrame()

        # generate LC here
        lc = self.step_lc(obs, gen_par, x1=x1, color=color)

        if self.verbose:
            print("daymax values", lc["daymax"].unique(), len(lc["daymax"].unique()))
            print(
                lc[
                    [
                        "daymax",
                        "z",
                        "flux",
                        "fluxerr_photo",
                        "flux_e_sec",
                        "flux_5",
                        self.m5_col,
                    ]
                ]
            )

        if len(lc) == 0:
            return None

        # get observing efficiencies and build sn for metric
        lc.index = lc.index.droplevel()

        # estimate efficiencies
        sn_effis = self.step_efficiencies(lc)

        # estimate nsn
        sn = self.step_nsn(sn_effis, dur_z)

        # estimate redshift completeness
        if metric == "zlim":
            sn["season_copy"] = sn["season"].values.copy()
            metric_values = (
                sn.groupby(["season"]).apply(lambda x: self.zlim(x), include_groups=False).reset_index()
            )

        if metric == "nsn":
            sn = sn.merge(zlim, left_on=["season"], right_on=["season"])
            sn["season_copy"] = sn["season"].values.copy()
            metric_values = (
                sn.groupby(["season"]).apply(lambda x: self.nsn(x), include_groups=False).reset_index()
            )

        return metric_values

    def z_season(self, seasons, data_slice):
        """
        Fill the z values per season

        Parameters
        --------------
        seasons: list
          seasons to process
        data_slice: array
          data to process

        """
        # if seasons = -1: process the seasons seen in data
        if seasons == [-1]:
            seasons = np.unique(data_slice[self.season_col])

        # `pd.DataFrame` with zmin, zmax, zstep per season
        zseason = pd.DataFrame(seasons, columns=["season"])
        zseason["zmin"] = self.zmin
        zseason["zmax"] = self.zmax
        zseason["zstep"] = self.zstep

        return zseason

    def z_season_allz(self, zseason):

        zseason["season_copy"] = zseason["season"].values.copy()
        zseason_allz = (
            zseason.groupby(["season"])
            .apply(
                lambda x: pd.DataFrame(
                    {"z": list(np.arange(x["zmin"].mean(), x["zmax"].mean(), x["zstep"].mean()))}
                ),
                include_groups=False,
            )
            .reset_index()
        )

        return zseason_allz[["season", "z"]]

    def nsn_from_rate(self, grp):
        """
        Method to estimate the expected number of supernovae

        Parameters
        ---------------
        grp: `pd.DataFrame`
          data to process

        Returns
        -----------
        `pd.DataFrame` with z and nsn_expected as cols

        """
        durinterp_z = interp1d(
            grp["z"],
            grp["season_length"],
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        zz, rate, err_rate, nsn, err_nsn = self.rate_sn(
            zmin=self.zmin,
            zmax=self.zmax,
            dz=self.zstep,
            duration_z=durinterp_z,
            # duration=self.duration_ref,
            survey_area=self.pix_area,
            account_for_edges=False,
        )

        nsn_expected = interp1d(zz, nsn, kind="linear", bounds_error=False, fill_value=0)
        nsn_res = nsn_expected(grp["z"])

        return pd.DataFrame({"nsn_expected": nsn_res, "z": grp["z"].to_list()})

    def coadd(self, obs):
        """
        Method to coadd data per band and per night

        Parameters
        ------------
        data : `pd.DataFrame`
            `pd.DataFrame` of observations

        Returns
        -------
        coadded data : `pd.DataFrame`

        """

        data = pd.DataFrame(np.copy(obs))
        keygroup = [self.filter_col, self.night_col]

        data.sort_values(by=keygroup, ascending=[True, True], inplace=True)

        # get the median single exptime
        exptime_single = data[self.exptime_col].median()
        coadd_df = (
            data.groupby(keygroup)
            .agg(
                {
                    self.nexp_col: ["sum"],
                    self.vistime_col: ["sum"],
                    self.exptime_col: ["sum"],
                    self.mjd_col: ["mean"],
                    self.m5_col: ["mean"],
                }
            )
            .reset_index()
        )

        coadd_df.columns = [
            self.filter_col,
            self.night_col,
            self.nexp_col,
            self.vistime_col,
            self.exptime_col,
            self.mjd_col,
            self.m5_col,
        ]

        coadd_df = coadd_df.sort_values(by=self.mjd_col)
        coadd_df[self.m5_col] += 1.25 * np.log10(coadd_df[self.exptime_col] / exptime_single)

        return coadd_df.to_records(index=False)

    def getseason(self, obs, season_gap=80.0, mjd_col="observationStartMJD"):
        """
        Method to estimate seasons

        Parameters
        ------------
        obs: `np.ndarray`
            array of observations
        season_gap: `float`, optional
            minimal gap required to define a season (default: 80 days)
        mjd_col: `str`, optional
            col name for MJD infos (default: observationStartMJD)

        Returns
        ---------
        obs : `np.ndarray`
            original numpy array with seasonnumber appended
        """

        # check whether season has already been estimated
        obs.sort(order=mjd_col)

        seasoncalc = np.ones(obs.size, dtype=int)

        if len(obs) > 1:
            diff = np.diff(obs[mjd_col])
            flag = np.where(diff > season_gap)[0]

            if len(flag) > 0:
                for i, indx in enumerate(flag):
                    seasoncalc[indx + 1 :] = i + 2

        obs = nlr.append_fields(obs, "season", seasoncalc)

        return obs

    def reducen_sn(self, metric_val):
        # At each slice_point, return the sum nSN value.

        return np.sum(metric_val["n_sn"])

    def reducezlim(self, metric_val):
        # At each slice_point, return the median zlim
        result = np.median(metric_val["zlim"])
        if result < 0:
            result = self.badval

        return result

    def nsn_expected_z(self, grp):
        """
        Method to estimate the expected nsn  vs redshift

        Parameters
        --------------
        grp: `pd.DataFrame` group
          data to process: season infos

        Returns
        ----------
        `pd.DataFrame` with season_length, z, nsn_expected cols

        """

        if len(grp) < 2:
            nsn = pd.DataFrame(grp["z"].to_list(), columns=["z"])
            nsn.loc[:, "nsn_expected"] = 0
        else:
            nsn = self.nsn_from_rate(grp)

        if self.verbose:
            print("dur_z", grp)
            print("nsn expected", nsn)

        dur_z = pd.DataFrame(grp)
        dur_z = dur_z.merge(nsn, left_on=["z"], right_on=["z"])

        # if "season" in dur_z.columns:
        #    dur_z = dur_z.drop(columns=["season"])

        return dur_z

    def zlim_or_nsn(self, effi, sntype="faint", zlim=-1.0):
        """
        Method to estimate the redshift limit or the number of sn

        Parameters
        ---------------
        effi : `pd.DataFrame`
            data to process
        sntype : `str`, opt
            type of SN to consider for estimation (default: faint)
        zlim : `float`, opt
            redshift limit

        Returns
        -----------
        if zlim<0: returns the redshift limit
        if zlim>0: returns the number of sn up to zlim


        """

        seleffi = effi[effi["sntype"] == sntype]
        seleffi = seleffi.reset_index(drop=True)
        seleffi = seleffi.sort_values(by=["z"])
        nsn_cum = np.cumsum(seleffi["nsn"].to_list())

        resa = -1.0
        if zlim < 0:
            df = pd.DataFrame(seleffi).reset_index(drop=True)
            df.loc[:, "nsn_cum"] = nsn_cum / nsn_cum[-1]
            index = df[df["nsn_cum"] < 1].index
            if len(index) == 0:
                return resa
            dfb = df[: index[-1] + 2]
            zlim = interp1d(
                dfb["nsn_cum"],
                dfb["z"],
                kind="linear",
                bounds_error=False,
                fill_value=0,
            )
            resa = zlim(self.zlim_coeff)
        else:
            effi = interp1d(
                seleffi["z"],
                seleffi["effi"],
                kind="linear",
                bounds_error=False,
                fill_value=0,
            )
            durinterp_z = interp1d(
                seleffi["z"],
                seleffi["season_length"],
                kind="linear",
                bounds_error=False,
                fill_value=0,
            )
            zmin, zstep = 0.1, 0.001
            resa = self.get_nsn(effi, durinterp_z, zmin, zlim, zstep)

        return np.round(resa, 6)

    def zlim(self, grp, sn_type="faint"):
        """
        Method to estimate the metric zcomp

        Parameters
        ---------------
        grp: pd group
        sn_type: str, opt
          type of SN to estimate zlim (default: faint)

        Returns
        ------------
        zcomp : `pd.DataFrame` with the metric as cols
        """
        zcomp = -1

        if grp["effi"].mean() > 0.02 and len(grp["effi"]) >= 2:
            zcomp = self.zlim_or_nsn(grp, sn_type, -1)

        return pd.DataFrame({"zcomp": [zcomp]})

    def nsn(self, grp, sn_type="medium"):
        """
        Method to estimate the metric nsn up to zlim

        Parameters
        ---------------
        grp: pd group
        sn_type: str, opt
          type of SN to estimate zlim (default: medium)

        Returns
        ------------
        nsn : `pd.DataFrame`
            Dataframe with the metric as cols
        """
        nsn = -1

        if grp["effi"].mean() > 0.02:
            nsn = self.zlim_or_nsn(grp, sn_type, grp["zcomp"].mean())

        return pd.DataFrame({"nsn": [nsn]})

    def get_nsn(self, effi, durinterp_z, zmin, zmax, zstep):
        """
        Method to estimate to total number of SN: NSN = Sum(effi(z)*rate(z))

        Parameters
        -----------
        effi : 1D interpolator
            efficiencies vs z
        durinterp_z : 1D interpolator
            duration vs z
        zmin : `float`
            redshift min
        zmax : `float`
            redshift max
        zstep : `float`
            redshift step

        Returns
        ----------
        tot_sn : `int`
            total number of SN up to zmax
        """

        zz, rate, err_rate, nsn, err_nsn = self.rate_sn(
            zmin=zmin,
            zmax=zmax + zstep,
            dz=zstep,
            duration_z=durinterp_z,
            # duration=180.,
            survey_area=self.pix_area,
            account_for_edges=False,
        )
        res = np.cumsum(effi(zz) * nsn)
        if np.size(res) > 0:
            return res[-1]
        else:
            return 0

    def check_dur_z(self, dur_z, nmin=2):
        """ "
        Method to remove seasons with a poor redshift range due
        to too low season length

        Parameters
        ----------------
        dur_z: `pd.DataFrame`
          data to process
        nmin: int, opt
          minimal number of redshift points per season (default: 2)

        Returns
        -----------
        dur_z_subset : `pd.DataFrame`
            dur_z but only with seasons having at least nmin points in redshift

        """

        dur_z["size"] = dur_z.groupby(["season"])["z"].transform("size")

        idx = dur_z["size"] >= nmin

        return pd.DataFrame(dur_z[idx])
