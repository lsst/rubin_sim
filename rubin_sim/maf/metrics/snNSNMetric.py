import numpy as np
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.utils.snNSNUtils import SN_Rate
from rubin_sim.maf.utils.snNSNUtils import load_sne_cached, LCfast_new
import pandas as pd
from scipy.interpolate import interp1d
import numpy.lib.recfunctions as nlr
import healpy as hp
from rubin_sim.photUtils import Dust_values

__all__ = ["SNNSNMetric"]


class SNNSNMetric(BaseMetric):
    """
    Measure zlim of type Ia supernovae.

    Parameters
    --------------
    metricName : `str`, opt
        metric name (default : SNSNRMetric)
    mjdCol : `str`, opt
        mjd column name (default : observationStartMJD)
    filterCol : `str`, opt
        filter column name (default: filter)
    m5Col : `str`, opt
        five-sigma depth column name (default : fiveSigmaDepth)
    exptimeCol : `str`, opt
        exposure time column name (default : visitExposureTime)
    nightCol : `str`, opt
        night column name (default : night)
    obsidCol : `str`, opt
        observation id column name (default : observationId)
    nexpCol : `str`, opt
        number of exposure column name (default : numExposures)
    vistimeCol : `str`, opt
        visit time column name (default : visitTime)
    seeingCol : `str`, opt
        seeing column name (default: seeingFwhmEff)
    noteCol : `str`, opt
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
      If set, cut any point on the sky that has an ebv extinction higher than the hard_dust_cut value.
      Default 0.25
    """

    def __init__(
        self,
        metricName="SNNSNMetric",
        mjdCol="observationStartMJD",
        filterCol="filter",
        m5Col="fiveSigmaDepth",
        exptimeCol="visitExposureTime",
        nightCol="night",
        obsidCol="observationId",
        nexpCol="numExposures",
        vistimeCol="visitTime",
        seeingCol="seeingFwhmEff",
        noteCol="note",
        season=[-1],
        coadd_night=True,
        zmin=0.1,
        zmax=0.5,
        zStep=0.03,
        daymaxStep=3.0,
        verbose=False,
        ploteffi=False,
        n_bef=3,
        n_aft=8,
        snr_min=1.0,
        n_phase_min=1,
        n_phase_max=1,
        sigmaC=0.04,
        zlim_coeff=0.95,
        bands="grizy",
        add_dust=False,
        hard_dust_cut=0.25,
        gammaName="gamma_WFD.hdf5",
        **kwargs,
    ):
        # n_bef / n_aft = 3/8 for WFD, 4/10 for DDF

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.exptimeCol = exptimeCol
        self.seasonCol = "season"
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol
        self.seeingCol = seeingCol
        self.noteCol = noteCol

        self.ploteffi = ploteffi
        self.T0s = "all"
        self.zlim_coeff = zlim_coeff
        self.bands = bands
        self.coadd_night = coadd_night
        self.add_dust = add_dust
        self.hard_dust_cut = hard_dust_cut

        maps = ["DustMap"]
        dust_properties = Dust_values()
        self.Ax1 = dust_properties.Ax1

        cols = [
            self.nightCol,
            self.m5Col,
            self.filterCol,
            self.mjdCol,
            self.obsidCol,
            self.nexpCol,
            self.vistimeCol,
            self.exptimeCol,
            self.noteCol,
        ]

        super(SNNSNMetric, self).__init__(
            col=cols, metricDtype="object", metricName=metricName, maps=maps, **kwargs
        )

        self.season = season

        # LC selection parameters
        self.n_bef = n_bef  # nb points before peak
        self.n_aft = n_aft  # nb points after peak
        self.snr_min = snr_min  # SNR cut for points before/after peak
        self.n_phase_min = n_phase_min  # nb of point with phase <=-5
        self.n_phase_max = n_phase_max  # nb of points with phase >=20
        self.sigmaC = sigmaC

        # loading reference LC files
        lc_reference = load_sne_cached(gammaName)
        # loading reference LC files

        self.lcFast = {}
        for key, vals in lc_reference.items():
            self.lcFast[key] = LCfast_new(
                vals,
                key[0],
                key[1],
                self.mjdCol,
                self.filterCol,
                self.exptimeCol,
                self.m5Col,
                self.seasonCol,
                self.nexpCol,
                self.seeingCol,
                self.snr_min,
                lightOutput=False,
            )
        # loading parameters
        self.zmin = zmin  # zmin for the study
        self.zmax = zmax  # zmax for the study
        self.zstep = zStep  # zstep
        # get redshift range for processing
        zrange = list(np.arange(self.zmin, self.zmax, self.zstep))
        if zrange[0] < 1.0e-6:
            zrange[0] = 0.01

        self.zrange = np.unique(zrange)

        self.daymaxStep = daymaxStep  # daymax step
        self.min_rf_phase = -20.0  # min ref phase for LC points selection
        self.max_rf_phase = 60.0  # max ref phase for LC points selection

        self.min_rf_phase_qual = -15.0  # min ref phase for bounds effects
        self.max_rf_phase_qual = 30.0  # max ref phase for bounds effects

        # snrate
        self.rateSN = SN_Rate(
            H0=70.0,
            Om0=0.3,
            min_rf_phase=self.min_rf_phase_qual,
            max_rf_phase=self.max_rf_phase_qual,
        )

        # verbose mode - useful for debug and code performance estimation
        self.verbose = verbose

        # supernovae parameters for fisher estimation
        self.params = ["x0", "x1", "daymax", "color"]

    def run(self, dataSlice, slicePoint):
        """
        Run method of the metric

        Parameters
        --------------
        dataSlice: `np.array`
            Observations to process (scheduler simulations)
        slicePoint: `bool`, opt
            Information about the location on the sky from the slicer

        Returns
        -------
        metricVal : `np.recarray`
            ['nSN', 'zlim'] at this point on the sky
        """
        # Hard dust cut
        if self.hard_dust_cut is not None:
            ebvofMW = slicePoint["ebv"]
            if ebvofMW > self.hard_dust_cut:
                return self.badval

        # get area on-sky in this slicePoint
        if "nside" in slicePoint:
            self.pixArea = hp.nside2pixarea(slicePoint["nside"], degrees=True)
        else:
            self.pixArea = 9.6

        # If we want to apply dust extinction.
        if self.add_dust:
            new_m5 = dataSlice[self.m5Col] * 0
            for filtername in np.unique(dataSlice[self.filterCol]):
                in_filt = np.where(dataSlice[self.filterCol] == filtername)[0]
                A_x = self.Ax1[filtername] * slicePoint["ebv"]
                new_m5[in_filt] = dataSlice[self.m5Col][in_filt] - A_x
            dataSlice[self.m5Col] = new_m5

        # select observations filter
        goodFilters = np.in1d(dataSlice[self.filterCol], list(self.bands))
        dataSlice = dataSlice[goodFilters]

        # coaddition per night and per band (if requested by the user)
        if self.coadd_night:
            dataSlice = self.coadd(dataSlice)

        # This seems unlikely, but is a possible bailout point
        if len(dataSlice) <= self.n_aft + self.n_bef:
            return self.badval

        # get season information (seasons calculated by gaps, not by place on sky)
        dataSlice = self.getseason(dataSlice, mjdCol=self.mjdCol)

        # get redshift values per season
        zseason = self.z_season(self.season, dataSlice)
        zseason_allz = self.z_season_allz(zseason)

        # estimate redshift completeness
        metricValues = self.metric(
            dataSlice, zseason_allz, x1=-2.0, color=0.2, zlim=-1, metric="zlim"
        )

        if metricValues is None:
            return self.badval

        if np.max(metricValues["zcomp"]) < 0:
            return self.badval

        # get redshift values per season up to zcomp
        zseason = pd.DataFrame(metricValues[["season", "zcomp"]])
        zseason.loc[:, "zmin"] = 0.01
        zseason.loc[:, "zstep"] = self.zstep
        zseason = zseason.rename(columns={"zcomp": "zmax"})
        zseason["zmax"] += self.zstep
        zseason_allz = self.z_season_allz(zseason)

        # get the total number of well-sampled SN up to zcomp
        nsn_zcomp = self.metric(
            dataSlice,
            zseason_allz,
            x1=0.0,
            color=0.0,
            zlim=metricValues[["season", "zcomp"]],
            metric="nsn",
        )

        # final results
        if nsn_zcomp is None:
            return self.badval
        metricValues = metricValues.merge(
            nsn_zcomp, left_on=["season"], right_on=["season"]
        )

        if self.verbose:
            print("metricValues", metricValues[["season", "zcomp", "nsn"]])

        idx = metricValues["zcomp"] > 0.0
        selmet = metricValues[idx]

        if len(selmet) > 0:
            zcomp = selmet["zcomp"].median()
            nSN = selmet["nsn"].sum()
            res = np.rec.fromrecords([(nSN, zcomp)], names=["nSN", "zlim"])
        else:
            res = self.badval

        if self.verbose:
            print("final result", res)

        return res

    def season_length(self, seasons, dataSlice, zseason):
        """
        Method to estimate season lengths vs z

        Parameters
        ---------------
        seasons : list(int)
          list of seasons to process
        dataSlice: numpy array
          array of observations

        Returns
        -----------
        seasons : list(int)
          list of seasons to process
        dur_z : pandas df
          season lengths vs z
        """
        # if seasons = -1: process the seasons seen in data
        if seasons == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # season infos
        dfa = pd.DataFrame(np.copy(dataSlice))
        dfa = pd.DataFrame(dfa[dfa["season"].isin(seasons)])

        season_info = self.get_season_info(dfa, zseason)

        if self.verbose:
            print("season_info", season_info)

        if season_info.empty:
            return [], pd.DataFrame()

        dur_z = (
            season_info.groupby("season", group_keys=False)
            .apply(lambda x: self.nsn_expected_z(x))
            .reset_index(drop=True)
        )
        return season_info["season"].to_list(), dur_z

    def get_season_info(self, dfa, zseason, min_duration=60.0):
        """
        method to get season infos vs z

        Parameters
        --------------
        dfa: pandas df
          dat to process
        zseason: pandas df
          redshift infos per season
        min_duration: float, opt
          min season length to be accepted (default: 60 days)

        Returns
        ----------
        pandas df with season length infos

        """

        season_info = (
            dfa.groupby("season")
            .apply(lambda x: self.seasonInfo(x, min_duration=min_duration))
            .reset_index()
        )

        # season_info.index = season_info.index.droplevel()
        season_info = season_info.drop(columns=["level_1"])

        season_info = season_info.merge(
            zseason, left_on=["season"], right_on=["season"]
        )

        season_info["T0_min"] = (
            season_info["MJD_min"] - (1.0 + season_info["z"]) * self.min_rf_phase_qual
        )
        season_info["T0_max"] = (
            season_info["MJD_max"] - (1.0 + season_info["z"]) * self.max_rf_phase_qual
        )
        season_info["season_length"] = season_info["T0_max"] - season_info["T0_min"]

        idx = season_info["season_length"] >= min_duration

        return pd.DataFrame(season_info[idx])

    def step_lc(self, obs, gen_par, x1=-2.0, color=0.2):
        """
        Method to generate lc

        Parameters
        ---------------
        obs: array
          observations
        gen_par: array
          simulation parameters
        x1: float, opt
           stretch value (default: -2.0)
        color: float, opt
          color value (default: 0.2)

        Returns
        ----------
        SN light curves (astropy table)

        """
        lc = obs.groupby(["season"]).apply(lambda x: self.genLC(x, gen_par, x1, color))
        return lc

    def step_efficiencies(self, lc):
        """
        Method to estimate observing efficiencies

        Parameter
        -------------
        lc: pandas df
           light curves

        Returns
        -----------
        pandas df with efficiencies

        """
        # sn_effis = lc.groupby(['healpixID', 'season', 'z', 'x1', 'color', 'sntype']).apply(
        #    lambda x: self.sn_effi(x)).reset_index()

        sn_effis = (
            lc.groupby(["season", "z", "x1", "color", "sntype"])
            .apply(lambda x: self.sn_effi(x))
            .reset_index()
        )

        # estimate efficiencies
        sn_effis["season"] = sn_effis["season"].astype(int)
        sn_effis["effi"] = sn_effis["nsel"] / sn_effis["ntot"]
        sn_effis["effi_err"] = (
            np.sqrt(sn_effis["nsel"] * (1.0 - sn_effis["effi"])) / sn_effis["ntot"]
        )

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
                plotNSN_effi(
                    sn_effis[idx], "effi", "effi_err", "Observing Efficiencies", ls="-"
                )

        return sn_effis

    def step_nsn(self, sn_effis, dur_z):
        """
        Method to estimate the number of supernovae from efficiencies

        Parameters
        ---------------
        sn_effis: pandas df
          data with efficiencies of observation
        dur_z:  array
          array of season length

        Returns
        ----------
        initial sn_effis appended with a set of infos (duration, nsn)

        """
        # add season length here
        sn_effis = sn_effis.merge(
            dur_z, left_on=["season", "z"], right_on=["season", "z"]
        )

        # estimate the number of supernovae
        sn_effis["nsn"] = sn_effis["effi"] * sn_effis["nsn_expected"]

        return sn_effis

    def seasonInfo(self, grp, min_duration):
        """
        Method to estimate seasonal info (cadence, season length, ...)

        Parameters
        --------------
        grp: pandas df group
        min_duration: float
          minimal duration for a season to be considered

        Returns
        ---------
        pandas df with the following cols:
        - Nvisits: number of visits for this group
        - N_xx:  number of visits in xx where xx is defined in self.bandstat

        """
        df = pd.DataFrame([len(grp)], columns=["Nvisits"])
        df["MJD_min"] = grp[self.mjdCol].min()
        df["MJD_max"] = grp[self.mjdCol].max()
        df["season_length"] = df["MJD_max"] - df["MJD_min"]
        df["cadence"] = 0.0

        if len(grp) > 5:
            # to = grp.groupby(['night'])[self.mjdCol].median().sort_values()
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
        grp: pandas df group
          data to process: season infos
        min_duration: float, opt
          min season length for a season to be processed (deafult: 60 days)

        Returns
        ----------
        pandas df with season_length, z, T0_min and T0_max cols

        """

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

    def calcDaymax(self, grp, daymaxStep):
        """
        Method to estimate T0 (daymax) values for simulation.

        Parameters
        --------------
        grp: group (pandas df sense)
         group of data to process with the following cols:
           T0_min: T0 min value (per season)
           T0_max: T0 max value (per season)
        daymaxStep: float
          step for T0 simulation

        Returns
        ----------
        pandas df with daymax, min_rf_phase, max_rf_phase values

        """

        if self.T0s == "all":
            T0_max = grp["T0_max"].values
            T0_min = grp["T0_min"].values
            num = (T0_max - T0_min) / daymaxStep
            if T0_max - T0_min > 10:
                df = pd.DataFrame(
                    np.linspace(T0_min, T0_max, int(num)), columns=["daymax"]
                )
            else:
                df = pd.DataFrame([-1], columns=["daymax"])
        else:
            df = pd.DataFrame([0.0], columns=["daymax"])

        df["minRFphase"] = self.min_rf_phase
        df["maxRFphase"] = self.max_rf_phase

        return df

    def genLC(self, grp, gen_par_orig, x1, color):
        """
        Method to generate light curves from observations

        Parameters
        ---------------
        grp: pandas group
          observations to process
        gen_par_orig: pandas df
          simulation parameters
        x1: float
          SN stretch
        color: float
          SN color

        Returns
        ----------
        light curves as pandas df

        """
        season = grp.name
        idx = gen_par_orig["season"] == season
        gen_par = gen_par_orig[idx].to_records(index=False)

        sntype = dict(zip([(-2.0, 0.2), (0.0, 0.0)], ["faint", "medium"]))
        res = pd.DataFrame()
        key = (np.round(x1, 1), np.round(color, 1))
        vals = self.lcFast[key]

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
        lc: pandas grp
          light curve

        Returns
        ----------
        pandas df of sn efficiencies vs z
        """

        if self.verbose:
            print("effi for", lc.name)

        lcarr = lc.to_records(index=False)

        idx = lcarr["snr_m5"] >= self.snr_min

        lcarr = np.copy(lcarr[idx])

        T0s = np.unique(lcarr["daymax"])
        T0s.sort()

        deltaT = lcarr["daymax"] - T0s[:, np.newaxis]

        flag = np.abs(deltaT) < 1.0e-5

        resdf = pd.DataFrame(T0s, columns=["daymax"])

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
            resdf[vv] = self.get_sum(lcarr, vv, len(deltaT), flag)

        nights = np.tile(lcarr["night"], (len(deltaT), 1))
        phases = np.tile(lcarr["phase"], (len(deltaT), 1))

        flagph = phases >= 0.0
        resdf["nepochs_aft"] = self.get_epochs(nights, flag, flagph)
        flagph = phases <= 0.0
        resdf["nepochs_bef"] = self.get_epochs(nights, flag, flagph)

        # replace NaN by 0
        resdf = resdf.fillna(0)

        # get selection efficiencies
        effis = self.efficiencies(resdf)

        return effis

    def get_sum(self, lcarr, varname, nvals, flag):
        """
        Method to get the sum of variables using broadcasting

        Parameters
        --------------
        lcarr: numpy array
          data to process
        varname: str
          col to process in lcarr
        nvals: int
          dimension for tiling
        flag: array(bool)
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
        nights: array
          night number array
        flag: array(bool)
          flag to apply
        flagph: array(bool)
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

    def sigmaSNparams(self, grp):
        """
        Method to estimate variances of SN parameters
        from inversion of the Fisher matrix

        Parameters
        ---------------
        grp: pandas df of flux derivatives wrt SN parameters
        Returns
        ----------
        Diagonal elements of the inverted matrix (as pandas df)
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
        Fisher_Big = np.zeros((npar * size, npar * size))
        Big_Diag = np.zeros((npar * size, npar * size))
        Big_Diag = []

        for iv in range(size):
            for ia, vala in enumerate(self.params):
                for jb, valb in enumerate(self.params):
                    if jb >= ia:
                        Fisher_Big[ia + npar * iv][jb + npar * iv] = parts[ia, jb][iv]

        # pprint.pprint(Fisher_Big)

        Fisher_Big = Fisher_Big + np.triu(Fisher_Big, 1).T
        Big_Diag = np.diag(np.linalg.inv(Fisher_Big))

        res = pd.DataFrame()
        for ia, vala in enumerate(self.params):
            indices = range(ia, len(Big_Diag), npar)
            res["Cov_{}{}".format(vala, vala)] = np.take(Big_Diag, indices)

        return res

    def efficiencies(self, dfo):
        """ "
        Method to estimate selection efficiencies

        Parameters
        ---------------
        df: pandas df
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

        badSN = pd.DataFrame(df.loc[~idx])
        goodSN = pd.DataFrame()
        if len(df[idx]) > 0:
            goodSN = pd.DataFrame(df.loc[idx].reset_index())
            sigma_Fisher = self.sigmaSNparams(goodSN)
            goodSN["Cov_colorcolor"] = sigma_Fisher["Cov_colorcolor"]

        allSN = pd.concat((goodSN, badSN))
        allSN["select"] &= allSN["Cov_colorcolor"] <= self.sigmaC**2
        idx = allSN["select"] == 1

        if self.verbose:
            if len(goodSN) > 0:
                print("good SN", len(goodSN), goodSN[["daymax", "Cov_colorcolor"]])
            else:
                print("no good SN")
        return pd.DataFrame({"ntot": [len(allSN)], "nsel": [len(allSN[idx])]})

    def metric(self, dataSlice, zseason, x1=-2.0, color=0.2, zlim=-1, metric="zlim"):
        """
        Method to run the metric

        Parameters
        ---------------
        dataSlice: array
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
        seasons, dur_z = self.season_length(self.season, dataSlice, zseason)
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
            gen_par = (
                dur_z.groupby(["z", "season"])
                .apply(lambda x: self.calcDaymax(x, self.daymaxStep))
                .reset_index()
            )
        else:
            gen_par = (
                dur_z.groupby(["z"])
                .apply(lambda x: self.calcDaymax(x, self.daymaxStep))
                .reset_index()
            )
            gen_par["season"] = gen_par["level_1"] * 0 + np.unique(seasons)

        if gen_par.empty:
            return None

        # select observations corresponding to seasons
        obs = pd.DataFrame(np.copy(dataSlice))
        obs = obs[obs["season"].isin(seasons)]

        # metric values in a DataFrame
        metricValues = pd.DataFrame()

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
                        self.m5Col,
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
            metricValues = (
                sn.groupby(["season"]).apply(lambda x: self.zlim(x)).reset_index()
            )

        if metric == "nsn":
            sn = sn.merge(zlim, left_on=["season"], right_on=["season"])
            metricValues = (
                sn.groupby(["season"]).apply(lambda x: self.nsn(x)).reset_index()
            )

        return metricValues

    def z_season(self, seasons, dataSlice):
        """
        Fill the z values per season

        Parameters
        --------------
        seasons: list
          seasons to process
        dataSlice: array
          data to process

        """
        # if seasons = -1: process the seasons seen in data
        if seasons == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # pandas df with zmin, zmax, zstep per season
        zseason = pd.DataFrame(seasons, columns=["season"])
        zseason["zmin"] = self.zmin
        zseason["zmax"] = self.zmax
        zseason["zstep"] = self.zstep

        return zseason

    def z_season_allz(self, zseason):

        zseason_allz = (
            zseason.groupby(["season"])
            .apply(
                lambda x: pd.DataFrame(
                    {
                        "z": list(
                            np.arange(
                                x["zmin"].mean(), x["zmax"].mean(), x["zstep"].mean()
                            )
                        )
                    }
                )
            )
            .reset_index()
        )

        return zseason_allz[["season", "z"]]

    def nsn_from_rate(self, grp):
        """
        Method to estimate the expected number of supernovae

        Parameters
        ---------------
        grp: pandas df
          data to process

        Returns
        -----------
        pandas df with z and nsn_expected as cols

        """
        durinterp_z = interp1d(
            grp["z"],
            grp["season_length"],
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(
            zmin=self.zmin,
            zmax=self.zmax,
            dz=self.zstep,
            duration_z=durinterp_z,
            # duration=self.duration_ref,
            survey_area=self.pixArea,
            account_for_edges=False,
        )

        nsn_expected = interp1d(
            zz, nsn, kind="linear", bounds_error=False, fill_value=0
        )
        nsn_res = nsn_expected(grp["z"])

        return pd.DataFrame({"nsn_expected": nsn_res, "z": grp["z"].to_list()})

    def coadd(self, obs):
        """
        Method to coadd data per band and per night

        Parameters
        ------------
        data : `pd.DataFrame`
            pandas df of observations

        Returns
        -------
        coadded data : `pd.DataFrame`

        """

        data = pd.DataFrame(np.copy(obs))
        keygroup = [self.filterCol, self.nightCol]

        data.sort_values(by=keygroup, ascending=[True, True], inplace=True)

        # get the median single exptime
        exptime_single = data[self.exptimeCol].median()
        coadd_df = (
            data.groupby(keygroup)
            .agg(
                {
                    self.nexpCol: ["sum"],
                    self.vistimeCol: ["sum"],
                    self.exptimeCol: ["sum"],
                    self.mjdCol: ["mean"],
                    self.m5Col: ["mean"],
                }
            )
            .reset_index()
        )

        coadd_df.columns = [
            self.filterCol,
            self.nightCol,
            self.nexpCol,
            self.vistimeCol,
            self.exptimeCol,
            self.mjdCol,
            self.m5Col,
        ]

        coadd_df = coadd_df.sort_values(by=self.mjdCol)
        coadd_df[self.m5Col] += 1.25 * np.log10(
            coadd_df[self.exptimeCol] / exptime_single
        )

        return coadd_df.to_records(index=False)

    def getseason(self, obs, season_gap=80.0, mjdCol="observationStartMJD"):
        """
        Method to estimate seasons

        Parameters
        ------------
        obs: `np.ndarray`
            array of observations
        season_gap: `float`, optional
            minimal gap required to define a season (default: 80 days)
        mjdCol: `str`, optional
            col name for MJD infos (default: observationStartMJD)

        Returns
        ---------
        obs : `np.ndarray`
            original numpy array with seasonnumber appended
        """

        # check whether season has already been estimated
        obs.sort(order=mjdCol)

        seasoncalc = np.ones(obs.size, dtype=int)

        if len(obs) > 1:
            diff = np.diff(obs[mjdCol])
            flag = np.where(diff > season_gap)[0]

            if len(flag) > 0:
                for i, indx in enumerate(flag):
                    seasoncalc[indx + 1 :] = i + 2

        obs = nlr.append_fields(obs, "season", seasoncalc)

        return obs

    def reducenSN(self, metricVal):

        # At each slicepoint, return the sum nSN value.

        return np.sum(metricVal["nSN"])

    def reducezlim(self, metricVal):

        # At each slicepoint, return the median zlim
        result = np.median(metricVal["zlim"])
        if result < 0:
            result = self.badval

        return result

    def nsn_expected_z(self, grp):
        """
        Method to estimate the expected nsn  vs redshift

        Parameters
        --------------
        grp: pandas df group
          data to process: season infos

        Returns
        ----------
        pandas df with season_length, z, nsn_expected cols

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
        effi: pandas df
          data to process
        sntype: str, opt
          type of SN to consider for estimation (default: faint)
        zlim: float, opt
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

    def zlim(self, grp, snType="faint"):
        """
        Method to estimate the metric zcomp

        Parameters
        ---------------
        grp: pandas group
        snType: str, opt
          type of SN to estimate zlim (default: faint)

        Returns
        ------------
        pandas df with the metric as cols
        """
        zcomp = -1

        if grp["effi"].mean() > 0.02 and len(grp["effi"]) >= 2:
            zcomp = self.zlim_or_nsn(grp, snType, -1)

        return pd.DataFrame({"zcomp": [zcomp]})

    def nsn(self, grp, snType="medium"):
        """
        Method to estimate the metric nsn up to zlim

        Parameters
        ---------------
        grp: pandas group
        snType: str, opt
          type of SN to estimate zlim (default: medium)

        Returns
        ------------
        pandas df with the metric as cols
        """
        nsn = -1

        if grp["effi"].mean() > 0.02:
            nsn = self.zlim_or_nsn(grp, snType, grp["zcomp"].mean())

        return pd.DataFrame({"nsn": [nsn]})

    def get_nsn(self, effi, durinterp_z, zmin, zmax, zstep):
        """
         Method to estimate to total number of SN: NSN = Sum(effi(z)*rate(z))

         Parameters
         ---------------
         effi: 1D interpolator
           efficiencies vs z
         durinterp_z: 1D interpolator
          duration vs z
         zmin: float
           redshift min
         zmax: float
           redshift max
         zstep: float
           redshift step

        Returns
        ----------
         total number of SN up to zmax


        """

        zz, rate, err_rate, nsn, err_nsn = self.rateSN(
            zmin=zmin,
            zmax=zmax + zstep,
            dz=zstep,
            duration_z=durinterp_z,
            # duration=180.,
            survey_area=self.pixArea,
            account_for_edges=False,
        )
        res = np.cumsum(effi(zz) * nsn)
        if np.size(res) > 0:
            return res[-1]
        else:
            return 0

    def check_dur_z(self, dur_z, nmin=2):
        """ "
        Method to remove seasons with a poor redshift range due to too low season length

        Parameters
        ----------------
        dur_z: pandas df
          data to process
        nmin: int, opt
          minimal number of redshift points per season (default: 2)

        Returns
        -----------
        pandas df with seasons having at least nmin points in redshift

        """

        dur_z["size"] = dur_z.groupby(["season"])["z"].transform("size")

        idx = dur_z["size"] >= nmin

        return pd.DataFrame(dur_z[idx])
