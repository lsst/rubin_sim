import numpy as np
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.utils.snNSNUtils import load_sne_cached, Telescope, LCfast
from rubin_sim.maf.utils.snNSNUtils import SN_Rate, CovColor
import pandas as pd
import numpy.lib.recfunctions as rf
import time
from scipy.interpolate import interp1d
import numpy.lib.recfunctions as nlr
from rubin_sim.photUtils import Dust_values

__all__ = ["SNNSNMetric"]


def effiObsdf(data, color_cut=0.04):
    """
    Method to estimate observing efficiencies for supernovae

    Parameters
    -----------
    data: pandas df - grp
        data to process

    Returns
    ----------
    pandas df with the following cols:
        - cols used to make the group
        - effi, effi_err: observing efficiency and associated error
    """

    # reference df to estimate efficiencies
    df = data.loc[lambda dfa: np.sqrt(dfa["Cov_colorcolor"]) < 100000.0, :]

    # selection on sigma_c<= 0.04
    df_sel = df.loc[lambda dfa: np.sqrt(dfa["Cov_colorcolor"]) <= color_cut, :]

    # make groups (with z)
    group = df.groupby("z")
    group_sel = df_sel.groupby("z")

    # Take the ratio to get efficiencies
    rb = group_sel.size() / group.size()
    err = np.sqrt(rb * (1.0 - rb) / group.size())
    var = rb * (1.0 - rb) * group.size()

    rb = rb.array
    err = err.array
    var = var.array

    rb[np.isnan(rb)] = 0.0
    err[np.isnan(err)] = 0.0
    var[np.isnan(var)] = 0.0

    return pd.DataFrame(
        {
            group.keys: list(group.groups.keys()),
            "effi": rb,
            "effi_err": err,
            "effi_var": var,
        }
    )


class SNNSNMetric(BaseMetric):
    """
    Estimate (nSN,zlim) of type Ia supernovae.

    Parameters
    -----------
    metricName : `str`, optional
        metric name (default : SNSNRMetric)
    mjdCol : `str`, optional
        mjd column name (default : observationStartMJD)
    RACol : `str`, optional
        Right Ascension column name (default : fieldRA)
    DecCol : `str`, optional
        Declination column name (default : fieldDec)
    filterCol : `str`, optional
        filter column name (default: filter)
    m5Col : `str`, optional
        five-sigma depth column name (default : fiveSigmaDepth)
    exptimeCol : `str`, optional
        exposure time column name (default : visitExposureTime)
    nightCol : `str`, optional
        night column name (default : night)
    obsidCol : `str`, optional
        observation id column name (default : observationId)
    nexpCol : `str`, optional
        number of exposure column name (default : numExposures)
    vistimeCol : `str`, optional
        visit time column name (default : visitTime)
    season : `list` of `int`, optional
        list of seasons to process (float)(default: -1 = all seasons)
    zmin : `float`, optional
        min redshift for the study (default: 0.0)
    zmax : `float`, optional
        max redshift for the study (default: 1.2)
    pixArea : `float`, optional
        pixel area (default: 9.6)
    verbose : `bool`, optional
        verbose mode (default: False)
    ploteffi : `bool`, optional
        to plot observing efficiencies vs z (default: False)
    n_bef : `int`, optional
        number of LC points LC before T0 (default:5)
    n_aft : `int`, optional
        number of LC points after T0 (default: 10)
    snr_min : `float`, optional
        minimal SNR of LC points (default: 5.0)
    n_phase_min : `int`, optional
        number of LC points with phase<= -5(default:1)
    n_phase_max : `int`, optional
        number of LC points with phase>= 20 (default: 1)
    templateDir: str, opt
        template directory for reference files (default: None)
    zlim_coeff: float, opt
      ith percentile to estimate the redshift completeness (default: 0.95)
    dust: bool, opt
      to apply dust (default: False)
     mjd_LSST_Start: float, opt
       mjd LSST start (this is just to have a first night estimator) (default: 60218.83514)
     bands: str, opt
       filters to consider (default: 'griz')
    """

    def __init__(
        self,
        metricName="SNNSNMetric",
        mjdCol="observationStartMJD",
        RACol="fieldRA",
        DecCol="fieldDec",
        filterCol="filter",
        m5Col="fiveSigmaDepth",
        exptimeCol="visitExposureTime",
        nightCol="night",
        obsidCol="observationId",
        nexpCol="numExposures",
        vistimeCol="visitTime",
        season=[-1],
        zmin=0.10,
        zmax=0.50,
        zStep=0.03,
        daymaxStep=4.,
        pixArea=9.6,
        verbose=True,
        ploteffi=False,
        n_bef=3,
        n_aft=8,
        snr_min=1.0,
        n_phase_min=1,
        n_phase_max=1,
        templateDir=None,
        zlim_coeff=0.95,
        dust=False,
        mjd_LSST_Start=60218.83514,
        bands='griz',
        **kwargs
    ):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RACol = RACol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = "season"
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol
        self.pixArea = pixArea
        self.zlim_coeff = zlim_coeff
        self.dust = dust
        if dust:
            maps = ["DustMap"]
            dust_properties = Dust_values()
            self.Ax1 = dust_properties.Ax1
        else:
            maps = []

        cols = [
            self.nightCol,
            self.m5Col,
            self.filterCol,
            self.mjdCol,
            self.obsidCol,
            self.nexpCol,
            self.vistimeCol,
            self.exptimeCol,
            'seeingFwhmEff',
            'seeingFwhmGeom',
            'airmass',
            'moonPhase'
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

        # loading reference LC files
        lc_reference = load_sne_cached()

        self.lcFast = {}
        telescope = Telescope(airmass=1.2)
        for key, vals in lc_reference.items():
            self.lcFast[key] = LCfast(
                vals,
                key[0],
                key[1],
                telescope,
                self.mjdCol,
                self.RACol,
                self.DecCol,
                self.filterCol,
                self.exptimeCol,
                self.m5Col,
                self.seasonCol,
                self.nexpCol,
                self.snr_min,
            )

        # loading parameters
        self.zmin = zmin  # zmin for the study
        self.zmax = zmax  # zmax for the study
        self.zStep = zStep  # zstep
        self.daymaxStep = daymaxStep  # daymax step
        self.min_rf_phase = -20.0  # min ref phase for LC points selection
        self.max_rf_phase = 60.0  # max ref phase for LC points selection

        self.min_rf_phase_qual = -15.0  # min ref phase for bounds effects
        self.max_rf_phase_qual = 30.0  # max ref phase for bounds effects

        # snrate
        self.rateSN = SN_Rate(H0=70., Om0=0.3,
                              min_rf_phase=self.min_rf_phase_qual, max_rf_phase=self.max_rf_phase_qual
                              )

        # verbose mode - useful for debug and code performance estimation
        self.verbose = verbose
        self.ploteffi = ploteffi

        # supernovae parameters
        self.params = ["x0", "x1", "daymax", "color"]

        # r = [(-1.0, -1.0)]
        self.bad = np.rec.fromrecords([(-1.0, -1.0)], names=["nSN", "zlim"])
        # self.bad = {'nSN': -1.0, 'zlim': -1.0}

        # LSST starting date
        self.mjd_LSST_Start = mjd_LSST_Start

        # filters
        self.bands = bands

    def run(self, dataSlice, slicePoint=None):
        idarray = None
        healpixID = -1
        pixRA = -1
        pixDec = -1
        if slicePoint is not None:
            if "nside" in slicePoint.keys():
                import healpy as hp

                self.pixArea = hp.nside2pixarea(
                    slicePoint["nside"], degrees=True)
                r = []
                names = []

                healpixID = hp.ang2pix(
                    slicePoint["nside"],
                    np.rad2deg(slicePoint["ra"]),
                    np.rad2deg(slicePoint["dec"]),
                    nest=True,
                    lonlat=True,
                )
                coord = hp.pix2ang(slicePoint["nside"],
                                   healpixID, nest=True, lonlat=True)
                pixRA = coord[0]
                pixDec = coord[1]
                for kk, vv in slicePoint.items():
                    r.append(vv)
                    names.append(kk)
                idarray = np.rec.fromrecords([r], names=names)
        else:
            idarray = np.rec.fromrecords([0.0, 0.0], names=["RA", "Dec"])

        # select obs with good filters
        goodFilters = np.in1d(dataSlice[self.filterCol], list(self.bands))
        dataSlice = dataSlice[goodFilters]

        dataSlice.sort(order=self.mjdCol)

        # Two things to do: concatenate data (per band, night) and estimate seasons
        dataSlice = rf.drop_fields(dataSlice, ["season"])

        dataSlice = self.coadd(pd.DataFrame(dataSlice))

        dataSlice = self.getseason(dataSlice, mjdCol=self.mjdCol)

        # If we want to apply dust extinction.

        if self.dust:
            new_m5 = dataSlice[self.m5Col] * 0
            for filtername in np.unique(dataSlice[self.filterCol]):
                in_filt = np.where(dataSlice[self.filterCol] == filtername)[0]
                A_x = self.Ax1[filtername] * slicePoint["ebv"]
                new_m5[in_filt] = dataSlice[self.m5Col][in_filt] - A_x
            dataSlice[self.m5Col] = new_m5

        # get the seasons
        seasons = self.season

        # if seasons = -1: process the seasons seen in data
        if self.season == [-1]:
            seasons = np.unique(dataSlice[self.seasonCol])

        # get redshift range for processing
        zRange = list(np.arange(self.zmin, self.zmax, self.zStep))
        if zRange[0] < 1.0e-6:
            zRange[0] = 0.01

        self.zRange = np.unique(zRange)
        # season infos
        dfa = pd.DataFrame(np.copy(dataSlice))

        season_info = dfa.groupby(["season"]).apply(
            lambda x: self.seasonInfo(x)).reset_index()

        # select seasons of at least 30 days
        idx = season_info["season_length"] >= 60.0
        season_info = season_info[idx]

        # If we don't have many observations, bail out
        if season_info.empty:
            return nlr.merge_arrays([idarray, self.bad], flatten=True)
        else:
            if self.verbose:
                print('season info', season_info)

        # check wether requested seasons can be processed
        test_season = season_info[season_info["season"].isin(seasons)]

        if test_season.empty:
            return nlr.merge_arrays([idarray, self.bad], flatten=True)
        else:
            seasons = test_season["season"]

        if self.verbose:
            print('info season', seasons)

        # get season length depending on the redshift
        dur_z = (
            season_info.groupby(["season"])
            .apply(lambda x: self.duration_z(x))
            .reset_index()
        )

        # remove dur_z with negative season lengths
        idx = dur_z["season_length"] >= 60.0
        dur_z = dur_z[idx]

        if self.verbose:
            print('duration', dur_z)
        # generating simulation parameters
        gen_par = (
            dur_z.groupby(["z", "season"])
            .apply(lambda x: self.calcDaymax(x))
            .reset_index()
        )

        if gen_par.empty:
            return nlr.merge_arrays([idarray, self.bad], flatten=True)
        resdf = pd.DataFrame()

        for seas in seasons:
            vara_df = self.run_season(dataSlice, [seas], gen_par, dur_z)
            if vara_df is not None:
                resdf = pd.concat((resdf, vara_df))

        # final result: median zlim for a faint sn
        # and nsn_med for z<zlim
        if resdf.empty:
            return nlr.merge_arrays([idarray, self.bad], flatten=True)

        resdf = resdf.round({"zlim": 5, "nsn_med": 5})

        if self.verbose:
            print('final result', resdf)

        x1_ref = -2.0
        color_ref = 0.2
        idx = np.abs(resdf["x1"] - x1_ref) < 1.0e-5
        idx &= np.abs(resdf["color"] - color_ref) < 1.0e-5
        idx &= resdf["zlim"] > 0

        if not resdf[idx].empty:
            zlim = resdf[idx]["zlim"].median()
            nSN = resdf[idx]["nsn_med"].sum()

            resd = np.rec.fromrecords(
                [(nSN, zlim, healpixID)], names=["nSN", "zlim", "healpixID"]
            )
            res = nlr.merge_arrays([idarray, resd], flatten=True)

        else:

            res = nlr.merge_arrays([idarray, self.bad], flatten=True)

        if self.verbose:
            print('returning', res)
        return res

    def reducenSN(self, metricVal):

        # At each slicepoint, return the sum nSN value.

        return np.sum(metricVal["nSN"])

    def reducezlim(self, metricVal):

        # At each slicepoint, return the median zlim
        result = np.median(metricVal["zlim"])
        if result < 0:
            result = self.badval

        return result

    def coadd(self, data):
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
                    self.RACol: ["min"],
                    self.DecCol: ["mean"],
                    self.m5Col: ["mean"]
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
            self.RACol,
            self.DecCol,
            self.m5Col,
        ]

        coadd_df = coadd_df.sort_values(by=self.mjdCol)
        coadd_df[self.m5Col] += 1.25 * \
            np.log10(coadd_df[self.exptimeCol]/exptime_single)

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
                    seasoncalc[indx + 1:] = i + 2

        obs = rf.append_fields(obs, "season", seasoncalc)

        return obs

    def seasonInfo(self, grp):
        """
        Method to estimate seasonal info (cadence, season length, ...)

        Parameters
        -----------
        grp: pandas df group

        Returns
        ---------
        df : `pd.DataFrame`
        """
        df = pd.DataFrame([len(grp)], columns=["Nvisits"])
        df["MJD_min"] = grp[self.mjdCol].min()
        df["MJD_max"] = grp[self.mjdCol].max()
        df["season_length"] = df["MJD_max"] - df["MJD_min"]
        df["cadence"] = 0.0

        if len(grp) > 5:
            to = grp.groupby(["night"])[self.mjdCol].median().sort_values()
            df["cadence"] = np.mean(to.diff())

        # print('result', df)
        return df

    def duration_z(self, grp):
        """
        Method to estimate the season length vs redshift
        This is necessary to take into account boundary effects
        when estimating the number of SN that can be detected
        daymin, daymax = min and max MJD of a season
        T0_min(z) =  daymin-(1+z)*min_rf_phase_qual
        T0_max(z) =  daymax-(1+z)*max_rf_phase_qual
        season_length(z) = T0_max(z)-T0_min(z)

        Parameters
        -----------
        grp: pandas df group
            data to process: season infos

        Returns
        --------
        pandas df with season_length, z, T0_min and T0_max cols
        """

        daymin = grp["MJD_min"].values
        daymax = grp["MJD_max"].values
        dur_z = pd.DataFrame(self.zRange, columns=["z"])
        dur_z["T0_min"] = daymin - (1.0 + dur_z["z"]) * self.min_rf_phase_qual
        dur_z["T0_max"] = daymax - (1.0 + dur_z["z"]) * self.max_rf_phase_qual
        dur_z["season_length"] = dur_z["T0_max"] - dur_z["T0_min"]

        return dur_z

    def calcDaymax(self, grp):
        """
        Method to estimate T0 (daymax) values for simulation.

        Parameters
        -----------
        grp: group (pandas df sense)
            group of data to process with the following cols:
            T0_min: T0 min value (per season)
            T0_max: T0 max value (per season)

        Returns
        --------
        pandas df with daymax, min_rf_phase, max_rf_phase values
        """

        T0_max = grp["T0_max"].values
        T0_min = grp["T0_min"].values
        num = (T0_max - T0_min) / self.daymaxStep
        if T0_max - T0_min > 10:
            df = pd.DataFrame(np.linspace(
                T0_min, T0_max, int(num)), columns=["daymax"])
        else:
            df = pd.DataFrame([-1], columns=["daymax"])

        df["min_rf_phase"] = self.min_rf_phase
        df["max_rf_phase"] = self.max_rf_phase

        return df

    def run_season(self, dataSlice, season, gen_par, dura_z):
        """
        Method to run on seasons

        Parameters
        -----------
        dataSlice : numpy array, optional
            data to process (scheduler simulations)
        seasons: list(int)
            list of seasons to process

        Returns
        -------
        effi_seasondf: pandas df
            efficiency curves
        zlimsdf: pandas df
            redshift limits and number of supernovae
        """

        time_ref = time.time()

        if self.verbose:
            print("#### Processing season", season)

        groupnames = ["season", "x1", "color"]

        gen_p = gen_par[gen_par["season"].isin(season)]

        if gen_p.empty:
            if self.verbose:
                print("No generator parameter found")
            return None
        else:
            if self.verbose:
                print("generator parameters", gen_p, len(gen_p))
        dur_z = dura_z[dura_z["season"].isin(season)]
        obs = pd.DataFrame(np.copy(dataSlice))
        obs = obs[obs["season"].isin(season)]

        obs = obs.sort_values(by=["night"])

        # simulate supernovae and lc
        if self.verbose:
            print("SN generation")
            print(season, obs)
        sn = self.genSN(obs.to_records(index=False),
                        gen_p.to_records(index=False))
        if np.size(sn) == 0:
            return None

        if self.verbose:
            idx = np.abs(sn["x1"] + 2) < 1.0e-5
            idx &= sn["z"] <= 0.1
            sel = sn[idx]
            sel = sel.sort_values(by=["z", "daymax"])

            print(
                "sn and lc",
                len(sn),
                sel.columns,
                sel[["x1", "color", "z", "daymax",
                    "Cov_colorcolor", "n_bef", "n_aft"]],
            )
            print('effidf', self.effidf(sel))

        # from these supernovae: estimate observation efficiency vs z
        effi_seasondf = self.effidf(sn)
        if effi_seasondf is None:
            return None

        if self.verbose:
            print('effi df', effi_seasondf)

        # zlims can only be estimated if efficiencies are ok
        idx = effi_seasondf["z"] <= 0.2
        x1ref = -2.0
        colorref = 0.2
        idx &= np.abs(effi_seasondf["x1"] - x1ref) < 1.0e-5
        idx &= np.abs(effi_seasondf["color"] - colorref) < 1.0e-5
        sel = effi_seasondf[idx]

        if np.mean(sel["effi"]) > 0.02:
            # estimate zlims
            zlimsdf = self.zlims(effi_seasondf, dur_z, groupnames)
            if self.verbose:
                print('zlims', zlimsdf)

            # estimate number of medium supernovae
            zlimsdf["nsn_med"], zlimsdf["var_nsn_med"] = zlimsdf.apply(
                lambda x: self.nsn_typedf(x, 0.0, 0.0, effi_seasondf, dur_z),
                axis=1,
                result_type="expand",
            ).T.values
        else:
            return None

        if self.verbose:
            print("#### SEASON processed", time.time() - time_ref, season)

        return zlimsdf

    def genSN(self, obs, gen_par):
        """
        Method to simulate LC and supernovae

        Parameters
        -----------
        obs: numpy array
            array of observations(from scheduler)
        gen_par: numpy array
            array of parameters for simulation
        """

        time_ref = time.time()
        # LC estimation

        sn_tot = pd.DataFrame()
        lc_tot = pd.DataFrame()
        for key, vals in self.lcFast.items():
            time_refs = time.time()
            gen_par_cp = np.copy(gen_par)
            if key == (-2.0, 0.2):
                idx = gen_par_cp["z"] < 0.9
                #idx &= np.abs(gen_par_cp['daymax']-61016.17090) < 1.e-5
                gen_par_cp = gen_par_cp[idx]

            lc = vals(obs, gen_par_cp, bands="grizy")
            """
            if key == (-2.0, 0.2):
                obs.sort(order=self.mjdCol)
                print('obs', obs[[self.mjdCol, self.filterCol]])
                self.plotLC_debug(lc, daymax=61016.17090)
            """
            if self.verbose:
                print("End of simulation", key, time.time() - time_refs)

            if self.verbose:
                print("End of simulation after concat",
                      key, time.time() - time_refs)

            # estimate SN

            sn = pd.DataFrame()
            if len(lc) > 0:
                sn = self.process(pd.DataFrame(lc))

            if self.verbose:
                print("End of supernova", time.time() - time_refs)

            if not sn.empty:
                sn_tot = pd.concat([sn_tot, pd.DataFrame(sn)], sort=False)

        if self.verbose:
            print("End of supernova - all", time.time() - time_ref)

        return sn_tot

    def process(self, tab):
        """
        Method to process LC: sigma_color estimation and LC selection

        Parameters
        ------------
        tab: pandas df of LC points with the following cols:
            flux:  flux
            fluxerr: flux error
            phase:  phase
            snr_m5: Signal-to-Noise Ratio
            time: time(MJD)
            mag: magnitude
            m5:  five-sigma depth
            magerr: magnitude error
            exposuretime: exposure time
            band: filter
            zp:  zero-point
            season: season number
            healpixID: pixel ID
            pixRA: pixel RA
            pixDec: pixel Dec
            z: redshift
            daymax: T0
            flux_e_sec: flux(in photoelec/sec)
            flux_5: 5-sigma flux(in photoelec/sec)
            F_x0x0, ...F_colorcolor: Fisher matrix elements
            x1: x1 SN
            color: color SN
            n_aft: number of LC points before daymax
            n_bef: number of LC points after daymax
            n_phmin: number of LC points with a phase < -5
            n_phmax:  number of LC points with a phase > 20
        """
        # now groupby
        tab = tab.round({"daymax": 5, "z": 3, "x1": 2, "color": 2})

        idx = tab['snr_m5'] >= self.snr_min
        tab = tab[idx]

        groups = tab.groupby(["daymax", "season", "z", "x1", "color"])

        tosum = []
        for ia, vala in enumerate(self.params):
            for jb, valb in enumerate(self.params):
                if jb >= ia:
                    tosum.append("F_" + vala + valb)
        tosum += ["n_aft", "n_bef", "n_phmin", "n_phmax"]
        # apply the sum on the group
        #sums = groups[tosum].sum().reset_index()
        sums = groups.apply(lambda x: self.sumIt(x, tosum)).reset_index()
        # select LC according to the number of points bef/aft peak
        idx = sums["n_aft"] >= self.n_aft
        idx &= sums["n_bef"] >= self.n_bef
        idx &= sums["n_phmin"] >= self.n_phase_min
        idx &= sums["n_phmax"] >= self.n_phase_max

        if self.verbose:
            print(
                "selection parameters",
                self.n_bef,
                self.n_aft,
                self.n_phase_min,
                self.n_phase_max,
            )
            print('for sel', sums)
        finalsn = pd.DataFrame()
        goodsn = pd.DataFrame(sums.loc[idx])
        if self.verbose:
            print('goodSN', len(goodsn))

        # estimate the color for SN that passed the selection cuts
        if len(goodsn) > 0:
            goodsn.loc[:, "Cov_colorcolor"] = CovColor(goodsn).Cov_colorcolor
            finalsn = pd.concat([finalsn, goodsn], sort=False)
            if self.verbose:
                idb = goodsn['z'] <= 0.1
                print('goodsn', len(goodsn[idb]), goodsn[idb])

        badsn = pd.DataFrame(sums.loc[~idx])

        # Supernovae that did not pass the cut have a sigma_color=10
        if len(badsn) > 0:
            badsn.loc[:, "Cov_colorcolor"] = 100.0
            finalsn = pd.concat([finalsn, badsn], sort=False)
            if self.verbose:
                print('finalsn', len(finalsn))
                idx = finalsn['z'] <= 0.1
                print('finalsn', len(finalsn[idx]), finalsn[idx])
        return finalsn

    def effidf(self, sn_tot, color_cut=0.04):
        """
        Method estimating efficiency vs z for a sigma_color

        Parameters
        -----------
        sn_tot: pandas df
            data used to estimate efficiencies
        color_cut: float, optional
            color selection cut(default: 0.04)

        Returns
        --------
        effi: pandas df with the following cols:
            season: season
            pixRA: RA of the pixel
            pixDec: Dec of the pixel
            healpixID: pixel ID
            x1: SN stretch
            color: SN color
            z: redshift
            effi: efficiency
            effi_err: efficiency error(binomial)
        """

        sndf = pd.DataFrame(sn_tot)

        listNames = ["season", "x1", "color"]
        groups = sndf.groupby(listNames)

        # estimating efficiencies
        effi = groups[["Cov_colorcolor", "z"]].apply(
            lambda x: effiObsdf(x, color_cut))

        if self.verbose:
            print('efficiencies', effi)
        if np.mean(effi["effi"]) == 0:
            return None
        effi = effi.reset_index(level=[0, 1, 2])

        # this is to plot efficiencies and also sigma_color vs z
        if self.ploteffi:
            import matplotlib.pylab as plt

            fig, ax = plt.subplots()
            figb, axb = plt.subplots()

            self.plot(ax, effi, "effi", "effi_err",
                      "Observing Efficiencies", ls="-")
            sndf["sigma_color"] = np.sqrt(sndf["Cov_colorcolor"])
            self.plot(axb, sndf, "sigma_color", None, "$\sigma_{color}$")
            # get efficiencies vs z

            plt.show()

        return effi

    def plot(self, ax, effi, vary, erry=None, legy="", ls="None"):
        """
        Simple method to plot vs z

        Parameters
        -----------
        ax: `matplotlib.Axes`
            axis where to plot
        effi: pandas df
            data to plot
        vary: str
            variable(column of effi) to plot
        erry: str, optional
            error on y-axis(default: None)
        legy: str, optional
            y-axis legend(default: '')
        """
        grb = effi.groupby(["x1", "color"])
        yerr = None
        for key, grp in grb:
            x1 = grp["x1"].unique()[0]
            color = grp["color"].unique()[0]
            if erry is not None:
                yerr = grp[erry]
            ax.errorbar(
                grp["z"],
                grp[vary],
                yerr=yerr,
                marker="o",
                label="(x1,color)=({},{})".format(x1, color),
                lineStyle=ls,
            )

        ftsize = 15
        ax.set_xlabel("z", fontsize=ftsize)
        ax.set_ylabel(legy, fontsize=ftsize)
        ax.xaxis.set_tick_params(labelsize=ftsize)
        ax.yaxis.set_tick_params(labelsize=ftsize)
        ax.legend(fontsize=ftsize)

    def zlims(self, effi_seasondf, dur_z, groupnames):
        """
        Method to estimate redshift limits

        Parameters
        -----------
        effi_seasondf: pandas df
            season: season
            pixRA: RA of the pixel
            pixDec: Dec of the pixel
            healpixID: pixel ID
            x1: SN stretch
            color: SN color
            z: redshift
            effi: efficiency
            effi_err: efficiency error (binomial)
        dur_z: pandas df with the following cols:
            season: season
            z: redshift
            T0_min: min daymax
            T0_max: max daymax
            season_length: season length
        groupnames: list(str)
            list of columns to use to define the groups

        Returns
        --------
        pandas df with the following cols:
            pixRA: RA of the pixel
            pixDec: Dec of the pixel
            healpixID: pixel ID
            season: season number
            x1: SN stretch
            color: SN color
            zlim: redshift limit
        """

        res = (
            effi_seasondf.groupby(groupnames)
            .apply(lambda x: self.zlimdf(x, dur_z))
            .reset_index(level=list(range(len(groupnames))))
        )

        return res

    def zlimdf(self, grp, duration_z):
        """Estimate redshift limits.

        Parameters
        -----------
        grp: `pd.DataFrameGroupBy`
            DataFrame with efficiencies to estimate redshift limits
            expected columns are season, pixRA, pixDec, healpixID, x1 (SNstretch), color (SN color),
            z (redshift), effi (efficiency), effi_err (efficiency error, binomial)
        duration_z: `pd.DataFrame`
            season: season
            z: redshift
            T0_min: min daymax
            T0_max: max daymax
            season_length: season length

        Returns
        ----------
        df: `pd.DataFrame`
            Dataframe with columns of zlimit (redshift limit)
        """

        zlimit = 0.0

        # z range for the study
        zplot = np.arange(self.zmin, self.zmax, 0.01)

        # print(grp['z'], grp['effi'])

        if len(grp["z"]) <= 3:
            return pd.DataFrame({"zlim": [zlimit]})
            # 'status': [int(status)]})
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            grp["z"], grp["effi"], kind="linear", bounds_error=False, fill_value=0.0
        )

        if self.zlim_coeff < 0.0:
            # in that case zlim is estimated from efficiencies
            # first step: identify redshift domain with efficiency decrease
            zlimit = self.zlim_from_effi(effiInterp, zplot)
            # status = self.status['ok']

        else:
            zlimit = self.zlim_from_cumul(
                grp, duration_z, effiInterp, zplot, rate='SN_rate')

        return pd.DataFrame({"zlim": [zlimit]})
        # 'status': [int(status)]})

    def zlim_from_cumul(self, grp, duration_z, effiInterp, zplot, rate="cte"):
        """
        Method to estimate the redshift limit from the cumulative
        The redshift limit is estimated to be the z value corresponding to:
        frac(NSN(z<zlimit))=zlimi_coeff

        Parameters
        ---------------
        grp: pandas group
            data to process
        duration_z: array
            duration as a function of the redshift
        effiInterp: interp1d
            interpolator for efficiencies
        zplot: interp1d
            interpolator for redshift values
        rate: str, optional
            rate to estimate the number of SN to estimate zlimit
            rate = cte: rate independent of z
            rate = SN_rate: rate from SN_Rate class

        Returns
        ----------
        zlimit: float
            the redshift limit
        """

        if rate == "SN_rate":
            # get rate
            season = np.median(grp["season"])
            idx = duration_z["season"] == season
            seas_duration_z = duration_z[idx]

            durinterp_z = interp1d(
                seas_duration_z["z"],
                seas_duration_z["season_length"],
                bounds_error=False,
                fill_value=0.0,
            )

            # estimate the rates and nsn vs z
            zz, rate, err_rate, nsn, err_nsn = self.rateSN(
                zmin=self.zmin,
                zmax=self.zmax,
                duration_z=durinterp_z,
                survey_area=self.pixArea,
                account_for_edges=False
            )

            # rate interpolation
            rateInterp = interp1d(
                zz, nsn, kind="linear", bounds_error=False, fill_value=0
            )
        else:
            # this is for a rate z-independent
            nsn = np.ones(len(zplot))
            rateInterp = interp1d(
                zplot, nsn, kind="linear", bounds_error=False, fill_value=0
            )

        nsn_cum = np.cumsum(effiInterp(zplot) * rateInterp(zplot))
        if self.verbose:
            print('zlim estimation ', self.pixArea, self.zlim_coeff)
            print('season', duration_z[['z', 'season_length']])
            print('effis', effiInterp(zplot))
            print('rate', rateInterp(zplot))
            print('nsn', nsn_cum)
        if nsn_cum[-1] >= 1.0e-5:
            nsn_cum_norm = nsn_cum / nsn_cum[-1]  # normalize
            zlim = interp1d(nsn_cum_norm, zplot)
            zlimit = zlim(self.zlim_coeff).item()

            if self.ploteffi:
                self.plot_NSN_cumul(grp, nsn_cum_norm, zplot)
        else:
            zlimit = 0.0

        return zlimit

    def plot_NSN_cumul(self, grp, nsn_cum_norm, zplot):
        """
        Method to plot the NSN cumulative vs redshift

        Parameters
        --------------
        grp: pandas group
            data to process
        """

        import matplotlib.pylab as plt

        fig, ax = plt.subplots()
        x1 = grp["x1"].unique()[0]
        color = grp["color"].unique()[0]

        ax.plot(zplot, nsn_cum_norm, label="(x1,color)=({},{})".format(x1, color))

        ftsize = 15
        ax.set_ylabel("NSN ($z<$)", fontsize=ftsize)
        ax.set_xlabel("z", fontsize=ftsize)
        ax.xaxis.set_tick_params(labelsize=ftsize)
        ax.yaxis.set_tick_params(labelsize=ftsize)
        ax.set_xlim((0.0, 0.8))
        ax.set_ylim((0.0, 1.05))
        ax.plot([0.0, 1.2], [0.95, 0.95], ls="--", color="k")
        plt.legend(fontsize=ftsize)
        plt.show()

    def zlim_from_effi(self, effiInterp, zplot):
        """
        Method to estimate the redshift limit from efficiency curves
        The redshift limit is defined here as the redshift value beyond
        which efficiency decreases up to zero.

        Parameters
        ---------------
        effiInterp: interpolator
            use to get efficiencies
        zplot: numpy array
            redshift values

        Returns
        -----------
        zlimit: float
            the redshift limit
        """

        # get efficiencies
        effis = effiInterp(zplot)
        # select data with efficiency decrease
        idx = np.where(np.diff(effis) < -0.005)[0]

        # Bail out if there is no data
        if np.size(idx) == 0:
            return 0

        z_effi = np.array(zplot[idx], dtype={
            "names": ["z"], "formats": [float]})
        # from this make some "z-periods" to avoid accidental zdecrease at low z
        z_gap = 0.05
        seasoncalc = np.ones(z_effi.size, dtype=int)
        diffz = np.diff(z_effi["z"])
        flag = np.where(diffz > z_gap)[0]

        if len(flag) > 0:
            for i, indx in enumerate(flag):
                seasoncalc[indx + 1:] = i + 2
        z_effi = rf.append_fields(z_effi, "season", seasoncalc)

        # now take the highest season (end of the efficiency curve)
        idd = z_effi["season"] == np.max(z_effi["season"])
        zlimit = np.min(z_effi[idd]["z"])

        return zlimit

    def zlimdf_deprecated(self, grp, duration_z):
        """
        Method to estimate redshift limits

        Parameters
        --------------
        grp: pandas df group
            efficiencies to estimate redshift limits;
            columns:
            season: season
            pixRA: RA of the pixel
            pixDec: Dec of the pixel
            healpixID: pixel ID
            x1: SN stretch
            color: SN color
            z: redshift
            effi: efficiency
            effi_err: efficiency error (binomial)
        duration_z: pandas df with the following cols:
            season: season
            z: redshift
            T0_min: min daymax
            T0_max: max daymax
            season_length: season length

        Returns
        ----------
        pandas df with the following cols:
            zlimit: redshift limit
        """

        zlimit = 0.0

        # z range for the study
        zplot = list(np.arange(self.zmin, self.zmax, 0.001))

        # print(grp['z'], grp['effi'])

        if len(grp["z"]) <= 3:
            return pd.DataFrame({"zlim": [zlimit]})
            # 'status': [int(status)]})
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            grp["z"], grp["effi"], kind="linear", bounds_error=False, fill_value=0.0
        )

        # get rate
        season = np.median(grp["season"])
        idx = duration_z["season"] == season
        seas_duration_z = duration_z[idx]

        durinterp_z = interp1d(
            seas_duration_z["z"],
            seas_duration_z["season_length"],
            bounds_error=False,
            fill_value=0.0,
        )

        # estimate the rates and nsn vs z
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(
            zmin=self.zmin,
            zmax=self.zmax,
            duration_z=durinterp_z,
            survey_area=self.pixArea,
        )

        # rate interpolation
        rateInterp = interp1d(zz, nsn, kind="linear",
                              bounds_error=False, fill_value=0)

        # estimate the cumulated number of SN vs z
        nsn_cum = np.cumsum(effiInterp(zplot) * rateInterp(zplot))

        if nsn_cum[-1] >= 1.0e-5:
            nsn_cum_norm = nsn_cum / nsn_cum[-1]  # normalize
            zlim = interp1d(nsn_cum_norm, zplot)
            zlimit = zlim(0.95).item()
            # status = self.status['ok']

            if self.ploteffi:
                import matplotlib.pylab as plt

                fig, ax = plt.subplots()
                x1 = grp["x1"].unique()[0]
                color = grp["color"].unique()[0]

                ax.plot(
                    zplot, nsn_cum_norm, label="(x1,color)=({},{})".format(
                        x1, color)
                )

                ftsize = 15
                ax.set_ylabel("NSN ($z<$)", fontsize=ftsize)
                ax.set_xlabel("z", fontsize=ftsize)
                ax.xaxis.set_tick_params(labelsize=ftsize)
                ax.yaxis.set_tick_params(labelsize=ftsize)
                ax.set_xlim((0.0, 1.2))
                ax.set_ylim((0.0, 1.05))
                ax.plot([0.0, 1.2], [0.95, 0.95], ls="--", color="k")
                plt.legend(fontsize=ftsize)
                plt.show()

        return pd.DataFrame({"zlim": [zlimit]})
        # 'status': [int(status)]})

    def nsn_typedf(self, grp, x1, color, effi_tot, duration_z, search=True):
        """
        Method to estimate the number of supernovae for a given type of SN

        Parameters
        --------------
        grp: pandas series with the following infos:
            pixRA: pixelRA
            pixDec: pixel Dec
            healpixID: pixel ID
            season: season
        x1: SN stretch
        color: SN color
        lim: redshift limit
            x1, color: SN params to estimate the number
        effi_tot: pandas df with columns:
            season: season
            pixRA: RA of the pixel
            pixDec: Dec of the pixel
            healpixID: pixel ID
            x1: SN stretch
            color: SN color
            z: redshift
            effi: efficiency
            effi_err: efficiency error (binomial)
        duration_z: pandas df with the following cols:
            season: season
            z: redshift
            T0_min: min daymax
            T0_max: max daymax
            season_length: season length

        Returns
        ----------
        nsn: float
            number of supernovae
        """

        # get rate
        season = np.median(grp["season"])
        idx = duration_z["season"] == season
        seas_duration_z = duration_z[idx]

        durinterp_z = interp1d(
            seas_duration_z["z"],
            seas_duration_z["season_length"],
            bounds_error=False,
            fill_value=0.0,
        )

        if search:
            effisel = effi_tot.loc[
                lambda dfa: (dfa["x1"] == x1) & (dfa["color"] == color), :
            ]
        else:
            effisel = effi_tot

        nsn, var_nsn = self.nsn(effisel, grp["zlim"], durinterp_z)

        return (nsn, var_nsn)

    def nsn(self, effi, zlim, duration_z):
        """
        Method to estimate the number of supernovae

        Parameters
        -----------
        effi: pandas df grp of efficiencies
            season: season
            pixRA: RA of the pixel
            pixDec: Dec of the pixel
            healpixID: pixel ID
            x1: SN stretch
            color: SN color
            z: redshift
            effi: efficiency
            effi_err: efficiency error (binomial)
        zlim: float
            redshift limit value
        duration_z: pandas df with the following cols:
            season: season
            z: redshift
            T0_min: min daymax
            T0_max: max daymax
            season_length: season length

        Returns
        ----------
        nsn, var_nsn : float
            number of supernovae (and variance) with z<zlim
        """

        if zlim < 1.0e-3:
            return (-1.0, -1.0)

        dz = 0.001
        zplot = list(np.arange(self.zmin, self.zmax, dz))
        # interpolate efficiencies vs z
        effiInterp = interp1d(
            effi["z"], effi["effi"], kind="linear", bounds_error=False, fill_value=0.0
        )
        # estimate the cumulated number of SN vs z
        zz, rate, err_rate, nsn, err_nsn = self.rateSN(
            zmin=self.zmin,
            zmax=self.zmax,
            dz=dz,
            duration_z=duration_z,
            survey_area=self.pixArea,
        )

        nsn_cum = np.cumsum(effiInterp(zplot) * nsn)

        nsn_interp = interp1d(zplot, nsn_cum)

        nsn = nsn_interp(zlim).item()

        return [nsn, 0.0]

    def sumIt_nonight(self, grpa, tosum):
        """
        Method to estimate the sum of some of the group col
        and also of the number of epochs before and after peak

        Parameters
        --------------
        grpa: pandas df
          pandas group to process
        tosum: list(str)
          list of the columns to sum

        Returns
        ----------
        pandas df with the summed columns and the estimation of the number of epochs (n_bef and n_aft)

        """

        grp = pd.DataFrame(grpa)
        grp = grp.reset_index()
        grp = grp.sort_values(by=['time'])

        gap = 12./24.  # in days
        df_time = grp['time'].diff()
        index = list((df_time[df_time > gap].index))
        index.insert(0, grp.index.min())
        index.insert(len(index), grp.index.max())
        grp['epoch'] = 1

        for i in range(0, len(index)-1):
            grp.loc[index[i]: index[i+1], 'epoch'] = i+1

        #grp = grp.sort_values(by=['epoch'])
        idx = grp['phase'] <= 0
        sel = grp[idx]

        n_bef = len(grp[idx]['epoch'].unique())

        idx = grp['phase'] >= 0
        n_aft = len(grp[idx]['epoch'].unique())

        sums = grpa[tosum].sum()
        sums['n_bef'] = n_bef
        sums['n_aft'] = n_aft

        return sums

    def sumIt(self, grp, tosum):
        """
        Method to estimate the sum of some of the group col
        and also of the number of epochs before and after peak

        Parameters
        --------------
        grp: pandas df
          pandas group to process
        tosum: list(str)
          list of the columns to sum

        Returns
        ----------
        pandas df with the summed columns and the estimation of the number of epochs (n_bef and n_aft)

        """
        sums = grp[tosum].sum()

        grp['night'] = grp['time']-self.mjd_LSST_Start+1.
        grp['night'] = grp['night'].astype(int)

        grp = grp.sort_values(by=['night'])
        idx = grp['phase'] <= 0
        sel = grp[idx]

        n_bef = len(grp[idx]['night'].unique())

        idx = grp['phase'] >= 0
        n_aft = len(grp[idx]['night'].unique())

        sums['n_bef'] = n_bef
        sums['n_aft'] = n_aft

        return sums

    def plotLC_debug(self, lc, daymax=60896.88112):
        """
        Method to plot LC for a given daymax (debug purpose)

        Parameters
        --------------
        lc: pandas df
          set of light curves
        daymax: float, opt
          T0 value (default: 60896.88112)

        """
        print('bands', np.unique(lc['band']))
        ido = np.abs(lc['daymax']-daymax) < 1.e-5
        ido &= np.abs(lc['x1']+2.0) < 1.e-5
        ido &= lc['snr_m5'] >= 1
        lcsel = lc[ido]
        lcsel = lcsel.sort_values(by=['phase'])
        print('light curve ',
              lcsel[['phase', 'flux', 'band', 'daymax', 'snr_m5', 'time']])
        """
        for ir, row in lcsel.iterrows():
            print(row['phase'], row['flux'],
                  row['band'], row['night'], row['z'], row['x1'], row['daymax'])
            print('light curve ',
                  lcsel[['phase', 'flux', 'band', 'night', 'daymax']])
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(lcsel['phase'], lcsel['flux'], 'ko')
        plt.show()
