from rubin_sim.data import get_data_dir
import numpy as np
import os
import h5py
from astropy.table import Table
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from astropy.cosmology import FlatLambdaCDM
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

STERADIAN2SQDEG = 180.0**2 / np.pi**2
# Mpc^3 -> Mpc^3/sr
norm = 1.0 / (4.0 * np.pi)

__all__ = [
    "LCfast_new",
    "Load_Reference",
    "GetReference",
    "SN_Rate",
    "CovColor",
    "load_sne_cached",
]


class LCfast_new:
    """
    class to simulate supernovae light curves in a fast way
    The method relies on templates and broadcasting to increase speed

    Parameters
    ---------------
    reference_lc: RegularGridData
        lc reference files
    x1: float
        SN stretch
    color: float
        SN color
    telescope: Telescope()
        telescope for the study
    mjdCol: str, opt
        name of the MJD col in data to simulate (default: observationStartMJD)
    filterCol: str, opt
        name of the filter col in data to simulate (default: filter)
    exptimeCol: str, opt
        name of the exposure time  col in data to simulate (default: visitExposureTime)
    m5Col: str, opt
        name of the fiveSigmaDepth col in data to simulate (default: fiveSigmaDepth)
    seasonCol: str, opt
        name of the season col in data to simulate (default: season)
    snr_min: float, opt
        minimal Signal-to-Noise Ratio to apply on LC points (default: 5)
    lightOutput: bool, opt
        to get a lighter output (ie lower number of cols) (default: True)
    bluecutoff: float,opt
        blue cutoff for SN (default: 380.0 nm)
    redcutoff: float, opt
        red cutoff for SN (default: 800.0 nm)
    """

    def __init__(
        self,
        reference_lc,
        x1,
        color,
        telescope,
        mjdCol="observationStartMJD",
        filterCol="filter",
        exptimeCol="visitExposureTime",
        m5Col="fiveSigmaDepth",
        seasonCol="season",
        nexpCol="numExposures",
        seeingCol="seeingFwhmEff",
        snr_min=5.0,
        lightOutput=True,
        ebvofMW=-1.0,
        bluecutoff=380.0,
        redcutoff=800.0,
    ):

        # grab all vals
        self.filterCol = filterCol
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.exptimeCol = exptimeCol
        self.seasonCol = seasonCol
        self.nexpCol = nexpCol
        self.seeingCol = seeingCol

        self.x1 = x1
        self.color = color
        self.lightOutput = lightOutput
        self.ebvofMW = ebvofMW

        # Loading reference file
        self.reference_lc = reference_lc

        self.mean_wavelength = telescope["mean_wavelengths"]
        self.zp = telescope["zp_s"]

        # This cutoffs are used to select observations:
        # phase = (mjd - DayMax)/(1.+z)
        # selection: min_rf_phase < phase < max_rf_phase
        # and        blue_cutoff < mean_rest_frame < red_cutoff
        # where mean_rest_frame = telescope.mean_wavelength/(1.+z)
        self.blue_cutoff = bluecutoff
        self.red_cutoff = redcutoff

        # SN parameters for Fisher matrix estimation
        self.param_Fisher = ["x0", "x1", "daymax", "color"]

        self.snr_min = snr_min


    def __call__(self, obs, ebvofMW, gen_par=None, bands="grizy"):
        """Simulation of the light curve


        Parameters
        ----------------
        obs: array
            array of observations
        ebvofMW: float
            E(B-V) for MW
        gen_par: array, opt
            simulation parameters (default: None)
        bands: str, opt
            filters to consider for simulation (default: grizy)


        Returns
        ------------
        astropy table with:
            columns: band, flux, fluxerr, snr_m5,flux_e,zp,zpsys,time
        metadata : SNID,RA,Dec,DayMax,X1,Color,z
        """

        if len(obs) == 0:
            return None

        # result in this df
        tab_tot = pd.DataFrame()

        # loop on the bands
        for band in bands:
            idx = obs[self.filterCol] == band
            if len(obs[idx]) > 0:
                tab_tot = pd.concat([tab_tot, self.processBand(obs[idx], ebvofMW, band, gen_par)],
                                    ignore_index=True)

        # return produced LC
        return tab_tot

    def processBand(self, sel_obs, ebvofMW, band, gen_par):
        """LC simulation of a set of obs corresponding to a band
        The idea is to use python broadcasting so as to estimate
        all the requested values (flux, flux error, Fisher components, ...)
        in a single path (i.e no loop!)

        Parameters
        ---------------
        sel_obs: array
            array of observations
        band: str
            band of observations
        gen_par: array
            simulation parameters

        Returns
        -------
        astropy table with fields corresponding to LC components
        """

        # if there are no observations in this filter: return None
        if len(sel_obs) == 0:
            return None

        # Get the fluxes (from griddata reference)

        # xi = MJD-T0
        xi = sel_obs[self.mjdCol] - gen_par["daymax"][:, np.newaxis]

        # yi = redshift simulated values
        # requested to avoid interpolation problems near boundaries
        yi = np.round(gen_par["z"], 4)
        # yi = gen_par['z']

        # p = phases of LC points = xi/(1.+z)
        p = xi / (1.0 + yi[:, np.newaxis])
        yi_arr = np.ones_like(p) * yi[:, np.newaxis]

        # get fluxes
        pts = (p, yi_arr)
        fluxes_obs = self.reference_lc.flux[band](pts)

        # Fisher components estimation

        dFlux = {}

        # loop on Fisher parameters
        for val in self.param_Fisher:
            dFlux[val] = self.reference_lc.param[band][val](pts)

        # Fisher matrix components estimation
        # loop on SN parameters (x0,x1,color)
        # estimate: dF/dxi*dF/dxj
        Derivative_for_Fisher = {}
        for ia, vala in enumerate(self.param_Fisher):
            for jb, valb in enumerate(self.param_Fisher):
                if jb >= ia:
                    Derivative_for_Fisher[vala + valb] = dFlux[vala] * dFlux[valb]

        flag = self.getFlag(sel_obs, gen_par, fluxes_obs, band, p)
        flag_idx = np.argwhere(flag)

        # now apply the flag to select LC points - masked arrays
        fluxes = self.marray(fluxes_obs, flag)
        phases = self.marray(p, flag)
        z_vals = gen_par["z"][flag_idx[:, 0]]
        daymax_vals = gen_par["daymax"][flag_idx[:, 0]]
        Fisher_Mat = {}
        for key, vals in Derivative_for_Fisher.items():
            Fisher_Mat[key] = self.marray(vals, flag)

        ndata = len(fluxes[~fluxes.mask])

        if ndata <= 0:
            return pd.DataFrame()

        nvals = len(phases)
        # masked - tile arrays

        cols = [
            self.mjdCol,
            self.seasonCol,
            self.m5Col,
            self.exptimeCol,
            self.nexpCol,
            self.seeingCol,
            "night",
        ]

        # take columns common with obs cols
        colcoms = set(cols).intersection(sel_obs.dtype.names)

        masked_tile = {}
        for vv in list(colcoms):
            masked_tile[vv] = self.tarray(sel_obs[vv], nvals, flag)

        # mag_obs = np.ma.array(mag_obs, mask=~flag)

        # put data in a pandas df
        lc = pd.DataFrame()

        lc["flux"] = fluxes[~fluxes.mask]
        lc["mag"] = -2.5 * np.log10(lc["flux"] / 3631.0)
        lc["phase"] = phases[~phases.mask]
        lc["band"] = ["LSST::" + band] * len(lc)
        lc["zp"] = self.zp[band]
        lc["zp"] = 2.5 * np.log10(3631)
        lc["zpsys"] = "ab"
        lc["z"] = z_vals
        lc["daymax"] = daymax_vals

        for key, vals in masked_tile.items():
            lc[key] = vals[~vals.mask]

        lc["time"] = lc[self.mjdCol]
        lc.loc[:, "x1"] = self.x1
        lc.loc[:, "color"] = self.color

        # estimate errors
        lc["flux_e_sec"] = self.reference_lc.mag_to_flux[band](
            (lc["mag"], lc[self.exptimeCol] / lc[self.nexpCol], lc[self.nexpCol])
        )
        lc["flux_5"] = 10 ** (-0.4 * (lc[self.m5Col] - self.zp[band]))
        lc["snr_m5"] = lc["flux_e_sec"] / np.sqrt(
            (lc["flux_5"] / 5.0) ** 2 + lc["flux_e_sec"] / lc[self.exptimeCol]
        )
        lc["fluxerr_photo"] = lc["flux"] / lc["snr_m5"]
        lc["magerr"] = (2.5 / np.log(10.0)) / lc["snr_m5"]
        lc["fluxerr"] = lc["fluxerr_photo"]

        # Fisher matrix components
        for key, vals in Fisher_Mat.items():
            lc.loc[:, "F_{}".format(key)] = vals[~vals.mask] / (
                lc["fluxerr_photo"].values ** 2
            )

        lc.loc[:, "n_aft"] = (np.sign(lc["phase"]) == 1) & (
            lc["snr_m5"] >= self.snr_min
        )
        lc.loc[:, "n_bef"] = (np.sign(lc["phase"]) == -1) & (
            lc["snr_m5"] >= self.snr_min
        )

        lc.loc[:, "n_phmin"] = lc["phase"] <= -5.0
        lc.loc[:, "n_phmax"] = lc["phase"] >= 20

        # if len(lc) > 0.:
        #    lc = self.dust_corrections(lc, ebvofMW)

        return lc

    def getFlag(self, sel_obs, gen_par, fluxes_obs, band, p):
        """
        Method to flag events corresponding to selection criteria

        Parameters
        ---------------
        sel_obs: array
          observations
        gen_par: array
           simulation parameters
        fluxes_obs: array
           LC fluxes
        band: str
          filter
        p: array
          LC phases
        """
        # remove LC points outside the restframe phase range
        min_rf_phase = gen_par["minRFphase"][:, np.newaxis]
        max_rf_phase = gen_par["maxRFphase"][:, np.newaxis]
        flag = (p >= min_rf_phase) & (p <= max_rf_phase)

        # remove LC points outside the (blue-red) range
        mean_restframe_wavelength = np.array(
            [self.mean_wavelength[band]] * len(sel_obs)
        )
        mean_restframe_wavelength = np.tile(
            mean_restframe_wavelength, (len(gen_par), 1)
        ) / (1.0 + gen_par["z"][:, np.newaxis])
        # flag &= (mean_restframe_wavelength > 0.) & (
        #    mean_restframe_wavelength < 1000000.)
        flag &= (mean_restframe_wavelength > self.blue_cutoff) & (
            mean_restframe_wavelength < self.red_cutoff
        )

        # remove points with neg flux
        flag &= fluxes_obs > 1.0e-10

        return flag

    def marray(self, val, flag):
        """
        method to estimate a masked array

        Parameters
        ---------------
        val: array
        flag: mask

        Returns
        ----------
        the masked array

        """
        return np.ma.array(val, mask=~flag)

    def tarray(self, arr, nvals, flag):
        """
        method to estimate a masked tiled array

        Parameters
        ---------------
        val: array
        nvals: int
          number of tiles
        flag: mask

        Returns
        ----------
        the masked tiled array

        """

        vv = np.ma.array(np.tile(arr, (nvals, 1)), mask=~flag)
        return vv


def load_sne_cached(gammaName="gamma.hdf5"):
    """Load up the SNe files with a simple function that caches the result so each metric
    doesn't need to load it on it's own
    """

    if hasattr(load_sne_cached, "data"):
        return load_sne_cached.data
    else:
        ref = Load_Reference(gammaName=gammaName).ref
        load_sne_cached.data = ref
        return ref


class Load_Reference:
    """
    class to load template files requested for LCFast
    These files should be stored in a reference_files directory

    Parameters
    ---------------
    server: str, optional
        where to get the files (default: https://me.lsst.eu/gris/DESC_SN_pipeline/Reference_Files)
    templateDir: str, optional
        where to put the files (default: reference_files)

    """

    def __init__(
        self,
        server="https://me.lsst.eu/gris/DESC_SN_pipeline",
        templateDir=None,
        gammaName="gamma.hdf5",
    ):

        if templateDir is None:
            sims_maf_contrib_dir = get_data_dir()
            templateDir = os.path.join(sims_maf_contrib_dir, "maf/SNe_data")

        self.server = server
        # define instrument
        self.Instrument = {}
        self.Instrument["name"] = "LSST"  # name of the telescope (internal)
        # dir of throughput
        self.Instrument["throughput_dir"] = os.path.join(
            get_data_dir(), "throughputs", "baseline"
        )
        self.Instrument["atmos_dir"] = os.path.join(
            get_data_dir(), "throughputs", "atmos"
        )
        self.Instrument["airmass"] = 1.2  # airmass value
        self.Instrument["atmos"] = True  # atmos
        self.Instrument["aerosol"] = False  # aerosol

        x1_colors = [(-2.0, 0.2), (0.0, 0.0)]

        lc_reference = {}

        # create this directory if it does not exist
        if not os.path.isdir(templateDir):
            os.system("mkdir {}".format(templateDir))

        list_files = [gammaName]
        for j in range(len(x1_colors)):
            x1 = x1_colors[j][0]
            color = x1_colors[j][1]
            fname = "LC_{}_{}_380.0_800.0_ebvofMW_0.0_vstack.hdf5".format(x1, color)
            list_files += [fname]

        self.check_grab(templateDir, list_files)

        # gamma_reference
        self.gamma_reference = os.path.join(templateDir, gammaName)

        # print('Loading reference files')

        resultdict = {}

        for j in range(len(x1_colors)):
            x1 = x1_colors[j][0]
            color = x1_colors[j][1]
            fname = "{}/LC_{}_{}_380.0_800.0_ebvofMW_0.0_vstack.hdf5".format(
                templateDir, x1, color
            )
            resultdict[j] = self.load(fname)

        for j in range(len(x1_colors)):
            if resultdict[j] is not None:
                lc_reference[x1_colors[j]] = resultdict[j]

        self.ref = lc_reference

    def load(self, fname):
        """
        Method to load reference files

        Parameters
        ---------------
        fname: str
            file name
        """
        lc_ref = GetReference(fname, self.gamma_reference, self.Instrument)

        return lc_ref

    def check_grab(self, templateDir, listfiles):
        """
        Method that check if files are on disk.
        If not: grab them from a server (self.server)

        Parameters
        ---------------
        templateDir: `str`
            directory where files are (or will be)
        listfiles: `list` [`str`]
            list of files that are (will be) in templateDir
        """

        for fi in listfiles:
            # check whether the file is available; if not-> get it!
            fname = "{}/{}".format(templateDir, fi)
            if not os.path.isfile(fname):
                if "gamma" in fname:
                    fullname = "{}/reference_files/{}".format(self.server, fi)
                else:
                    fullname = "{}/Template_LC/{}".format(self.server, fi)
                print("wget path:", fullname)
                cmd = "wget --no-clobber --no-verbose {} --directory-prefix {}".format(
                    fullname, templateDir
                )
                os.system(cmd)


class GetReference:
    """
    Class to load reference data
    used for the fast SN simulator

    Parameters
    ----------------
    lcName: `str`
        name of the reference file to load (lc)
    gammaName: `str`
        name of the reference file to load (gamma)
    tel_par: `dict`
        telescope parameters
    param_Fisher : `list` [`str`], optional
        list of SN parameter for Fisher estimation to consider
        (default: ['x0', 'x1', 'color', 'daymax'])

    Returns
    -----------
    The following dict can be accessed:
    mag_to_flux_e_sec : Interp1D of mag to flux(e.sec-1)  conversion
    flux : dict of RegularGridInterpolator of fluxes (key: filters, (x,y)=(phase, z), result=flux)
    fluxerr : dict of RegularGridInterpolator of flux errors (key: filters, (x,y)=(phase, z), result=fluxerr)
    param : dict of dict of RegularGridInterpolator of flux derivatives wrt SN parameters
        (key: filters plus param_Fisher parameters; (x,y)=(phase, z), result=flux derivatives)
    gamma : dict of RegularGridInterpolator of gamma values (key: filters)
    """

    def __init__(
        self, lcName, gammaName, tel_par, param_Fisher=["x0", "x1", "color", "daymax"]
    ):

        # Load the file - lc reference

        f = h5py.File(lcName, "r")
        # lc_ref_tot = Table.read(filename, path=keys[0])
        lc_ref_tot = Table.from_pandas(pd.read_hdf(lcName))

        idx = lc_ref_tot["z"] > 0.005
        lc_ref_tot = np.copy(lc_ref_tot[idx])

        # Load the file - gamma values
        if not os.path.exists(gammaName):
            print("gamma file {} does not exist")
            print("will generate it - few minutes")
            mag_range = np.arange(15.0, 38.0, 1.0)
            exptimes = np.arange(1.0, 3000.0, 10.0)
            Gamma(
                "ugrizy", telescope, gammaName, mag_range=mag_range, exptimes=exptimes
            )
            print("end of gamma estimation")

        # Load references needed for the following
        self.lc_ref = {}
        self.gamma_ref = {}
        self.gamma = {}
        self.m5_ref = {}
        self.mag_to_flux_e_sec = {}
        self.mag_to_flux = {}

        self.flux = {}
        self.fluxerr = {}
        self.param = {}

        bands = np.unique(lc_ref_tot["band"])
        mag_range = np.arange(10.0, 38.0, 0.01)

        method = "linear"

        # for each band: load data to be used for interpolation
        for band in bands:
            idx = lc_ref_tot["band"] == band
            lc_sel = Table(lc_ref_tot[idx])

            lc_sel["z"] = lc_sel["z"].data.round(decimals=2)
            lc_sel["phase"] = lc_sel["phase"].data.round(decimals=1)

            """
               select phases between -20 and 50 only
            """
            idx = lc_sel["phase"] < 60.0
            idx &= lc_sel["phase"] > -20.0
            lc_sel = lc_sel[idx]

            # these reference data will be used for griddata interp.
            self.lc_ref[band] = lc_sel
            self.gamma_ref[band] = lc_sel["gamma"][0]
            self.m5_ref[band] = np.unique(lc_sel["m5"])[0]

            # Another interpolator, faster than griddata: regulargridinterpolator

            # Fluxes and errors
            zmin, zmax, zstep, nz = self.limVals(lc_sel, "z")
            phamin, phamax, phastep, npha = self.limVals(lc_sel, "phase")

            zstep = np.round(zstep, 1)
            phastep = np.round(phastep, 1)

            zv = np.linspace(zmin, zmax, nz)
            # zv = np.round(zv,2)
            # print(band,zv)
            phav = np.linspace(phamin, phamax, npha)

            print("Loading ", lcName, band, len(lc_sel), npha, nz)
            index = np.lexsort((lc_sel["z"], lc_sel["phase"]))
            flux = np.reshape(lc_sel[index]["flux"], (npha, nz))
            fluxerr = np.reshape(lc_sel[index]["fluxerr"], (npha, nz))

            self.flux[band] = RegularGridInterpolator(
                (phav, zv), flux, method=method, bounds_error=False, fill_value=-1.0
            )
            self.fluxerr[band] = RegularGridInterpolator(
                (phav, zv), fluxerr, method=method, bounds_error=False, fill_value=-1.0
            )

            # Flux derivatives
            self.param[band] = {}
            for par in param_Fisher:
                valpar = np.reshape(lc_sel[index]["d{}".format(par)], (npha, nz))
                self.param[band][par] = RegularGridInterpolator(
                    (phav, zv),
                    valpar,
                    method=method,
                    bounds_error=False,
                    fill_value=0.0,
                )

            # gamma estimator

            rec = Table.read(gammaName, path="gamma_{}".format(band))

            rec["mag"] = rec["mag"].data.round(decimals=4)
            rec["single_exptime"] = rec["single_exptime"].data.round(decimals=4)

            magmin, magmax, magstep, nmag = self.limVals(rec, "mag")
            expmin, expmax, expstep, nexpo = self.limVals(rec, "single_exptime")
            nexpmin, nexpmax, nexpstep, nnexp = self.limVals(rec, "nexp")
            mag = np.linspace(magmin, magmax, nmag)
            exp = np.linspace(expmin, expmax, nexpo)
            nexp = np.linspace(nexpmin, nexpmax, nnexp)

            index = np.lexsort(
                (rec["nexp"], np.round(rec["single_exptime"], 4), rec["mag"])
            )
            gammab = np.reshape(rec[index]["gamma"], (nmag, nexpo, nnexp))
            fluxb = np.reshape(rec[index]["flux_e_sec"], (nmag, nexpo, nnexp))
            self.gamma[band] = RegularGridInterpolator(
                (mag, exp, nexp),
                gammab,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )

            self.mag_to_flux[band] = RegularGridInterpolator(
                (mag, exp, nexp),
                fluxb,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )

    def limVals(self, lc, field):
        """Get unique values of a field in  a table

        Parameters
        ----------
        lc: Table
            astropy Table (here probably a LC)
        field: str
            name of the field of interest

        Returns
        -------
        vmin: float
            min value of the field
        vmax: float
            max value of the field
        vstep: float
            step value for this field (median)
        nvals: int
            number of unique values
        """

        lc.sort(field)
        vals = np.unique(lc[field].data.round(decimals=4))
        # print(vals)
        vmin = np.min(vals)
        vmax = np.max(vals)
        vstep = np.median(vals[1:] - vals[:-1])

        return vmin, vmax, vstep, len(vals)


class SN_Rate:
    """
    Estimate production rates of typeIa SN
    Available rates: Ripoche, Perrett, Dilday

    Parameters
    ----------
    rate :  str, optional
        type of rate chosen (Ripoche, Perrett, Dilday) (default : Perrett)
    H0 : float, optional
        Hubble constant value :math:`H_{0}` (default : 70.)
    Om0 : float, optional
        matter density value :math:`\Omega_{0}` (default : 0.25)
    min_rf_phase : float, optional
        min rest-frame phase (default : -15.)
    max_rf_phase : float, optional
        max rest-frame phase (default : 30.)
    """

    def __init__(
        self, rate="Perrett", H0=70, Om0=0.25, min_rf_phase=-15.0, max_rf_phase=30.0
    ):

        self.astropy_cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        self.rate = rate
        self.min_rf_phase = min_rf_phase
        self.max_rf_phase = max_rf_phase

    def __call__(
        self,
        zmin=0.1,
        zmax=0.2,
        dz=0.01,
        survey_area=9.6,
        bins=None,
        account_for_edges=False,
        duration=140.0,
        duration_z=None,
    ):
        """
        Parameters
        ----------------
        zmin : `float`, optional
            minimal redshift (default : 0.1)
        zmax : `float`, optional
            max redshift (default : 0.2)
        dz : `float`, optional
            redshift bin (default : 0.001)
        survey_area : `float`, optional
            area of the survey (:math:`deg^{2}`) (default : 9.6 :math:`deg^{2}`)
        bins : `list` [`float`], optional
            redshift bins (default : None)
        account_for_edges : `bool`
            to account for season edges.
            If true, duration of the survey will be reduced by (1+z)*(maf_rf_phase-min_rf_phase)/365.25
            (default : False)
        duration : `float`, optional
            survey duration (in days) (default : 140 days)
        duration_z : `list` [`float`], optional
            survey duration (as a function of z) (default : None)

        Returns
        -----------
        Lists :
        zz : `float`
            redshift values
        rate : `float`
            production rate
        err_rate : `float`
            production rate error
        nsn : `float`
            number of SN
        err_nsn : `float`
            error on the number of SN
        """

        if bins is None:
            thebins = np.arange(zmin, zmax + dz, dz)
            zz = 0.5 * (thebins[1:] + thebins[:-1])
        else:
            zz = bins
            thebins = bins

        rate, err_rate = self.SNRate(zz)

        area = survey_area / STERADIAN2SQDEG
        # or area= self.survey_area/41253.

        dvol = norm * self.astropy_cosmo.comoving_volume(thebins).value

        dvol = dvol[1:] - dvol[:-1]

        if account_for_edges:
            margin = (1.0 + zz) * (self.max_rf_phase - self.min_rf_phase) / 365.25
            effective_duration = duration / 365.25 - margin
            effective_duration[effective_duration <= 0.0] = 0.0
        else:
            # duration in days!
            effective_duration = duration / 365.25
            if duration_z is not None:
                effective_duration = duration_z(zz) / 365.25

        normz = 1.0 + zz
        nsn = rate * area * dvol * effective_duration / normz
        err_nsn = err_rate * area * dvol * effective_duration / normz

        return zz, rate, err_rate, nsn, err_nsn

    def RipocheRate(self, z):
        """The SNLS SNIa rate according to the (unpublished) Ripoche et al study.

        Parameters
        --------------
        z : `float`
            redshift

        Returns
        ----------
        rate : `float`
        error_rate : `float`
        """
        rate = 1.53e-4 * 0.343
        expn = 2.14
        my_z = np.copy(z)
        my_z[my_z > 1.0] = 1.0
        rate_sn = rate * np.power((1 + my_z) / 1.5, expn)
        return rate_sn, 0.2 * rate_sn

    def PerrettRate(self, z):
        """The SNLS SNIa rate according to (Perrett et al, 201?)

        Parameters
        --------------
        z : `float`
            redshift

        Returns
        ----------
        rate : `float`
        error_rate : `float`
        """
        rate = 0.17e-4
        expn = 2.11
        err_rate = 0.03e-4
        err_expn = 0.28
        my_z = np.copy(z)
        rate_sn = rate * np.power(1 + my_z, expn)
        err_rate_sn = np.power(1 + my_z, 2.0 * expn) * np.power(err_rate, 2.0)
        err_rate_sn += np.power(rate_sn * np.log(1 + my_z) * err_expn, 2.0)

        return rate_sn, np.power(err_rate_sn, 0.5)

    def DildayRate(self, z):
        """The Dilday rate according to

        Parameters
        --------------
        z : `float`
            redshift

        Returns
        ----------
        rate : `float`
        error_rate : `float`
        """

        rate = 2.6e-5
        expn = 1.5
        # err_rate = 0.01
        err_expn = 0.6
        my_z = np.copy(z)
        my_z[my_z > 1.0] = 1.0
        rate_sn = rate * np.power(1 + my_z, expn)
        err_rate_sn = rate_sn * np.log(1 + my_z) * err_expn
        return rate_sn, err_rate_sn

    def SNRate(self, z):
        """SN rate estimation

        Parameters
        --------------
        z : `float`
            redshift

        Returns
        ----------
        rate : `float`
        error_rate : `float`
        """
        if self.rate == "Ripoche":
            return self.RipocheRate(z)
        if self.rate == "Perrett":
            return self.PerrettRate(z)
        if self.rate == "Dilday":
            return self.DildayRate(z)


class CovColor:
    """
    class to estimate CovColor from lc using Fisher matrix element

    Parameters
    ---------------
    lc : `pandas.DataFrame`
        lc to process. Should contain the Fisher matrix components
        ie the sum of the derivative of the fluxes wrt SN parameters
    """

    def __init__(self, lc):

        self.Cov_colorcolor = self.varColor(lc)

    def varColor(self, lc):
        """
        Method to estimate the variance color from matrix element

        Parameters
        --------------
        lc : `pandas.DataFrame`
            data to process containing the derivative of the flux with respect to SN parameters

        Returns
        ----------
        Cov_colorcolor : `float`
        """
        a1 = lc["F_x0x0"]
        a2 = lc["F_x0x1"]
        a3 = lc["F_x0daymax"]
        a4 = lc["F_x0color"]

        b1 = a2
        b2 = lc["F_x1x1"]
        b3 = lc["F_x1daymax"]
        b4 = lc["F_x1color"]

        c1 = a3
        c2 = b3
        c3 = lc["F_daymaxdaymax"]
        c4 = lc["F_daymaxcolor"]

        d1 = a4
        d2 = b4
        d3 = c4
        d4 = lc["F_colorcolor"]

        detM = a1 * self.det(b2, b3, b4, c2, c3, c4, d2, d3, d4)
        detM -= b1 * self.det(a2, a3, a4, c2, c3, c4, d2, d3, d4)
        detM += c1 * self.det(a2, a3, a4, b2, b3, b4, d2, d3, d4)
        detM -= d1 * self.det(a2, a3, a4, b2, b3, b4, c2, c3, c4)

        res = (
            -a3 * b2 * c1
            + a2 * b3 * c1
            + a3 * b1 * c2
            - a1 * b3 * c2
            - a2 * b1 * c3
            + a1 * b2 * c3
        )

        return res / detM

    def det(self, a1, a2, a3, b1, b2, b3, c1, c2, c3):
        """
        Method to estimate the det of a matrix from its values

        Parameters
        -------------
        Values of the matrix
        (a1 a2 a3)
        (b1 b2 b3)
        (c1 c2 c3)

        Returns
        -----------
        det value
        """
        resp = a1 * b2 * c3 + b1 * c2 * a3 + c1 * a2 * b3
        resm = a3 * b2 * c1 + b3 * c2 * a1 + c3 * a2 * b1

        return resp - resm
