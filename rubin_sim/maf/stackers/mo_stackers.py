__all__ = (
    "BaseMoStacker",
    "MoMagStacker",
    "AppMagStacker",
    "CometAppMagStacker",
    "SNRStacker",
    "EclStacker",
)

import warnings

import numpy as np

from .base_stacker import BaseStacker
from .mo_phase import phase__halley_marcus

# Willmer 2018, ApJS 236, 47
# VEGA V mag and AB mag of sun (LSST-equivalent bandpasses)
VMAG_SUN = -26.76  # Vega mag
AB_SUN = {"u": -25.30, "g": -26.52, "r": -26.93, "i": -27.05, "z": -27.07, "y": -27.07}
KM_PER_AU = 149597870.7


class BaseMoStacker(BaseStacker):
    """Base class for moving object (SSobject)  stackers.
    Relevant for MoSlicer ssObs (pd.dataframe).

    Provided to add moving-object specific API for 'run'
    method of moving object stackers.
    """

    def run(self, sso_obs, href, hval=None):
        # Redefine this here, as the API does not match BaseStacker.
        if hval is None:
            hval = href
        if len(sso_obs) == 0:
            return sso_obs
        # Add the columns.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sso_obs, cols_present = self._add_stacker_cols(sso_obs)
        # Here we don't really care about cols_present, because almost
        # every time we will be readding
        # columns anymore (for different H values).
        return self._run(sso_obs, href, hval)


class MoMagStacker(BaseMoStacker):
    """Add columns relevant to SSobject apparent magnitudes and
    visibility to the slicer ssoObs
    dataframe, given a particular Href and current h_val.

    Specifically, this stacker adds magLimit, appMag, SNR, and vis.
    magLimit indicates the appropriate limiting magnitude to consider
    for a particular object in a particular
    observation, when combined with the losses due to detection
    (dmag_detect) or trailing (dmagTrail).
    appMag adds the apparent magnitude in the filter of the current object,
    at the current h_val.
    SNR adds the SNR of this object, given the magLimit.
    vis adds a flag (0/1) indicating whether an object was visible
    (assuming a 5sigma threshhold including
    some probabilistic determination of visibility).

    Parameters
    ----------
    m5Col : `str`, optional
        Name of the column describing the 5 sigma depth of each visit.
        Default fiveSigmaDepth.
    lossCol : `str`, optional
        Name of the column describing the magnitude losses,
        due to trailing (dmagTrail) or detection (dmag_detect).
        Default dmag_detect.
    gamma : `float`, optional
        The 'gamma' value for calculating SNR. Default 0.038.
        LSST range under normal conditions is about 0.037 to 0.039.
    sigma : `float`, optional
        The 'sigma' value for probabilistic prediction of whether or not
        an object is visible at 5sigma.
        Default 0.12.
        The probabilistic prediction of visibility is based on
        Fermi-Dirac completeness formula (see SDSS, eqn 24, Stripe82 analysis:
        http://iopscience.iop.org/0004-637X/794/2/120/pdf/apj_794_2_120.pdf).
    randomSeed: `int` or None, optional
        If set, then used as the random seed for the numpy random number
        generation for the dither offsets.
        Default: None.
    """

    cols_added = ["appMag", "SNR", "vis"]

    def __init__(
        self,
        magtype="asteroid",
        v_mag_col="magV",
        color_col="dmag_color",
        loss_col="dmag_detect",
        m5_col="fiveSigmaDepth",
        seeing_col="seeingFwhmGeom",
        filter_col="filter",
        gamma=0.038,
        sigma=0.12,
        random_seed=None,
    ):
        if magtype == "asteroid":
            self.mag_stacker = AppMagStacker(v_mag_col=v_mag_col, color_col=color_col, loss_col=loss_col)
            self.cols_req = [m5_col, v_mag_col, color_col, loss_col]
        elif magtype.startswith("comet"):
            # magtype should be = comet_oort comet_short or comet_mbc
            comet_type = magtype.split("_")[-1]
            self.mag_stacker = CometAppMagStacker(
                comet_type=comet_type,
                ap=0.04,
                rh_col="helio_dist",
                delta_col="geo_dist",
                phase_col="phase",
                seeing_col=seeing_col,
                ap_scale=1,
                filter_col=filter_col,
                v_mag_col=v_mag_col,
                color_col=color_col,
                loss_col=loss_col,
            )
            self.cols_req = [
                m5_col,
                v_mag_col,
                color_col,
                loss_col,
                "helio_dist",
                "geo_dist",
                "phase",
                seeing_col,
                filter_col,
            ]
        else:
            self.mag_stacker = AppMagNullStacker()
        self.snr_stacker = SNRStacker(
            app_mag_col="appMag",
            m5_col=m5_col,
            gamma=gamma,
            sigma=sigma,
            random_seed=random_seed,
        )

        self.units = ["mag", "mag", "SNR", ""]

    def _run(self, sso_obs, href, hval):
        # hval = current H value (useful if cloning over H range),
        # href = reference H value from orbit.
        # Without cloning, href = hval.
        # add apparent magnitude
        self.mag_stacker._run(sso_obs, href, hval)
        # add snr
        self.snr_stacker._run(sso_obs, href, hval)
        return sso_obs


class AppMagNullStacker(BaseMoStacker):
    """Do nothing to calculate an apparent magnitude.

    This assumes an apparent magnitude was part of the input data and
    does not need to be modified (no
    cloning, color terms, trailing losses, etc). Just return the appMag column.

    This would not be necessary in general, but appMag is treated as a
    special column (because we must have an
    apparent magnitude for most of the basic moving object metrics,
    and it must be calculated before SNR
    if that is also needed).
    """

    cols_added = ["appMag"]

    def __init__(self, app_mag_col="appMag"):
        self.app_mag_col = app_mag_col
        self.units = [
            "mag",
        ]
        self.cols_req = [self.app_mag_col]

    def _run(self, sso_obs, href, hval):
        sso_obs["appMag"] = sso_obs[self.app_mag_col]
        return sso_obs


class AppMagStacker(BaseMoStacker):
    """Add apparent magnitude of an object for the current h_val
    (compared to Href in the orbit file),
    incorporating the magnitude losses due to trailing/detection,
    as well as the color of the object.

    This is calculated from the reported mag_v in the input observation
    file (calculated assuming Href) as:
    .. codeblock::python

        ssoObs['appMag'] = ssoObs[self.vMagCol] + ssoObs[self.colorCol] +
        ssoObs[self.lossCol] + h_val - Href

    Using the vMag reported in the input observations implicitly uses
    the phase curve coded in at that point;
    for Oorb this is an H/G phase curve, with G=0.15 unless otherwise
    specified in the orbit file.
    See sims_movingObjects for more details on the color and loss quantities.

    Parameters
    ----------
    v_mag_col : `str`, optional
        Name of the column containing the base V magnitude for the
        object at H=Href.
    loss_col : `str`, optional
        Name of the column describing the magnitude losses,
        due to trailing (dmagTrail) or detection (dmag_detect).
        Default dmag_detect.
    color_col : `str`, optional
        Name of the column describing the color correction
        (into the observation filter, from V).
        Default dmag_color.
    """

    cols_added = ["appMag"]

    def __init__(self, v_mag_col="magV", color_col="dmag_color", loss_col="dmag_detect"):
        self.v_mag_col = v_mag_col
        self.color_col = color_col
        self.loss_col = loss_col
        self.cols_req = [self.v_mag_col, self.color_col, self.loss_col]
        self.units = [
            "mag",
        ]

    def _run(self, sso_obs, href, hval):
        # hval = current H value (useful if cloning over H range),
        # href = reference H value from orbit.
        # Without cloning, href = hval.
        sso_obs["appMag"] = (
            sso_obs[self.v_mag_col] + sso_obs[self.color_col] + sso_obs[self.loss_col] + hval - href
        )
        return sso_obs


class CometAppMagStacker(BaseMoStacker):
    """Add a cometary apparent magnitude, including nucleus and coma,
    based on a calculation of
    Afrho (using the current h_val) and a Halley-Marcus phase curve
    for the coma brightness.

    Parameters
    ----------
    cometType : `str`, optional
        Type of comet - short, oort, or mbc.
        This setting also sets the value of Afrho1 and k:
        short = Afrho1 / R^2 = 100 cm/km2, k = -4
        oort = Afrho1 / R^2 = 1000 cm/km2, k = -2
        mbc = Afrho1 / R^2 = 4000 cm/km2, k = -6.
        Default = 'oort'.
        It is also possible to pass this a dictionary instead:
        the dictionary should contain 'k' and
        'afrho1_const' keys, which will be used to set these values directly.
        (e.g. cometType = {'k': -3.5, 'afrho1_const': 1500}).
    ap : `float`, optional
        The albedo for calculating the object's size. Default 0.04
    rh_col : `str`, optional
        The column name for the heliocentric distance (in AU).
        Default 'helio_dist'.
    delta_col : `str`, optional
        The column name for the geocentric distance (in AU).
        Default 'geo_dist'.
    phase_col : `str`, optional
        The column name for the phase value (in degrees).
        Default 'phase'.
    """

    cols_added = ["appMag"]

    def __init__(
        self,
        comet_type="oort",
        ap=0.04,
        rh_col="helio_dist",
        delta_col="geo_dist",
        phase_col="phase",
        seeing_col="FWHMgeom",
        ap_scale=1,
        filter_col="filter",
        v_mag_col="magV",
        color_col="dmag_color",
        loss_col="dmag_detect",
    ):
        self.units = ["mag"]  # new column units
        # Set up k and Afrho1 constant values.
        comet_types = {
            "short": {"k": -4, "Afrho1_const": 100},
            "oort": {"k": -2, "Afrho1_const": 1000},
            "mbc": {"k": -6, "Afrho1_const": 4000},
        }
        self.k = None
        self.afrho1_const = None
        if isinstance(comet_type, str):
            if comet_type in comet_types:
                self.k = comet_types[comet_type]["k"]
                self.afrho1_const = comet_types[comet_type]["Afrho1_const"]
        if isinstance(comet_type, dict):
            if "k" in comet_type:
                self.k = comet_type["k"]
            if "Afrho1_const" in comet_type:
                self.afrho1_const = comet_type["Afrho1_const"]
        if self.k is None or self.afrho1_const is None:
            raise ValueError(
                f"comet_type must be a string {comet_types} or "
                f'dict containing "k" and "Afrho1_const" - but received {comet_type}'
            )
        # Phew, now set the simple stuff.
        self.ap = ap
        self.rh_col = rh_col
        self.delta_col = delta_col
        self.phase_col = phase_col
        self.seeing_col = seeing_col
        self.ap_scale = ap_scale
        self.filter_col = filter_col
        self.v_mag_col = v_mag_col
        self.color_col = color_col
        self.loss_col = loss_col
        # names of required columns
        self.cols_req = [
            self.rh_col,
            self.delta_col,
            self.phase_col,
            self.seeing_col,
            self.filter_col,
            self.v_mag_col,
            self.color_col,
            self.loss_col,
        ]

    def _run(self, sso_obs, href, hval):
        # Calculate radius from the current H value (hval).
        radius = 10 ** (0.2 * (VMAG_SUN - hval)) / np.sqrt(self.ap) * KM_PER_AU
        # Calculate expected Afrho - this is a value that describes
        # how the brightness of the coma changes
        afrho1 = self.afrho1_const * radius**2
        phase_val = phase__halley_marcus(sso_obs[self.phase_col])
        # afrho is equivalent to a sort of 'absolute' magnitude of the coma
        afrho = afrho1 * sso_obs[self.rh_col] ** self.k * phase_val
        # rho describes the projected area on the sky
        # (project the aperture into cm on-sky)
        # Using the seeing * apScale as the aperture
        radius_aperture = sso_obs[self.seeing_col] * self.ap_scale
        rho = 725e5 * sso_obs[self.delta_col] * radius_aperture
        # Calculate the expected apparent of the comet coma = sun + correction
        delta = sso_obs[self.delta_col] * KM_PER_AU * 1000  # delta in cm
        dm = -2.5 * np.log10(afrho * rho / (2 * sso_obs[self.rh_col] * delta) ** 2)
        coma = np.zeros(len(sso_obs), float)
        # This calculates a coma mag that scales with the sun's color in
        # each bandpass, but the coma
        # modification is gray (i.e. it's just reflecting sunlight)
        for f in sso_obs[self.filter_col]:
            match = np.where(sso_obs[self.filter_col] == f)
            coma[match] = AB_SUN[f] + dm[match]
        # Calculate cometary nucleus magnitude -- we'll use the
        # apparent V mag adapted from OOrb as well as
        # the object's color - these are generally assumed to be
        # D type (which was used in sims_movingObjects)
        nucleus = sso_obs[self.v_mag_col] + sso_obs[self.color_col] + sso_obs[self.loss_col] + hval - href
        # add coma and nucleus then ready for calculation of SNR, etc.
        sso_obs["appMag"] = -2.5 * np.log10(10 ** (-0.4 * coma) + 10 ** (-0.4 * nucleus))
        return sso_obs


class SNRStacker(BaseMoStacker):
    """Add SNR and visibility for a particular object,
    given the five sigma depth of the image and the
    apparent magnitude (whether from AppMagStacker or CometAppMagStacker etc).

    The SNR simply calculates the SNR based on the five sigma depth and
    the apparent magnitude.
    The 'vis' column is a probabilistic flag (0/1) indicating whether
    the object was detected, assuming
    a 5-sigma SNR threshold and then applying a probabilistic cut on whether
    it was detected or not (i.e.
    there is a gentle roll-over in 'vis' from 1 to 0 depending on the
    SNR of the object).
    This is based on the Fermi-Dirac completeness formula as described
    in equation 24 of the Stripe 82 SDSS
    analysis here:
    http://iopscience.iop.org/0004-637X/794/2/120/pdf/apj_794_2_120.pdf.

    Parameters
    ----------
    app_mag_col : `str`, optional
        Name of the column describing the apparent magnitude of the object.
        Default 'appMag'.
    m5_col : `str`, optional
        Name of the column describing the 5 sigma depth of each visit.
        Default fiveSigmaDepth.
    gamma : `float`, optional
        The 'gamma' value for calculating SNR. Default 0.038.
        LSST range under normal conditions is about 0.037 to 0.039.
    sigma : `float`, optional
        The 'sigma' value for probabilistic prediction of whether or not
        an object is visible at 5sigma.
        Default 0.12.
        The probabilistic prediction of visibility is based on Fermi-Dirac
        completeness formula (see SDSS, eqn 24, Stripe82 analysis:
        http://iopscience.iop.org/0004-637X/794/2/120/pdf/apj_794_2_120.pdf).
    random_seed: `int` or None, optional
        If set, then used as the random seed for the numpy random number
        generation for the probability of detection. Default: None.
    """

    cols_added = ["SNR", "vis"]

    def __init__(
        self,
        app_mag_col="appMag",
        m5_col="fiveSigmaDepth",
        gamma=0.038,
        sigma=0.12,
        random_seed=None,
    ):
        self.app_mag_col = app_mag_col
        self.m5_col = m5_col
        self.gamma = gamma
        self.sigma = sigma
        self.random_seed = random_seed
        self.cols_req = [self.app_mag_col, self.m5_col]
        self.units = ["SNR", ""]

    def _run(self, sso_obs, href, hval):
        # hval = current H value (useful if cloning over H range),
        # href = reference H value from orbit.
        # Without cloning, href = hval.
        xval = np.power(10, 0.5 * (sso_obs[self.app_mag_col] - sso_obs[self.m5_col]))
        sso_obs["SNR"] = 1.0 / np.sqrt((0.04 - self.gamma) * xval + self.gamma * xval * xval)
        completeness = 1.0 / (1 + np.exp((sso_obs[self.app_mag_col] - sso_obs[self.m5_col]) / self.sigma))
        if not hasattr(self, "_rng"):
            if self.random_seed is not None:
                self._rng = np.random.RandomState(self.random_seed)
            else:
                self._rng = np.random.RandomState(734421)
        probability = self._rng.random_sample(len(sso_obs[self.app_mag_col]))
        sso_obs["vis"] = np.where(probability <= completeness, 1, 0)
        return sso_obs


class EclStacker(BaseMoStacker):
    """
    Add ecliptic latitude/longitude (ecLat/ecLon) to the slicer ssoObs
    (in degrees).

    Parameters
    -----------
    ra_col : `str`, optional
        Name of the RA column to convert to ecliptic lat/long. Default 'ra'.
    dec_col : `str`, optional
        Name of the Dec column to convert to ecliptic lat/long. Default 'dec'.
    in_deg : `bool`, optional
        Flag indicating whether RA/Dec are in degrees. Default True.
    """

    cols_added = ["ecLat", "ecLon"]

    def __init__(self, ra_col="ra", dec_col="dec", in_deg=True):
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.in_deg = in_deg
        self.cols_req = [self.ra_col, self.dec_col]
        self.units = ["deg", "deg"]
        self.ecnode = 0.0
        self.ecinc = np.radians(23.439291)

    def _run(self, sso_obs, href, hval):
        ra = sso_obs[self.ra_col]
        dec = sso_obs[self.dec_col]
        if self.in_deg:
            ra = np.radians(ra)
            dec = np.radians(dec)
        x = np.cos(ra) * np.cos(dec)
        y = np.sin(ra) * np.cos(dec)
        z = np.sin(dec)
        xp = x
        yp = np.cos(self.ecinc) * y + np.sin(self.ecinc) * z
        zp = -np.sin(self.ecinc) * y + np.cos(self.ecinc) * z
        sso_obs["ecLat"] = np.degrees(np.arcsin(zp))
        sso_obs["ecLon"] = np.degrees(np.arctan2(yp, xp))
        sso_obs["ecLon"] = sso_obs["ecLon"] % 360
        return sso_obs
