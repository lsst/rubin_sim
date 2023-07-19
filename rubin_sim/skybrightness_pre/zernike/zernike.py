"""Sky brightnes approzimation using Zernike polynomials

The form and notation used here follow:

Thibos, L. N., Applegate, R. A., Schwiegerling, J. T., Webb, R. & VSIA
Standards Taskforce Members. Vision science and its
applications. Standards for reporting the optical aberrations of
eyes. J Refract Surg 18, S652-660 (2002).
"""

__all__ = ("ZernikeSky", "SkyModelZernike", "SkyBrightnessPreData")

import logging
import os
import warnings
from functools import lru_cache
from glob import glob

# imports
from math import factorial

import healpy
import numpy as np
import palpy
import pandas as pd
import scipy.optimize
from numexpr import NumExpr
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

import rubin_sim.utils as utils
from rubin_sim.data import get_data_dir

# constants

logging.basicConfig(format="%(asctime)s %(message)s")
LOGGER = logging.getLogger(__name__)

TELESCOPE = utils.Site("LSST")
SIDEREAL_TIME_SAMPLES_RAD = np.radians(np.arange(361, dtype=float))
BANDS = ("u", "g", "r", "i", "z", "y")


# exception classes

# interface functions


def fit_pre(npy_fname, npz_fname, *args, **kwargs):
    """Fit Zernike coefficients to a pre-computed data set

    Parameters
    ----------
    npy_fname : `str`
        File name of the SkyBrightessPre <MJD>_<MDJ>.npy file
    npz_fname : `str`
        File name of the SkyBrightessPre <MJD>_<MDJ>.npz file

    other arguments are passed to the ZernikeSky constructor.

    Returns
    -------
    zernike_coeffs : `pd.DataFrame`
        A DataFrame with the coefficients, indexed by band and mjd.
    """
    # Load the pre-computed data
    npz = np.load(npz_fname, allow_pickle=True)
    npz_hdr = npz["header"][()]
    npz_data = npz["dict_of_lists"][()]
    pre_sky = np.load(npy_fname, allow_pickle=True)

    mjds = npz_data["mjds"]
    alt = npz_hdr["alt"]
    az = npz_hdr["az"]
    zernike_coeffs_by_band = []
    zernike_sky = ZernikeSky(*args, **kwargs)
    for band in pre_sky.dtype.fields.keys():
        LOGGER.info("Starting %s band", band)
        zernike_coeff_arrays = []
        for mjd_idx, mjd in enumerate(mjds):
            zernike_coeff_arrays.append(zernike_sky.fit_coeffs(alt, az, pre_sky[band][mjd_idx], mjd))
            if mjd_idx % 1000 == 0:
                msg = f"Finished {mjd_idx*100.0/float(len(mjds)):.2f}%"
                LOGGER.debug(msg)
        zernike_coeffs_by_band.append(
            pd.DataFrame(
                zernike_coeff_arrays,
                columns=np.arange(len(zernike_coeff_arrays[0])),
                index=pd.MultiIndex.from_arrays(
                    [np.full_like(mjds, band, dtype=type(band)), mjds],
                    names=["band", "mjd"],
                ),
            )
        )

    zernike_coeffs = pd.concat(zernike_coeffs_by_band)
    return zernike_coeffs


def bulk_zernike_fit(data_dir, out_fname, *args, **kwargs):
    """Fit Zernike coeffs to all SkyBrightnessPre files in a directory.

    Parameters
    ----------
    data_dir : `str`
        Name of the directory in which to look for SkyBrightnessPre
        data files.
    out_fname: `str`
        Name of the file in which to save fit coefficients.

    other arguments are passed to the ZernikeSky constructor.

    Returns
    -------
    zernike_coeffs : `pd.DataFrame`
        A DataFrame with the coefficients, indexed by band and mjd.
    """
    zernike_coeff_batches = []
    for npz_fname in glob(os.path.join(data_dir, "?????_?????.npz")):
        LOGGER.info("Processing %s", npz_fname)
        npy_fname = os.path.splitext(npz_fname)[0] + ".npy"
        zernike_coeff_batch = fit_pre(npy_fname, npz_fname, *args, **kwargs)
        zernike_coeff_batches.append(zernike_coeff_batch)

    zernike_coeffs = pd.concat(zernike_coeff_batches)
    zernike_coeffs.sort_index(level="mjd", inplace=True)

    if out_fname is not None:
        zernike_coeffs.to_hdf(out_fname, "zernike_coeffs", complevel=6)

        zernike_sky = ZernikeSky(*args, **kwargs)
        zernike_metadata = pd.Series({"order": zernike_sky.order, "max_zd": zernike_sky.max_zd})
        zernike_metadata.to_hdf(out_fname, "zernike_metadata")

    return zernike_coeffs

    # classes


class ZernikeSky:
    """Zernike sky approximator.

    Parameters
    ----------
    order : `int`, optional
        The order of the Zernike polynomial to use. Default is 6.
    nside : `int`, optional
        The nside of the healpix array to pre-compute Zernike Z terms for.
        Default is 32.
    max_zd : `float`, optional
        The maximum zenith distance, in degrees. This value will correspond
        to rho=1 in the Thibos et al. (2002) notation.
        Default is 67.
    dtype : `type`: optional
        The numpy type to use for all calculations. Default is `np.float64`.
    """

    def __init__(self, order=6, nside=32, max_zd=67, dtype=np.float64):
        self.order = order
        self.dtype = dtype
        self.nside = nside

        # Sets the value of zd where rho (radial coordinate of the
        # unit disk in which Zernike polynomials are orthogonal) = 1
        self.max_zd = max_zd

        # a list of functions to calculate big Z given rho, phi,
        # following eqn 1 of Thibos et al. (2002). The jth element of
        # the list returns the jth Z, following the indexing
        # convertions of Thibos et al. eqn 4.
        #
        # Should switch to using functools.cached_property in python 3.8
        self._z_function = self._build_z_functions()

        # A function that calculates the full Zernike approximation,
        # taking rho and phi as arguments.
        #
        # numexpr can only compile functions with a limited number of
        # arguments. If the order is too high, sum the terms
        # separately
        if order <= 7:
            self._zern_function = self._build_zern_function()
        else:
            self._zern_function = self._compute_sky_by_sum

        # big Z values for all m,n at all rho, phi in the
        # pre-defined healpix coordinate, following eqn 1 of Thibos et
        # al. (2002) The array returned should be indexed with j,
        # Should switch to using functools.cached_property in python 3.8
        self.healpix_z = self._compute_healpix_z()
        self._interpolate_healpix_z = interp1d(
            SIDEREAL_TIME_SAMPLES_RAD, self.healpix_z, axis=0, kind="nearest"
        )

        # A pd.DataFrame of zernike coeffs, indexed by mjd, providing the
        # Zernike polynomial coefficients for the approximation of the
        # sky at that time. That is, self._coeffs[5, 3] is the
        # j=3 coefficient of the approximation of the sky at
        # mjd=self.mjds[5], where j is defined as in Thibos et al. eqn 4.
        self._coeffs = pd.DataFrame()

    def load_coeffs(self, fname, band):
        """Load Zernike coefficients from a file.

        Parameters
        ----------
        fname : `str`
            The file name of the hdf5 file with the Zernike coeffs.
        band : `str`
            The band to load.

        """
        zernike_metadata = pd.read_hdf(fname, "zernike_metadata")
        assert self.order == zernike_metadata["order"]
        assert self.max_zd == zernike_metadata["max_zd"]
        all_zernike_coeffs = pd.read_hdf(fname, "zernike_coeffs")
        self._coeffs = all_zernike_coeffs.loc[band]
        self._coeff_calc_func = interp1d(self._coeffs.index.values, self._coeffs.values, axis=0)

    def compute_sky(self, alt, az, mjd=None):
        """Estimate sky values

        Parameters
        ----------
        alt : `np.ndarray`, (N)
            An array of altitudes above the horizon, in degrees
        az : `np.ndarray`, (N)
            An array of azimuth coordinates, in degrees
        mjd : `float`
            The time (floating point MJD) at which to estimate the sky.

        Returns
        -------
        `np.ndarray` (N) of sky brightnesses (mags/asec^2)
        """
        rho = self._calc_rho(alt)
        phi = self._calc_phi(az)
        result = self._zern_function(rho, phi, *tuple(self.coeffs(mjd)))
        return result

    def _compute_sky_by_sum(self, rho, phi, *coeffs):
        z = self._compute_z(rho, phi)
        if len(z.shape) == 2:
            result = np.sum(np.array(coeffs) * z, axis=1)
        else:
            result = np.sum(np.array(coeffs) * z)
        return result

    def compute_healpix(self, hpix, mjd=None):
        """Estimate sky values

        Parameters
        ----------
        hpix : `int`, (N)
            Array of healpix indexes of the desired coordinates.
        mjd : `float`
            The time (floating point MJD) at which to estimate the sky.

        Returns
        -------
        `np.ndarray` (N) of sky brightnesses (mags/asec^2)
        """
        interpolate_healpix_z = self._interpolate_healpix_z
        gmst = palpy.gmst(mjd)
        mjd_healpix_z = interpolate_healpix_z(gmst)
        # mjd_healpix_z = self.healpix_z[int(np.degrees(gmst))]
        if hpix is None:
            result = np.sum(self.coeffs(mjd) * mjd_healpix_z, axis=1)
        else:
            result = np.sum(self.coeffs(mjd) * mjd_healpix_z[hpix], axis=1)
        return result

    def coeffs(self, mjd):
        """Zerinke coefficients at a time

        Parameters
        ----------
        mjd : `float`
            The time (floating point MJD) at which to estimate the sky.

        Returns
        -------
        `np.ndarray` of Zernike coefficients following the OSA/ANSI
        indexing convention described in Thibos et al. (2002).
        """
        if len(self._coeffs) == 1:
            these_coeffs = self._coeffs.loc[mjd]
        else:
            calc_these_coeffs = self._coeff_calc_func
            these_coeffs = calc_these_coeffs(mjd)
        return these_coeffs

    def fit_coeffs(self, alt, az, sky, mjd, min_moon_sep=10, maxdiff=False):
        """Fit Zernike coefficients to a set of points

        Parameters
        ----------
        alt : `np.ndarray`, (N)
            An array of altitudes above the horizon, in degrees
        az : `np.ndarray`, (N)
            An array of azimuth coordinates, in degrees
        sky : `np.ndarray`, (N)
            An array of sky brightness values (mags/asec^2)
        mjd : `float`
            The time (floating point MJD) at which to estimate the sky.
        maxdiff : `bool`
            Minimize the maximum difference between the estimate and data,
            rather than the default RMS.

        """

        # Do not fit too close to the moon
        alt_rad, az_rad = np.radians(alt), np.radians(az)
        gmst_rad = palpy.gmst(mjd)
        lst_rad = gmst_rad + TELESCOPE.longitude_rad
        moon_ra_rad, moon_decl_rad, moon_diam = palpy.rdplan(
            mjd, 3, TELESCOPE.longitude_rad, TELESCOPE.latitude_rad
        )
        moon_ha_rad = lst_rad - moon_ra_rad
        moon_az_rad, moon_el_rad = palpy.de2h(moon_ha_rad, moon_decl_rad, TELESCOPE.latitude_rad)
        moon_sep_rad = palpy.dsepVector(
            np.full_like(az_rad, moon_az_rad),
            np.full_like(alt_rad, moon_el_rad),
            az_rad,
            alt_rad,
        )
        moon_sep = np.degrees(moon_sep_rad)

        rho = self._calc_rho(alt)
        phi = self._calc_phi(az)
        good_points = np.logical_and(rho <= 1.0, moon_sep > min_moon_sep)

        rho = rho[good_points]
        phi = phi[good_points]
        sky = sky[good_points]
        alt = alt[good_points]
        az = az[good_points]
        num_points = len(alt)
        assert len(az) == num_points
        assert len(sky) == num_points

        z = np.zeros((num_points, self._number_of_terms), dtype=self.dtype)
        for j in np.arange(self._number_of_terms):
            compute_z = self._z_function[j]
            z[:, j] = compute_z(rho, phi)

        # If the points being fit were evenly distributed across the sky,
        # we might be able to get away with a multiplication rather than
        # a linear regression, but we might be asked to fit masked data
        zern_fit = LinearRegression(fit_intercept=False).fit(z, sky)

        fit_coeffs = zern_fit.coef_

        if maxdiff:

            def max_abs_diff(test_coeffs):
                max_resid = np.max(np.abs(np.sum(test_coeffs * z, axis=1) - sky))
                return max_resid

            min_fit = scipy.optimize.minimize(max_abs_diff, fit_coeffs)
            fit_coeffs = min_fit.x

        self._coeffs = pd.DataFrame(
            [fit_coeffs],
            columns=np.arange(len(fit_coeffs)),
            index=pd.Index([mjd], name="mjd"),
        )
        return fit_coeffs

    def _compute_healpix_z(self):
        # Compute big Z values for all m,n at all rho, phi in the
        # pre-defined healpix coordinate, following eqn 1 of Thibos et
        # al. (2002) The array returned should be indexed with j,
        # following the conventions of eqn 4.
        sphere_npix = healpy.nside2npix(self.nside)
        sphere_ipix = np.arange(sphere_npix)
        ra, decl = healpy.pix2ang(self.nside, sphere_ipix, lonlat=True)

        num_st = len(SIDEREAL_TIME_SAMPLES_RAD)
        healpix_z = np.full([num_st, sphere_npix, self._number_of_terms], np.nan)
        for st_idx, gmst_rad in enumerate(SIDEREAL_TIME_SAMPLES_RAD):
            lst_rad = gmst_rad + TELESCOPE.longitude_rad
            ha_rad = lst_rad - np.radians(ra)
            az_rad, alt_rad = palpy.de2hVector(ha_rad, np.radians(decl), TELESCOPE.latitude_rad)
            sphere_az, sphere_alt = np.degrees(az_rad), np.degrees(alt_rad)

            # We only need the half sphere above the horizen
            visible_ipix = sphere_ipix[sphere_alt > 0]
            alt, az = sphere_alt[visible_ipix], sphere_az[visible_ipix]
            rho = self._calc_rho(alt)
            phi = self._calc_phi(az)
            healpix_z[st_idx, visible_ipix] = self._compute_z(rho, phi)

        return healpix_z

    def _compute_horizan_healpix_z(self):
        # Compute big Z values for all m,n at all rho, phi in the
        # pre-defined healpix coordinate, following eqn 1 of Thibos et
        # al. (2002) The array returned should be indexed with j,
        # following the conventions of eqn 4.
        sphere_npix = healpy.nside2npix(self.nside)
        sphere_ipix = np.arange(sphere_npix)
        sphere_az, sphere_alt = healpy.pix2ang(self.nside, sphere_ipix, lonlat=True)

        # We only need the half sphere above the horizen
        ipix = sphere_ipix[sphere_alt > 0]
        alt, phi_deg = sphere_alt[ipix], sphere_az[ipix]
        rho = self._calc_rho(alt)
        rho, phi = (90.0 - alt) / self.max_zd, np.radians(phi_deg)
        healpix_z = self._compute_z(rho, phi)
        return healpix_z

    def _compute_z(self, rho, phi):
        # Compute big Z values for all m,n at rho, phi
        # following eqn 1 of Thibos et al. (2002)
        # The array returned should be indexed with j,
        # following the conventions of eqn 4.
        try:
            npix = len(rho)
            z = np.zeros((npix, self._number_of_terms), dtype=self.dtype)
            for j in np.arange(self._number_of_terms):
                compute_z = self._z_function[j]
                z[:, j] = compute_z(rho, phi)
        except TypeError:
            z = np.zeros(self._number_of_terms, dtype=self.dtype)
            for j in np.arange(self._number_of_terms):
                compute_z = self._z_function[j]
                z[j] = compute_z(rho, phi)
        return z

    def _build_z_functions(self):
        z_functions = []
        for j in np.arange(self._number_of_terms):
            z_functions.append(self._make_z_function(j))
        return z_functions

    def _build_zern_function(self):
        coeffs = [f"c{j}" for j in np.arange(self._number_of_terms)]

        expression = ""
        for j, coeff in enumerate(coeffs):
            zern_z_expr = self._make_z_expression(j)
            if zern_z_expr == "(1)":
                term = f"{coeff}"
            else:
                term = f"{coeff}*({zern_z_expr})"
            if expression == "":
                expression = term
            else:
                expression += f" + {term}"

        arg_types = []
        if expression.find("rho") >= 0:
            arg_types.append(
                ("rho", self.dtype),
            )
        if expression.find("phi") >= 0:
            arg_types.append(
                ("phi", self.dtype),
            )
        for coeff in coeffs:
            arg_types.append(
                (coeff, self.dtype),
            )
        arg_types = tuple(arg_types)

        zern_function = NumExpr(expression, arg_types)
        return zern_function

    @property
    def _number_of_terms(self):
        n_terms = np.sum(np.arange(self.order) + 1)
        return n_terms

    def _make_r_expression(self, m, n):
        if (n - m) % 2 == 1:
            return 0
        assert n >= m
        assert m >= 0
        m = int(m)
        n = int(n)
        num_terms = 1 + (n - m) // 2
        expression = "("
        for k in range(num_terms):
            # From eqn 2 of Thibos et al. (2002)
            coeff = (((-1) ** k) * factorial(n - k)) / (
                factorial(k) * factorial(int((n + m) / 2 - k)) * factorial(int((n - m) / 2 - k))
            )
            assert coeff == int(coeff)
            coeff = int(coeff)
            power = n - 2 * k
            if len(expression) > 1:
                expression += " + "
            if power == 0:
                expression += f"{coeff}"
            elif power == 1:
                expression += f"{coeff}*rho"
            else:
                expression += f"{coeff}*rho**{power}"
        expression += ")"
        return expression

    def _make_z_expression(self, j=None, mprime=None, n=None):
        if j is None:
            assert mprime is not None
            assert n is not None
        else:
            assert mprime is None
            assert n is None
            # From eqn 5 in Thibos et al. (2002)
            n = np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2).astype(int)

            # From eqn 6 in Thibos et al. (2002)
            mprime = 2 * j - n * (n + 2)

        m = np.abs(mprime)
        r = self._make_r_expression(m, n)

        # From eqn. 3 of Thibos et al. 2002, again
        delta = 1 if m == 0 else 0
        big_nsq = 2 * (n + 1) / (1 + delta)
        assert int(big_nsq) == big_nsq
        big_nsq = int(big_nsq)

        if mprime == 0:
            expression = f"sqrt({big_nsq})*{r}"
        elif mprime > 0:
            expression = f"sqrt({big_nsq})*{r}*cos({m}*phi)"
        elif mprime < 0:
            expression = f"sqrt({big_nsq})*{r}*sin({m}*phi)"
        else:
            assert False

        return expression

    def _make_z_function(self, j=None, mprime=None, n=None):
        expression = self._make_z_expression(j, mprime, n)

        arg_types = []
        if expression.find("rho") >= 0:
            arg_types.append(
                ("rho", self.dtype),
            )
        if expression.find("phi") >= 0:
            arg_types.append(
                ("phi", self.dtype),
            )
        arg_types = tuple(arg_types)

        raw_z_function = NumExpr(expression, arg_types)

        # Create functions with dummy arguments so that
        # terms that do not require both phi and rho can
        # still accept them, such that all z_functions
        # can be called in the same way.
        if len(arg_types) == 0:

            def z_function(rho=None, phi=None):
                return raw_z_function()

        elif len(arg_types) == 1:

            def z_function(rho, phi=None):
                return raw_z_function(rho)

        else:
            z_function = raw_z_function

        return z_function

    def _calc_rho(self, alt):
        zd = 90.0 - alt
        if np.isscalar(alt) and zd > self.max_zd:
            return np.nan

        rho = zd / self.max_zd

        if not np.isscalar(alt):
            rho[zd > self.max_zd] = np.nan

        return rho

    def _calc_phi(self, az):
        phi = np.radians(az)
        return phi


class SkyBrightnessPreData:
    """Manager for raw pre-computed sky brightness data

    Parameters
    ----------
    base_fname : `str`
        Base name for data files to load.
    bands: `List` [`str`]
        Name of bands to read.
    pre_data_dir : `str`
        Name of source directory for pre-computed sky brightness data.
    max_num_mjds : `int`
        If there are more than this number of MJDs in the requested
        data files, sample this many out of the total.
    """

    def __init__(self, fname_base, bands, pre_data_dir=None, max_num_mjds=None):
        if pre_data_dir is None:
            try:
                self.pre_data_dir = os.environ["SIMS_SKYBRIGHTNESS_DATA"]
            except KeyError:
                self.pre_data_dir = "."
        else:
            self.pre_data_dir = pre_data_dir

        self.fname_base = fname_base
        self.max_num_mjds = max_num_mjds
        self.times = None
        self.sky = None
        self.metadata = {}
        self.load(fname_base, bands)

    def load(self, fname_base, bands="ugrizy"):
        """Load pre-computed sky values.

        Parameters
        ----------
        base_fname : `str`
            Base name for data files to load.
        bands: `List` [`str`]
            Name of bands to read.
        """

        npz_fname = os.path.join(self.pre_data_dir, fname_base + "." + "npz")
        npy_fname = os.path.join(self.pre_data_dir, fname_base + "." + "npy")
        npz = np.load(npz_fname, allow_pickle=True)
        npz_hdr = npz["header"][()]
        npz_data = npz["dict_of_lists"][()]
        pre_sky = np.load(npy_fname, allow_pickle=True)

        alt = npz_hdr["alt"]
        az = npz_hdr["az"]
        alt_rad, az_rad = np.radians(alt), np.radians(az)

        self.metadata = npz_hdr

        self.times = pd.DataFrame(
            {k: npz_data[k] for k in npz_data.keys() if npz_data[k].shape == npz_data["mjds"].shape}
        )

        read_mjds = len(self.times)
        if self.max_num_mjds is not None:
            read_mjd_idxs = pd.Series(np.arange(read_mjds))
            mjd_idxs = read_mjd_idxs.sample(self.max_num_mjds)
        else:
            mjd_idxs = np.arange(read_mjds)

        skies = []
        for mjd_idx in mjd_idxs:
            mjd = npz_data["mjds"][mjd_idx]
            gmst_rad = palpy.gmst(mjd)
            lst_rad = gmst_rad + TELESCOPE.longitude_rad
            ha_rad, decl_rad = palpy.dh2eVector(az_rad, alt_rad, TELESCOPE.latitude_rad)
            ra_rad = (lst_rad - ha_rad) % (2 * np.pi)
            moon_ra_rad = npz_data["moonRAs"][mjd_idx]
            moon_decl_rad = npz_data["moonDecs"][mjd_idx]
            moon_ha_rad = lst_rad - moon_ra_rad
            moon_az_rad, moon_el_rad = palpy.de2h(moon_ha_rad, moon_decl_rad, TELESCOPE.latitude_rad)
            moon_sep = palpy.dsepVector(
                np.full_like(az_rad, moon_az_rad),
                np.full_like(alt_rad, moon_el_rad),
                az_rad,
                alt_rad,
            )
            for band in bands:
                skies.append(
                    pd.DataFrame(
                        {
                            "band": band,
                            "mjd": npz_data["mjds"][mjd_idx],
                            "gmst": np.degrees(gmst_rad),
                            "lst": np.degrees(lst_rad),
                            "alt": alt,
                            "az": az,
                            "ra": np.degrees(ra_rad),
                            "decl": np.degrees(decl_rad),
                            "moon_ra": np.degrees(npz_data["moonRAs"][mjd_idx]),
                            "moon_decl": np.degrees(npz_data["moonDecs"][mjd_idx]),
                            "moon_alt": np.degrees(npz_data["moonAlts"][mjd_idx]),
                            "moon_az": np.degrees(moon_az_rad),
                            "moon_sep": np.degrees(moon_sep),
                            "sun_ra": np.degrees(npz_data["sunRAs"][mjd_idx]),
                            "sun_decl": np.degrees(npz_data["sunDecs"][mjd_idx]),
                            "sun_alt": np.degrees(npz_data["sunAlts"][mjd_idx]),
                            "sky": pre_sky[band][mjd_idx],
                        }
                    )
                )

        self.sky = pd.concat(skies).set_index(["band", "mjd", "alt", "az"], drop=False)
        self.sky.sort_index(inplace=True)

        if self.max_num_mjds is not None:
            self.times = self.times.iloc[mjd_idxs]

    def __getattr__(self, name):
        return self.metadata[name]


class SkyModelZernike:
    """Interface to zernike sky that is more similar to SkyModelPre

    Parameters
    ----------
    data_file : `str`, optional
        File name from which to load Zernike coefficients. Default None uses default data directory.

    """

    def __init__(self, data_file=None, **kwargs):
        if data_file is None:
            if "SIMS_SKYBRIGHTNESS_DATA" in os.environ:
                data_dir = os.environ["SIMS_SKYBRIGHTNESS_DATA"]
            else:
                data_dir = os.path.join(get_data_dir(), "sims_skybrightness_pre")

            data_file = os.path.join(data_dir, "zernike", "zernike.h5")

            zernike_metadata = pd.read_hdf(data_file, "zernike_metadata")

            order = int(zernike_metadata["order"])
            if "order" in kwargs:
                assert order == kwargs["order"]
            else:
                kwargs["order"] = order

            max_zd = zernike_metadata["max_zd"]
            if "max_zd" in kwargs:
                assert max_zd == kwargs["max_zd"]
            else:
                kwargs["max_zd"] = max_zd

        self.zernike_model = {}
        for band in BANDS:
            sky = ZernikeSky(**kwargs)
            sky.load_coeffs(data_file, band)
            self.zernike_model[band] = sky
        self.nside = sky.nside

    def return_mags(
        self,
        mjd,
        indx=None,
        badval=healpy.UNSEEN,
        filters=["u", "g", "r", "i", "z", "y"],
        extrapolate=False,
    ):
        """
        Return a full sky map or individual pixels for the input mjd

        Parameters
        ----------
        mjd : float
            Modified Julian Date to interpolate to
        indx : List of int(s) (None)
            indices to interpolate the sky values at. Returns full sky if None. If the class was
            instatiated with opsimFields, indx is the field ID, otherwise it is the healpix ID.
        badval : float (-1.6375e30)
            Mask value. Defaults to the healpy mask value.
        filters : list
            List of strings for the filters that should be returned.
        extrapolate : bool (False)
            In indx is set, extrapolate any masked pixels to be the same as the nearest non-masked
            value from the full sky map.

        Returns
        -------
        sbs : dict
            A dictionary with filter names as keys and np.arrays as values which
            hold the sky brightness maps in mag/sq arcsec.
        """
        sky_brightness = {}

        sun_el = _calc_sun_el(mjd)
        if sun_el > 0:
            warnings.warn("Requested MJD between sunrise and sunset")
            if indx is None:
                nside = self.zernike_model[filters[0]].nside
                npix = healpy.nside2npix(nside)
            else:
                npix = len(indx)

            for band in filters:
                sky_brightness[band] = np.full(npix, badval)

            return sky_brightness

        if extrapolate:
            raise NotImplementedError

        for band in filters:
            band_brightness = self.zernike_model[band].compute_healpix(indx, mjd)
            badval_idxs = np.where(~np.isfinite(band_brightness))
            band_brightness[badval_idxs] = badval
            sky_brightness[band] = band_brightness

        return sky_brightness


def cut_pre_dataset(
    fname_base="59823_60191",
    num_mjd=3,
    pre_dir="/data/des91.b/data/neilsen/LSST/horizon_skybrightness_pre",
    cut_dir=".",
):
    """Cut a per-computed dataset to specified number of MJDs

    The purpose of this command is to create small input datafiles
    that can be used for testing

    Parameters
    ----------
    fname_base : `str`
        The base from which to construct file names
    num_mjd : `int`
        The number of MJDs to include in the cut file.
    pre_dir : `str`
        The directory from which to read data files
    cut_dur : `str`
        The directory into which to write cut data files
    """
    npy_fname = os.path.join(pre_dir, fname_base + ".npy")
    npz_fname = os.path.join(pre_dir, fname_base + ".npz")

    npz = np.load(npz_fname, allow_pickle=True)
    npz_hdr = npz["header"][()]
    npz_data = npz["dict_of_lists"][()]
    pre_sky = np.load(npy_fname, allow_pickle=True)

    kept_mjds = np.sort(npz_hdr["required_mjds"])[:num_mjd].copy()
    kept_mjd_idxs = np.where(np.in1d(npz_data["mjds"], kept_mjds))

    npz_hdr["required_mjds"] = kept_mjds
    del npz_hdr["version"]
    del npz_hdr["fingerprint"]

    for data_key in npz_data.keys():
        npz_data[data_key] = npz_data[data_key][kept_mjd_idxs].copy()

    pre_sky = pre_sky[kept_mjd_idxs].copy()

    min_mjd = int(np.floor(kept_mjds.min()))
    max_mjd = int(np.floor(kept_mjds.max()))
    out_fname_base = f"{min_mjd}_{max_mjd}"

    cut_npz_fname = os.path.join(cut_dir, out_fname_base + ".npz")
    np.savez(cut_npz_fname, header=npz_hdr, dict_of_lists=npz_data)

    cut_npy_fname = os.path.join(cut_dir, out_fname_base + ".npy")
    np.save(cut_npy_fname, pre_sky)


# internal functions & classes


@lru_cache()
def _calc_moon_az_rad(mjd):
    ra_rad, decl_rad, diam = palpy.rdplan(mjd, 3, TELESCOPE.longitude_rad, TELESCOPE.latitude_rad)
    ha_rad = palpy.gmst(mjd) + TELESCOPE.longitude_rad - ra_rad
    az_rad, el_rad = palpy.de2h(ha_rad, decl_rad, TELESCOPE.latitude_rad)
    return az_rad


@lru_cache()
def _calc_sun_el(mjd):
    ra_rad, decl_rad, diam = palpy.rdplan(mjd, 0, TELESCOPE.longitude_rad, TELESCOPE.latitude_rad)
    ha_rad = palpy.gmst(mjd) + TELESCOPE.longitude_rad - ra_rad
    az_rad, el_rad = palpy.de2h(ha_rad, decl_rad, TELESCOPE.latitude_rad)
    el = np.degrees(el_rad)
    return el
