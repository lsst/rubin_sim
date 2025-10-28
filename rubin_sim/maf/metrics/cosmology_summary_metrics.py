__all__ = (
    "StaticProbesFoMEmulatorMetricSimple",
    "TotalPowerMetric",
    "TomographicClusteringSigma8biasMetric",
    "MultibandMeanzBiasMetric",
    "get_stats_by_region",
    "stripiness_test_statistic",
    "StripinessMetric",
    "UniformAreaFoMFractionMetric",
    "UDropoutsNumbersShallowestDepth",
    "UDropoutsArea",
)

import warnings
from copy import deepcopy

import healpy as hp
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from scipy import interpolate
from scipy.stats import median_abs_deviation
from sklearn.cluster import KMeans

from ..maf_contrib.static_probes_fom_summary_metric import StaticProbesFoMEmulatorMetric
from .area_summary_metrics import AreaThresholdMetric
from .base_metric import BaseMetric
from .simple_metrics import PercentileMetric

# Cosmology-related summary metrics.
# These generally calculate a FoM for various DESC metrics.


class TotalPowerMetric(BaseMetric):
    """Calculate the total power in the angular power spectrum,
    between lmin/lmax.

    Parameters
    ----------
    lmin : `float`, optional
        Minimum ell value to include when calculating total power.
    lmax : `float`, optional
        Maximum ell value to include when calculating total power.
    remove_monopole : `bool`, optional
        Flag to remove monopole when calculating total power.
    remove_dipole : `bool`, optional
        Flag  to remove dipole when calculating total power.
    col : `str`, optional
        The column name to operate on.
        For summary metrics, this is almost always `metricdata`.
    mask_val : `float` or np.nan, optional
        The mask value to apply to the metric values when passed.
        If this attribute exists, the metric_values will be passed
        using metric_values.filled(mask_val).
        If mask_val is `None` for a metric, metric_values will be passed
        using metric_values.compressed().
    """

    def __init__(
        self,
        col="metricdata",
        lmin=100.0,
        lmax=300.0,
        remove_monopole=True,
        remove_dipole=True,
        mask_val=np.nan,
        **kwargs,
    ):
        self.lmin = lmin
        self.lmax = lmax
        self.remove_monopole = remove_monopole
        self.remove_dipole = remove_dipole
        super(TotalPowerMetric, self).__init__(col=col, mask_val=mask_val, **kwargs)

    def run(self, data_slice, slice_point=None):
        # Calculate the power spectrum.
        data = data_slice[self.colname]
        if self.remove_monopole:
            data = hp.remove_monopole(data, verbose=False, bad=self.mask_val)
        if self.remove_dipole:
            cl = hp.anafast(hp.remove_dipole(data_slice[self.colname]))
        else:
            cl = hp.anafast(data_slice[self.colname])
        ell = np.arange(np.size(cl))
        condition = np.where((ell <= self.lmax) & (ell >= self.lmin))[0]
        totalpower = np.sum(cl[condition] * (2 * ell[condition] + 1))
        return totalpower


class StaticProbesFoMEmulatorMetricSimple(BaseMetric):
    """This calculates the Figure of Merit for the combined
    static probes (3x2pt, i.e., Weak Lensing, LSS, Clustering).
    This FoM is purely statistical and does not factor in systematics.

    Parameters
    ----------
    year : `int`, optional
        The year of the survey to calculate FoM.
        This calibrates expected depth and area.

    Returns
    -------
    result : `float`
        The simple 3x2pt FoM emulator value, for the
        years where the correlation between area/depth and value is defined.

    Notes
    -----
    This FoM is purely statistical and does not factor in systematics.
    The implementation here is simpler than in
    `rubin_sim.maf.mafContrib.StaticProbesFoMEmulatorMetric`, and that
    more sophisticated version should replace this metric.

    This version of the emulator was used to generate the results in
    https://ui.adsabs.harvard.edu/abs/2018arXiv181200515L/abstract

    Note that this is truly a summary metric and should be run on the
    output of Exgalm5_with_cuts.
    """

    def __init__(self, year=10, col=None, **kwargs):

        super().__init__(col=col, **kwargs)
        if col is None:
            self.col = "metricdata"
        self.year = year
        # Set a mask_val so that all metric data is passed
        # for summary calculation, even masked values.
        self.mask_val = -1

    def run(self, data_slice, slice_point=None):
        """
        Args:
            data_slice (ndarray): Values passed to metric by the slicer,
                which the metric will use to calculate metric values
                at each slice_point.
            slice_point (Dict): Dictionary of slice_point metadata passed
                to each metric.
        Returns:
             float: Interpolated static-probe statistical Figure-of-Merit.
        Raises:
             ValueError: If year is not one of the 4 for which a FoM is
             calculated
        """
        nside = hp.npix2nside(len(data_slice))
        # Chop off any outliers
        good_pix = np.where(data_slice[self.col] > 0)[0]
        if np.size(good_pix) == 0:
            return self.badval

        # Calculate area and med depth from
        area = hp.nside2pixarea(nside, degrees=True) * np.size(good_pix)
        median_depth = np.median(data_slice[self.col][good_pix])
        # FoM is calculated at the following values
        if self.year == 1:
            areas = [7500, 13000, 16000]
            depths = [24.9, 25.2, 25.5]
            fom_arr = [
                [1.212257e02, 1.462689e02, 1.744913e02],
                [1.930906e02, 2.365094e02, 2.849131e02],
                [2.316956e02, 2.851547e02, 3.445717e02],
            ]
        elif self.year == 3:
            areas = [10000, 15000, 20000]
            depths = [25.5, 25.8, 26.1]
            fom_arr = [
                [1.710645e02, 2.246047e02, 2.431472e02],
                [2.445209e02, 3.250737e02, 3.516395e02],
                [3.173144e02, 4.249317e02, 4.595133e02],
            ]

        elif self.year == 6:
            areas = [10000, 15000, 2000]
            depths = [25.9, 26.1, 26.3]
            fom_arr = [
                [2.346060e02, 2.414678e02, 2.852043e02],
                [3.402318e02, 3.493120e02, 4.148814e02],
                [4.452766e02, 4.565497e02, 5.436992e02],
            ]

        elif self.year == 10:
            areas = [10000, 15000, 20000]
            depths = [26.3, 26.5, 26.7]
            fom_arr = [
                [2.887266e02, 2.953230e02, 3.361616e02],
                [4.200093e02, 4.292111e02, 4.905306e02],
                [5.504419e02, 5.624697e02, 6.441837e02],
            ]
        else:
            warnings.warn("FoMEmulator is not defined for this year")
            return self.badval

        # Interpolate FoM to the actual values for this sim
        areas = [[i] * 3 for i in areas]
        depths = [depths] * 3
        f = interpolate.RectBivateateSpline(np.ravel(areas), np.ravel(depths), np.ravel(fom_arr))
        fom = f(area, median_depth)[0]
        return fom


class TomographicClusteringSigma8biasMetric(BaseMetric):
    """Compute bias on sigma8 due to spurious contamination of density maps.
    Run as summary metric on NestedLinearMultibandModelMetric.

    Points of contact / contributors: Boris Leistedt

    Parameters
    ----------
    density_tomograph_model : `dict`
        dictionary containing models calculated for fiducial N(z)s and Cells:
        lmax : numpy.array of int, of shape (Nbins, )
            lmax corresponding to kmax of 0.05
        poly1d_coefs_loglog : numpy.array of float, of shape (Nbins, )
            polynomial fits to log(C_ell) vs log(ell) computed for CCL
        sigma8square_model (float)
            value of sigma8^2 used as fiducal model for CCL
    power_multiplier : `float`, optional
        fraction of power (variance) which is uncorrected
        and thus biases sigma8
    lmin : `int`, optional
        lmin for the analysis
    convert_to_sigma8 : `str`, optional
        Convert the bias to sigma8 instead of sigma8^2
        (via change of variables for the uncertainty)

    Returns
    -------
    result : `float`
        Value of sigma8 bias calculated from this model:
        (sigma8^2_obs - sigma^2_model) / error on sigma8^2_obs
        if `convert_to_sigma8` is True,
        then it is about sigma8 instead of sigma8^2.

    Notes
    -----
    This is a summary metric to be run on the results
    of the NestedLinearMultibandModelMetric.

    NestedLinearMultibandModelMetric converts 6-band depth maps into
    a set of maps (e.g tomographic redshift bins) which describe
    spurious density fluctuations in each bin.

    This summary metric multiplies the maps by the parameter power_multiplier,
    which can be used to describe the fraction of power uncorrected by
    systematics mitigation schemes, computes the total power
    (via angular power spectra with lmin-lmax limits)
    and then infers sigma8^2 via a model of the angular power spectra.
    By taking sigma8_obs minus sigma8_model divided by the uncertainty,
    one derives a bias.
    """

    def __init__(
        self,
        density_tomography_model,
        power_multiplier=0.1,
        lmin=10,
        convert_to_sigma8=True,
        badval=hp.UNSEEN,
        **kwargs,
    ):
        super().__init__(col="metricdata", **kwargs)
        # Set mask_val, so that we receive metric_values.filled(mask_val)
        self.mask_val = badval
        self.badval = badval

        self.convert_to_sigma8 = convert_to_sigma8

        self.power_multiplier = power_multiplier
        self.lmin = lmin
        self.density_tomography_model = density_tomography_model
        # to compute angular power spectra and total power,
        # initialize an array of metrics, with the right lmin and lmax.
        self.totalPowerMetrics = [
            TotalPowerMetric(lmin=lmin, lmax=lmax, mask_val=self.mask_val)
            for lmax in density_tomography_model["lmax"]
        ]
        self.areaThresholdMetric = AreaThresholdMetric(
            col="metricdata",
            metric_name="FootprintFraction_bin",
            lower_threshold=badval,
            upper_threshold=np.inf,
            mask_val=self.mask_val,
        )

    def run(self, data_slice, slice_point=None):
        # need to define an array of bad values for the masked pixels
        badval_arr = np.repeat(self.badval, len(self.density_tomography_model["lmax"]))
        # converts the input recarray to an array
        data_slice_list = [
            badval_arr if isinstance(x, float) else x for x in data_slice["metricdata"].tolist()
        ]
        # should be (nbins, npix)
        data_slice_arr = np.asarray(data_slice_list, dtype=float).T
        data_slice_arr[~np.isfinite(data_slice_arr)] = (
            self.badval
        )  # need to work with TotalPowerMetric and healpix

        # measure valid sky fractions and total power
        # (via angular power spectra) in each bin.
        # The original metric returns an array at each slice_point (of the
        # original slicer) -- so there is a bit of "rearrangement" that
        # has to happen to be able to pass a np.array with right dtype
        # (i.e. dtype = [("metricdata", float)]) to each call to
        # the AreaThresholdMetric and TotalPowerMetric `run` methods.
        totalsky = 42000
        fskys = np.array(
            [
                self.areaThresholdMetric.run(np.core.records.fromrecords(x, dtype=[("metricdata", float)]))
                / totalsky
                for x in data_slice_arr
            ]
        )  # sky fraction
        spuriousdensitypowers = (
            np.array(
                [
                    self.totalPowerMetrics[i].run(
                        np.core.records.fromrecords(x, dtype=[("metricdata", float)])
                    )
                    for i, x in enumerate(data_slice_arr)
                ]
            )
            / fskys
        )
        # some gymnastics needed to convert each slice into a recarray.
        # this could probably be avoided if recarrays were returned
        # by the original nested/vector metric,
        # except that we would need to manipulate the names in various
        # places, which I wanted to avoid,
        # so for now the main metric returns an array per healpix pixel
        # (not a recarray) and puts together
        # healpix maps which we need to convert to a recarray to pass to
        # AreaThresholdMetric and TotalPowerMetric

        def solve_for_multiplicative_factor(spurious_powers, model_cells, fskys, lmin, power_multiplier):
            """
            Infer multiplicative factor sigma8^2 (and uncertainty)
            from the model Cells and observed total powers
            since it os a Gaussian posterior distribution.
            """
            # solve for multiplicative sigma8^2 term between
            # measured angular power spectra
            # (spurious measured Cells times power_multiplier)
            # and model ones (polynomial model from CCL).
            n_bins = model_cells["lmax"].size
            assert len(spurious_powers) == n_bins
            assert len(fskys) == n_bins
            assert model_cells["poly1d_coefs_loglog"].shape[0] == n_bins
            totalvar_mod = np.zeros((n_bins, 1))
            totalvar_obs = np.zeros((n_bins, 1))
            totalvar_var = np.zeros((n_bins, 1))
            # loop over tomographic bins
            # hardcoded; assumed CCL cosmology
            sigma8square_model = model_cells["sigma8square_model"]
            for i in range(n_bins):
                # get model Cells from polynomial model (in log log space)
                ells = np.arange(lmin, model_cells["lmax"][i])
                polynomial_model = np.poly1d(model_cells["poly1d_coefs_loglog"][i, :])
                cells_model = np.exp(polynomial_model(np.log(ells)))

                # model variance is sum of cells x (2l+1)
                totalvar_mod[i, 0] = np.sum(cells_model * (2 * ells + 1))

                # observations is spurious power  noiseless model
                totalvar_obs[i, 0] = totalvar_mod[i, 0] + spurious_powers[i] * power_multiplier

                # simple model variance of cell based on Gaussian covariance
                cells_var = 2 * cells_model**2 / (2 * ells + 1) / fskys[i]
                totalvar_var[i, 0] = np.sum(cells_var * (2 * ells + 1) ** 2)

            # model assumed sigma8 = 0.8
            # (add CCL cosmology here? or how I obtained them + documentation)

            # model Cell variance divided by sigma8^2,
            # which is the common normalization
            transfers = totalvar_mod / sigma8square_model

            # model ratio: formula for posterior distribution on unknown
            # multiplicative factor in multivariate Gaussian likelihood
            # Gaussian likelihood
            FOT = np.sum(transfers[:, 0] * totalvar_obs[:, 0] / totalvar_var[:, 0])
            FTT = np.sum(transfers[:, 0] * transfers[:, 0] / totalvar_var[:, 0])
            # mean and stddev of multiplicative factor
            sigma8square_fit = FOT / FTT
            sigma8square_error = FTT**-0.5

            return sigma8square_fit, sigma8square_error, sigma8square_model

        # solve for the gaussian posterior distribution on sigma8^2
        sigma8square_fit, sigma8square_error, sigma8square_model = solve_for_multiplicative_factor(
            spuriousdensitypowers, self.density_tomography_model, fskys, self.lmin, self.power_multiplier
        )

        results_sigma8_square_bias = (sigma8square_fit - sigma8square_model) / sigma8square_error

        if not self.convert_to_sigma8:
            return results_sigma8_square_bias

        else:

            # turn result into bias on sigma8,
            # via change of variable and simple propagation of uncertainty.
            sigma8_fit = sigma8square_fit**0.5
            sigma8_model = sigma8square_model**0.5
            sigma8_error = 0.5 * sigma8square_error * sigma8_fit / sigma8square_fit
            results_sigma8_bias = (sigma8_fit - sigma8_model) / sigma8_error
            return results_sigma8_bias


class MultibandMeanzBiasMetric(BaseMetric):
    """
    Run as summary metric on MultibandExgalM5.

    Parameters
    ----------
    year : `int`, optional
        The year of the survey to calculate the bias.
        This is used to derive the dm/dz derivative used to
        translate m5 rms into dz rms.

    meanz_tomograph_model : `dict`
        dictionary containing models calculated for fiducial N(z):

        meanz: numpy.float
            the meanz within a tomographic bin at a given band
        dz_dm5: numpy.float
            the absolute value of the derivative of z in a bin as a function
            of the depth, m5 magnitude. This is later used to
            translate fluctuations
            in depth to fluctuations in tomographic redshift

    Returns
    -------
    result : `float` array
        The ratio of this bias to the desired DESC y1 upper
        bound on the bias, and the ratio
        between the clbias and the y10 DESC SRD requirement.
        Desired values are less than 1 by Y10.

    Notes
    -----
    This is a summary metric to be run on the results
    of the MultibandExgalM5.

    MultibandExgalM5 provides the m5 depth in all LSST bands
    given a specific slice.

    This summary metric takes those depths and reads the derivatives
    from the tomogrpahic model,
    computes the bias in shear signal Cl and then computes the ratio of
    that bias to the Y1 goal and
    Y10 science requirement.
    """

    def __init__(
        self,
        meanz_tomography_model,
        filter_list=["u", "g", "r", "i", "z", "y"],
        year=10,
        zbins=4,
        **kwargs,
    ):

        super().__init__(col="metricdata", **kwargs)
        # Set mask_val, so that we receive metric_values.filled(mask_val)
        self.mask_val = np.nan
        self.badval = np.nan
        self.rms_func = np.nanstd
        self.year = year
        self.filter_list = filter_list
        self.zbins = zbins
        self.tomo_model = meanz_tomography_model

    def run(self, data_slice, slice_point=None):

        def compute_dzfromdm(zbins, band_name, year):
            """This computes the dm/dz relationship calibrated from simulations
            by Jeff Newmann and Qianjun Hang, which forms the
            meanz_tomographic_model.

            Parameters
            ----------
            zbins : `int`
                The number of tomographic bins considered. For now
                this is zbins < 5
            band_name : `str`
                The assumed filter band
            model_z: dict
                The meanz_tomographic_model assumed in this work

            Returns
            -------
            dzdminterp : `float` array
                The interpolated value of the derivative dz/dm
            meanzinterp : `float` array
                The meanz in each tomographic bin.

            """
            meanzinterp = np.zeros(zbins)
            dzdminterp = np.zeros(zbins)

            for z in range(zbins):
                meanzinterp[z] = self.tomo_model["year%i" % (year + 1)]["meanz"][z][band_name]
                dzdminterp[z] = self.tomo_model["year%i" % (year + 1)]["dz_dm5"][z][band_name]

            return dzdminterp, meanzinterp

        # Not entirely sure if we should include these
        # here or in a separate module/file
        def use_zbins(meanz_vals, figure_9_mean_z=np.array([0.2, 0.4, 0.7, 1.0]), figure_9_width=0.2):
            """This computes which redshift bands are within the range
            specified in https://arxiv.org/pdf/2305.15406.pdf and
            can safely be used
            to compute what Cl bias result from z fluctuations
            caused by rms variations in the m5.


            Parameters
            ----------
            meanz_vals : `float` array
                Array of meanz values to be used.

            Returns
            -------
            use_bins : `boolean` array
                An array of boolean values of length meanz_vals

            """
            max_z_use = np.max(figure_9_mean_z) + 2 * figure_9_width
            use_bins = meanz_vals < max_z_use

            return use_bins

        def compute_Clbias(meanz_vals, scatter_mean_z_values):
            """This computes the Cl bias
            that results z fluctuations caused by rms variations in the m5.



            Parameters
            ----------
            meanz_vals : `float` array
                Array of meanz values to be used.

            scatter_mean_z_values : `float` array
                Array of rms values of the z fluctuations


            Returns
            -------
            clbiasvals : `float` array
                An array of values of the clbias

            mean_z_values_use :  `float` array
                An array of the meanz values that are within the
                interpolation range of 2305.15406

            Notes
            ------
            This interpolates from the Figure 9 in
            https://arxiv.org/pdf/2305.15406.pdf

            """

            figure_9_mean_z = np.array([0.2, 0.4, 0.7, 1.0])
            figure_9_Clbias = np.array([1e-3, 2e-3, 5e-3, 1.1e-2])
            figure_9_width = 0.2
            figure_9_mean_z_scatter = 0.02

            mzvals = np.array([float(mz) for mz in meanz_vals])
            sctz = np.array([float(sz) for sz in scatter_mean_z_values])

            fit_res = np.polyfit(figure_9_mean_z, figure_9_Clbias, 2)
            poly_fit = np.poly1d(fit_res)
            use_bins = use_zbins(meanz_vals, figure_9_mean_z, figure_9_width)

            mean_z_values_use = mzvals[use_bins]
            sctz_use = sctz[use_bins]

            Clbias = poly_fit(mean_z_values_use)
            rescale_fac = sctz_use / figure_9_mean_z_scatter
            Clbias *= rescale_fac
            fit_res_bias = np.polyfit(mean_z_values_use, Clbias, 1)
            poly_fit_bias = np.poly1d(fit_res_bias)

            clbiasvals = poly_fit_bias(mean_z_values_use)
            return clbiasvals, mean_z_values_use

        result = np.empty(self.zbins, dtype=[("name", np.str_, 20), ("y1ratio", float), ("y10ratio", float)])
        result["name"][0] = "MultibandMeanzBiasMetric"

        # Technically don't need this for now (isn't used in previous one)
        # need to define an array of bad values for the masked pixels
        badval_arr = np.repeat(self.badval, len(self.filter_list))
        # converts the input recarray to an array
        data_slice_list = [
            badval_arr if isinstance(x, float) else x for x in data_slice["metricdata"].tolist()
        ]
        # should be (nbins, npix)
        data_slice_arr = np.asarray(data_slice_list, dtype=float).T
        data_slice_arr[~np.isfinite(data_slice_arr)] = np.nan

        # measure rms in each bin.

        # XXX--this is probably meant to be run on
        # only valid values, so changing to nanstd
        rmsarray = self.rms_func(data_slice_arr, axis=1)

        # totdz = 0
        avmeanz = 0
        totclbias = 0

        for i, filt in enumerate(self.filter_list):
            dzdminterp, meanzinterp = compute_dzfromdm(self.zbins, filt, self.year)
            stdz = [float(np.abs(dz)) * float(rmsarray[i]) for dz in dzdminterp]

            clbias, meanz_use = compute_Clbias(meanzinterp, stdz)

            # totdz += [float(st**2) for st in stdz]
            totclbias += clbias
            avmeanz += meanzinterp

        # These should be included in self rather than hard coded
        y10_req = 0.003
        y1_goal = 0.013

        # clbiastot = np.max(clbias) # if adding doesn't
        # work over z range -- CHECK
        y10ratio = totclbias / y10_req
        y1ratio = totclbias / y1_goal

        result["y1ratio"] = y1ratio
        result["y10ratio"] = y10ratio
        return result


# Let's make code that pulls out the northern/southern galactic regions,
# and gets statistics of the footprint by region.
def _is_ngp(ra, dec):
    c = SkyCoord(ra=ra * u.degree, dec=dec * u.degree, frame="icrs")
    lat = c.galactic.b.deg
    return lat >= 0


def get_stats_by_region(use_map, nside, maskval=0, region="all"):
    if region not in ["all", "north", "south"]:
        raise ValueError("Invalid region %s" % region)
    to_use = use_map > maskval

    if region != "all":
        # Find the north/south part of the map as requested
        npix = hp.nside2npix(nside)
        ra, dec = hp.pix2ang(hp.npix2nside(npix), range(npix), lonlat=True)
        ngp_mask = _is_ngp(ra, dec)
        if region == "north":
            reg_mask = ngp_mask
        else:
            reg_mask = ~ngp_mask
        to_use = to_use & reg_mask

    # Calculate the desired stats
    reg_mad = median_abs_deviation(use_map[to_use])
    reg_median = np.median(use_map[to_use])
    reg_std = np.std(use_map[to_use])

    # Return the values
    return (reg_mad, reg_median, reg_std)


def stripiness_test_statistic(data_slice, nside):
    """
    A utility to find whether a particular routine has stripey
    features in the exposure time map.
    """
    # Analyze the exposure time map to get MAD, median, std in north/south.
    mad = {}
    med = {}
    frac_scatter = {}
    regions = ["north", "south"]
    for region in regions:
        mad[region], med[region], _ = get_stats_by_region(data_slice, nside, region=region)
        frac_scatter[region] = mad[region] / med[region]
    test_statistic = frac_scatter["north"] / frac_scatter["south"] - 1
    return test_statistic


class StripinessMetric(BaseMetric):
    """
    Run as summary metric on NestedRIZExptimeExgalM5Metric.

    This metric uses maps of the combined RIZ exposure time
    to identify stripes or residual rolling features,
    for the UniformAreaFoMFractionMetric.
    The number returned quantifies the stripiness of the field.
    In UniformAreaFoMFractionMetric it is a test statistic:
    if it is outside of +-0.7/np.sqrt(year) then
    the RIZ exposure time map is considered to have stripes.

    Points of contact / contributors: Rachel Mandelbaum, Boris Leistedt

    Parameters
    ----------
    year: `int`
        year of observation, in order to adjust the cut on
        the depth fluctuations for the stipe detection
    nside: `int`
        must be the nside at which the base metric
        NestedRIZExptimeExgalM5Metric is calculated at
    verbose: bool, optional
        if true, will display the segmentation maps and the areas and
        FOMs of the two regions found
    Returns
    -------
    result: `float`
        A number quantifying the stripiness of the RIZ exposure time.
    """

    def __init__(
        self,
        year,
        nside,
        verbose=True,
        **kwargs,
    ):
        self.year = year
        self.mask_val = hp.UNSEEN
        self.verbose = verbose
        self.nside = nside
        names = ["exgal_m5", "riz_exptime"]
        types = [float] * 2
        self.mask_val_arr = np.zeros(1, dtype=list(zip(names, types)))
        self.mask_val_arr["exgal_m5"] = self.mask_val
        self.mask_val_arr["riz_exptime"] = self.mask_val

        super().__init__(col="metricdata", **kwargs)

    def run(self, data_slice, slice_point=None):
        data_slice_list = [
            self.mask_val_arr if isinstance(x, float) else x for x in data_slice["metricdata"].tolist()
        ]
        data_slice_arr = np.asarray(data_slice_list, dtype=self.mask_val_arr.dtype)
        # a bit of gymnastics to make sure all bad values
        # (nan, -666) are recast as hp.UNSEEN
        ind = data_slice_arr["riz_exptime"] == -666
        ind |= ~np.isfinite(data_slice_arr["riz_exptime"])
        data_slice_arr["riz_exptime"][ind.ravel()] = hp.UNSEEN
        # sanity check
        nside = hp.npix2nside(data_slice["metricdata"].size)
        assert nside == self.nside

        test_statistic = stripiness_test_statistic(data_slice_arr["riz_exptime"].ravel(), nside)

        return test_statistic


class UniformAreaFoMFractionMetric(BaseMetric):
    """
    Run as summary metric on NestedRIZExptimeExgalM5Metric.

    This metric uses maps of the combined RIZ exposure time and i-band depth
    maps (with a consistent set of area cuts)
    to identify potential reductions in cosmological constraining power due to
    substantial large-scale power
    in non-uniform coadds at a particular data release.
    The RIZ exposure time map is used to identify whether there
    are residual rolling features.
    Under the hood, this runs the StripinessMetric.
    If not, the metric returns 1. If there are such features, then the region
    is segmented into similar-depth regions
    and the one with the largest cosmological constraining power is presumed
    to be used for science.
    In that case, the metric returns the 3x2pt FoM
    (StaticProbesFoMEmulatorMetric,
    quantifying weak lensing and large-scale structure constraining power)
    for the largest of those regions,
    divided by that for the full region if it had been usable.



    Points of contact / contributors: Rachel Mandelbaum, Boris Leistedt

    Parameters
    ----------
    year: `int`
        year of observation, in order to adjust the
        cut on the depth fluctuations
        for the stipe detection
    nside: `int`
        must be the nside at which the base metric
        NestedRIZExptimeExgalM5Metric
        is calculated at
    verbose: bool, optional
        if true, will display the segmentation maps and the areas and FOMs of
        the two regions found
    Returns
    -------
    result: `float`
        The ratio of the FOMs of the two areas found
    """

    def __init__(
        self,
        year,
        nside,
        verbose=True,
        **kwargs,
    ):
        self.year = year
        self.mask_val = hp.UNSEEN
        self.verbose = verbose
        self.nside = nside
        names = ["exgal_m5", "riz_exptime"]
        types = [float] * 2
        self.mask_val_arr = np.zeros(1, dtype=list(zip(names, types)))
        self.mask_val_arr["exgal_m5"] = self.mask_val
        self.mask_val_arr["riz_exptime"] = self.mask_val

        self.threebyTwoSummary = StaticProbesFoMEmulatorMetric(
            nside=nside, metric_name="3x2ptFoM", col="exgal_m5"
        )
        super().__init__(col="metricdata", **kwargs)

    def run(self, data_slice, slice_point=None):
        data_slice_list = [
            self.mask_val_arr if isinstance(x, float) else x for x in data_slice["metricdata"].tolist()
        ]
        data_slice_arr = np.asarray(data_slice_list, dtype=self.mask_val_arr.dtype)
        # a bit of gymnastics to make sure all bad values
        # (nan, -666) are recast as hp.UNSEEN
        ind = data_slice_arr["riz_exptime"] == -666
        ind |= ~np.isfinite(data_slice_arr["riz_exptime"])
        ind = data_slice_arr["exgal_m5"] == -666
        ind |= ~np.isfinite(data_slice_arr["exgal_m5"])
        data_slice_arr["exgal_m5"][ind.ravel()] = hp.UNSEEN
        data_slice_arr["riz_exptime"][ind.ravel()] = hp.UNSEEN
        # sanity check
        nside = hp.npix2nside(data_slice["metricdata"].size)
        assert nside == self.nside

        # Check for stripiness
        threshold = 0.7 / np.sqrt(self.year)
        test_statistic = stripiness_test_statistic(data_slice_arr["riz_exptime"].ravel(), nside)

        if np.abs(test_statistic) < np.abs(threshold):
            stripes = False
        else:
            stripes = True

        def apply_clustering(clustering_data):
            # A thin wrapper around sklearn routines
            # (can swap out which one we are using systematically).
            # We fix parameters like `n_clusters` since realistically
            # we know for rolling that we should expect 2 clusters.
            # from sklearn.cluster import SpectralClustering
            # clustering = SpectralClustering(n_clusters=2,
            # assign_labels='discretize', random_state=0).fit(clustering_data)

            clustering = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(clustering_data)
            labels = clustering.labels_ + 1
            return labels

        def expand_labels(depth_map, labels, maskval=0):
            # A utility to apply the labels from a masked
            # version of a depth map back to the entire depth map.
            expanded_labels = np.zeros(hp.nside2npix(nside))
            cutval = maskval + 0.1
            expanded_labels[depth_map > cutval] = labels
            return expanded_labels

        def get_area_stats(depth_map, labels, maskval=0, n_clusters=2):
            # A routine to get some statistics of the
            # clustering: area fractions, median map values
            expanded_labels = expand_labels(depth_map, labels, maskval=maskval)
            cutval = maskval + 0.1
            area_frac = []
            med_val = []
            for i in range(n_clusters):
                new_map = depth_map.copy()
                new_map[expanded_labels != i + 1] = maskval
                this_area_frac = len(new_map[new_map > cutval]) / len(depth_map[depth_map > cutval])
                this_med_val = np.median(new_map[new_map > cutval])
                area_frac.append(this_area_frac)
                med_val.append(this_med_val)
            return area_frac, med_val

        def show_clusters(depth_map, labels, maskval=0, n_clusters=2, min=500, max=3000):
            # A routine to show the clusters found by the unsupervised
            # clustering algorithm (start with original map then 2 clusters).
            expanded_labels = expand_labels(depth_map, labels, maskval=maskval)
            hp.visufunc.mollview(depth_map, min=min, max=max)
            for i in range(n_clusters):
                new_map = depth_map.copy()
                new_map[expanded_labels != i + 1] = maskval
                hp.visufunc.mollview(new_map, min=min, max=max)
            return get_area_stats(depth_map, labels, maskval=maskval, n_clusters=n_clusters)

        def make_clustering_dataset(depth_map, maskval=0, priority_fac=0.9, nside=64):
            """
            A utility routine to get a dataset for unsupervised clustering.
            Note:
            - We want the unmasked regions of the depth map only.
            - We assume masked regions are set to `maskval`,
              and cut 0.1 magnitudes above that.
            - We really want it to look at depth fluctuations.
              So, we have to rescale the
              RA/dec dimensions to avoid them being prioritized
              because their values are larger and
              have more variation than depth.  Currently we rescale RA/dec
              such that their standard deviations are 1-priority_fac times
              the standard deviation of the depth map. That's why
              priority_fac is a tunable parameter; it should be between 0 and 1
            """
            if priority_fac < 0 or priority_fac >= 1:
                raise ValueError("priority_fac must lie between 0 and 1")
            theta, phi = hp.pixelfunc.pix2ang(nside, ipix=np.arange(hp.nside2npix(nside)))
            # theta is 0 at the north pole, pi/2 at equator,
            # pi at south pole; phi maps to RA
            ra = np.rad2deg(phi)
            dec = np.rad2deg(0.5 * np.pi - theta)

            # Make a 3D numpy array containing the unmasked regions,
            # including a rescaling factor to prioritize the depth
            n_unmasked = len(depth_map[depth_map > 0.1])
            my_data = np.zeros((n_unmasked, 3))
            cutval = 0.1 + maskval
            my_data[:, 0] = (
                ra[depth_map > cutval]
                * (1 - priority_fac)
                * np.std(depth_map[depth_map > cutval])
                / np.std(ra[depth_map > cutval])
            )
            my_data[:, 1] = (
                dec[depth_map > cutval]
                * (1 - priority_fac)
                * np.std(depth_map[depth_map > cutval])
                / np.std(dec[depth_map > cutval])
            )
            my_data[:, 2] = depth_map[depth_map > cutval]
            return my_data

        if not stripes:
            return 1
        else:
            # Do the clustering if we got to this point
            if self.verbose:
                print("Verbose mode - Carrying out the clustering exercise for this map")
            clustering_data = make_clustering_dataset(data_slice_arr["riz_exptime"].ravel(), nside=nside)
            labels = apply_clustering(clustering_data)
            area_frac, med_val = get_area_stats(data_slice_arr["riz_exptime"].ravel(), labels)
            if self.verbose:
                print("Verbose mode - showing original map and clusters identified for this map")
                show_clusters(data_slice_arr["riz_exptime"].ravel(), labels)
                print("Area fractions", area_frac)
                print("Median exposure time values", med_val)
                print("Median exposure time ratio", np.max(med_val) / np.min(med_val))
                print("Verbose mode - proceeding with area cuts")
            # Get the FoM without/with cuts.  We want to check the FoM for each
            # area, if we're doing cuts, and
            # return the higher one. This will typically be for the
            # larger area, but not necessarily, if the smaller area
            # is deeper.
            expanded_labels = expand_labels(data_slice_arr["riz_exptime"].ravel(), labels)
            my_hpid_1 = expanded_labels == 1  # np.where(expanded_labels == 1)[0]
            my_hpid_2 = expanded_labels == 2  # np.where(expanded_labels == 2)[0]
            # should this be labels or expanded_labels?!

            # copies need to be recarrays
            data_slice_subset_2 = deepcopy(data_slice_arr)
            data_slice_subset_1 = deepcopy(data_slice_arr)
            data_slice_subset_1[my_hpid_2] = self.mask_val_arr
            data_slice_subset_2[my_hpid_1] = self.mask_val_arr
            fom1 = self.threebyTwoSummary.run(data_slice_subset_1)
            fom2 = self.threebyTwoSummary.run(data_slice_subset_2)
            fom = np.max((fom1, fom2))
            fom_total = self.threebyTwoSummary.run(data_slice_arr)
            if self.verbose:
                print("FOMs:", fom1, fom2, fom, fom_total)
            return fom / fom_total


class UDropoutsNumbersShallowestDepth(BaseMetric):
    """
    Run as summary metric on MultibandExgalM5.



    Parameters
    ----------
    year : `int`, optional
        The year of the survey to calculate the bias.
        This is used to derive the dm/dz derivative used to translate
        m5 rms into dz rms.

    Returns
    -------
    result : `float` array
        The ratio of this bias to the desired DESC y1 upper bound on
        the bias, and the ratio
        between the clbias and the y10 DESC SRD requirement.
        Desired values are less than 1 by Y10.


    Notes
    -----
    This is a summary metric to be run on the results
    of the MultibandExgalM5.

    MultibandExgalM5 provides the m5 depth in all LSST
    bands given a specific slice.

    """

    def __init__(self, filter_list=["u", "g", "r", "i", "z", "y"], **kwargs):
        super().__init__(col="metricdata", percentile=5, **kwargs)
        # Set mask_val, so that we receive metric_values.filled(mask_val)
        self.mask_val = hp.UNSEEN
        self.badval = hp.UNSEEN
        self.percentileMetric = PercentileMetric(percentile=5, name="percentile")
        self.filter_list = filter_list

    def run(self, data_slice, slice_point=None):
        # Technically don't need this for now (isn't used in previous one)
        # need to define an array of bad values for the masked pixels
        badval_arr = np.repeat(self.badval, len(self.filter_list))
        # converts the input recarray to an array
        data_slice_list = [
            badval_arr if isinstance(x, float) else x for x in data_slice["metricdata"].tolist()
        ]
        # should be (nbins, npix)
        data_slice_arr = np.asarray(data_slice_list, dtype=float).T
        data_slice_arr[~np.isfinite(data_slice_arr)] = (
            hp.UNSEEN
        )  # need to work with TotalPowerMetric and healpix

        # measure rms in each bin.
        # The original metric returns an array at each slice_point (of the
        # original slicer) -- so there is a bit of "rearrangement" that
        # has to happen to be able to pass a np.array with right dtype
        # (i.e. dtype = [("metricdata", float)]) to each call to
        # the rmsMetric `run` methods.
        percentiles = np.array(
            [
                self.percentileMetric.run(np.core.records.fromrecords(x, dtype=[("metricdata", float)]))
                for x in data_slice_arr
            ]
        )  # 6 numbers

        import astropy.units as u
        from astropy.cosmology import Planck18
        from astropy.modeling.models import Schechter1D

        def schecter_lf(
            m_grid: np.ndarray,
            log_phi_star: float = -2.84,
            M_star: float = -20.91,
            alpha: float = -1.68,
            redshift: float = 3,
        ) -> np.ndarray:
            """Schecter Luminosity Function on grid of apparent magnitudes.

            Defaults are for z~3 u-dropout Luminosity Function from Table 6
            of Harikane et al. 2022.

            Parameters
            ----------
            m_grid: np.ndarray
                Array of apparent AB magnitudes on which to calculate the
                luminosity function.
            log_phi_star: float, default=-2.84
                Natural log of phi_star, the normalization of the luminosity
                function in units of mag^-1 Mpc^-3
            M_star: float, default=-20.91
                The characteristic absolute magnitude where the power-law form
                of the luminosity function cuts off.
            alpha: float, default=-1.68
                The power law index, also known as the faint-end slope.
            redshift: float, default=3
                Redshift used for converting apparent magnitudes into absolute
                magnitudes.

            Returns
            -------
            np.ndarray
                Number density in units mag^-1 Mpc^-3
            """
            # Convert observed magnitudes to absolute
            DL = Planck18.luminosity_distance(redshift).to(u.pc).value  # Lum. Dist. in pc
            M_grid = m_grid - 5 * np.log10(DL / 10) + 2.5 * np.log10(1 + redshift)

            # Calculate luminosity function in absolute magnitudes
            schechter = Schechter1D(10**log_phi_star, M_star, alpha)

            return schechter(M_grid)

        def number_density(
            m_grid: np.ndarray,
            LF: np.ndarray,
            redshift: float = 3,
            dz: float = 1,
        ) -> float:
            """Calculate number density per deg^2.

            Parameters
            ----------
            m_grid: np.ndarray
                Array of apparent AB magnitudes corresponding to the
                luminosity function values.
            LF: np.ndarray
                The luminosity function evaluated on m_grid, in units
                mag^-1 Mpc^-3.
            redshift: float, default=3
                The central redshift used for evaluating comoving quantities.
            dz: float, default=1
                The full width of the redshift bin

            Returns
            -------
            float
                The total number density of galaxies in units deg^-2.
            """
            # Calculate comoving depth of redshift bin (Mpc)
            chi_far = Planck18.comoving_distance(redshift + dz / 2)
            chi_near = Planck18.comoving_distance(redshift - dz / 2)
            dchi = chi_far - chi_near

            # Calculate number density (mag^-1 Mpc^-2)
            n_dm = (LF / u.Mpc**3) * dchi

            # Convert to mag^-1 deg^-2
            deg_per_Mpc = Planck18.arcsec_per_kpc_comoving(redshift).to(u.deg / u.Mpc)
            n_dm /= deg_per_Mpc**2

            # Integrate the luminosity function
            n = np.trapz(n_dm, m_grid)

            return n.value

        u5 = percentiles[self.filter_list == "u"]  # uband m5
        u3 = u5 + 2.5 * np.log10(5 / 3)  # convert to uband 3-sigma
        g_cut = u3 - 1.5  # cut on g band
        # Calculate LF up to g_cut
        m_grid = np.linspace(20, g_cut)
        LF = schecter_lf(m_grid)
        n_deg2 = number_density(m_grid, LF)  # number of objects per sq deg

        return n_deg2


class UDropoutsArea(BaseMetric):
    """
    Run as summary metric on MultibandExgalM5.

    Parameters
    ----------
    year : `int`, optional
        The year of the survey to calculate the bias.
        This is used to derive the dm/dz derivative used to translate m5
        rms into dz rms.

    Returns
    -------
    result : `float` array
        The ratio of this bias to the desired DESC y1 upper bound on
        the bias, and the ratio
        between the clbias and the y10 DESC SRD requirement.
        Desired values are less than 1 by Y10.


    Notes
    -----
    This is a summary metric to be run on the results
    of the MultibandExgalM5.

    MultibandExgalM5 provides the m5 depth in all LSST
    bands given a specific slice.

    """

    def __init__(self, filter_list=["u", "g", "r", "i", "z", "y"], **kwargs):
        super().__init__(col="metricdata", percentile=5, **kwargs)
        # Set mask_val, so that we receive metric_values.filled(mask_val)
        self.mask_val = hp.UNSEEN
        self.badval = hp.UNSEEN
        self.percentileMetric = PercentileMetric(percentile=5, name="percentile")
        self.filter_list = filter_list

    def run(self, data_slice, slice_point=None):
        # Technically don't need this for now (isn't used in previous one)
        # need to define an array of bad values for the masked pixels
        badval_arr = np.repeat(self.badval, len(self.filter_list))
        # converts the input recarray to an array
        data_slice_list = [
            badval_arr if isinstance(x, float) else x for x in data_slice["metricdata"].tolist()
        ]
        # should be (nbins, npix)
        data_slice_arr = np.asarray(data_slice_list, dtype=float).T
        data_slice_arr[~np.isfinite(data_slice_arr)] = (
            hp.UNSEEN
        )  # need to work with TotalPowerMetric and healpix

        # measure rms in each bin.
        # The original metric returns an array at each slice_point (of the
        # original slicer) -- so there is a bit of "rearrangement" that
        # has to happen to be able to pass a np.array with right dtype
        # (i.e. dtype = [("metricdata", float)]) to each call to
        # the rmsMetric `run` methods.
        percentiles = np.array(
            [
                self.percentileMetric.run(np.core.records.fromrecords(x, dtype=[("metricdata", float)]))
                for x in data_slice_arr
            ]
        )  # 6 numbers
        return percentiles
