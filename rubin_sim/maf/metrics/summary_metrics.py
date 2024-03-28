__all__ = (
    "FootprintFractionMetric",
    "FOArea",
    "FONv",
    "IdentityMetric",
    "NormalizeMetric",
    "ZeropointMetric",
    "TotalPowerMetric",
    "StaticProbesFoMEmulatorMetricSimple",
)

import warnings

import healpy as hp
import numpy as np
from scipy import interpolate

from .base_metric import BaseMetric

# Metrics which are primarily intended to be used as summary statistics.


class FootprintFractionMetric(BaseMetric):
    """Calculate fraction of a desired footprint got covered.
    Helpful to check if everything was covered in first year

    Parameters
    ----------
    footprint : `np.ndarray`, (N,)
        The HEALpix footprint to compare to.
        Nside of the footprint should match nside of the slicer.
    n_min : `int`
        The number of visits to require to consider an area covered
    """

    def __init__(self, footprint=None, n_min=1, **kwargs):
        super().__init__(**kwargs)
        self.footprint = footprint
        self.nside = hp.npix2nside(footprint.size)
        self.npix = np.where(self.footprint > 0)[0].size
        # get whole array passed
        self.mask_val = 0
        self.n_min = n_min

    def run(self, data_slice, slice_point=None):
        overlap = np.where((self.footprint > 0) & (data_slice["metricdata"] >= self.n_min))[0]
        result = overlap.size / self.npix
        return result


class FONv(BaseMetric):
    """Given asky area, what is the minimum and median NVISITS obtained over
    that area?
    (chooses the portion of the sky with the highest number of visits first).

    Parameters
    ----------
    col : `str` or `list` of `strs`, optional
        Name of the column in the numpy recarray passed to the summary metric.
    asky : `float`, optional
        Area of the sky to base the evaluation of number of visits over.
    nside : `int`, optional
        Nside parameter from healpix slicer, used to set the physical
        relationship between on-sky area and number of healpixels.
    n_visit : `int`, optional
        Number of visits to use as the benchmark value, if choosing to return
        a normalized n_visit value.
    norm : `bool`, optional
        Normalize the returned "n_visit" (min / median) values by n_visit,
        if true.
    metric_name : `str`, optional
        Name of the summary metric. Default FONv.
    """

    def __init__(self, col="metricdata", asky=18000.0, nside=128, n_visit=825, norm=False, **kwargs):
        """asky = square degrees"""
        super().__init__(col=col, **kwargs)
        self.nvisit = n_visit
        self.nside = nside
        # Determine how many healpixels are included in asky sq deg.
        self.asky = asky
        self.scale = hp.nside2pixarea(self.nside, degrees=True)
        self.npix__asky = int(np.ceil(self.asky / self.scale))
        self.norm = norm

    def run(self, data_slice, slice_point=None):
        result = np.empty(2, dtype=[("name", np.str_, 20), ("value", float)])
        result["name"][0] = "MedianNvis"
        result["name"][1] = "MinNvis"
        # If there is not even as much data as needed to cover Asky:
        if len(data_slice) < self.npix__asky:
            result["value"][0] = self.badval
            result["value"][1] = self.badval
            return result
        # Otherwise, calculate median and mean Nvis:
        nvis_sorted = np.sort(data_slice[self.colname])
        # Find the Asky's worth of healpixels with the largest # of visits.
        nvis__asky = nvis_sorted[-self.npix__asky :]
        result["value"][0] = np.median(nvis__asky)
        result["value"][1] = np.min(nvis__asky)
        if self.norm:
            result["value"] /= float(self.nvisit)
        return result


class FOArea(BaseMetric):
    """Given an n_visit threshold, how much AREA receives at least that many
    visits?

    Parameters
    ----------
    col : `str` or `list` of `strs`, optional
        Name of the column in the numpy recarray passed to the summary metric.
    n_visit : `int`, optional
        Number of visits to use as the minimum required --
        metric calculated area that has this many visits.
    asky : `float`, optional
        Area to use as the benchmark area value,
        if choosing to return a normalized Area value.
    nside : `int`, optional
        Nside parameter from healpix slicer, used to set the physical
        relationship between on-sky area and number of healpixels.
    norm : `bool`, optional
        If true, normalize the returned area value by asky.
    """

    def __init__(
        self,
        col="metricdata",
        n_visit=825,
        asky=18000.0,
        nside=128,
        norm=False,
        **kwargs,
    ):
        """asky = square degrees"""
        super().__init__(col=col, **kwargs)
        self.nvisit = n_visit
        self.nside = nside
        self.asky = asky
        self.scale = hp.nside2pixarea(self.nside, degrees=True)
        self.norm = norm

    def run(self, data_slice, slice_point=None):
        nvis_sorted = np.sort(data_slice[self.colname])
        # Identify the healpixels with more than Nvisits.
        nvis_min = nvis_sorted[np.where(nvis_sorted >= self.nvisit)]
        if len(nvis_min) == 0:
            result = self.badval
        else:
            result = nvis_min.size * self.scale
            if self.norm:
                result /= float(self.asky)
        return result


class IdentityMetric(BaseMetric):
    """Return the metric value.

    This is primarily useful as a summary statistic for UniSlicer metrics,
    to propagate the ~MetricBundle.metric_value into the results database.
    """

    def run(self, data_slice, slice_point=None):
        if len(data_slice[self.colname]) == 1:
            result = data_slice[self.colname][0]
        else:
            result = data_slice[self.colname]
        return result


class NormalizeMetric(BaseMetric):
    """
    Return a metric values divided by 'norm_val'.
    Useful for turning summary statistics into fractions.
    """

    def __init__(self, col="metricdata", norm_val=1, **kwargs):
        super(NormalizeMetric, self).__init__(col=col, **kwargs)
        self.norm_val = float(norm_val)

    def run(self, data_slice, slice_point=None):
        result = data_slice[self.colname] / self.norm_val
        if len(result) == 1:
            return result[0]
        else:
            return result


class ZeropointMetric(BaseMetric):
    """
    Return a metric values with the addition of 'zp'.
    Useful for altering the zeropoint for summary statistics.
    """

    def __init__(self, col="metricdata", zp=0, **kwargs):
        super(ZeropointMetric, self).__init__(col=col, **kwargs)
        self.zp = zp

    def run(self, data_slice, slice_point=None):
        result = data_slice[self.colname] + self.zp
        if len(result) == 1:
            return result[0]
        else:
            return result


class TotalPowerMetric(BaseMetric):
    """
    Calculate the total power in the angular power spectrum between lmin/lmax.
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

    This version of the emulator was used to generate the results in
    https://ui.adsabs.harvard.edu/abs/2018arXiv181200515L/abstract

    A newer version is being created. This version has been renamed
    Simple in anticipation of the newer, more sophisticated metric
    replacing it.

    Note that this is truly a summary metric and should be run on the output of
    Exgalm5_with_cuts.
    """

    def __init__(self, nside=128, year=10, col=None, **kwargs):
        """
        Args:
            nside (int): healpix resolution
            year (int): year of the FoM emulated values,
                can be one of [1, 3, 6, 10]
            col (str): column name of metric data.
        """
        self.nside = nside
        super().__init__(col=col, **kwargs)
        if col is None:
            self.col = "metricdata"
        self.year = year

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
        # Chop off any outliers
        good_pix = np.where(data_slice[self.col] > 0)[0]
        if np.size(good_pix) == 0:
            return self.badval

        # Calculate area and med depth from
        area = hp.nside2pixarea(self.nside, degrees=True) * np.size(good_pix)
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


class TomographicClusteringSigma8bias(BaseMetric):
    """
    Compute bias on sigma8 due to spurious contamination of density maps.
    This is a summary metric to be interfaced with NestedLinearMultibandModelMetric.
    NestedLinearMultibandModelMetric converts 6-band depth maps into a set of maps (e.g tomographic redshift bins)
    which describes spurious density fluctuations in each bin.
    This summary metric multiplies the maps by the parameter power_multiplier,
    which can be used to describe the fraction of power uncorrected by systematics mitigation schemes,
    computes the total power (via angular power spectra with lmin-lmax limits)
    and then infers sigma8^2 via a model of the angular power spectra.
    By taking sigma8_obs minus sigma8_model divided by the uncertainty, one derives a bias.
    """

    def __init__(
        self,
        density_tomography_model,
        power_multiplier=0.1,
        lmin=10,
        badval=np.nan,
        convert_to_sigma8=True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        density_tomography_model: dict
            dictionary containing models calculated for fiducial N(z)s and Cells:
                lmax: numpy.array of int, of shape (Nbins, )
                    lmax corresponding to kmax of 0.05
                poly1d_coefs_loglog: numpy.array of float, of shape (Nbins, )
                    polynomial fits to log(C_ell) vs log(ell) computed for CCL
                sigma8square_model (float)
                    value of sigma8^2 used as fiducal model for CCL
        power_multiplier: `float`, optional
            fraction of power (variance) which is uncorrected and thus biases sigma8
        lmin: `int`, optional
            lmin for the analysis
        convert_to_sigma8: `str`, optional
            Convert the bias to sigma8 instead of sigma8^2 (via change of variables for the uncertainty)
        badval: `float`, optional
            value to return for bad pixels (e.g. pixels not passing cuts)

        Returns (with the method 'run')
        -------
        result: `float`
            value of sigma8 bias calculated from this model: (sigma8^2_obs - sigma^2_model) / error on sigma8^2_obs
            if convert_to_sigma8=Then then it is about sigma8 instead of sigma8^2.

        """
        super().__init__(**kwargs)

        self.convert_to_sigma8 = convert_to_sigma8
        self.badval = badval
        self.mask_val = badval
        self.power_multiplier = power_multiplier
        self.lmin = lmin
        self.density_tomography_model = density_tomography_model
        # to compute angular power spectra and total power, initialize an array of metrics, with the right lmin and lmax.
        self.totalPowerMetrics = [
            TotalPowerMetric(
                col="metricdata", lmin=lmin, lmax=lmax, metric_name="TotalPower_bin", mask_val=hp.UNSEEN
            )
            for lmax in density_tomography_model["lmax"]
        ]
        self.areaThresholdMetric = AreaThresholdMetric(
            col="metricdata",
            metric_name="FootprintFraction_bin",
            lower_threshold=hp.UNSEEN,
            upper_threshold=np.inf,
        )

    def run(self, data_slice, slice_point=None):

        # need to define an array of bad values for the masked pixels
        badval_arr = np.repeat(self.badval, len(self.density_tomography_model["lmax"]))
        # converts the input recarray to an array
        data_slice_list = [
            badval_arr if isinstance(x, float) else x for x in data_slice["metricdata"].tolist()
        ]
        data_slice_arr = np.asarray(data_slice_list, dtype=float).T  # should be (nbins, npix)
        data_slice_arr[~np.isfinite(data_slice_arr)] = (
            hp.UNSEEN
        )  # need to work with TotalPowerMetric and healpix

        # measure valid sky fractions and total power (via angular power spectra) in each bin.
        fskys = np.array(
            [
                self.areaThresholdMetric.run(np.core.records.fromrecords(x, dtype=[("metricdata", float)]))
                / 42000
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
        # this could probably be avoided if recarrays were returned by the original nested/vector metric,
        # except that we would need to manipulate the names in various places, which I wanted to avoid,
        # so for now the main metric returns an array per healpix pixel (not a recarray) and puts together
        # healpix maps which we need to convert to a recarray to pass to AreaThresholdMetric and TotalPowerMetric

        def solve_for_multiplicative_factor(spurious_powers, model_cells, fskys, lmin, power_multiplier):
            """
            Infer multiplicative factor sigma8^2 (and uncertainty) from the model Cells and observed total powers
            since it os a Gaussian posterior distribution.
            """
            # solve for multiplicative sigma8^2 term between
            # measured angular power spectra (spurious measured Cells times power_multiplier)
            # and model ones (polynomial model from CCL).
            n_bins = model_cells["lmax"].size
            assert len(spurious_powers) == n_bins
            assert len(fskys) == n_bins
            assert model_cells["poly1d_coefs_loglog"].shape[0] == n_bins
            totalvar_mod = np.zeros((n_bins, 1))
            totalvar_obs = np.zeros((n_bins, 1))
            totalvar_var = np.zeros((n_bins, 1))
            # loop over tomographic bins
            sigma8square_model = model_cells["sigma8square_model"]  # hardcoded; assumed CCL cosmology
            for i in range(n_bins):
                # get model Cells from polynomial model (in log log space)
                ells = np.arange(lmin, model_cells["lmax"][i])
                polynomial_model = np.poly1d(model_cells["poly1d_coefs_loglog"][i, :])
                cells_model = np.exp(polynomial_model(np.log(ells)))

                # model variance is sum of cells x (2l+1)
                totalvar_mod[i, 0] = np.sum(cells_model * (2 * ells + 1))

                # observations is spurious power  noiseless model
                totalvar_obs[i, 0] = totalvar_mod[i, 0] + spurious_powers[i] * power_multiplier

                # simple model variance of cell baased on Gaussian covariance
                cells_var = 2 * cells_model**2 / (2 * ells + 1) / fskys[i]
                totalvar_var[i, 0] = np.sum(cells_var * (2 * ells + 1) ** 2)

            # model assumed sigma8 = 0.8 (add CCL cosmology here? or how I obtained them + documentation
            results_fractional_spurious_power = totalvar_obs / totalvar_mod - 1.0
            transfers = (
                totalvar_mod / sigma8square_model
            )  # model Cell variance divided by sigma8^2, which is the common normalization

            # model ratio: formula for posterior distribution on unknown multiplicative factor in multivariate Gaussian likelihood
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

        # turn result into bias on sigma8, via change of variable and simple propagation of uncertainty.
        sigma8_fit = sigma8square_fit**0.5
        sigma8_model = sigma8square_model**0.5
        sigma8_error = 0.5 * sigma8square_error * sigma8_fit / sigma8square_fit
        results_sigma8_bias = (sigma8_fit - sigma8_model) / sigma8_error

        if self.convert_to_sigma8:
            return results_sigma8_bias
        else:
            return results_sigma8_square_bias
