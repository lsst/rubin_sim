__all__ = (
    "TotalPowerMetric",
    "StaticProbesFoMEmulatorMetricSimple",
    "TomographicClusteringSigma8biasMetric",
)

import warnings

import healpy as hp
import numpy as np
from scipy import interpolate

from .area_summary_metrics import AreaThresholdMetric
from .base_metric import BaseMetric

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
        lmin=100.0,
        lmax=300.0,
        remove_monopole=True,
        remove_dipole=True,
        col="metricdata",
        mask_val=np.nan,
        **kwargs,
    ):
        self.lmin = lmin
        self.lmax = lmax
        self.remove_monopole = remove_monopole
        self.remove_dipole = remove_dipole
        super().__init__(col=col, mask_val=mask_val, **kwargs)

    def run(self, data_slice, slice_point=None):
        # Calculate the power spectrum.
        data = data_slice[self.colname]
        if self.remove_monopole:
            data = hp.remove_monopole(data, verbose=False, bad=self.mask_val)
        if self.remove_dipole:
            data = hp.remove_dipole(data, verbose=False, bad=self.mask_val)
        cl = hp.anafast(data)
        ell = np.arange(np.size(cl))
        condition = np.where((ell <= self.lmax) & (ell >= self.lmin))[0]
        totalpower = np.sum(cl[condition] * (2 * ell[condition] + 1))
        return totalpower


class StaticProbesFoMEmulatorMetricSimple(BaseMetric):
    """Calculate the FoM for the combined static probes
    (3x2pt, i.e. Weak Lensing, LSS, Clustering).

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

    def __init__(self, year=10, **kwargs):
        self.year = year
        super().__init__(col="metricdata", mask_val=-666, **kwargs)

    def run(self, data_slice, slice_point=None):
        # derive nside from length of data slice
        nside = hp.npix2nside(len(data_slice))
        pix_area = hp.nside2pixarea(nside, degrees=True)

        # Chop off any outliers (and also the masked value)
        good_pix = np.where(data_slice[self.col] > 0)[0]

        # Calculate area and med depth from
        area = pix_area * np.size(good_pix)
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
        f = interpolate.interp2d(areas, depths, fom_arr, bounds_error=False)
        fom = f(area, median_depth)[0]
        return fom


class TomographicClusteringSigma8biasMetric(BaseMetric):
    """Compute bias on sigma8 due to spurious contamination of density maps.
    Run as summary metric on NestedLinearMultibandModelMetric.

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
        **kwargs,
    ):
        super().__init__(col="metricdata", **kwargs)
        # Set mask_val, so that we receive metric_values.filled(mask_val)
        self.mask_val = hp.UNSEEN

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
            lower_threshold=hp.UNSEEN,
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
            hp.UNSEEN
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

                # simple model variance of cell baased on Gaussian covariance
                cells_var = 2 * cells_model**2 / (2 * ells + 1) / fskys[i]
                totalvar_var[i, 0] = np.sum(cells_var * (2 * ells + 1) ** 2)

            # model assumed sigma8 = 0.8
            # (add CCL cosmology here? or how I obtained them + documentation)
            # results_fractional_spurious_power =
            # totalvar_obs / totalvar_mod - 1.0

            # model Cell variance divided by sigma8^2,
            # which is the common normalization
            transfers = totalvar_mod / sigma8square_model

            # model ratio: formula for posterior distribution on unknown
            # multiplicative factor in multivariate Gaussian likelihood
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


class UniformAreaFoMFractionMetric(BaseMetric):
    """?
    Run as summary metric on RIZDetectionCoaddExposureTime.

    Parameters
    ----------
    ?

    Returns
    -------
    ?

    Notes
    -----
    ?
    """

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(col="metricdata", **kwargs)
        # Set mask_val, so that we receive metric_values.filled(mask_val)
        self.mask_val = hp.UNSEEN

    def run(self, data_slice, slice_point=None):

        def make_clustering_dataset(depth_map, maskval=0, priority_fac=0.9, nside=64):
            # A utility routine to get a dataset for unsupervised clustering.  Note:
            # - We want the unmasked regions of the depth map only.
            # - We assume masked regions are set to `maskval`, and cut 0.1 magnitudes above that.
            # - We really want it to look at depth fluctuations.  So, we have to rescale the
            #   RA/dec dimensions to avoid them being prioritized because their values are larger and
            #   have more variation than depth.  Currently we rescale RA/dec such that their
            #   standard deviations are 1-priority_fac times the standard deviation of the depth map.
            #   That's why priority_fac is a tunable parameter; it should be between 0 and 1
            if priority_fac < 0 or priority_fac >= 1:
                raise ValueError("priority_fac must lie between 0 and 1")
            theta, phi = hp.pixelfunc.pix2ang(nside, ipix=np.arange(hp.nside2npix(nside)))
            # theta is 0 at the north pole, pi/2 at equator, pi at south pole; phi maps to RA
            ra = np.rad2deg(phi)
            dec = np.rad2deg(0.5 * np.pi - theta)

            # Make a 3D numpy array containing the unmasked regions, including a rescaling factor to prioritize the depth
            n_unmasked = len(depth_map[depth_map > 0.1])
            my_data = np.zeros((n_unmasked, 3))
            cutval = 0.1 + maskval
            my_data[:, 0] = ra[depth_map > cutval] * (1 - priority_fac) * np.std(
                depth_map[depth_map > cutval]) / np.std(ra[depth_map > cutval])
            my_data[:, 1] = dec[depth_map > cutval] * (1 - priority_fac) * np.std(
                depth_map[depth_map > cutval]) / np.std(dec[depth_map > cutval])
            my_data[:, 2] = depth_map[depth_map > cutval]
            return my_data

            # Check for stripiness

        nside = hp.npix2nside(data_slice.size)
        self.threebyTwoSummary = StaticProbesFoMEmulatorMetric(nside=nside, metric_name="3x2ptFoM")
        stripes = has_stripes(data_slice, nside)

        if not stripes:
            return 1
        else:
            # Do the clustering if we got to this point
            if verbose: print("Verbose mode - Carrying out the clustering exercise for this map")
            clustering_data = make_clustering_dataset(map)
            labels = apply_clustering(clustering_data)
            area_frac, med_val = get_area_stats(data_slice, labels)
            if verbose:
                print("Verbose mode - showing original map and clusters identified for this map")
                show_clusters(data_slice, labels)
                print("Area fractions", area_frac)
                print("Median exposure time values", med_val)
                print("Median exposure time ratio", np.max(med_val)/np.min(med_val))
                print("Verbose mode - proceeding with area cuts")
            # Get the FoM without/with cuts.  We want to check the FoM for each area, if we're doing cuts, and
            # return the higher one. This will typically be for the larger area, but not necessarily, if the smaller area
            # is deeper.
            expanded_labels = expand_labels(data_slice, labels)
            my_hpid_1 = np.where(labels == 1)[0]
            my_hpid_2 = np.where(labels == 2)[0]
            data_slice_subset_1 = copy(data_slice)
            data_slice[my_hpid_2] = hp.UNSEEN
            data_slice_subset_2 = copy(data_slice)
            data_slice[my_hpid_1] = hp.UNSEEN
            fom1 = self.threebyTwoSummary.run(data_slice_subset_1)
            fom2 = self.threebyTwoSummary.run(data_slice_subset_2)
            fom = np.max((fom1, fom2))
            fom_total = self.threebyTwoSummary.run(data_slice)
            return fom / fom_total

