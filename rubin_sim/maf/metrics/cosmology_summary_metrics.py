__all__ = (
    "TotalPowerMetric",
    "StaticProbesFoMEmulatorMetricSimple",
    "TomographicClusteringSigma8biasMetric",
    "UniformMeanzBiasMetric",
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

class UniformMeanzBiasMetric(BaseMetric):
    import maf

    """This calculates the bias in the weak lensing power given 
       the scatter in the redshift of the tomographic sample
    induced by survey non-uniformity. 

    Parameters
    ----------
    year : `int`, optional
        The year of the survey to calculate the bias.
        This is used to derive the dm/dz derivative used to translate m5 rms into dz rms.

    Returns
    -------
    result : `float` array
        The ratio of this bias to the desired DESC y1 upper bound on the bias, and the ratio 
        between the clbias and the y10 DESC SRD requirement. 
        Desired values are less than 1 by Y10.

    Notes
    -----

    Note that this is truly a summary metric and should be run on the
    output of Exgalm5_with_cuts.
    """

    def __init__(self, filter_list="filters",year=10, n_filters=6,**kwargs):
        self.year = year
        self.filter_list = filter_list
        self.exgal_m5 = ExgalM5(m5_col=m5_col, units=units)
        
        super().__init__(col="metricdata", mask_val=-666, **kwargs)


    def run(self, data_slice, slice_point=None):

        result = np.empty(1, dtype=[("name", np.str_, 20), ("value", float)])
        result["name"][0] = "UniformMeanzBiasMetric"

        def compute_dzfromdm(zbins, band_ind, year):
            """ This computes the dm/dz relationship calibrated from simulations
            by Jeff Newmann.

            Parameters
            ----------
            zbins : `int`
                The number of tomographic bins considered. For now this is zbins < 5
            filter : `str`
                The assumed filter band 

            Returns
            -------
            dzdminterp : `float` 
                The interpolated value of the derivative dz/dm
            meanzinterp : `float` array
                The meanz in each tomographic bin.
            
            """
            import pandas as pd

            filter_list=["u","g","r","i","z","y"]
            band_ind =filter_list.index(filter)
            
            deriv = pd.read_pickle('uniformity_pkl/meanzderiv.pkl')
            # pkl file of derivatives with 10 years, 7 bands (ugrizY and combined), 5 bins
            zvals = pd.read_pickle('uniformity_pkl/meanzsy%i.pkl'%(year+1)) 
            # pkl file of mean z values for a given year over 5 z bins, 7 bands (ugrizY and combined),
            # for a fixed delta density index (index 5 assumed below is for zero m_5 shift)
            meanzinterp = zvals[0:zbins,band_ind,5]
            dzdminterp = np.abs(deriv[year,band_ind,0:zbins])

            return dzdminterp, meanzinterp

        def use_zbins(meanz_vals, figure_9_mean_z=np.array([0.2, 0.4, 0.7, 1.0]),  figure_9_width=0.2):
            """ This computes which redshift bands are within the range 
            specified in https://arxiv.org/pdf/2305.15406.pdf and can safely be used
            to compute what Cl bias result from z fluctuations caused by rms variations in the m5.


            Parameters
            ----------
            meanz_vals : `float` array
                Array of meanz values to be used.
            
            Returns
            -------
            use_bins : `boolean` array
                An array of boolean values of length meanz_vals 
            
            """
            max_z_use = np.max(figure_9_mean_z)+2*figure_9_width
            use_bins = meanz_vals < max_z_use
            
            return use_bins

        def compute_Clbias(meanz_vals,scatter_mean_z_values):
            """ This computes the Cl bias 
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
                An array of the meanz values that are within the interpolation range of 2305.15406
            
            Notes
            ------
            This interpolates from the Figure 9 in https://arxiv.org/pdf/2305.15406.pdf

            """
            import numpy as np
            figure_9_mean_z=np.array([0.2, 0.4, 0.7, 1.0])
            figure_9_Clbias =np.array([1e-3, 2e-3, 5e-3, 1.1e-2])
            figure_9_width=0.2
            figure_9_mean_z_scatter = 0.02

            mzvals= np.array([float(mz) for mz in meanz_vals])
            sctz = np.array([float(sz)for sz in scatter_mean_z_values])
            
            fit_res = np.polyfit(figure_9_mean_z, figure_9_Clbias, 2)
            poly_fit = np.poly1d(fit_res)
            use_bins = use_zbins(meanz_vals,figure_9_mean_z, figure_9_width)

            mean_z_values_use = mzvals[use_bins]
            sctz_use = sctz[use_bins]

            Clbias = poly_fit(mean_z_values_use)
            rescale_fac =  sctz_use / figure_9_mean_z_scatter
            Clbias *= rescale_fac
            fit_res_bias = np.polyfit(mean_z_values_use, Clbias, 1)
            poly_fit_bias = np.poly1d(fit_res_bias)

            clbiasvals = poly_fit_bias(mean_z_values_use)
            return clbiasvals, mean_z_values_use

        totdz=0
        avmeanz=0
        clbiastot=0
        for filt in self.filter_list:
            d_s = data_slice[data_slice[self.filter_col] == filt]
            # calculate the lsstFilter-band coadded depth
            coadd_depth = self.exgal_m5.run(d_s, slice_point)

            rmsval = np.std(coadd_depth)

            dzdminterp, meanzinterp=compute_dzfromdm(self.zbins, filt,self.year)
            stdz = [float(np.abs(dz))*float(rmsval) for dz in dzdminterp]
    
            clbias, meanz_use = compute_Clbias(meanzinterp,stdz)

            totdz+=[float(st**2) for st in stdz]
            totclbias+=clbias
            avmeanz+=meanzinterp
        

        y10_req = 0.003
        y1_goal = 0.013

        clbiastot = np.max(clbias)
        y10ratio = clbiastot/y10_req
        y1ratio = clbiastot/y1_goal

        result["y1ratio"]=y1ratio
        result["y10ratio"]=y1ratio
        
        return result

    