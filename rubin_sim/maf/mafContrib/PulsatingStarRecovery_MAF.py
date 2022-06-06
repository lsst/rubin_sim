import os
import copy
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from gatspy import periodic
from scipy.optimize import leastsq

from rubin_sim.maf.utils import m52snr
from rubin_sim.photUtils import Dust_values
from rubin_sim.maf.metrics import BaseMetric
from rubin_sim.maf.data import get_data_dir

__all__ = ["meanmag_antilog", "mag_antilog", "PulsatingStarRecovery"]


def meanmag_antilog(mag):
    # Convert mags to flux, take mean of flux, return as mag
    mag = np.asarray(mag)
    flux = 10.0 ** (-mag / 2.5)
    if len(flux) > 0:
        result = (-2.5) * np.log10(sum(flux) / len(flux))
    else:
        result = 9999.0
    return result


def mag_antilog(mag):
    # convert mags to fluxes (assume zp = 0), return fluxes
    mag = np.asarray(mag)
    flux = 10.0 ** (-mag / 2.5)
    return flux


class PulsatingStarRecovery(BaseMetric):
    """
    Evaluate how well the period and shape of a multi-band lightcurve (from a template in a csv file)
    can be fit.
    Returns a dictionary with the results of Lcsampling, Lcperiod, Lcfitting:Lcsampling
    This metric  studies how well a given cadence stategy
    is able to recover the period and the shape of a light curve (from a template given in .csv file)
    in a  given point  of the sky. Returns a dictionary with the results of Lcsampling,Lcperiod, Lcfitting

    Parameters
    ----------
    lc_filename : `str`
        CSV file containing the light curve of pulsating star. The file must
        contain nine columns -
        ['time', 'Mbol', 'u_lsst','g_lsst','r_lsst','i_lsst','z_lsst','y_lsst', 'P']
        Default uses $RUBIN_SIM_DATA_DIR/maf/pulsatingStars/RRc.csv
    do_deblend : `bool`, opt
        if True, use stellarcatalog (or default version) to add de-blending
    stellarcatalog : `str`
        Catalog from TRILEGAL lsst_sim.simdr2, containing magnitudes of the nearest stars.
        Query example shown in example notebook in rubin_sim_notebooks/maf/science/PulsatingStarRecovery
        Default is $RUBIN_SIM_DATA_DIR/maf/pulsatingStars/simdr2_270.9_-30.0.hdf
    dmod : `float`
        Distance modulus. If this is also set in the slicer, the slicer value will override.
    sigma_for_noise : `int`
        Add noise to the simulated light give with this 'sigma'
        the number of sigma used to generate the noise for the simulated light curve.
    remove_saturated : `bool`
        If True, remove observations where the saturation magnitude is above the predicted LC magnitude
    numberOfHarmonics : `int`
        defines the number of harmonics used in LcFitting
    factorForDimensionGap : `float`
        fraction of the size of the largest gap in the phase distribution
        that is used to calculate numberGaps_X (see LcSampling)
    """

    def __init__(
        self,
        lc_filename=None,
        add_blend=True,
        stellarcatalog=None,
        dmod=14.5,
        sigma_for_noise=1,
        remove_saturated=True,
        number_of_harmonics=3,
        factor_for_dimension_gap=0.5,
        seed=42,
        mjdCol="observationStartMJD",
        fiveSigmaDepthCol="fiveSigmaDepth",
        filterCol="filter",
        nightCol="night",
        visitExposureTimeCol="visitExposureTime",
        skyBrightnessCol="skyBrightness",
        numExposuresCol="numExposures",
        seeingCol="seeingFwhmEff",
        airmassCol="airmass",
        **kwargs,
    ):
        # Set up defaults if not specified
        default_dir = os.path.join(get_data_dir(), "maf/pulsatingStars")
        if lc_filename is None:
            self.lc_filename = os.path.join(default_dir, "RRc.csv")
        # Read lc_file
        self.read_lc_ascii()

        self.add_blend = add_blend
        if self.add_blend:
            if stellarcatalog is None:
                self.stellarcatalog = os.path.join(
                    default_dir, "simdr2_270.9_-30.0.hdf"
                )
            else:
                self.stellarcatalog = stellarcatalog
        else:
            self.stellarcatalog = None
        # Read stellarcatalog
        if self.stellarcatalog is not None:
            self.stellar_cat = pd.read_hdf(self.stellarcatalog)

        self.sigma_for_noise = sigma_for_noise
        self.remove_saturated = remove_saturated
        self.number_of_harmonics = number_of_harmonics
        self.factor_for_dimension_gap = factor_for_dimension_gap
        self.dmod = dmod
        # Find dust extinction values per band
        self.R_x = Dust_values().R_x

        self.mjdCol = mjdCol
        self.fiveSigmaDepthCol = fiveSigmaDepthCol
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.visitExposureTimeCol = visitExposureTimeCol
        self.skyBrightnessCol = skyBrightnessCol
        self.numExposuresCol = numExposuresCol
        self.seeingCol = seeingCol
        self.airmassCol = airmassCol
        self.saturationCol = "saturation_mag"

        cols = [
            self.mjdCol,
            self.fiveSigmaDepthCol,
            self.filtersCol,
            self.nightCol,
            self.visitExposureTimeCol,
            self.skyBrightnessCol,
            self.numExposuresCol,
            self.seeingCol,
            self.airmassCol,
            self.saturationCol,
        ]

        maps = ["DustMap"]

        # Add a random noise generator, so the metric can be repeatable
        self.rng = np.random.default_rng(seed)
        # Value for adding noise - currently always 0 (maybe not in the future?)
        self.blend_percent = 0

        super().__init__(
            col=cols,
            maps=maps,
            units="#",
            **kwargs,
            metricName="PulsatingStarRecovery",
            metricDtype="object",
        )

    def run(self, dataSlice, slicePoint):

        # If the slicer has a distance value in the slicepoint, override the self.dmod with that
        if "distance" in slicePoint:
            dmod = (5 * np.log10(slicePoint["distance"]) * 10**6) - 5
        else:
            dmod = self.dmod

        # Modify a copy of the lightcurve template to place it at the desired distance and dust extinction
        lc_model, means = self.modify_lightcurve_template(dmod, slicePoint["ebv"])

        # Generate the light curve - check that dataslice is in the correct time ordering
        dataSlice.sort(order=self.mjdCol)
        (
            lc_mags_obs,
            dmags_obs,
            snr_obs,
            times_obs,
            phase_obs,
            filters_obs,
            period_final,
        ) = self.generate_lightcurve_obs(
            dataSlice[self.mjdCol], dataSlice[self.filterCol], lc_model, means
        )

        # Analyze the uniformity of the observations, using a variety of methods
        maxGap, numberOfGaps, uniformity, uniformityKS = self.Lcsampling(
            times_obs,
            filters_obs,
            period_final,
            self.factorForDimensionGap,
        )

        # the function 'LcPeriod' analyse the periodogram with Gatspy and gives:
        # 1)the best period (best_per_temp)
        # 2)the difference between the recovered period and the  model's period(P) and
        # 3)diffper_abs=(DeltaP/P)*100
        # 4)diffcicli= DeltaP/P*1/number of cycle
        best_per_temp, diffper, diffper_abs, diffcicli = self.LcPeriodLight(
            times_obs, filters_obs, lc_mags_obs, dmags_obs, period_final
        )
        period = best_per_temp

        # The function 'LcFitting' fit the simulated light curve with number of
        # harmonics=numberOfHarmonics.
        # Return a dictionary with mean magnitudes, amplitudes and chi of the fits
        finalResult = self.LcFitting(
            LcTeoLSST_noised, index_notsaturated, period, self.numberOfHarmonics
        )

        # Some useful figure of merit on the recovery of the:
        # and shape.Difference between observed and derived mean magnitude (after fitting the light curve)
        deltamag_u = lcModel_noblend["meanu"] - finalResult["mean_u"]
        deltamag_g = lcModel_noblend["meang"] - finalResult["mean_g"]
        deltamag_r = lcModel_noblend["meanr"] - finalResult["mean_r"]
        deltamag_i = lcModel_noblend["meani"] - finalResult["mean_i"]
        deltamag_z = lcModel_noblend["meanz"] - finalResult["mean_z"]
        deltamag_y = lcModel_noblend["meany"] - finalResult["mean_y"]
        # the same can be done for the amplitudes (without the effect of blending for the momment.)
        deltaamp_u = lcModel_noblend["amplu"] - finalResult["ampl_u"]
        deltaamp_g = lcModel_noblend["amplg"] - finalResult["ampl_g"]
        deltaamp_r = lcModel_noblend["amplr"] - finalResult["ampl_r"]
        deltaamp_i = lcModel_noblend["ampli"] - finalResult["ampl_i"]
        deltaamp_z = lcModel_noblend["amplz"] - finalResult["ampl_z"]
        deltaamp_y = lcModel_noblend["amply"] - finalResult["ampl_y"]
        # Chi of the fit-->finalResult['chi_u']....

        if self.df.empty:
            output_metric = {
                "n_u": uni_meas["n_u"],
                "n_g": uni_meas["n_g"],
                "n_r": uni_meas["n_r"],
                "n_i": uni_meas["n_i"],
                "n_z": uni_meas["n_z"],
                "n_y": uni_meas["n_y"],
                "maxGap_u": uni_meas["maxGap_u"],
                "maxGap_g": uni_meas["maxGap_g"],
                "maxGap_r": uni_meas["maxGap_r"],
                "maxGap_i": uni_meas["maxGap_i"],
                "maxGap_z": uni_meas["maxGap_z"],
                "maxGap_y": uni_meas["maxGap_y"],
                "numberGaps_u": uni_meas["numberGaps_u"],
                "numberGaps_g": uni_meas["numberGaps_g"],
                "numberGaps_r": uni_meas["numberGaps_r"],
                "numberGaps_i": uni_meas["numberGaps_i"],
                "numberGaps_z": uni_meas["numberGaps_z"],
                "numberGaps_y": uni_meas["numberGaps_y"],
                "uniformity_u": uni_meas["uniformity_u"],
                "uniformity_g": uni_meas["uniformity_g"],
                "uniformity_r": uni_meas["uniformity_r"],
                "uniformity_i": uni_meas["uniformity_i"],
                "uniformity_z": uni_meas["uniformity_z"],
                "uniformity_y": uni_meas["uniformity_y"],
                "uniformityKS_u": uni_meas["uniformityKS_u"],
                "uniformityKS_g": uni_meas["uniformityKS_g"],
                "uniformityKS_r": uni_meas["uniformityKS_r"],
                "uniformityKS_i": uni_meas["uniformityKS_i"],
                "uniformityKS_z": uni_meas["uniformityKS_z"],
                "uniformityKS_y": uni_meas["uniformityKS_y"],
                "P_gatpsy": best_per_temp,
                "Delta_Period": diffper,
                "Delta_Period_abs": diffper_abs,
                "Delta_Period_abs_cicli": diffcicli,
                "deltamag_u": deltamag_u,
                "deltamag_g": deltamag_g,
                "deltamag_r": deltamag_r,
                "deltamag_i": deltamag_i,
                "deltamag_z": deltamag_z,
                "deltamag_y": deltamag_y,
                "deltaamp_u": deltaamp_u,
                "deltaamp_g": deltaamp_g,
                "deltaamp_r": deltaamp_r,
                "deltaamp_i": deltaamp_i,
                "deltaamp_z": deltaamp_z,
                "deltaamp_y": deltaamp_y,
                "chi_u": finalResult["chi_u"],
                "chi_g": finalResult["chi_g"],
                "chi_r": finalResult["chi_r"],
                "chi_i": finalResult["chi_i"],
                "chi_z": finalResult["chi_z"],
                "chi_y": finalResult["chi_y"],
            }
        else:

            # The function 'Lcsampling' analize the sampling of the simulated light curve. Give a dictionary with UniformityPrameters obtained with three different methods
            # 1) for each filter X calculates the number of points (n_X), the size in phase of the largest gap (maxGap_X) and the number of gaps largest than factorForDimensionGap*maxGap_X (numberGaps_X)
            # 2) the uniformity parameters from Barry F. Madore and Wendy L. Freedman 2005 ApJ 630 1054 (uniformity_X)  useful for n_X<20
            # 3) a modified version of UniformityMetric by Peter Yoachim (https://sims-maf.lsst.io/_modules/lsst/sims/maf/metrics/cadenceMetrics.html#UniformityMetric.run). Calculate how uniformly the observations are spaced in phase (not time)using KS test.Returns a value between 0 (uniform sampling) and 1 . uniformityKS_X

            period_model_blend = LcTeoLSST_blend["p_model"]
            uni_meas_blend = self.Lcsampling(
                LcTeoLSST_noised_blend,
                period_model_blend,
                index_notsaturated_blend,
                self.factorForDimensionGap,
            )

            # the function 'LcPeriod' analyse the periodogram with Gatspy and gives:
            # 1)the best period (best_per_temp)
            # 2)the difference between the recovered period and the  model's period(P) and
            # 3)diffper_abs=(DeltaP/P)*100
            # 4)diffcicli= DeltaP/P*1/number of cycle
            (
                best_per_temp_blend,
                diffper_blend,
                diffper_abs_blend,
                diffcicli_blend,
            ) = self.LcPeriodLight(
                mv, LcTeoLSST_blend, LcTeoLSST_noised_blend, index_notsaturated_blend
            )
            period_blend = (
                best_per_temp_blend  # or period_model or fitLS_multi.best_period
            )
            # period=LcTeoLSST['p_model']

            # The function 'LcFitting' fit the simulated light curve with number of harmonics=numberOfHarmonics.Return a dictionary with mean magnitudes, amplitudes and chi of the fits

            finalResult_blend = self.LcFitting(
                LcTeoLSST_noised_blend,
                index_notsaturated_blend,
                period_blend,
                self.numberOfHarmonics,
            )

            # Some useful figure of merit on the recovery of the:
            # and shape.Difference between observed and derived mean magnitude (after fitting the light curve)
            deltamag_u_blend = lcModel_blend["meanu"] - finalResult_blend["mean_u"]
            deltamag_g_blend = lcModel_blend["meang"] - finalResult_blend["mean_g"]
            deltamag_r_blend = lcModel_blend["meanr"] - finalResult_blend["mean_r"]
            deltamag_i_blend = lcModel_blend["meani"] - finalResult_blend["mean_i"]
            deltamag_z_blend = lcModel_blend["meanz"] - finalResult_blend["mean_z"]
            deltamag_y_blend = lcModel_blend["meany"] - finalResult_blend["mean_y"]
            # the same can be done for the amplitudes (without the effect of blending for the momment.)
            deltaamp_u_blend = lcModel_blend["amplu"] - finalResult_blend["ampl_u"]
            deltaamp_g_blend = lcModel_blend["amplg"] - finalResult_blend["ampl_g"]
            deltaamp_r_blend = lcModel_blend["amplr"] - finalResult_blend["ampl_r"]
            deltaamp_i_blend = lcModel_blend["ampli"] - finalResult_blend["ampl_i"]
            deltaamp_z_blend = lcModel_blend["amplz"] - finalResult_blend["ampl_z"]
            deltaamp_y_blend = lcModel_blend["amply"] - finalResult_blend["ampl_y"]

            output_metric = {
                "maxGap_u": uni_meas["maxGap_u"],
                "maxGap_g": uni_meas["maxGap_g"],
                "maxGap_r": uni_meas["maxGap_r"],
                "maxGap_i": uni_meas["maxGap_i"],
                "maxGap_z": uni_meas["maxGap_z"],
                "maxGap_y": uni_meas["maxGap_y"],
                "numberGaps_u": uni_meas["numberGaps_u"],
                "numberGaps_g": uni_meas["numberGaps_g"],
                "numberGaps_r": uni_meas["numberGaps_r"],
                "numberGaps_i": uni_meas["numberGaps_i"],
                "numberGaps_z": uni_meas["numberGaps_z"],
                "numberGaps_y": uni_meas["numberGaps_y"],
                "uniformity_u": uni_meas["uniformity_u"],
                "uniformity_g": uni_meas["uniformity_g"],
                "uniformity_r": uni_meas["uniformity_r"],
                "uniformity_i": uni_meas["uniformity_i"],
                "uniformity_z": uni_meas["uniformity_z"],
                "uniformity_y": uni_meas["uniformity_y"],
                "uniformityKS_u": uni_meas["uniformityKS_u"],
                "uniformityKS_g": uni_meas["uniformityKS_g"],
                "uniformityKS_r": uni_meas["uniformityKS_r"],
                "uniformityKS_i": uni_meas["uniformityKS_i"],
                "uniformityKS_z": uni_meas["uniformityKS_z"],
                "uniformityKS_y": uni_meas["uniformityKS_y"],
                "P_gatpsy": best_per_temp,
                "Delta_Period": diffper,
                "Delta_Period_abs": diffper_abs,
                "Delta_Period_abs_cicli": diffcicli,
                "deltamag_u": deltamag_u,
                "deltamag_g": deltamag_g,
                "deltamag_r": deltamag_r,
                "deltamag_i": deltamag_i,
                "deltamag_z": deltamag_z,
                "deltamag_y": deltamag_y,
                "deltaamp_u": deltaamp_u,
                "deltaamp_g": deltaamp_g,
                "deltaamp_r": deltaamp_r,
                "deltaamp_i": deltaamp_i,
                "deltaamp_z": deltaamp_z,
                "deltaamp_y": deltaamp_y,
                "chi_u": finalResult["chi_u"],
                "chi_g": finalResult["chi_g"],
                "chi_r": finalResult["chi_r"],
                "chi_i": finalResult["chi_i"],
                "chi_z": finalResult["chi_z"],
                "chi_y": finalResult["chi_y"],
                "P_gatpsy_blend": best_per_temp_blend,
                "Delta_Period_blend": diffper_blend,
                "Delta_Period_abs_blend": diffper_abs_blend,
                "Delta_Period_abs_cicli_blend": diffcicli_blend,
                "deltamag_u_blend": deltamag_u_blend,
                "deltamag_g_blend": deltamag_g_blend,
                "deltamag_r_blend": deltamag_r_blend,
                "deltamag_i_blend": deltamag_i_blend,
                "deltamag_z_blend": deltamag_z_blend,
                "deltamag_y_blend": deltamag_y_blend,
                "deltaamp_u_blend": deltaamp_u_blend,
                "deltaamp_g_blend": deltaamp_g_blend,
                "deltaamp_r_blend": deltaamp_r_blend,
                "deltaamp_i_blend": deltaamp_i_blend,
                "deltaamp_z_blend": deltaamp_z_blend,
                "deltaamp_y_blend": deltaamp_y_blend,
                "chi_u_blend": finalResult_blend["chi_u"],
                "chi_g_blend": finalResult_blend["chi_g"],
                "chi_r_blend": finalResult_blend["chi_r"],
                "chi_i_blend": finalResult_blend["chi_i"],
                "chi_z_blend": finalResult_blend["chi_z"],
                "chi_y_blend": finalResult_blend["chi_y"],
            }
        return output_metric

    def reduceP_gatpsy(self, metricValue):
        return metricValue["P_gatpsy"]

    def get_deltamag_u(self, metricValue):
        return metricValue["deltamag_u"]

    def get_deltaamp_u(self, metricValue):
        return metricValue["deltaamp_u"]

    def get_chi_u(self, metricValue):
        return metricValue["chi_u"]

    def get_deltamag_g(self, metricValue):
        return metricValue["deltamag_g"]

    def get_deltaamp_g(self, metricValue):
        return metricValue["deltaamp_g"]

    def get_chi_g(self, metricValue):
        return metricValue["chi_g"]

    def get_deltamag_r(self, metricValue):
        return metricValue["deltamag_r"]

    def get_deltaamp_r(self, metricValue):
        return metricValue["deltaamp_r"]

    def get_chi_r(self, metricValue):
        return metricValue["chi_r"]

    def get_deltamag_i(self, metricValue):
        return metricValue["deltamag_i"]

    def get_deltaamp_i(self, metricValue):
        return metricValue["deltaamp_i"]

    def get_chi_i(self, metricValue):
        return metricValue["chi_i"]

    def get_deltamag_z(self, metricValue):
        return metricValue["deltamag_z"]

    def get_deltaamp_z(self, metricValue):
        return metricValue["deltaamp_z"]

    def get_chi_z(self, metricValue):
        return metricValue["chi_z"]

    def get_deltamag_y(self, metricValue):
        return metricValue["deltamag_y"]

    def get_deltaamp_y(self, metricValue):
        return metricValue["deltaamp_y"]

    def get_chi_y(self, metricValue):
        return metricValue["chi_y"]

    def get_P_gatpsy_blend(self, metricValue):
        return metricValue["P_gatpsy_blend"]

    def get_deltamag_u_blend(self, metricValue):
        return metricValue["deltamag_u_blend"]

    def get_deltaamp_u_blend(self, metricValue):
        return metricValue["deltaamp_u_blend"]

    def get_chi_u_blend(self, metricValue):
        return metricValue["chi_u_blend"]

    def get_deltamag_g_blend(self, metricValue):
        return metricValue["deltamag_g_blend"]

    def get_deltaamp_g_blend(self, metricValue):
        return metricValue["deltaamp_g_blend"]

    def get_chi_g_blend(self, metricValue):
        return metricValue["chi_g_blend"]

    def get_deltamag_r_blend(self, metricValue):
        return metricValue["deltamag_r_blend"]

    def get_deltaamp_r_blend(self, metricValue):
        return metricValue["deltaamp_r_blend"]

    def get_chi_r_blend(self, metricValue):
        return metricValue["chi_r_blend"]

    def get_deltamag_i_blend(self, metricValue):
        return metricValue["deltamag_i_blend"]

    def get_deltaamp_i_blend(self, metricValue):
        return metricValue["deltaamp_i_blend"]

    def get_chi_i_blend(self, metricValue):
        return metricValue["chi_i_blend"]

    def get_deltamag_z_blend(self, metricValue):
        return metricValue["deltamag_z_blend"]

    def get_deltaamp_z_blend(self, metricValue):
        return metricValue["deltaamp_z_blend"]

    def get_chi_z_blend(self, metricValue):
        return metricValue["chi_z_blend"]

    def get_deltamag_y_blend(self, metricValue):
        return metricValue["deltamag_y_blend"]

    def get_deltaamp_y_blend(self, metricValue):
        return metricValue["deltaamp_y_blend"]

    def get_chi_y_blend(self, metricValue):
        return metricValue["chi_y_blend"]

    def read_lc_ascii(self):
        """
        Reads in an ascii CSV file the light curve of the pulsating stars that we want simulate.
        Expecting to find the following columns:
        `time,Mbol,u_lsst,g_lsst,r_lsst,i_lsst,z_lsst,y_lsst,P`
        Time should be in units of days,
        the per-band brightness values in magnitudes,
        period is units of days.
        """
        lc = pd.read_csv(self.lc_filename)
        # Modify period to seconds
        lc["period"] = lc["P"] * 24 * 60 * 60
        lc = lc.drop("P", axis=1)
        # Add a phased version of the time
        lc["phase"] = ((lc["time"] - lc["time"][0]) / lc["period"]) % 1
        # Rename the columns
        name_mapper = {f"{f}_lsst": f"{f}" for f in "ugrizy"}
        lc.rename(columns=name_mapper, inplace=True)
        self.lc_model = lc
        return

    def modify_lightcurve_template(self, dmod, ebv):
        """
        Add distance modulus and dust extinction to the magnitudes in the lc_model.
        If the stellar catalog is provided, take into account blends caused by nearby stars.

        Parameters
        -----------
        dmod: float
            distance modulus
        ebv: float
            E(B-V)=slicePoint('ebv')

        """
        # 'self' contains lc_model (pd.DataFrame with lightcurve template)
        # 'self' also has R_x per band
        lc_model = copy.deepcopy(self.lc_model)

        # Add distance modulus and dust extinction
        for f in "ugrizy":
            lc_model[f] = lc_model[f] + dmod + self.R_x[f] * 3.1 * ebv

        # Set up to calculate the intensity means
        if self.add_blend:
            flux_blend = {}
            for f in "ugrizy":
                flux_blend[f] = mag_antilog(self.stellar_cat[f"{f}mag"])

        # Compute the intensity means
        means = {}
        for f in "ugrizy":
            if self.add_blend:
                model_flux = mag_antilog(lc_model[f]) + sum(flux_blend[f])
            else:
                model_flux = mag_antilog(lc_model[f])
            means[f"mean_flux_{f}"] = np.mean(model_flux)
            mags = -2.5 * np.log10(model_flux)
            means[f"mean_{f}"] = meanmag_antilog(mags)
            means[f"amp_{f}"] = max(mags) - min(mags)

        # Add a repeat of the first value, at the end of the template
        repeat = pd.DataFrame(lc_model.iloc[0]).T
        repeat["phase"] = 1
        lc_model = pd.concat([lc_model, repeat]).reset_index(drop=True)
        return lc_model, means

    def generate_lightcurve_obs(
        self,
        times,
        filters,
        m5_mags,
        saturation_mags,
        lc_model,
        means,
        period_true=None,
        ampl_true=1,
        do_normalize=False,
    ):
        """
        Generate the observed temporal series and light curve from template  and opsim

        Parameters
        -----------
        time : `np.ndarray`
        filters : `np.ndarray`
        lc_model : `pd.Dataframe`
        period_true : `float`, opt
        ampl_true : `float`, opt
        do_normalize : `bool`, opt
            Default False

        """
        # Choose the desired period (default is to use period from lightcurve)
        if period_true is None:
            # Convert the period in the lightcurve model template to days
            period_final = (lc_model["period"].iloc[0]) / 24 / 60 / 60
        else:
            period_final = period_true
        # The amplitude can be increased by a factor of ampl_true

        # Can normalize the lc model, subtracting the mean magnitude (will be added back later), and
        # dividing by the amplitude in g band (will not be muliplied by in, unless ampl_true != 1
        if do_normalize:
            mean_mags = {}
            for f in "ugrizy":
                # Normalize by subtracting mean (new mean = 0) and making amplitude in g = 1
                lc_model[f] = (lc_model[f] - means[f"mean_{f}"]) / means["amp_g"]
                # And then keep a record of the mean_mags, so we add them back to the LC later
                mean_mags = means[f"mean_{f}"]
        else:
            # If did not normalize, then do not need to add offsets back in later.
            mean_mags = {}
            for f in "ugrizy":
                mean_mags[f] = 0

        # Create interpolators for each bandpass using the PHASE and MAGNITUDE
        interp = {}
        for f in "ugrizy":
            interp[f] = interp1d(lc_model["phase"], lc_model[f])

        # Calculate the expected magnitude in each visit
        t_time_0 = np.min(times)
        lc_mags_obs = np.zeros(len(times), float)
        phase_obs = np.zeros(len(times), float)
        for f in "ugrizy":
            match = np.where(filters == f)[0]
            # Calculate the expected phase values for these observations
            phase_obs[match] = ((times[match] - t_time_0) / period_final) % 1.0
            lc_mags_obs[match] = mean_mags[f] + ampl_true * interp[f](phase_obs[match])

        # Calculate SNR for each observation (this just uses m5 from each visit)
        snr_obs = m52snr(lc_mags_obs, m5_mags)
        # Calculate noise, using self.sigma_for_noise to scale level of noise to be added
        # (adds uniform distribution of noise from -sigma to sigma)
        dmags_obs = 2.5 * np.log10(1.0 + 1.0 / snr_obs)  # dmag_obs ~ uncertainty
        if self.blend_percent > 0:
            dmags_obs = np.sqrt(2) * dmags_obs
        noise = (
            self.rng.random.uniform(-self.sigma_for_noise, self.sigma_for_noise)
            * dmags_obs
        )
        lc_mags_obs += noise

        # Remove saturated observations
        if self.remove_saturated:
            # Sometimes the saturation_mag could be np.nan -- entire image saturated
            not_sat = np.where(lc_mags_obs < saturation_mags, False, True)
            lc_mags_obs = lc_mags_obs[not_sat]
            snr_obs = snr_obs[not_sat]
            times_obs = times[not_sat]
            filters_obs = filters[not_sat]
            phase_obs = phase_obs[not_sat]

        return (
            lc_mags_obs,
            dmags_obs,
            snr_obs,
            times_obs,
            phase_obs,
            filters_obs,
            period_final,
        )

    def Lcsampling(self, times_obs, filters_obs, period, factor1):
        """
        Analyse the sampling of the simulated light curve (with the period=period_model)
        Returns a dictionary with UniformityParameters obtained with three different methods
        1) for each filter X calculates the number of points (n_X),
        the size in phase of the largest gap (maxGap_X)
        and the number of gaps largest than factorForDimensionGap*maxGap_X (numberGaps_X)
        2) the uniformity parameters from Barry F. Madore and Wendy L. Freedman 2005 ApJ 630 1054
        (uniformity_X)  useful for n_X<20
        3) a modified version of UniformityMetric by Peter Yoachim
        (https://sims-maf.lsst.io/_modules/lsst/sims/maf/metrics/cadenceMetrics.html#UniformityMetric.run).
        Calculate how uniformly the observations are spaced in phase (not time)using KS test.
        Returns a value between 0 (uniform sampling) and 1 . uniformityKS_X

        Parameters:
        -----------
        data:
        period:
        index:
        factor1:
            factorForDimensionGap
        """
        maxGap = {}
        numberOfGaps = {}
        uniformity = {}
        uniformityKS = {}
        for f in "ugrizy":
            match = np.where(filters_obs == f)
            maxGap[f], numberOfGaps[f] = self.qualityCheck(
                times_obs[match], period, factor1
            )
            uniformity[f] = self.qualityCheck2(times_obs[match], period)
            uniformityKS[f] = self.qualityCheck3(times_obs[match], period)

        return maxGap, numberOfGaps, uniformity, uniformityKS

    def qualityCheck(self, time, period, factor1):
        if (len(time)) > 0:
            # period=param[0]
            phase = ((time - time[0]) / period) % 1
            indexSorted = np.argsort(phase)

            distances = []
            indexStart = []
            indexStop = []
            leftDistance = phase[indexSorted[0]]
            rightDistance = 1 - phase[indexSorted[len(indexSorted) - 1]]
            for i in range(len(phase) - 1):
                dist = phase[indexSorted[i + 1]] - phase[indexSorted[i]]
                distances.append(dist)

            # factor=sum(distances)/len(distances)*factor1
            distancesTotal = distances
            distancesTotal.append(leftDistance)
            distancesTotal.append(rightDistance)
            # factor=sum(distancesTotal)/len(distancesTotal)*factor1
            maxDistance = max(distancesTotal)
            factor = maxDistance * factor1
            for i in range(len(phase) - 1):
                dist = phase[indexSorted[i + 1]] - phase[indexSorted[i]]
                distances.append(dist)
                if dist > factor:
                    indexStart.append(indexSorted[i])
                    indexStop.append(indexSorted[i + 1])
            a = len(indexStart)
        else:
            maxDistance = 999.0
            a = 999.0
        return maxDistance, a

    def qualityCheck2(self, time, period):
        # This is based on Madore and Freedman (Apj 2005), uniformity definition
        if (len(time)) <= 20 and (len(time)) > 1:
            phase = ((time - time[0]) / period) % 1
            indexSorted = np.argsort(phase)

            distances = []
            leftDistance = phase[indexSorted[0]]
            rightDistance = 1 - phase[indexSorted[len(indexSorted) - 1]]
            sumDistances = 0
            for i in range(len(phase) - 1):
                dist = phase[indexSorted[i + 1]] - phase[indexSorted[i]]
                sumDistances = sumDistances + pow(dist, 2)
                distances.append(dist)

            distancesTotal = distances
            distancesTotal.append(leftDistance)
            distancesTotal.append(rightDistance)

            # uniformity parameter
            u = len(time) / (len(time) - 1) * (1 - sumDistances)
        else:
            u = 999.0
        return u

    def qualityCheck3(self, time, period):
        # This is based on how a KS-test works: look at the cumulative distribution of observation dates,
        #    and compare to a perfectly uniform cumulative distribution.
        #    Perfectly uniform observations = 0, perfectly non-uniform = 1.
        if (len(time)) > 1:
            phase = ((time - time[0]) / period) % 1
            phase_sort = np.sort(phase)
            n_cum = np.arange(1, len(phase) + 1) / float(len(phase))
            D_max = np.max(np.abs(n_cum - phase_sort - phase_sort[0]))
        else:
            D_max = 999.0
        return D_max

    def LcPeriodLight(
        self, times_obs, filters_obs, lc_mags_obs, dmags_obs, period_model
    ):
        """
        Compute the period using Gatpsy and return differences with the period of the model.
        """
        #########################################################################

        minper_opt = period_model - 0.9 * period_model
        maxper_opt = period_model + 0.9 * period_model
        periods = np.linspace(minper_opt, maxper_opt, 1000)

        #########This is to measure the noise of the periodogramm but is not used yet

        LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=0)
        LS_multi.fit(times_obs, lc_mags_obs, dmags_obs, filters_obs)

        P_multi = LS_multi.periodogram(periods)
        periodogram_noise = np.median(P_multi)
        periodogram_noise_mean = np.mean(P_multi)

        # print("Noise level (median vs mean)")
        # print(periodogram_noise, periodogram_noise_mean)

        ########This is to measure the best period
        fitLS_multi = periodic.LombScargleMultiband(fit_period=True)
        fitLS_multi.optimizer.period_range = (minper_opt, maxper_opt)
        fitLS_multi.fit(times_obs, lc_mags_obs, dmags_obs, filters_obs)
        best_per_temp = fitLS_multi.best_period

        tmin = np.min(times_obs)
        tmax = np.max(times_obs)
        cicli = (tmax - tmin) / period_model

        diffper = best_per_temp - period_model
        diffper_abs = abs(best_per_temp - period_model) / period_model * 100
        diffcicli = abs(best_per_temp - period_model) / period_model * 1 / cicli

        return best_per_temp, diffper, diffper_abs, diffcicli

    def LcFitting(self, data, index, period, numberOfHarmonics):
        """
        Fit of the light curve and gives a dictionary with the mean amplitudes and magnitudes of the
        fitting curve
        """

        fitting = self.computingLcModel(data, period, numberOfHarmonics, index)
        # timeForModel=np.arange(data['timeu'][0],data['timeu'][0]+2*period,0.01)
        # computing the magModelFromFit
        if len(fitting["u"]) > 1:
            timeForModel = np.arange(
                data["timeu"][0], data["timeu"][0] + 2 * period, 0.01
            )
            magModelFromFit_u = self.modelToFit(timeForModel, fitting["u"])
            ampl_u = max(magModelFromFit_u) - min(magModelFromFit_u)
        else:
            magModelFromFit_u = [9999.0]
            ampl_u = 9999.0
        # timeForModel=np.arange(data['timeg'][0],data['timeg'][0]+2*period,0.01)
        if len(fitting["g"]) > 1:
            timeForModel = np.arange(
                data["timeg"][0], data["timeg"][0] + 2 * period, 0.01
            )
            # magModelFromFit_g=self.modelToFit(data['timeg'],fitting['g'])
            magModelFromFit_g = self.modelToFit(timeForModel, fitting["g"])
            ampl_g = max(magModelFromFit_g) - min(magModelFromFit_g)
        else:
            magModelFromFit_g = [9999.0]
            ampl_g = 9999.0
        # timeForModel=np.arange(data['timer'][0],data['timer'][0]+2*period,0.01)
        if len(fitting["r"]) > 1:
            timeForModel = np.arange(
                data["timer"][0], data["timer"][0] + 2 * period, 0.01
            )
            # magModelFromFit_r=modelToFit(data['timer'],fitting['r'])
            magModelFromFit_r = self.modelToFit(timeForModel, fitting["r"])
            ampl_r = max(magModelFromFit_r) - min(magModelFromFit_r)
        else:
            magModelFromFit_r = [9999.0]
            ampl_r = 9999.0
        # timeForModel=np.arange(data['timei'][0],data['timei'][0]+2*period,0.01)

        if len(fitting["i"]) > 1:
            timeForModel = np.arange(
                data["timei"][0], data["timei"][0] + 2 * period, 0.01
            )
            magModelFromFit_i = self.modelToFit(timeForModel, fitting["i"])

            # if len(magModelFromFit_i)>0:
            ampl_i = max(magModelFromFit_i) - min(magModelFromFit_i)
            # else:
            # ampl_i=9999.
        else:
            magModelFromFit_i = [9999.0]
            ampl_i = 9999.0
        # timeForModel=np.arange(data['timez'][0],data['timez'][0]+2*period,0.01)
        if len(fitting["z"]) > 1:
            timeForModel = np.arange(
                data["timez"][0], data["timez"][0] + 2 * period, 0.01
            )
            magModelFromFit_z = self.modelToFit(timeForModel, fitting["z"])
            ampl_z = max(magModelFromFit_z) - min(magModelFromFit_z)
        else:
            magModelFromFit_z = [9999.0]
            ampl_z = 9999.0
        # timeForModel=np.arange(data['timey'][0],data['timey'][0]+2*period,0.01)
        if len(fitting["y"]) > 1:
            timeForModel = np.arange(
                data["timey"][0], data["timey"][0] + 2 * period, 0.01
            )
            magModelFromFit_y = self.modelToFit(timeForModel, fitting["y"])
            ampl_y = max(magModelFromFit_y) - min(magModelFromFit_y)
        else:
            magModelFromFit_y = [9999.0]
            ampl_y = 9999.0

        meanMag_u = self.meanmag_antilog(magModelFromFit_u)
        meanMag_g = self.meanmag_antilog(magModelFromFit_g)
        meanMag_r = self.meanmag_antilog(magModelFromFit_r)
        meanMag_i = self.meanmag_antilog(magModelFromFit_i)
        meanMag_z = self.meanmag_antilog(magModelFromFit_z)
        meanMag_y = self.meanmag_antilog(magModelFromFit_y)

        finalResult = {
            "mean_u": meanMag_u,
            "mean_g": meanMag_g,
            "mean_r": meanMag_r,
            "mean_i": meanMag_i,
            "mean_z": meanMag_z,
            "mean_y": meanMag_y,
            "ampl_u": ampl_u,
            "ampl_g": ampl_g,
            "ampl_r": ampl_r,
            "ampl_i": ampl_i,
            "ampl_z": ampl_z,
            "ampl_y": ampl_y,
            "chi_u": fitting["chi_u"],
            "chi_g": fitting["chi_g"],
            "chi_r": fitting["chi_r"],
            "chi_i": fitting["chi_i"],
            "chi_z": fitting["chi_z"],
            "chi_y": fitting["chi_y"],
            "fittingParametersAllband": fitting,
        }

        return finalResult

    def modelToFit(self, time, param):
        # time in days
        magModel = []
        amplitudes = []
        phases = []
        numberOfHarmonics = int((len(param) - 2) / 2)

        zp = param[1]
        period = param[0]
        for i in range(numberOfHarmonics):
            amplitudes.append(param[2 + i])
            phases.append(param[numberOfHarmonics + 2 + i])
        for i in range(len(time)):
            y = zp
            for j in range(0, int(numberOfHarmonics)):
                y = y + amplitudes[j] * np.cos(
                    (2 * np.pi / period) * (j + 1) * (time[i]) + phases[j]
                )
            magModel.append(y)
        return magModel

    def chisqr(self, residual, Ndat, Nvariable):
        chi = sum(pow(residual, 2)) / (Ndat - Nvariable)
        return chi

    def chisqr2(self, datax, datay, fitparameters, Ndat, Nvariable):
        residuals = self.modelToFit(datax, fitparameters) - datay
        chi2 = (
            sum(pow(residuals, 2) / self.modelToFit(datax, fitparameters))
            * 1
            / (Ndat - Nvariable)
        )
        return chi2

    def residuals(self, datax, datay, fitparameters):
        residuals = self.modelToFit(datax, fitparameters) - datay
        return residuals

    def computingLcModel(self, data, period, numberOfHarmonics, index):
        def modelToFit2_fit(coeff):
            fit = self.modelToFit(x, coeff)
            return fit - y_proc

        time_u = data["timeu"][index["ind_notsaturated_u"]]  # time must be in days
        time_g = data["timeg"][index["ind_notsaturated_g"]]
        time_r = data["timer"][index["ind_notsaturated_r"]]
        time_i = data["timei"][index["ind_notsaturated_i"]]
        time_z = data["timez"][index["ind_notsaturated_z"]]
        time_y = data["timey"][index["ind_notsaturated_y"]]
        mag_u = data["magu"][index["ind_notsaturated_u"]]
        mag_g = data["magg"][index["ind_notsaturated_g"]]
        mag_r = data["magr"][index["ind_notsaturated_r"]]
        mag_i = data["magi"][index["ind_notsaturated_i"]]
        mag_z = data["magz"][index["ind_notsaturated_z"]]
        mag_y = data["magy"][index["ind_notsaturated_y"]]

        parametersForLcFit = [period, 1]  # period,zp
        for i in range(numberOfHarmonics):
            parametersForLcFit.append(1)  # added ampl
            parametersForLcFit.append(1)  # added phase
        x = time_u

        y_proc = np.copy(mag_u)
        if len(y_proc) > (numberOfHarmonics * 2) + 2:
            print("fitting u band")
            fit_u, a = leastsq(modelToFit2_fit, parametersForLcFit)
            residual = self.residuals(x, y_proc, fit_u)
            chi_u = self.chisqr2(x, y_proc, fit_u, len(x), len(fit_u))
        else:
            fit_u = [9999.0]
            chi_u = 9999.0
        x = time_g
        y_proc = np.copy(mag_g)
        if len(y_proc) > (numberOfHarmonics * 2) + 2:
            print("fitting g band")
            fit_g, a = leastsq(modelToFit2_fit, parametersForLcFit)
            residual = self.residuals(x, y_proc, fit_g)
            chi_g = self.chisqr2(x, y_proc, fit_g, len(x), len(fit_g))
        else:
            fit_g = [9999.0]
            chi_g = 9999.0
        y_proc = np.copy(mag_r)
        x = time_r
        if len(y_proc) > (numberOfHarmonics * 2) + 2:
            print("fitting r band")
            fit_r, a = leastsq(modelToFit2_fit, parametersForLcFit)
            residual = self.residuals(x, y_proc, fit_r)
            chi_r = self.chisqr2(x, y_proc, fit_r, len(x), len(fit_r))
        else:
            fit_r = [9999.0]
            chi_r = 9999.0
        x = time_i
        y_proc = np.copy(mag_i)
        if len(y_proc) > (numberOfHarmonics * 2) + 2:
            print("fitting i band")
            fit_i, a = leastsq(modelToFit2_fit, parametersForLcFit)
            residual = self.residuals(x, y_proc, fit_i)
            chi_i = self.chisqr2(x, y_proc, fit_i, len(x), len(fit_i))
        else:
            fit_i = [9999.0]
            chi_i = 9999.0
        x = time_z
        y_proc = np.copy(mag_z)
        if len(y_proc) > (numberOfHarmonics * 2) + 2:
            print("fitting z band")
            fit_z, a = leastsq(modelToFit2_fit, parametersForLcFit)
            residual = self.residuals(x, y_proc, fit_z)
            chi_z = self.chisqr2(x, y_proc, fit_z, len(x), len(fit_z))
        else:
            fit_z = [9999.0]
            chi_z = 9999.0
        x = time_y
        y_proc = np.copy(mag_y)
        if len(y_proc) > (numberOfHarmonics * 2) + 2:
            print("fitting y band")
            fit_y, a = leastsq(modelToFit2_fit, parametersForLcFit)
            residual = self.residuals(x, y_proc, fit_y)
            chi_y = self.chisqr2(x, y_proc, fit_y, len(x), len(fit_y))
        else:
            fit_y = [9999.0]
            chi_y = 9999.0

        results = {
            "u": fit_u,
            "g": fit_g,
            "r": fit_r,
            "i": fit_i,
            "z": fit_z,
            "y": fit_y,
            "chi_u": chi_u,
            "chi_g": chi_g,
            "chi_r": chi_r,
            "chi_i": chi_i,
            "chi_z": chi_z,
            "chi_y": chi_y,
        }

        return results
