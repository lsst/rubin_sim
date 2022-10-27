import numpy as np
from scipy.interpolate import interp1d
from rubin_sim.maf.metrics import BaseMetric
import healpy as hp

# Modifying from Knut Olson's fork at:
# https://github.com/knutago/sims_maf_contrib/blob/master/tutorials/CrowdingMetric.ipynb

__all__ = ["CrowdingM5Metric", "CrowdingMagUncertMetric", "NstarsMetric"]


def _comp_crowd_error(mag_vector, lum_func, seeing, single_mag=None):
    """
    Compute the photometric crowding error given the luminosity function and best seeing.

    Parameters
    ----------
    mag_vector : np.array
        Stellar magnitudes.
    lum_func : np.array
        Stellar luminosity function.
    seeing : float
        The best seeing conditions. Assuming forced-photometry can use the best seeing conditions
        to help with confusion errors.
    single_mag : float (None)
        If single_mag is None, the crowding error is calculated for each mag in mag_vector. If
        single_mag is a float, the crowding error is interpolated to that single value.

    Returns
    -------
    np.array
        Magnitude uncertainties.

    Equation from Olsen, Blum, & Rigaut 2003, AJ, 126, 452
    """
    lum_area_arcsec = 3600.0**2
    lum_vector = 10 ** (-0.4 * mag_vector)
    coeff = np.sqrt(np.pi / lum_area_arcsec) * seeing / 2.0
    my_int = (np.add.accumulate((lum_vector**2 * lum_func)[::-1]))[::-1]
    temp = np.sqrt(my_int) / lum_vector
    if single_mag is not None:
        interp = interp1d(mag_vector, temp)
        temp = interp(single_mag)
    crowd_error = coeff * temp
    return crowd_error


class CrowdingM5Metric(BaseMetric):
    """Return the magnitude at which the photometric error exceeds crowding_error threshold."""

    def __init__(
        self,
        crowding_error=0.1,
        filtername="r",
        seeing_col="seeingFwhmGeom",
        metric_name=None,
        maps=["StellarDensityMap"],
        **kwargs,
    ):
        """
        Parameters
        ----------
        crowding_error : float, optional
            The magnitude uncertainty from crowding in magnitudes. Default 0.1 mags.
        filtername: str, optional
            The bandpass in which to calculate the crowding limit. Default r.
        seeing_col : str, optional
            The name of the seeing column.
        m5Col : str, optional
            The name of the m5 depth column.
        maps : list of str, optional
            Names of maps required for the metric.

        Returns
        -------
        float
        The magnitude of a star which has a photometric error of `crowding_error`
        """

        cols = [seeing_col]
        units = "mag"
        self.crowding_error = crowding_error
        self.filtername = filtername
        self.seeing_col = seeing_col
        if metric_name is None:
            metric_name = "Crowding to Precision %.2f" % (crowding_error)
        super().__init__(
            col=cols, maps=maps, units=units, metric_name=metric_name, **kwargs
        )

    def run(self, data_slice, slice_point=None):
        # Set mag_vector to the same length as starLumFunc (lower edge of mag bins)
        mag_vector = slice_point[f"starMapBins_{self.filtername}"][1:]
        # Pull up density of stars at this point in the sky
        lum_func = slice_point[f"starLumFunc_{self.filtername}"]
        # Calculate the crowding error using the best seeing value (in any filter?)
        crowd_error = _comp_crowd_error(
            mag_vector, lum_func, seeing=min(data_slice[self.seeing_col])
        )
        # Locate at which point crowding error is greater than user-defined limit
        above_crowd = np.where(crowd_error >= self.crowding_error)[0]

        if np.size(above_crowd) == 0:
            result = max(mag_vector)
        else:
            crowd_mag = mag_vector[max(above_crowd[0] - 1, 0)]
            result = crowd_mag

        return result


class NstarsMetric(BaseMetric):
    """Return the number of stars visible above some uncertainty limit,
    taking image depth and crowding into account.
    """

    def __init__(
        self,
        crowding_error=0.1,
        filtername="r",
        seeing_col="seeingFwhmGeom",
        m5_col="fiveSigmaDepth",
        metric_name=None,
        maps=["StellarDensityMap"],
        ignore_crowding=False,
        **kwargs,
    ):
        """
        Parameters
        ----------
        crowding_error : float, optional
            The magnitude uncertainty from crowding in magnitudes. Default 0.1 mags.
        filtername: str, optional
            The bandpass in which to calculate the crowding limit. Default r.
        seeing_col : str, optional
            The name of the seeing column.
        m5_col : str, optional
            The name of the m5 depth column.
        maps : list of str, optional
            Names of maps required for the metric.
        ignore_crowding : bool (False)
            Ignore the cowding limit.

        Returns
        -------
        float
        The number of stars above the error limit
        """

        cols = [seeing_col, m5_col]
        units = "N stars"
        self.crowding_error = crowding_error
        self.m5_col = m5_col
        self.filtername = filtername
        self.seeing_col = seeing_col
        self.ignore_crowding = ignore_crowding
        if metric_name is None:
            metric_name = "N stars to Precision %.2f" % (crowding_error)
        super().__init__(
            col=cols, maps=maps, units=units, metric_name=metric_name, **kwargs
        )

    def run(self, data_slice, slice_point=None):

        pix_area = hp.nside2pixarea(slice_point["nside"], degrees=True)
        # Set mag_vector to the same length as starLumFunc (lower edge of mag bins)
        mag_vector = slice_point[f"starMapBins_{self.filtername}"][1:]
        # Pull up density of stars at this point in the sky
        lum_func = slice_point[f"starLumFunc_{self.filtername}"]
        # Calculate the crowding error using the best seeing value (in any filter?)
        crowd_error = _comp_crowd_error(
            mag_vector, lum_func, seeing=min(data_slice[self.seeing_col])
        )
        # Locate at which point crowding error is greater than user-defined limit
        above_crowd = np.where(crowd_error >= self.crowding_error)[0]

        if np.size(above_crowd) == 0:
            crowd_mag = max(mag_vector)
        else:
            crowd_mag = mag_vector[max(above_crowd[0] - 1, 0)]

        # Compute the coadded depth, and the mag where that depth hits the error specified
        coadded_depth = 1.25 * np.log10(np.sum(10.0 ** (0.8 * data_slice[self.m5_col])))
        mag_limit = (
            -2.5 * np.log10(1.0 / (self.crowding_error * (1.09 * 5))) + coadded_depth
        )

        # Use the shallower depth, crowding or coadded
        if self.ignore_crowding:
            min_mag = mag_limit
        else:
            min_mag = np.min([crowd_mag, mag_limit])

        # Interpolate to the number of stars
        result = (
            np.interp(
                min_mag,
                slice_point[f"starMapBins_{self.filtername}"][1:],
                slice_point[f"starLumFunc_{self.filtername}"],
            )
            * pix_area
        )

        return result


class CrowdingMagUncertMetric(BaseMetric):
    """
    Given a stellar magnitude, calculate the mean uncertainty on the magnitude from crowding.
    """

    def __init__(
        self,
        rmag=20.0,
        seeing_col="seeingFwhmGeom",
        units="mag",
        metric_name=None,
        filtername="r",
        maps=["StellarDensityMap"],
        **kwargs,
    ):
        """
        Parameters
        ----------
        rmag : float
            The magnitude of the star to consider.

        Returns
        -------
        float
            The uncertainty in magnitudes caused by crowding for a star of rmag.
        """

        self.filtername = filtername
        self.seeing_col = seeing_col
        self.rmag = rmag
        if metric_name is None:
            metric_name = "CrowdingError at %.2f" % (rmag)
        super().__init__(
            col=[seeing_col], maps=maps, units=units, metric_name=metric_name, **kwargs
        )

    def run(self, data_slice, slice_point=None):
        mag_vector = slice_point[f"starMapBins_{self.filtername}"][1:]
        lum_func = slice_point[f"starLumFunc_{self.filtername}"]
        # Magnitude uncertainty given crowding
        dmag_crowd = _comp_crowd_error(
            mag_vector, lum_func, data_slice[self.seeing_col], single_mag=self.rmag
        )
        result = np.mean(dmag_crowd)
        return result
