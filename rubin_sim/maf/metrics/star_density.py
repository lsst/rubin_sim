__all__ = ("StarDensityMetric",)

import warnings

from scipy.interpolate import interp1d

from .base_metric import BaseMetric


class StarDensityMetric(BaseMetric):
    """Interpolate the stellar luminosity function to return the number of
    stars per square arcsecond brighter than the mag_limit.
    Note that the map is built from CatSim stars in the range 20 < r < 28.
    mag_limit values outside that the range of the map's starMapBins will
    return self.badval

    The stellar density maps are available in any bandpass, but bandpasses
    other than r band must use a pre-configured StellarDensityMap (not just the
    default). In other words, when setting up the metric bundle for an i-band
    stellar density using (as an example) a HealpixSlicer:
    ```
    map = maf.StellarDensityMap(filtername='i')
    metric = maf.StarDensityMetric(filtername='i', mag_limit=25.0)
    slicer = maf.HealpixSlicer()
    bundle = maf.MetricBundle(metric, slicer, "", mapsList=[map])
    ```

    Parameters
    ----------
    mag_limit : `float`, opt
        Magnitude limit at which to evaluate the stellar luminosity function.
        Returns number of stars per square arcsecond brighter than this limit.
        Default 25.
    filtername : `str`, opt
        Which filter to evaluate the luminosity function in; Note that using
        bands other than r will require setting up a custom (rather than
        default) version of the stellar density map.
        Default r.
    units : `str`, opt
        Units for the output values. Default "stars/sq arcsec".
    maps : `list` of `str`, opt
        Names for the maps required. Default "StellarDensityMap".

    Returns
    -------
    result : `float`
        Number of stars brighter than mag_limit in filtername, based on the
        stellar density map.
    """

    def __init__(
        self, mag_limit=25.0, filtername="r", units="stars/sq arcsec", maps=["StellarDensityMap"], **kwargs
    ):
        super(StarDensityMetric, self).__init__(col=[], maps=maps, units=units, **kwargs)
        self.mag_limit = mag_limit
        if "rmagLimit" in kwargs:
            warnings.warn(
                "rmagLimit is deprecated; please use mag_limit instead "
                "(will use the provided rmagLimit for now)."
            )
            self.mag_limit = kwargs["rmagLimit"]
        self.filtername = filtername

    def run(self, data_slice, slice_point=None):
        # Interpolate the data to the requested mag
        interp = interp1d(
            slice_point["starMapBins_%s" % self.filtername][1:],
            slice_point["starLumFunc_%s" % self.filtername],
        )
        # convert from stars/sq degree to stars/sq arcsec
        try:
            result = interp(self.mag_limit) / (3600.0**2)
        except ValueError:
            # This probably means the interpolation went out of range
            # (magLimit <15 or >28)
            return self.badval
        return result
