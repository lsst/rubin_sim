import warnings
from .baseMetric import BaseMetric
from scipy.interpolate import interp1d

__all__ = ["StarDensityMetric"]


class StarDensityMetric(BaseMetric):
    """Interpolate the stellar luminosity function to return the number of
    stars per square arcsecond brighter than the magLimit.
    Note that the map is built from CatSim stars in the range 20 < r < 28.
    magLimit values outside that the range of the map's starMapBins will return self.badval

    The stellar density maps are available in any bandpass, but bandpasses other
    than r band must use a pre-configured StellarDensityMap (not just the default).
    In other words, when setting up the metric bundle for an i-band stellar density
    using (as an example) a HealpixSlicer:
    ```
    map = maf.StellarDensityMap(filtername='i')
    metric = maf.StarDensityMetric(filtername='i', magLimit=25.0)
    slicer = maf.HealpixSlicer()
    bundle = maf.MetricBundle(metric, slicer, "", mapsList=[map])
    ```

    Parameters
    ----------
    magLimit : `float`, opt
        Magnitude limit at which to evaluate the stellar luminosity function.
        Returns number of stars per square arcsecond brighter than this limit.
        Default 25.
    filtername : `str`, opt
        Which filter to evaluate the luminosity function in; Note that using bands other than r
        will require setting up a custom (rather than default) version of the stellar density map.
        Default r.
    units : `str`, opt
        Units for the output values. Default "stars/sq arcsec".
    maps : `list` of `str`, opt
        Names for the maps required. Default "StellarDensityMap".

    Returns
    -------
    result : `float`
        Number of stars brighter than magLimit in filtername, based on the stellar density map.
    """

    def __init__(
        self,
        magLimit=25.0,
        filtername="r",
        units="stars/sq arcsec",
        maps=["StellarDensityMap"],
        **kwargs
    ):

        super(StarDensityMetric, self).__init__(
            col=[], maps=maps, units=units, **kwargs
        )
        self.magLimit = magLimit
        if "rmagLimit" in kwargs:
            warnings.warn(
                "rmagLimit is deprecated; please use magLimit instead "
                "(will use the provided rmagLimit for now)."
            )
            self.magLimit = kwargs["rmagLimit"]
        self.filtername = filtername

    def run(self, dataSlice, slicePoint=None):
        # Interpolate the data to the requested mag
        interp = interp1d(
            slicePoint["starMapBins_%s" % self.filtername][1:],
            slicePoint["starLumFunc_%s" % self.filtername],
        )
        # convert from stars/sq degree to stars/sq arcsec
        try:
            result = interp(self.magLimit) / (3600.0**2)
        except ValueError:
            # This probably means the interpolation went out of range (magLimit <15 or >28)
            return self.badval
        return result
