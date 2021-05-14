from .baseMetric import BaseMetric
from scipy.interpolate import interp1d

__all__ = ['StarDensityMetric']


class StarDensityMetric(BaseMetric):
    """Interpolate the stellar luminosity function to return the number of
    stars per square arcsecond brighter than the rmagLimit. Note that the
    map is built from CatSim stars in the range 20 < r < 28."""

    def __init__(self, rmagLimit=25., units='stars/sq arcsec', filtername='r',
                 maps=['StellarDensityMap'], **kwargs):

        super(StarDensityMetric, self).__init__(col=[],
                                                maps=maps, units=units, **kwargs)
        self.rmagLimit = rmagLimit
        self.filtername = filtername

    def run(self, dataSlice, slicePoint=None):
        # Interpolate the data to the requested mag
        interp = interp1d(slicePoint['starMapBins_%s' % self.filtername][1:], slicePoint['starLumFunc_%s' % self.filtername])
        # convert from stars/sq degree to stars/sq arcsec
        result = interp(self.rmagLimit)/(3600.**2)
        return result
