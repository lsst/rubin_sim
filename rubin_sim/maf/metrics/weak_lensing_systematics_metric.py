import numpy as np
from .base_metric import BaseMetric
from .exgal_m5 import ExgalM5

__all__ = ["ExgalM5_with_cuts", "WeakLensingNvisits"]


class ExgalM5_with_cuts(BaseMetric):
    """
    Calculate co-added five-sigma limiting depth, but apply dust extinction and depth cuts.
    This means that places on the sky that don't meet the dust extinction, coadded depth, or filter coverage
    cuts will have masked values on those places.

    This metric is useful for DESC static science and weak lensing metrics.
    In particular, it is required as input for the StaticProbesFoMEmulatorMetricSimple
    (a summary metric to emulate a 3x2pt FOM).

    Note: this metric calculates the depth after dust extinction in band 'lsstFilter', but because
    it looks for coverage in all bands, there should generally be no filter-constraint on the sql query.
    """

    def __init__(
        self,
        m5Col="fiveSigmaDepth",
        filterCol="filter",
        metricName="ExgalM5_with_cuts",
        units="mag",
        lsstFilter="i",
        extinction_cut=0.2,
        depth_cut=25.9,
        nFilters=6,
        **kwargs
    ):
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.extinction_cut = extinction_cut
        self.depth_cut = depth_cut
        self.nFilters = nFilters
        self.lsstFilter = lsstFilter
        # I thought about inheriting from ExGalM5 instead, but the columns specification is more complicated
        self.exgalM5 = ExgalM5(m5Col=m5Col, units=units)
        super().__init__(
            col=[self.m5Col, self.filterCol],
            metricName=metricName,
            units=units,
            maps=self.exgalM5.maps,
            **kwargs
        )

    def run(self, dataSlice, slicePoint):
        # exclude areas with high extinction
        if slicePoint["ebv"] > self.extinction_cut:
            return self.badval

        # check to make sure there is at least some coverage in the required number of bands
        nFilters = len(set(dataSlice[self.filterCol]))
        if nFilters < self.nFilters:
            return self.badval

        # if coverage and dust criteria are valid, move forward with only lsstFilter-band visits
        dS = dataSlice[dataSlice[self.filterCol] == self.lsstFilter]
        # calculate the lsstFilter-band coadded depth
        dustdepth = self.exgalM5.run(dS, slicePoint)

        # exclude areas that are shallower than the depth cut
        if dustdepth < self.depth_cut:
            return self.badval
        else:
            return dustdepth


class WeakLensingNvisits(BaseMetric):
    """A proxy metric for WL systematics. Higher values indicate better systematics mitigation.

    Weak Lensing systematics metric : Computes the average number of visits per point on a HEALPix grid
    after a maximum E(B-V) cut and a minimum co-added depth cut.
    Intended to be used to count visits in gri, but can be any filter combination as long as it
    includes `lsstFilter` band visits.

    """

    def __init__(
        self,
        m5Col="fiveSigmaDepth",
        expTimeCol="visitExposureTime",
        filterCol="filter",
        lsstFilter="i",
        depth_cut=24.5,
        ebvlim=0.2,
        min_expTime=15,
        **kwargs
    ):
        # Set up the coadd metric (using ExgalM5 adds galactic dust extinction)
        self.exgalM5 = ExgalM5(m5Col=m5Col, filterCol=filterCol)
        self.lsstFilter = lsstFilter
        self.depth_cut = depth_cut
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.expTimeCol = expTimeCol
        self.ebvlim = ebvlim
        self.min_expTime = min_expTime
        super().__init__(
            col=[self.m5Col, self.expTimeCol, filterCol],
            maps=self.exgalM5.maps,
            **kwargs
        )

    def run(self, dataSlice, slicePoint):
        # If the sky is too dusty here, stop.
        if slicePoint["ebv"] > self.ebvlim:
            return self.badval
        # Check that coadded depths meet requirements:
        dS = dataSlice[dataSlice[self.filterCol] == self.lsstFilter]
        coaddDepth = self.exgalM5.run(dataSlice=dS, slicePoint=slicePoint)
        if coaddDepth < self.depth_cut:
            return self.badval
        nvisits = len(np.where(dataSlice[self.expTimeCol] > self.min_expTime)[0])
        return nvisits
