import numpy as np
from .baseMetric import BaseMetric
from .exgalM5 import ExgalM5
from .vectorMetrics import VectorMetric

__all__ = ["ExgalM5_with_cuts", "WeakLensingNvisits", "RIZDetectionCoaddExposureTime"]


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


class RIZDetectionCoaddExposureTime(VectorMetric):
    """A metric computing the total exposure time of an riz coadd.

    This metric is intended to be used as a proxy for depth fluctuations in
    catalogs detected from coadds of the r, i and z bands together. This
    coadding + detection scheme is used by metadetection (weak lensing
    shear estimator) and will likely be adopted by the Rubin science
    pipelines.

    It counts the total exposure time in all three bands, excluding dusty
    regions, exposures that are too short, or areas where not all bands
    ugrizY are present. We do not make a depth cut explicitly since that is
    circular (and thus confuses MRB's feeble mind :/).

    TODO maybe:
     - apply some sort of inverse variance weighting to the coadd based on sky
       level?
     - use some sort of effective exposure time that accounts for the PSF?
     - @rhiannonlynne nicely suggested this Teff computation:
       rubin_sim/rubin_sim/maf/metrics/technicalMetrics.py

    However, given the unknown nature of detection will look like in LSST,
    a simple sum of exposure time is probably ok.

    Parameters
    ----------
    bins : list of float
        The bin edges. Typically this will be a list of nights for which to
        compute the riz coadd exposure times.
    binCol : str, optional
        The column to bin on. The default is 'night'.
    expTimeCol : str, optional
        The column name for the exposure time.
    filterCol : str, optional
        The column name for the filter name.
    ebvlim : float, optional
        The upper limit on E(B-V). Regions with E(B-V) greater than this
        limit are excluded.
    min_expTime : float, optional
        The minimal exposure time for a visit to contribute to a coadd.
    det_bands : list of str, optional
        If not None, the bands to use for detection. If None, defaults to riz.
    min_bands : list of str, optional
        If not None, the bands whose presence is used to cut the survey data.
        If None, defaults to ugrizY.
    """

    def __init__(
        self,
        *,
        bins,
        binCol="night",
        expTimeCol="visitExposureTime",
        filterCol="filter",
        ebvlim=0.2,
        min_expTime=15,
        det_bands=None,
        min_bands=None,
        **kwargs
    ):
        # Set up the coadd metric (using ExgalM5 adds galactic dust extinction)
        self.filterCol = filterCol
        self.expTimeCol = expTimeCol
        self.ebvlim = ebvlim
        self.min_expTime = min_expTime
        self.det_bands = det_bands or ["r", "i", "z"]
        self.min_bands = set(min_bands or ["u", "g", "r", "i", "z", "y"])
        super().__init__(
            bins=bins,
            binCol=binCol,
            col=[self.expTimeCol, self.filterCol],
            metricName="riz_detcoadd_exptime",
            units="seconds",
            maps=["DustMap"],
            **kwargs
        )

    def run(self, dataSlice, slicePoint):
        res = np.zeros(self.shape, dtype=self.metricDtype)

        # If the sky is too dusty here, stop.
        if slicePoint["ebv"] > self.ebvlim:
            res[:] = self.badval
            return res

        dataSlice.sort(order=self.binCol)
        cutinds = np.searchsorted(dataSlice[self.binCol], self.bins[1:], side="right")
        maxcutind = dataSlice.shape[0]
        cutinds = np.clip(cutinds, 0, maxcutind)

        # find all entries where exposure time is long enough and
        # in the detection bands
        exptime_msk = dataSlice[self.expTimeCol] > self.min_expTime
        filter_msk = np.in1d(dataSlice[self.filterCol], self.det_bands)
        tot_msk = exptime_msk & filter_msk

        for i, cutind in enumerate(cutinds):
            if cutind == 0:
                res[i] = self.badval
                continue

            # check to make sure there is at least some
            # coverage in the required bands
            filters = set(dataSlice[self.filterCol][:cutind])
            if filters != self.min_bands:
                res[i] = self.badval
                continue

            # if nothing passes for detection, we exclude this region
            if not np.any(tot_msk[:cutind]):
                res[i] = self.badval
                continue

            res[i] = np.sum(dataSlice[self.expTimeCol][:cutind][tot_msk[:cutind]])

        return res
