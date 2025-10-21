__all__ = ("ExgalM5WithCuts", "WeakLensingNvisits", "RIZDetectionCoaddExposureTime")

import numpy as np

from .base_metric import BaseMetric
from .exgal_m5 import ExgalM5


class ExgalM5WithCuts(BaseMetric):
    """
    Calculate co-added five-sigma limiting depth, but apply dust extinction and
    depth cuts. This means that places on the sky that don't meet the dust
    extinction, coadded depth, or filter coverage cuts will have masked values
    on those places.

    This metric is useful for DESC static science and weak lensing metrics. In
    particular, it is required as input for StaticProbesFoMEmulatorMetricSimple
    (a summary metric to emulate a 3x2pt FOM).

    Note: this metric calculates the depth after dust extinction in band
    'lsst_filter', but because it looks for coverage in all bands, there should
    generally be no filter-constraint on the sql query.
    """

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        metric_name="Exgalm5WithCuts",
        units="mag",
        lsst_filter="i",
        extinction_cut=0.2,
        depth_cut=25.9,
        n_filters=6,
        **kwargs,
    ):
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.extinction_cut = extinction_cut
        self.depth_cut = depth_cut
        self.n_filters = n_filters
        self.lsst_filter = lsst_filter
        # I thought about inheriting from ExGalM5 instead, but the columns
        # specification is more complicated
        self.exgal_m5 = ExgalM5(m5_col=m5_col, units=units)
        super().__init__(
            col=[self.m5_col, self.filter_col],
            metric_name=metric_name,
            units=units,
            maps=self.exgal_m5.maps,
            **kwargs,
        )

    def run(self, data_slice, slice_point):
        # exclude areas with high extinction
        if slice_point["ebv"] > self.extinction_cut:
            return self.badval

        # check to make sure there is at least some coverage in the required
        # number of bands
        n_filters = len(set(data_slice[self.filter_col]))
        if n_filters < self.n_filters:
            return self.badval

        # if coverage and dust criteria are valid, move forward with only
        # lsstFilter-band visits
        d_s = data_slice[data_slice[self.filter_col] == self.lsst_filter]
        # calculate the lsstFilter-band coadded depth
        dustdepth = self.exgal_m5.run(d_s, slice_point)

        # exclude areas that are shallower than the depth cut
        if dustdepth < self.depth_cut:
            return self.badval
        else:
            return dustdepth


class WeakLensingNvisits(BaseMetric):
    """A proxy metric for WL systematics. Higher values indicate better
    systematics mitigation.

    Weak Lensing systematics metric : Computes the average number of visits per
    point on a HEALPix grid after a maximum E(B-V) cut and a minimum co-added
    depth cut. Intended to be used to count visits in gri, but can be any
    filter combination as long as it includes `lsst_filter` band visits.

    """

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        exp_time_col="visitExposureTime",
        filter_col="filter",
        lsst_filter="i",
        depth_cut=24.5,
        ebvlim=0.2,
        min_exp_time=15,
        **kwargs,
    ):
        # Set up the coadd metric (using ExgalM5 adds galactic dust extinction)
        self.exgal_m5 = ExgalM5(m5_col=m5_col, filter_col=filter_col)
        self.lsst_filter = lsst_filter
        self.depth_cut = depth_cut
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.exp_time_col = exp_time_col
        self.ebvlim = ebvlim
        self.min_exp_time = min_exp_time
        super().__init__(col=[self.m5_col, self.exp_time_col, filter_col], maps=self.exgal_m5.maps, **kwargs)

    def run(self, data_slice, slice_point):
        # If the sky is too dusty here, stop.
        if slice_point["ebv"] > self.ebvlim:
            return self.badval
        # Check that coadded depths meet requirements:
        d_s = data_slice[data_slice[self.filter_col] == self.lsst_filter]
        coadd_depth = self.exgal_m5.run(data_slice=d_s, slice_point=slice_point)
        if coadd_depth < self.depth_cut:
            return self.badval
        nvisits = len(np.where(data_slice[self.exp_time_col] > self.min_exp_time)[0])
        return nvisits


class RIZDetectionCoaddExposureTime(BaseMetric):
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
    bin_col : str, optional
        The column to bin on. The default is 'night'.
    exp_time_col : str, optional
        The column name for the exposure time.
    filter_col : str, optional
        The column name for the filter name.
    ebvlim : float, optional
        The upper limit on E(B-V). Regions with E(B-V) greater than this
        limit are excluded.
    min_exp_time : float, optional
        The minimal exposure time for a visit to contribute to a coadd.
    det_bands : list of str, optional
        If not None, the bands to use for detection. If None, defaults to riz.
    min_bands : list of str, optional
        If not None, the bands whose presence is used to cut the survey data.
        If None, defaults to ugrizY.
    """

    def __init__(
        self,
        exp_time_col="visitExposureTime",
        filter_col="filter",
        ebvlim=0.2,
        min_expTime=15,
        det_bands=None,
        min_bands=None,
        metric_name="riz_detcoadd_exptime",
        **kwargs,
    ):
        # Set up the coadd metric (using ExgalM5 adds galactic dust extinction)
        self.filter_col = filter_col
        self.exp_time_col = exp_time_col
        self.ebvlim = ebvlim
        self.min_exp_time = min_expTime
        self.det_bands = det_bands or ["r", "i", "z"]
        self.min_bands = set(min_bands or ["u", "g", "r", "i", "z", "y"])
        super().__init__(
            col=[self.exp_time_col, self.filter_col],
            metric_name=metric_name,
            units="seconds",
            maps=["DustMap"],
            **kwargs,
        )

    def run(self, data_slice, slice_point):
        # If the sky is too dusty here, stop.
        if slice_point["ebv"] > self.ebvlim:
            res = self.badval
            return res
        # find all entries where exposure time is long enough and
        # in the detection bands
        exptime_msk = data_slice[self.exp_time_col] > self.min_exp_time
        filter_msk = np.isin(data_slice[self.filter_col], self.det_bands)
        tot_msk = exptime_msk & filter_msk

        res = np.sum(data_slice[self.exp_time_col][tot_msk])

        return res
