__all__ = ("SingleLinearMultibandModelMetric", "NestedLinearMultibandModelMetric")

import healpy as hp
import numpy as np

from .base_metric import BaseMetric
from .exgal_m5 import ExgalM5
from .weak_lensing_systematics_metric import ExgalM5WithCuts, RIZDetectionCoaddExposureTime


class SingleLinearMultibandModelMetric(BaseMetric):
    """Calculate a single linear combination of depths.

    This is useful for calculating density or redshift fluctuations
    from depth fluctuations on the sky,
    for a single redshift bins (thus it is a single metric).
    For multiple bins, see NestedLinearMultibandModelMetric.

    Parameters
    ----------
    arr_of_model_dicts : `dict`
        Linear coefficients to be applied to the M5 depth values.
        Keys should be filter names + 'cst' if there is a constant offset.
    post_processing_fn : lambda
        Post processing function to apply to the linear combination of inputs.
        Default `lambda x: x` simply returns the output.
    extinction_cut : `float`, optional
        E(B-V) cut on extinction (0.2 by default).
    n_filters : `int`, optional
        Cut on the number of filters required (6 by default).
    min_depth_cut : `dict`, optional
        Cut the areas of low depths, based on cuts in this dict.
        Keys should be filter names. Default is no cuts.
    max_depth_cut : `dict`, optional
        Cut the areas of high depths, based on cuts in this dict.
        Keys should be filter names. Default is no cuts.
    mean_depth : `dict`, optional
        Mean depths, which will be subtracted from the inputs
        before applying model.
        Keys should be filter names. Default is zero in each band.
        In a lot of cases, instead of feeding mean_depths,
        one can remove the monopole in the constructed healpix maps.
    m5_col : `str`, optional
        Column name for the m5 depth.
    filter_col : `str`, optional
        Column name for the filter.
    units : `str`, optional
        Label for "units" in the output, for use in plots.
    badval : `float`, optional
        Value to return for metric failure.
        Specified here so that exgalm5 uses the same value.

    Returns
    -------
    result : `float`
        Linear combination of the M5 depths (minus mean depth, if provided).
        result = np.sum([
                model_dict[band] * M5_depth[band]
                for band in ["u", "g", "r", "i", "z", "y"]
                ])
        if post_processing_fn is provided:
        result => post_processing_fn(result)
    """

    def __init__(
        self,
        model_dict,
        post_processing_fn=lambda x: x,
        extinction_cut=0.2,
        n_filters=6,
        min_depth_cut=None,
        max_depth_cut=None,
        mean_depth=None,
        m5_col="fiveSigmaDepth",
        filter_col="band",
        units="mag",
        badval=np.nan,
        **kwargs,
    ):
        if min_depth_cut is None:
            min_depth_cut = {}
        self.min_depth_cut = min_depth_cut
        if max_depth_cut is None:
            max_depth_cut = {}
        self.max_depth_cut = max_depth_cut
        if mean_depth is None:
            # Set mean_depth to 0 if not specified
            mean_depth = dict([(f, 0) for f in "ugrizy"])
        self.mean_depth = mean_depth

        self.n_filters = n_filters
        self.extinction_cut = extinction_cut
        if "cst" in model_dict:
            self.cst = model_dict["cst"]
        else:
            self.cst = 0.0
        lsst_filters = ["u", "g", "r", "i", "z", "y"]
        self.bands = [x for x in model_dict.keys() if x in lsst_filters]
        self.model_arr = np.array([model_dict[x] for x in self.bands])[:, None]

        self.m5_col = m5_col
        self.filter_col = filter_col
        self.post_processing_fn = post_processing_fn

        self.exgal_m5 = ExgalM5(m5_col=m5_col, filter_col=filter_col, badval=badval)
        super().__init__(
            col=[m5_col, filter_col], units=units, maps=self.exgal_m5.maps, badval=badval, **kwargs
        )

    def run(self, data_slice, slice_point=None):
        """ """

        if slice_point["ebv"] > self.extinction_cut:
            return self.badval

        n_filters = len(set(data_slice[self.filter_col]))
        if n_filters < self.n_filters:
            return self.badval

        # add extinction_cut and depth cuts? and n_filters cuts
        depths = np.vstack(
            [
                self.exgal_m5.run(data_slice[data_slice[self.filter_col] == lsst_filter], slice_point)
                for lsst_filter in self.bands
            ]
        ).T
        for i, lsst_filter in enumerate(self.bands):
            if lsst_filter in self.mean_depth:
                depths[i] -= self.mean_depth[lsst_filter]
        # Don't raise Exceptions in the middle of metrics -
        # assert depths.shape[0] >= 1
        if depths.shape[0] < 1:
            return self.badval

        if np.any(depths == self.badval):
            return self.badval

        for i, lsst_filter in enumerate(self.bands):
            if (
                lsst_filter in self.min_depth_cut
                and depths[0, i] < self.min_depth_cut[lsst_filter] - self.mean_depth[lsst_filter]
            ):
                return self.badval
            if (
                lsst_filter in self.max_depth_cut
                and depths[0, i] > self.max_depth_cut[lsst_filter] - self.mean_depth[lsst_filter]
            ):
                return self.badval

        val = self.post_processing_fn(self.cst + np.dot(depths, self.model_arr))
        return val


class NestedLinearMultibandModelMetric(BaseMetric):
    """Calculate multiple linear combinations of depths.

    This is useful for calculating density or redshift fluctuations
    from depth fluctuations on the sky,
    for multiple redshift bins (thus it is a nested metric).
    For a single bin, see LinearMultibandModelMetric.

    Points of contact / contributors: Boris Leistedt

    Parameters
    ----------
    arr_of_model_dicts : `list` [ `dicts` ]
        Array of linear coefficients to be applied to the M5 depth values.
        Keys should be filter names + 'cst' if there is a constant offset.
    post_processing_fn : lambda
        Post processing function to apply to the linear combination of inputs.
        Default `lambda x: x` simply returns the output.
    extinction_cut : `float`, optional
        E(B-V) cut on extinction (0.2 by default).
    n_filters : `int`, optional
        Cut on the number of filters required (6 by default).
    min_depth_cut : `dict`, optional
        Cut the areas of low depths, based on cuts in this dict.
        Keys should be filter names. Default is no cuts.
    max_depth_cut : `dict`, optional
        Cut the areas of high depths, based on cuts in this dict.
        Keys should be filter names. Default is no cuts.
    mean_depth : `dict`, optional
        Mean depths, which will be subtracted from the inputs
        before applying model.
        Keys should be filter names. Default is zero in each band.
        In a lot of cases, instead of feeding mean_depths,
        one can remove the monopole in the constructed healpix maps.
    m5_col : `str`, optional
        Column name for the m5 depth.
    filter_col : `str`, optional
        Column name for the filter.
    units : `str`, optional
        Label for "units" in the output, for use in plots.
    badval : `float`, optional
        Value to return for metric failure.
        Specified here so that exgalm5 uses the same value.

    Returns
    -------
    result : `list` [ `float` ]
        List is the same size as arr_of_model_dicts.
        For each element, return a linear combination of the M5 depths
        (minus mean depth, if provided).
        for i, model_dict in enumerate(arr_of_model_dicts):
            result[i] = np.sum([
                model_dict[band] * M5_depth[band]
                for band in ["u", "g", "r", "i", "z", "y"]
                ])
        if post_processing_fn is provided:
        result[i] => post_processing_fn(result[i])
    """

    def __init__(
        self,
        arr_of_model_dicts,
        post_processing_fn=lambda x: x,
        extinction_cut=0.2,
        n_filters=6,
        min_depth_cut=None,
        max_depth_cut=None,
        mean_depth=None,
        m5_col="fiveSigmaDepth",
        filter_col="band",
        units="mag",
        badval=np.nan,
        **kwargs,
    ):
        self.arr_of_model_dicts = arr_of_model_dicts
        self.n_filters = n_filters
        self.extinction_cut = extinction_cut
        if min_depth_cut is None:
            min_depth_cut = {}
        self.min_depth_cut = min_depth_cut
        if max_depth_cut is None:
            max_depth_cut = {}
        self.max_depth_cut = max_depth_cut
        if mean_depth is None:
            mean_depth = {}
        self.mean_depth = mean_depth

        self.m5_col = m5_col
        self.badval = badval
        self.mask_val = hp.UNSEEN
        self.filter_col = filter_col
        self.post_processing_fn = post_processing_fn

        self.n_bins = len(self.arr_of_model_dicts)
        self.bad_val_arr = np.repeat(self.badval, self.n_bins)

        self.exgal_m5 = ExgalM5(
            m5_col=m5_col,
            filter_col=filter_col,
            badval=self.badval,
        )
        super().__init__(
            col=[m5_col, filter_col],
            badval=self.badval,
            units=units,
            maps=self.exgal_m5.maps,
            metric_dtype="object",
            **kwargs,
        )

    def run(self, data_slice, slice_point=None):
        # apply extinction and n_filters cut
        if slice_point["ebv"] > self.extinction_cut:
            return self.bad_val_arr

        n_filters = len(set(data_slice[self.filter_col]))
        if n_filters < self.n_filters:
            return self.bad_val_arr

        lsst_filters = ["u", "g", "r", "i", "z", "y"]

        # initialize dictionary of outputs
        # types = [float]*n_bins
        result_arr = np.zeros((self.n_bins,), dtype=float)

        for binnumber, model_dict in enumerate(self.arr_of_model_dicts):

            # put together a vector of depths
            depths = np.vstack(
                [
                    self.exgal_m5.run(data_slice[data_slice[self.filter_col] == lsst_filter], slice_point)
                    for lsst_filter in lsst_filters
                ]
            ).T

            # if there are depth cuts, apply them
            for i, lsst_filter in enumerate(lsst_filters):
                if lsst_filter in self.min_depth_cut and depths[0, i] < self.min_depth_cut[lsst_filter]:
                    depths[0, i] = self.badval
                if lsst_filter in self.max_depth_cut and depths[0, i] > self.max_depth_cut[lsst_filter]:
                    depths[0, i] = self.badval

            # if provided, subtract the mean
            for i, lsst_filter in enumerate(lsst_filters):
                if lsst_filter in self.mean_depth:
                    depths[i] -= self.mean_depth[lsst_filter]

            # now assemble the results
            model_arr = np.array([model_dict[x] if x in model_dict else 0.0 for x in lsst_filters])
            cst = model_dict["cst"] if "cst" in model_dict else 0.0

            result_arr[binnumber] = self.post_processing_fn(cst + np.dot(depths, model_arr))

        # returns array where each element of the original
        # input arr_of_model_dicts contains a scalar value resulting
        # from a linear combination of the six depths
        return result_arr


class NestedRIZExptimeExgalM5Metric(BaseMetric):
    """TODO"""

    def __init__(
        self,
        m5_col="fiveSigmaDepth",
        filter_col="band",
        exptime_col="visitExposureTime",
        extinction_cut=0.2,
        n_filters=6,
        depth_cut=24,
        metric_name="new_nested_FoM",
        lsst_filter="i",
        badval=np.nan,
        **kwargs,
    ):
        maps = ["DustMap"]
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.exptime_col = exptime_col
        self.lsst_filter = lsst_filter
        self.n_filters = n_filters

        cols = [self.m5_col, self.filter_col, self.exptime_col]
        super().__init__(cols, metric_name=metric_name, maps=maps, badval=badval, **kwargs)
        self.riz_exptime_metric = RIZDetectionCoaddExposureTime(ebvlim=extinction_cut)
        self.exgalm5_metric = ExgalM5WithCuts(
            m5_col=m5_col,
            filter_col=filter_col,
            extinction_cut=extinction_cut,
            depth_cut=depth_cut,
            lsst_filter=lsst_filter,
            badval=badval,
            n_filters=n_filters,
        )

        self.metric_dtype = "object"

    def run(self, data_slice, slice_point):

        # set up array to hold the results
        names = ["exgal_m5", "riz_exptime"]
        types = [float] * 2
        result = np.zeros(1, dtype=list(zip(names, types)))
        result["exgal_m5"] = self.exgalm5_metric.run(data_slice, slice_point)
        result["riz_exptime"] = self.riz_exptime_metric.run(
            data_slice[data_slice[self.filter_col] == "i"], slice_point
        )

        return result


class MultibandExgalM5(BaseMetric):
    """Calculate multiple linear combinations of depths."""

    def __init__(
        self,
        extinction_cut=0.2,
        n_filters=6,
        m5_col="fiveSigmaDepth",
        filter_col="band",
        units="mag",
        badval=np.nan,
        **kwargs,
    ):
        self.n_filters = n_filters
        self.extinction_cut = extinction_cut

        self.m5_col = m5_col
        self.badval = badval
        self.filter_col = filter_col

        self.lsst_filters = ["u", "g", "r", "i", "z", "y"]
        self.bad_val_arr = np.repeat(self.badval, len(self.lsst_filters))

        self.exgal_m5 = ExgalM5(
            m5_col=m5_col,
            filter_col=filter_col,
            badval=self.badval,
        )
        super().__init__(
            col=[m5_col, filter_col],
            badval=self.badval,
            units=units,
            maps=self.exgal_m5.maps,
            metric_dtype="object",
            **kwargs,
        )

    def run(self, data_slice, slice_point=None):
        # apply extinction and n_filters cut
        if slice_point["ebv"] > self.extinction_cut:
            return self.bad_val_arr

        n_filters = len(set(data_slice[self.filter_col]))
        if n_filters < self.n_filters:
            return self.bad_val_arr

        depths = np.vstack(
            [
                self.exgal_m5.run(data_slice[data_slice[self.filter_col] == lsst_filter], slice_point)
                for lsst_filter in self.lsst_filters
            ]
        ).T

        return depths.ravel()
