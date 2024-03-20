__all__ = "LinearMultibandModelMetric"

import numpy as np
from rubin_sim.maf.metrics.base_metric import BaseMetric
from rubin_sim.maf.metrics.exgal_m5 import ExgalM5
import healpy as hp


class LinearMultibandModelMetric(BaseMetric):
    """ """

    def __init__(
        self,
        model_dict,
        metric_name="LinearMultibandModel",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        post_processing_fn=lambda x : x,
        units="mag",
        extinction_cut=0.2,
        min_depth_cut={'i': 25.0},
        max_depth_cut={},
        mean_depth={"u": 0.0, "g": 0.0, "r": 0.0, "i": 0.0, "z": 0.0, "y": 0.0},
        n_filters=6,
        **kwargs,
    ):
        """
        Args:

        """
        self.n_filters = n_filters
        self.extinction_cut = extinction_cut
        self.min_depth_cut = min_depth_cut
        self.max_depth_cut = max_depth_cut
        if "cst" in model_dict:
            self.cst = model_dict["cst"]
        else:
            self.cst = 0.0
        lsst_filters = ["u", "g", "r", "i", "z", "y"]
        self.bands = [x for x in model_dict.keys() if x in lsst_filters]
        self.model_arr = np.array([model_dict[x] for x in self.bands])[:, None]
        self.mean_depth = mean_depth
        self.m5_col = m5_col
        self.filter_col = filter_col
        self.post_processing_fn = post_processing_fn
        self.exgal_m5 = ExgalM5(m5_col=m5_col, units=units)

        self.exgal_m5 = ExgalM5(
            m5_col=m5_col,
            filter_col=filter_col,
            units=units,
            metric_name=metric_name+"ExgalM5",
        )
        super().__init__(
            col=[m5_col, filter_col], metric_name=metric_name, units=units, maps=self.exgal_m5.maps, **kwargs
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
        assert depths.shape[0] >= 1

        if np.any(depths == self.badval):
            return self.badval
        for i, lsst_filter in enumerate(self.bands):
            if lsst_filter in self.min_depth_cut and depths[0, i] < self.min_depth_cut[lsst_filter] - self.mean_depth[lsst_filter]:
                return self.badval
            if lsst_filter in self.max_depth_cut and depths[0, i] > self.max_depth_cut[lsst_filter] - self.mean_depth[lsst_filter]:
                return self.badval

        val = self.post_processing_fn(
            self.cst + np.dot(depths, self.model_arr)
        )

        return val




class NestedLinearMultibandModelMetric(BaseMetric):
    """ """

    def __init__(
        self,
        dict_of_model_dicts,
        metric_name="NestedLinearMultibandModelMetric",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        post_processing_fn=lambda x : x,
        units="mag",
        extinction_cut=0.2,
        n_filters=6,
        min_depth_cut={},
        max_depth_cut={},
        mean_depth={},
        **kwargs,
    ):

        """
        Args:
            data_slice (ndarray): Values passed to metric by the slicer,
                which the metric will use to calculate metric values
                at each slice_point.
            slice_point (Dict): Dictionary of slice_point metadata passed
                to each metric.
        """
        self.dict_of_model_dicts = dict_of_model_dicts
        self.n_filters = n_filters
        self.extinction_cut = extinction_cut
        self.min_depth_cut = min_depth_cut
        self.max_depth_cut = max_depth_cut
        self.mean_depth = mean_depth
        self.m5_col = m5_col
        self.badval = np.nan
        self.filter_col = filter_col
        self.post_processing_fn = post_processing_fn

        self.exgal_m5 = ExgalM5(
            m5_col=m5_col,
            filter_col=filter_col,
            units=units,
            metric_name=metric_name+"_ExgalM5",
        )
        super().__init__(
            col=[m5_col, filter_col], metric_name=metric_name, units=units, maps=self.exgal_m5.maps, **kwargs
        )
        # magic line so MAF knows we're returning something complicated
        self.metric_dtype = "object"

    def run(self, data_slice, slice_point=None):
        "Returns a dictionary containing the values of the model evaluated at the data_slice"

        n_bins = len(self.dict_of_model_dicts)
        bad_val_arr = np.repeat(self.badval, n_bins)

        # apply extinction and n_filters cut
        if slice_point["ebv"] > self.extinction_cut:
            return bad_val_arr
        n_filters = len(set(data_slice[self.filter_col]))
        if n_filters < self.n_filters:
            return bad_val_arr

        lsst_filters = ["u", "g", "r", "i", "z", "y"]

        # initialize dictionary of outputs
        names = self.dict_of_model_dicts.keys()
        types = [float]*n_bins
        result_dict = np.zeros(1, dtype=list(zip(names, types)))

        for binname, model_dict in self.dict_of_model_dicts.items():

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
            for i, lsst_filter in enumerate(self.bands):
                if lsst_filter in self.mean_depth:
                    depths[i] -= self.mean_depth[lsst_filter]

            # now assemble the results
            model_arr = np.array([model_dict[key] if key in model_dict else 0.0 for x in lsst_filters])
            cst = model_dict['cst'] if 'cst' in model_dict else 0.0

            result_dict[binname] = self.post_processing_fn(
                cst + np.dot(depths, model_arr)
            )

        # returns dictionary where each key of the original input dict_of_model_dicts
        # contains a scalar value resulting from a linear combination of the six depths
        #
        return result_dict

