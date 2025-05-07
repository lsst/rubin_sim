__all__ = (
    "BaseMetric",
    "MeanMetric",
    "CountMetric",
    "CoaddM5Metric",
    "CoaddM5ExtinctionMetric",
    "VectorMetric",
)

import warnings
from functools import cache

import numpy as np

from rubin_sim.maf_proto.utils import eb_v_hp
from rubin_sim.phot_utils import DustValues

# XXX--How do we want to track units?
UNIT_LOOKUP_DICT = {"night": "Days", "fiveSigmaDepth": "mag", "airmass": "airmass"}


class BaseMetric(object):
    """Example of a simple metric."""

    def __init__(self, col="night", unit=None, name="name"):
        self.shape = None
        self.dtype = float
        self.col = col
        self.name = name
        if unit is None:
            self.unit = UNIT_LOOKUP_DICT[self.col]
        else:
            self.unit = unit

    def add_info(self, info):
        info["metric: name"] = self.__class__.__name__
        if hasattr(self, "col"):
            info["metric: col"] = self.col
        if hasattr(self, "unit"):
            info["metric: unit"] = self.unit
        return info

    def __call__(self, visits, slice_point=None):
        raise NotImplementedError("Metric __call__ method not implemented")

    def call_cached(self, hashable, visits=None, slice_point=None):
        """hashable should be something like a frozenset of
        visitIDs. If you use call_cached when slicepoint has
        important data (extinction, stellar density, etc),
        then this can give a different (wrong) result
        """
        self.visits = visits
        self.slice_point = slice_point
        return self.call_cached_post(hashable)

    # XXX--danger, this simple caching can cause a memory leak.
    # Probably need to do the caching in the slicer as before.
    @cache
    def call_cached_post(self, hashable):
        return self.__call__(self.visits, slice_point=self.slice_point)


class MeanMetric(BaseMetric):
    def __init__(self, col="night", unit=None, name="name"):
        super().__init__(col=col, unit=unit, name=name)

    def __call__(self, visits, slice_point=None):
        return np.mean(visits[self.col])


class CountMetric(BaseMetric):
    def __init__(self, col="night", unit="#", name="Count"):
        super().__init__(col=col, unit=unit, name=name)

    def __call__(self, visits, slice_point=None):
        return np.size(visits[self.col])


class CoaddM5Metric(BaseMetric):
    def __init__(self, filtername, col="fiveSigmaDepth", unit=None, name="CoaddDepth"):
        self.col = col
        self.filtername = filtername
        self.name = name
        if unit is None:
            self.unit = "Coadded Depth, %s (mags)" % self.filtername
        else:
            self.unit = unit

    @staticmethod
    def coadd(single_visit_m5s):
        return 1.25 * np.log10(np.sum(10.0 ** (0.8 * single_visit_m5s)))

    def __call__(self, visits, slice_point=None):
        if np.size(np.unique(visits["filter"])) > 1:
            warnings.warn("Coadding depths in different filters")

        result = self.coadd(visits[self.col])
        return result


class CoaddM5ExtinctionMetric(CoaddM5Metric):
    """ """

    def __init__(self, filtername, col="fiveSigmaDepth", unit=None, name="CoaddDepthExtinction"):
        super().__init__(filtername, col=col, unit=unit, name=name)

        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1[self.filtername]

    def __call__(self, visits, slice_point=None):
        if np.size(np.unique(visits["filter"])) > 1:
            warnings.warn("Coadding depths in different filters")

        extinction = eb_v_hp(slice_point["nside"], pixels=slice_point["sid"])
        a_x = self.ax1 * extinction

        result = self.coadd(visits[self.col]) - a_x
        return result


class FancyMetric(MeanMetric):
    """Example of returning multiple values in a metric"""

    def __init__(self, col="night"):
        self.shape = None
        self.dtype = list(zip(["mean", "std"], [float, float]))
        self.col = col
        self.empty = np.empty(1, dtype=self.dtype)

    def __call__(self, visits, slice_point=None):
        result = self.empty.copy()
        result["mean"] = np.mean(visits[self.col])
        result["std"] = np.std(visits[self.col])
        return result


class VectorMetric(MeanMetric):
    """Example of returning a vector"""

    def __init__(self, times=np.arange(60), col="night", time_col="night", function=np.add):
        self.shape = np.size(times)
        self.dtype = float
        self.col = col
        self.function = function
        self.time_col = time_col
        self.times = times

    def add_info(self, info):
        info["metric: name, MeanMetric"]
        info["metric: times"] = self.times
        return info

    def __call__(self, visits, slice_point):

        visit_times = visits[self.time_col]
        visit_times.sort()
        to_count = np.ones(visit_times.size, dtype=int)
        result = self.function.accumulate(to_count)
        indices = np.searchsorted(visit_times, self.times, side="right")
        indices[np.where(indices >= np.size(result))] = np.size(result) - 1
        result = result[indices]
        return result
