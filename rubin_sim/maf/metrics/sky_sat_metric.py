__all__ = ("SkySaturationMetric",)

import numpy as np

from .base_metric import BaseMetric


class SkySaturationMetric(BaseMetric):
    """Check if the sky would saturate a visit in an exposure"""

    def __init__(self, metric_name="SkySaturation", units="#", **kwargs):
        super().__init__(col=["saturation_mag"], units=units, metric_name=metric_name, **kwargs)

    def run(self, data_slice, slice_point):
        # Saturation stacker returns NaN if the sky saturates
        finite = np.isfinite(data_slice["saturation_mag"])
        result = np.size(np.where(finite == False)[0])
        return result
