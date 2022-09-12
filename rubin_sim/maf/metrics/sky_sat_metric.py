import numpy as np
from .base_metric import BaseMetric


__all__ = ["SkySaturationMetric"]


class SkySaturationMetric(BaseMetric):
    """Check if the sky would saturate a visit in an exposure"""

    def __init__(self, metricName="SkySaturation", units="#", **kwargs):
        super().__init__(
            col=["saturation_mag"], units=units, metricName=metricName, **kwargs
        )

    def run(self, dataSlice, slicePoint):
        # Saturation stacker returns NaN if the sky saturates
        finite = np.isfinite(dataSlice["saturation_mag"])
        result = np.size(np.where(finite == False)[0])
        return result
