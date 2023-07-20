__all__ = ("ChipVendorMetric",)

import numpy as np

from .base_metric import BaseMetric


class ChipVendorMetric(BaseMetric):
    """
    See what happens if we have chips from different vendors
    """

    def __init__(self, cols=None, **kwargs):
        if cols is None:
            cols = []
        super(ChipVendorMetric, self).__init__(
            col=cols, metric_dtype=float, units="1,2,3:v1,v2,both", **kwargs
        )

    def _chip_names2vendor_id(self, chip_name):
        """
        given a list of chipnames, convert to 1 or 2, representing
        different vendors
        """
        vendors = []
        for chip in chip_name:
            # Parse the chip_name string.
            if int(chip[2]) % 2 == 0:
                vendors.append(1)
            else:
                vendors.append(2)
        return vendors

    def run(self, data_slice, slice_point=None):
        if "chipNames" not in list(slice_point.keys()):
            raise ValueError("No chipname info, need to set use_camera=True with a spatial slicer.")

        uvendor_i_ds = np.unique(self._chip_names2vendor_id(slice_point["chipNames"]))
        if np.size(uvendor_i_ds) == 1:
            result = uvendor_i_ds
        else:
            result = 3
        return result
