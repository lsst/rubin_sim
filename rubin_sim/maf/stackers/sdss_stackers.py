__all__ = ("SdssRADecStacker",)


import numpy as np

from .base_stacker import BaseStacker
from .coord_stackers import wrap_ra


class SdssRADecStacker(BaseStacker):
    """convert the p1,p2,p3... columns to radians and wrap them"""

    cols_added = ["RA1", "Dec1", "RA2", "Dec2", "RA3", "Dec3", "RA4", "Dec4"]

    def __init__(self, pcols=["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"]):
        """The p1,p2 columns represent the corners of chips.
        Could generalize this a bit."""
        self.units = ["rad"] * 8
        self.cols_req = pcols

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Assume this is unusual enough to run that you really mean it.
            pass
        for pcol, newcol in zip(self.cols_req, self.cols_added):
            if newcol[0:2] == "RA":
                sim_data[newcol] = wrap_ra(np.radians(sim_data[pcol]))
            else:
                sim_data[newcol] = np.radians(sim_data[pcol])
        return sim_data
