__all__ = ("SlewContributionMetric", "AveSlewFracMetric")

import numpy as np

from .base_metric import BaseMetric

# Metrics for dealing with things from the SlewActivities table


class SlewContributionMetric(BaseMetric):
    def __init__(
        self, col="actDelay", activity=None, active_col="activity", in_crit_col="inCriticalPath", **kwargs
    ):
        """
        Return the average time, multiplied by fraction of slew --
        considering critical path activities only.
        """
        self.col = col
        self.in_crit_col = in_crit_col
        col = [col, in_crit_col]
        col.append(active_col)
        self.active_col = active_col
        self.activity = activity
        super(SlewContributionMetric, self).__init__(col=col, **kwargs)
        self.comment = "Average time for %s activity (in seconds) when in the critical path, " % (activity)
        self.comment += "multiplied by the percent of total slews in the critical path."

    def run(self, data_slice, slice_point=None):
        # Activities of this type, in critical path.
        good_in_crit = np.where(
            (data_slice[self.active_col] == self.activity) & (data_slice[self.in_crit_col] == "True")
        )[0]
        if len(good_in_crit) == 0:
            result = 0.0
        else:
            # All activities in critical path.
            in_crit = np.where((data_slice[self.in_crit_col] == "True"))[0]
            # Calculate fraction of total in-critical-path slew activities that this activity represents.
            result = np.sum(data_slice[self.col][good_in_crit]) / np.sum(data_slice[self.col][in_crit])
            #  and multiply by the mean time required by this activity.
            result *= np.mean(data_slice[self.col][good_in_crit])
        return result


class AveSlewFracMetric(BaseMetric):
    def __init__(
        self, col="actDelay", activity=None, active_col="activity", id_col="SlewHistory_slewCount", **kwargs
    ):
        """
        Return the average time multiplied by fraction of slews.
        """
        self.col = col
        self.id_col = id_col
        col = [col, id_col]
        col.append(active_col)
        self.active_col = active_col
        self.activity = activity
        super(AveSlewFracMetric, self).__init__(col=col, **kwargs)
        self.comment = "Average time for %s activity (in seconds), multiplied by percent of total slews." % (
            activity
        )

    def run(self, data_slice, slice_point=None):
        good = np.where(data_slice[self.active_col] == self.activity)[0]
        if len(good) == 0:
            result = 0.0
        else:
            result = np.mean(data_slice[self.col][good])
            nslews = np.size(np.unique(data_slice[self.id_col]))
            result = result * np.size(good) / np.float(nslews)
        return result
