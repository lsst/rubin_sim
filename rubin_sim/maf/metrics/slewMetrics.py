import numpy as np
from .baseMetric import BaseMetric

# Metrics for dealing with things from the SlewActivities table

__all__ = ['SlewContributionMetric', 'AveSlewFracMetric']


class SlewContributionMetric(BaseMetric):
    def __init__(self, col='actDelay', activity=None, activeCol='activity',
                 inCritCol='inCriticalPath', **kwargs):
        """
        Return the average time, multiplied by fraction of slew --
        considering critical path activities only.
        """
        self.col = col
        self.inCritCol = inCritCol
        col = [col, inCritCol]
        col.append(activeCol)
        self.activeCol = activeCol
        self.activity = activity
        super(SlewContributionMetric, self).__init__(col=col, **kwargs)
        self.comment = 'Average time for %s activity (in seconds) when in the critical path, ' %(activity)
        self.comment += 'multiplied by the percent of total slews in the critical path.'

    def run(self, dataSlice, slicePoint=None):
        # Activities of this type, in critical path.
        goodInCrit = np.where((dataSlice[self.activeCol] == self.activity) &
                              (dataSlice[self.inCritCol] == 'True'))[0]
        if len(goodInCrit) == 0:
            result = 0.0
        else:
            # All activities in critical path.
            inCrit = np.where((dataSlice[self.inCritCol] == 'True'))[0]
            # Calculate fraction of total in-critical-path slew activities that this activity represents.
            result = np.sum(dataSlice[self.col][goodInCrit]) / np.sum(dataSlice[self.col][inCrit])
            #  and multiply by the mean time required by this activity.
            result *= np.mean(dataSlice[self.col][goodInCrit])
        return result


class AveSlewFracMetric(BaseMetric):
    def __init__(self, col='actDelay', activity=None, activeCol='activity',
                 idCol='SlewHistory_slewCount', **kwargs):
        """
        Return the average time multiplied by fraction of slews.
        """
        self.col = col
        self.idCol = idCol
        col = [col, idCol]
        col.append(activeCol)
        self.activeCol = activeCol
        self.activity = activity
        super(AveSlewFracMetric, self).__init__(col=col, **kwargs)
        self.comment = 'Average time for %s activity (in seconds), multiplied by percent of total slews.' %(activity)

    def run(self, dataSlice, slicePoint=None):
        good = np.where(dataSlice[self.activeCol] == self.activity)[0]
        if len(good) == 0:
            result = 0.0
        else:
            result = np.mean(dataSlice[self.col][good])
            nslews = np.size(np.unique(dataSlice[self.idCol]))
            result = result * np.size(good)/np.float(nslews)
        return result
