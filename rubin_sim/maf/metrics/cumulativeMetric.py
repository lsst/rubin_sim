import numpy as np
from .baseMetric import BaseMetric


__all__ = ["CumulativeMetric"]


class CumulativeMetric(BaseMetric):
    """For plotting up the cumulative number of observations.
    Expected to be used with a UniSlicer or UserPointSlicer with one point.

    Parameters
    ----------
    bins : `np.array` (None)
        The points to interpolate the cumulative number of observations to. If None,
        then the range of the data is used.
    """

    def __init__(
        self,
        metricName="Cumulative",
        time_col="observationStartMJD",
        night_col="night",
        bins=None,
        **kwargs
    ):
        super().__init__(
            col=[time_col, night_col],
            metricName=metricName,
            metricDtype="object",
            **kwargs
        )
        self.time_col = time_col
        self.night_col = night_col
        self.bins = bins
        self.plotDict = {"xlabel": "MJD (days)", "ylabel": "N obs"}

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=self.time_col)
        if self.bins is None:
            bins = np.arange(
                dataSlice[self.night_col].min(), dataSlice[self.night_col].max() + 1, 1
            )
        cumulative_number = np.arange(dataSlice.size) + 1
        yresult = np.interp(bins, dataSlice[self.night_col], cumulative_number)
        xresult = bins
        return {"x": xresult, "y": yresult, "plotDict": self.plotDict}
