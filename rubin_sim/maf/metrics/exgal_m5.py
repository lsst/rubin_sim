from .base_metric import BaseMetric
from .simple_metrics import Coaddm5Metric
from rubin_sim.phot_utils import DustValues

__all__ = ["ExgalM5"]


class ExgalM5(BaseMetric):
    """
    Calculate co-added five-sigma limiting depth after dust extinction.

    Uses phot_utils to calculate dust extinction.

    Parameters
    ----------
    m5Col : `str`, optional
        Column name for five sigma depth. Default 'fiveSigmaDepth'.
    unit : `str`, optional
        Label for units. Default 'mag'.
    """

    def __init__(
        self,
        m5Col="fiveSigmaDepth",
        metricName="ExgalM5",
        units="mag",
        filterCol="filter",
        **kwargs
    ):
        # Set the name for the dust map to use. This is gathered into the MetricBundle.
        maps = ["DustMap"]
        self.m5Col = m5Col
        self.filterCol = filterCol
        super().__init__(
            col=[self.m5Col, self.filterCol],
            maps=maps,
            metricName=metricName,
            units=units,
            **kwargs
        )
        # Set the default wavelength limits for the lsst filters. These are approximately correct.
        dust_properties = DustValues()
        self.Ax1 = dust_properties.Ax1
        # We will call Coaddm5Metric to calculate the coadded depth. Set it up here.
        self.Coaddm5Metric = Coaddm5Metric(m5Col=m5Col)

    def run(self, dataSlice, slicePoint):
        """
        Compute the co-added m5 depth and then apply dust extinction to that magnitude.
        """
        m5 = self.Coaddm5Metric.run(dataSlice)
        if m5 == self.Coaddm5Metric.badval:
            return self.badval
        # Total dust extinction along this line of sight. Correct default A to this EBV value.
        A_x = self.Ax1[dataSlice[self.filterCol][0]] * slicePoint["ebv"]
        return m5 - A_x
