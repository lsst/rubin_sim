__all__ = ("ExgalM5",)

from rubin_sim.phot_utils import DustValues

from .base_metric import BaseMetric
from .simple_metrics import Coaddm5Metric


class ExgalM5(BaseMetric):
    """
    Calculate co-added five-sigma limiting depth after dust extinction.

    Uses phot_utils to calculate dust extinction.

    Parameters
    ----------
    m5_col : `str`, optional
        Column name for five sigma depth. Default 'fiveSigmaDepth'.
    unit : `str`, optional
        Label for units. Default 'mag'.

    Returns
    -------
    coadd_m5 : `float`
        Coadded m5 value, corrected for galactic dust extinction.
    """

    def __init__(
        self, m5_col="fiveSigmaDepth", metric_name="ExgalM5", units="mag", filter_col="filter", **kwargs
    ):
        # Set the name for the dust map to use.
        # This is gathered into the MetricBundle.
        maps = ["DustMap"]
        self.m5_col = m5_col
        self.filter_col = filter_col
        super().__init__(
            col=[self.m5_col, self.filter_col], maps=maps, metric_name=metric_name, units=units, **kwargs
        )
        # Set the default wavelength limits for the lsst filters.
        # These are approximately correct.
        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1
        # We will call Coaddm5Metric to calculate the coadded depth.
        # Set it up here.
        self.coaddm5_metric = Coaddm5Metric(m5_col=m5_col)

    def run(self, data_slice, slice_point):
        """Compute the co-added m5 depth and then apply
        dust extinction to that magnitude.
        """
        m5 = self.coaddm5_metric.run(data_slice)
        if m5 == self.coaddm5_metric.badval:
            return self.badval
        # Total dust extinction along this line of sight.
        # Correct default A to this EBV value.
        a_x = self.ax1[data_slice[self.filter_col][0]] * slice_point["ebv"]
        return m5 - a_x
