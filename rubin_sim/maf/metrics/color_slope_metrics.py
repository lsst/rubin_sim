__all__ = ["CheckColorSlope", "ColorSlopeMetric", "ColorSlope2NightMetric"]

import numpy as np
from rubin_scheduler.utils import int_binned_stat

from .base_metric import BaseMetric


class CheckColorSlope(object):
    """Check if the data has a color and a slope

    Parameters
    ----------
    color_length : `float`
        The maximum length of time different filters be observed
        to still count as a color (hours). Default 1 hour.
    slope_length : `float`
        The length of time to demand observations in the
        same filter be greater than (hours). Default 3 hours.
    """

    def __init__(
        self, color_length=1.0, slope_length=3.0, filter_col="filter", mjd_col="observationStartMJD"
    ):
        self.color_length = color_length / 24.0
        self.slope_length = slope_length / 24.0

        self.filter_col = filter_col
        self.mjd_col = mjd_col

    def __call__(self, data_slice):
        has_color = False
        has_slope = False

        if np.size(data_slice) < 3:
            return 0
        filters = data_slice[self.filter_col]

        u_filters = np.unique(filters)

        for filtername in u_filters:
            in_filt = np.where(data_slice[self.filter_col] == filtername)[0]
            time_gap = (
                data_slice[self.mjd_col][in_filt].max() - data_slice[self.mjd_col][in_filt][np.newaxis].min()
            )
            if time_gap >= self.slope_length:
                has_slope = True
                break
        for filtername1 in u_filters:
            for filtername2 in u_filters:
                if filtername1 != filtername2:
                    in_filt1 = np.where(filters == filtername1)[0]
                    in_filt2 = np.where(filters == filtername2)[0]
                    time_gaps = (
                        data_slice[self.mjd_col][in_filt1] - data_slice[self.mjd_col][in_filt2][np.newaxis].T
                    )
                    time_gaps = time_gaps[np.where(time_gaps > 0)]
                    if time_gaps.size > 0:
                        if np.min(time_gaps[np.where(time_gaps > 0)]) <= self.color_length:
                            has_color = True
                            break
        if has_color & has_slope:
            return 1
        else:
            return 0


class ColorSlopeMetric(BaseMetric):
    """How many times do we get a color and slope in a night

    A proxy metric for seeing how many times
    there would be the possibility of identifying and
    classifying a transient.

    Parameters
    ----------
    mag : `dict`
        Dictionary with filternames as keys and minimum depth m5
        magnitudes as values. If None, defaults to mag 20 in ugrizy.
    color_length : `float`
        The maximum length of time different filters be observed
        to still count as a color (hours). Default 1 hour.
    slope_length : `float`
        The length of time to demand observations in the
        same filter be greater than (hours). Default 3 hours.
    """

    def __init__(
        self,
        mag=None,
        night_col="night",
        filter_col="filter",
        m5_col="fiveSigmaDepth",
        color_length=1.0,
        slope_length=3.0,
        time_col="observationStartMJD",
        units="#",
        metric_name="ColorSlope",
        **kwargs,
    ):
        cols = [filter_col, night_col, m5_col, time_col]

        if mag is None:
            mag = {"u": 20, "g": 20, "r": 20, "i": 20, "z": 20, "y": 20}

        self.night_col = night_col
        self.filter_col = filter_col
        self.m5_col = m5_col
        self.mag = mag
        self.time_col = time_col

        super().__init__(col=cols, units=units, metric_name=metric_name, **kwargs)

        self.sequence_checker = CheckColorSlope(color_length=color_length, slope_length=slope_length)

    def run(self, data_slice, slice_point=None):
        result = 0
        deep_enough = np.zeros(data_slice.size, dtype=bool)
        for filtername in np.unique(data_slice[self.filter_col]):
            in_filt = np.where(data_slice[self.filter_col] == filtername)[0]
            indx = np.where(data_slice[self.m5_col][in_filt] > self.mag[filtername])[0]
            deep_enough[in_filt[indx]] = True
        data = data_slice[deep_enough]
        if data.size > 0:
            _night, result = int_binned_stat(data[self.night_col], data, statistic=self.sequence_checker)

        return np.sum(result)


class ColorSlope2NightMetric(ColorSlopeMetric):
    """Like ColorSlopeMetric, but span over 2 nights

    Parameters
    ----------
    mag : `dict`
        Dictionary with filternames as keys and minimum depth m5
        magnitudes as values. If None, defaults to mag 20 in ugrizy.
    color_length : `float`
        The maximum length of time different filters be observed
        to still count as a color (hours). Default 1 hour.
    slope_length : `float`
        The length of time to demand observations in the
        same filter be greater than (hours). Default 15 hours.
    """

    def __init__(
        self,
        mag=None,
        night_col="night",
        filter_col="filter",
        m5_col="fiveSigmaDepth",
        color_length=1.0,
        slope_length=15.0,
        time_col="observationStartMJD",
        units="#",
        metric_name="ColorSlope2Night",
        **kwargs,
    ):
        super().__init__(
            mag=mag,
            night_col=night_col,
            filter_col=filter_col,
            m5_col=m5_col,
            color_length=color_length,
            slope_length=slope_length,
            time_col=time_col,
            units=units,
            metric_name=metric_name,
            **kwargs,
        )

    def run(self, data_slice, slice_point=None):
        result = 0
        deep_enough = np.zeros(data_slice.size, dtype=bool)
        for filtername in np.unique(data_slice[self.filter_col]):
            in_filt = np.where(data_slice[self.filter_col] == filtername)[0]
            indx = np.where(data_slice[self.m5_col][in_filt] > self.mag[filtername])[0]
            deep_enough[in_filt[indx]] = True
        data = data_slice[deep_enough]
        if data.size > 0:
            # Send in nights as pairs, (0,1) (2,3), (4,5), etc
            night_id = np.floor(data[self.night_col] / 2).astype(int)
            _night, result1 = int_binned_stat(night_id, data, statistic=self.sequence_checker)

            # Now to do pairs (1,2), (3,4)
            night_id = np.ceil(data[self.night_col] / 2).astype(int)
            _night, result2 = int_binned_stat(night_id, data, statistic=self.sequence_checker)

            result = np.sum(result1) + np.sum(result2)

        return result
