"""A group of metrics that work together to evaluate season characteristics
(length, number, etc).
In addition, these support the time delay metric for strong lensing.
"""

__all__ = (
    "find_season_edges",
    "SeasonLengthMetric",
    "CampaignLengthMetric",
    "MeanCampaignFrequencyMetric",
    "TdcMetric",
)

import numpy as np
from rubin_scheduler.utils import calc_season

from rubin_sim.phot_utils import DustValues

from .base_metric import BaseMetric


def find_season_edges(seasons):
    """Given the seasons, return the indexes of each start/end of the season.

    Parameters
    ----------
    seasons : `np.ndarray`, (N,)
        Seasons, such as calculated by calc_season.
        Note that seasons should be sorted!!

    Returns
    -------
    first, last : `np.ndarray`, (N,), `np.ndarray`, (N,)
        The indexes of the first and last date in the season.
    """
    int_seasons = np.floor(seasons)
    # Get the unique seasons, so that we can separate each one
    season_list = np.unique(int_seasons)
    # Find the first and last observation of each season.
    first_of_season = np.searchsorted(int_seasons, season_list)
    last_of_season = np.searchsorted(int_seasons, season_list, side="right") - 1
    return first_of_season, last_of_season


class SeasonLengthMetric(BaseMetric):
    """Calculate the length of LSST seasons, in days.

    Parameters
    ----------
    min_exp_time : `float`, optional
        Minimum visit exposure time to count for a 'visit', in seconds.
        Default 20.
    reduce_func : function, optional
       Function that can operate on array-like structures.
       Typically numpy function.
       This reduces the season length in each season from 10 separate
       values to a single value.
       Default np.median.

    Returns
    -------
    seasonlength : `float`
        The (reduceFunc) of the length of each season, in days.
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        exp_time_col="visitExposureTime",
        min_exp_time=16,
        reduce_func=np.median,
        metric_name="SeasonLength",
        **kwargs,
    ):
        units = "days"
        self.mjd_col = mjd_col
        self.exp_time_col = exp_time_col
        self.min_exp_time = min_exp_time
        self.reduce_func = reduce_func
        super().__init__(
            col=[self.mjd_col, self.exp_time_col], units=units, metric_name=metric_name, **kwargs
        )

    def run(self, data_slice, slice_point):
        # Order data Slice/times and exclude visits which are too short.
        long = np.where(data_slice[self.exp_time_col] > self.min_exp_time)
        if len(long[0]) == 0:
            return self.badval
        data = np.sort(data_slice[long], order=self.mjd_col)
        # SlicePoints ra/dec are always in radians -
        # convert to degrees to calculate season
        seasons = calc_season(np.degrees(slice_point["ra"]), data[self.mjd_col])
        first_of_season, last_of_season = find_season_edges(seasons)
        seasonlengths = data[self.mjd_col][last_of_season] - data[self.mjd_col][first_of_season]
        result = self.reduce_func(seasonlengths)
        return result


class CampaignLengthMetric(BaseMetric):
    """Calculate the number of seasons (roughly, years) a pointing is observed.

    This corresponds to the 'campaign length' for lensed quasar time delays.
    """

    def __init__(
        self, mjd_col="observationStartMJD", exp_time_col="visitExposureTime", min_exp_time=20, **kwargs
    ):
        units = ""
        self.exp_time_col = exp_time_col
        self.min_exp_time = min_exp_time
        self.mjd_col = mjd_col
        super().__init__(col=[self.mjd_col, self.exp_time_col], units=units, **kwargs)

    def run(self, data_slice, slice_point):
        # Order data Slice/times and exclude visits which are too short.
        long = np.where(data_slice[self.exp_time_col] > self.min_exp_time)
        if len(long[0]) == 0:
            return self.badval
        data = np.sort(data_slice[long], order=self.mjd_col)
        seasons = calc_season(np.degrees(slice_point["ra"]), data[self.mjd_col])
        int_seasons = np.floor(seasons)
        count = len(np.unique(int_seasons))
        return count


class MeanCampaignFrequencyMetric(BaseMetric):
    """Calculate the mean separation between nights, within a season -
    then the mean over the campaign.

    Calculate per season, to avoid any influence from season gaps.
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        exp_time_col="visitExposureTime",
        min_exp_time=20,
        night_col="night",
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.exp_time_col = exp_time_col
        self.min_exp_time = min_exp_time
        self.night_col = night_col
        units = "nights"
        super().__init__(col=[self.mjd_col, self.exp_time_col, self.night_col], units=units, **kwargs)

    def run(self, data_slice, slice_point):
        # Order data Slice/times and exclude visits which are too short.
        long = np.where(data_slice[self.exp_time_col] > self.min_exp_time)
        if len(long[0]) == 0:
            return self.badval
        data = np.sort(data_slice[long], order=self.mjd_col)
        # SlicePoints ra/dec are always in radians -
        # convert to degrees to calculate season
        seasons = calc_season(np.degrees(slice_point["ra"]), data[self.mjd_col])
        first_of_season, last_of_season = find_season_edges(seasons)
        season_means = np.zeros(len(first_of_season), float)
        for i, (first, last) in enumerate(zip(first_of_season, last_of_season)):
            if first < last:
                n = data[self.night_col][first : last + 1]
                delta_nights = np.diff(np.unique(n))
                if len(delta_nights) > 0:
                    season_means[i] = np.mean(delta_nights)
        return np.mean(season_means)


class TdcMetric(BaseMetric):
    """Calculate the Time Delay Challenge metric,
    as described in Liao et al 2015 (https://arxiv.org/pdf/1409.1254.pdf).

    This combines the MeanCampaignFrequency/MeanNightSeparation,
    the SeasonLength, and the CampaignLength
    metrics above, but rewritten to calculate season information only once.

    cad_norm = in units of days
    sea_norm = in units of months
    camp_norm = in units of years

    This metric also adds a requirement to achieve limiting magnitudes
    after galactic dust extinction, in various bandpasses,
    in order to exclude visits which are not useful for detecting quasars
    (due to being short or having high sky brightness, etc.) and to
    reject regions with high galactic dust extinction.

    Parameters
    ----------
    mjd_col : `str`, optional
        Column name for mjd. Default observationStartMJD.
    night_col : `str`, optional
        Column name for night. Default night.
    filter_col : `str`, optional
        Column name for filter. Default filter.
    m5_col : `str`, optional
        Column name for five-sigma depth. Default fiveSigmaDepth.
    mag_cuts : `dict`, optional
        Dictionary with filtername:mag limit (after dust extinction).
        Default None in kwarg.
        Defaults set within metric:
        {'u': 22.7, 'g': 24.1, 'r': 23.7, 'i': 23.1, 'z': 22.2, 'y': 21.4}
    metricName : `str`, optional
        Metric Name. Default TDC.
    cad_norm : `float`, optional
        Cadence normalization constant, in units of days. Default 3.
    sea_norm : `float`, optional
        Season normalization constant, in units of months. Default 4.
    camp_norm : `float`, optional
        Campaign length normalization constant, in units of years. Default 5.
    badval : `float`, optional
        Return this value instead of the dictionary for bad points.

    Returns
    -------
    TDCmetrics : `dict`
        Dictionary of values for {'rate', 'precision', 'accuracy'}
        at this point in the sky.
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        night_col="night",
        filter_col="filter",
        m5_col="fiveSigmaDepth",
        mag_cuts=None,
        metric_name="TDC",
        cad_norm=3.0,
        sea_norm=4.0,
        camp_norm=5.0,
        badval=-999,
        **kwargs,
    ):
        # Save the normalization values.
        self.cad_norm = cad_norm
        self.sea_norm = sea_norm
        self.camp_norm = camp_norm
        self.mjd_col = mjd_col
        self.m5_col = m5_col
        self.night_col = night_col
        self.filter_col = filter_col
        if mag_cuts is None:
            self.mag_cuts = {
                "u": 22.7,
                "g": 24.1,
                "r": 23.7,
                "i": 23.1,
                "z": 22.2,
                "y": 21.4,
            }
        else:
            self.mag_cuts = mag_cuts
            if not isinstance(self.mag_cuts, dict):
                raise Exception("mag_cuts should be a dictionary")
        # Set up dust map requirement
        maps = ["DustMap"]
        # Set the default wavelength limits for the lsst filters.
        # These are approximately correct.
        dust_properties = DustValues()
        self.ax1 = dust_properties.ax1
        super().__init__(
            col=[self.mjd_col, self.m5_col, self.night_col, self.filter_col],
            badval=badval,
            maps=maps,
            metric_name=metric_name,
            units="%s" % ("%"),
            **kwargs,
        )

    def run(self, data_slice, slice_point):
        # Calculate dust-extinction limiting magnitudes for each visit.
        filterlist = np.unique(data_slice[self.filter_col])
        m5_dust = np.zeros(len(data_slice), float)
        for f in filterlist:
            match = np.where(data_slice[self.filter_col] == f)
            a_x = self.ax1[f] * slice_point["ebv"]
            m5_dust[match] = data_slice[self.m5_col][match] - a_x
            m5_dust[match] = np.where(m5_dust[match] > self.mag_cuts[f], m5_dust[match], -999)
        idxs = np.where(m5_dust > -998)
        if len(idxs[0]) == 0:
            return self.badval
        data = np.sort(data_slice[idxs], order=self.mjd_col)
        # SlicePoints ra/dec are always in radians -
        # convert to degrees to calculate season
        seasons = calc_season(np.degrees(slice_point["ra"]), data[self.mjd_col])
        int_seasons = np.floor(seasons)
        first_of_season, last_of_season = find_season_edges(seasons)
        # Campaign length
        camp = len(np.unique(int_seasons))
        # Season length
        seasonlengths = data[self.mjd_col][last_of_season] - data[self.mjd_col][first_of_season]
        sea = np.median(seasonlengths)
        # Convert to months
        sea = sea / 30.0
        # Campaign frequency / mean night separation
        season_means = np.zeros(len(first_of_season), float)
        for i, (first, last) in enumerate(zip(first_of_season, last_of_season)):
            n = data[self.night_col][first : last + 1]
            delta_nights = np.diff(np.unique(n))
            if len(delta_nights) > 0:
                season_means[i] = np.mean(delta_nights)
        cad = np.mean(season_means)
        # Evaluate precision and accuracy for TDC
        if sea == 0 or cad == 0 or camp == 0:
            return self.badval
        else:
            accuracy = 0.06 * (self.sea_norm / sea) * (self.camp_norm / camp) ** (1.1)
            precision = (
                4.0
                * (cad / self.cad_norm) ** (0.7)
                * (self.sea_norm / sea) ** (0.3)
                * (self.camp_norm / camp) ** (0.6)
            )
            rate = (
                30.0
                * (self.cad_norm / cad) ** (0.4)
                * (sea / self.sea_norm) ** (0.8)
                * (self.camp_norm / camp) ** (0.2)
            )
        return {
            "accuracy": accuracy,
            "precision": precision,
            "rate": rate,
            "cadence (days)": cad,
            "season (months)": sea,
            "campaign": camp,
        }

    def reduce_accuracy(self, metric_value):
        return metric_value["accuracy"]

    def reduce_precision(self, metric_value):
        return metric_value["precision"]

    def reduce_rate(self, metric_value):
        return metric_value["rate"]

    def reduce_cadence(self, metric_value):
        return metric_value["cadence (days)"]

    def reduce_season(self, metric_value):
        return metric_value["season (months)"]

    def reduce_campaign(self, metric_value):
        return metric_value["campaign"]
