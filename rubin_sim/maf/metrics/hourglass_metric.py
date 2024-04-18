__all__ = ("HourglassMetric",)

import numpy as np
from astroplan import Observer
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
from astropy.time import Time
from rubin_scheduler.utils import Site

from .base_metric import BaseMetric


def nearest_val(A, val):
    return A[np.argmin(np.abs(np.array(A) - val))]


class HourglassMetric(BaseMetric):
    """Plot the filters used as a function of time.
    Must be used with the Hourglass Slicer.
    Will totally fail in the arctic circle.
    """

    def __init__(
        self,
        telescope="LSST",
        mjd_col="observationStartMJD",
        filter_col="filter",
        night_col="night",
        delta_t=60.0,
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.filter_col = filter_col
        self.night_col = night_col
        cols = [self.mjd_col, self.filter_col, self.night_col]
        super(HourglassMetric, self).__init__(col=cols, metric_dtype="object", **kwargs)
        self.telescope = Site(name=telescope)
        self.delta_t = delta_t / 60.0 / 24.0
        self.location = EarthLocation(
            lat=self.telescope.latitude_rad * u.rad,
            lon=self.telescope.longitude_rad * u.rad,
            height=self.telescope.height * u.m,
        )
        self.observer = Observer(location=self.location)

    def run(self, data_slice, slice_point=None):
        data_slice.sort(order=self.mjd_col)
        unights, uindx = np.unique(data_slice[self.night_col], return_index=True)

        mjds = np.arange(np.min(data_slice[self.mjd_col]), np.max(data_slice[self.mjd_col]) + 1, 0.5)

        # Define the breakpoints as where either the filter changes OR
        # there's more than a 2 minute gap in observing
        good = np.where(
            (data_slice[self.filter_col] != np.roll(data_slice[self.filter_col], 1))
            | (
                np.abs(np.roll(data_slice[self.mjd_col], 1) - data_slice[self.mjd_col])
                > 120.0 / 3600.0 / 24.0
            )
        )[0]
        good = np.concatenate((good, [0], [len(data_slice[self.filter_col])]))
        good = np.unique(good)
        left = good[:-1]
        right = good[1:] - 1
        good = np.ravel(list(zip(left, right)))

        names = ["mjd", "midnight", "filter"]
        types = ["float", "float", (np.str_, 1)]
        perfilter = np.zeros((good.size), dtype=list(zip(names, types)))
        perfilter["mjd"] = data_slice[self.mjd_col][good]
        perfilter["filter"] = data_slice[self.filter_col][good]

        # brute force compute midnight times for all days between
        # start and enc of data_slice
        times = Time(mjds, format="mjd")
        # let's just find the midnight before and after each of the
        # pre_night MJD values
        m_after = self.observer.midnight(times, "next")
        m_before = self.observer.midnight(times, "previous")
        try:
            midnights = np.unique(np.concatenate([m_before.mjd, m_after.mjd]).filled(np.nan))
        except AttributeError:
            midnights = np.unique(np.concatenate([m_before.mjd, m_after.mjd]))
        # calculating midnight can return nans? That seems bad.
        midnights = midnights[np.isfinite(midnights)]
        # chop off any repeats. Need to round because observe.midnight
        # values are not repeatable
        m10 = np.round(midnights * 10)
        _temp, indx = np.unique(m10, return_index=True)
        midnights = midnights[indx]
        names = [
            "mjd",
            "midnight",
            "moonPer",
            "twi6_rise",
            "twi6_set",
            "twi12_rise",
            "twi12_set",
            "twi18_rise",
            "twi18_set",
        ]
        types = ["float"] * len(names)
        pernight = np.zeros(len(midnights), dtype=list(zip(names, types)))
        pernight["midnight"] = midnights
        pernight["mjd"] = midnights

        # now for each perfilter, find the closes midnight
        indx = np.searchsorted(midnights, perfilter["mjd"])
        d1 = np.abs(perfilter["mjd"] - midnights[indx - 1])
        indx[np.where(indx >= midnights.size)] -= 1
        d2 = np.abs(perfilter["mjd"] - midnights[indx])

        perfilter["midnight"] = midnights[indx]
        temp_indx = np.where(d1 < d2)
        perfilter["midnight"][temp_indx] = midnights[indx - 1][temp_indx]
        mtime = Time(pernight["midnight"], format="mjd")

        pernight["twi12_rise"] = self.observer.twilight_morning_nautical(mtime, which="next").mjd
        pernight["twi12_set"] = self.observer.twilight_evening_nautical(mtime, which="previous").mjd

        pernight["twi18_rise"] = self.observer.twilight_morning_astronomical(mtime, which="next").mjd
        pernight["twi18_set"] = self.observer.twilight_evening_astronomical(mtime, which="previous").mjd

        aa = AltAz(location=self.location, obstime=mtime)
        moon_coords = get_body("moon", mtime).transform_to(aa)
        sun_coords = get_sun(mtime).transform_to(aa)
        ang_dist = sun_coords.separation(moon_coords)
        pernight["moonPer"] = ang_dist.deg / 180 * 100

        return {"pernight": pernight, "perfilter": perfilter}
