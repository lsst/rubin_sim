import numpy as np
from .base_metric import BaseMetric
from rubin_sim.utils import Site
from astropy.coordinates import get_sun, get_moon, EarthLocation, AltAz
from astropy import units as u
from astropy.time import Time
from astroplan import Observer


__all__ = ["HourglassMetric"]


def nearestVal(A, val):
    return A[np.argmin(np.abs(np.array(A) - val))]


class HourglassMetric(BaseMetric):
    """Plot the filters used as a function of time. Must be used with the Hourglass Slicer.
    Will totally fail in the arctic circle."""

    def __init__(
        self,
        telescope="LSST",
        mjdCol="observationStartMJD",
        filterCol="filter",
        nightCol="night",
        delta_t=60.0,
        **kwargs
    ):
        self.mjdCol = mjdCol
        self.filterCol = filterCol
        self.nightCol = nightCol
        cols = [self.mjdCol, self.filterCol, self.nightCol]
        super(HourglassMetric, self).__init__(col=cols, metricDtype="object", **kwargs)
        self.telescope = Site(name=telescope)
        self.delta_t = delta_t / 60.0 / 24.0
        self.location = EarthLocation(
            lat=self.telescope.latitude_rad * u.rad,
            lon=self.telescope.longitude_rad * u.rad,
            height=self.telescope.height * u.m,
        )
        self.observer = Observer(location=self.location)

    def run(self, dataSlice, slicePoint=None):

        dataSlice.sort(order=self.mjdCol)
        unights, uindx = np.unique(dataSlice[self.nightCol], return_index=True)

        mjds = np.arange(
            np.min(dataSlice[self.mjdCol]), np.max(dataSlice[self.mjdCol]) + 1, 0.5
        )

        # Define the breakpoints as where either the filter changes OR
        # there's more than a 2 minute gap in observing
        good = np.where(
            (dataSlice[self.filterCol] != np.roll(dataSlice[self.filterCol], 1))
            | (
                np.abs(np.roll(dataSlice[self.mjdCol], 1) - dataSlice[self.mjdCol])
                > 120.0 / 3600.0 / 24.0
            )
        )[0]
        good = np.concatenate((good, [0], [len(dataSlice[self.filterCol])]))
        good = np.unique(good)
        left = good[:-1]
        right = good[1:] - 1
        good = np.ravel(list(zip(left, right)))

        names = ["mjd", "midnight", "filter"]
        types = ["float", "float", (np.str_, 1)]
        perfilter = np.zeros((good.size), dtype=list(zip(names, types)))
        perfilter["mjd"] = dataSlice[self.mjdCol][good]
        perfilter["filter"] = dataSlice[self.filterCol][good]

        # brute force compute midnight times for all days between start and enc of dataSlice
        times = Time(mjds, format="mjd")
        # let's just find the midnight before and after each of the pre_night MJD values
        m_after = self.observer.midnight(times, "next")
        m_before = self.observer.midnight(times, "previous")
        midnights = np.unique(np.concatenate([m_before.mjd, m_after.mjd]))
        # calculating midnight can return nans? That seems bad.
        midnights = midnights[np.isfinite(midnights)]
        # chop off any repeats. Need to round because observe.midnight values are not repeatable
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
        try:
            mtime = Time(pernight["midnight"], format="mjd")
        except:
            import pdb

            pdb.set_trace()
        pernight["twi12_rise"] = self.observer.twilight_morning_nautical(
            mtime, which="next"
        ).mjd
        pernight["twi12_set"] = self.observer.twilight_evening_nautical(
            mtime, which="previous"
        ).mjd

        pernight["twi18_rise"] = self.observer.twilight_morning_astronomical(
            mtime, which="next"
        ).mjd
        pernight["twi18_set"] = self.observer.twilight_evening_astronomical(
            mtime, which="previous"
        ).mjd

        aa = AltAz(location=self.location, obstime=mtime)
        moon_coords = get_moon(mtime).transform_to(aa)
        sun_coords = get_sun(mtime).transform_to(aa)
        ang_dist = sun_coords.separation(moon_coords)
        pernight["moonPer"] = ang_dist.deg / 180 * 100

        return {"pernight": pernight, "perfilter": perfilter}
