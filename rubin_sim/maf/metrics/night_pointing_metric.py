__all__ = ("NightPointingMetric",)

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
from astropy.time import Time
from rubin_scheduler.utils import Site

from .base_metric import BaseMetric


class NightPointingMetric(BaseMetric):
    """
    Gather relevant information for a night to plot.
    """

    def __init__(
        self,
        alt_col="altitude",
        az_col="azimuth",
        filter_col="filter",
        mjd_col="observationStartMJD",
        metric_name="NightPointing",
        telescope="LSST",
        **kwargs,
    ):
        cols = [alt_col, az_col, filter_col, mjd_col]
        super(NightPointingMetric, self).__init__(
            col=cols, metric_name=metric_name, metric_dtype="object", **kwargs
        )
        self.telescope = Site(name=telescope)
        self.alt_col = alt_col
        self.az_col = az_col
        self.filter_col = filter_col
        self.mjd_col = mjd_col

        self.location = EarthLocation(
            lat=self.telescope.latitude_rad * u.rad,
            lon=self.telescope.longitude_rad * u.rad,
            height=self.telescope.height * u.m,
        )

    def run(self, data_slice, slice_point=None):
        pad = 30.0 / 60.0 / 24.0
        mjd_min = data_slice[self.mjd_col].min() - pad
        mjd_max = data_slice[self.mjd_col].max() + pad

        # How often to plot the moon and things
        step = 20.0 / 60.0 / 24.0
        mjds = Time(np.arange(mjd_min, mjd_max + step, step), format="mjd")

        aa = AltAz(location=self.location, obstime=mjds)

        moon_coords = get_body("moon", mjds).transform_to(aa)
        sun_coords = get_sun(mjds).transform_to(aa)

        moon_alts = np.array(moon_coords.alt.rad)
        moon_azs = np.array(moon_coords.az.rad)
        mjds = np.array(mjds)
        sun_alts = np.array(sun_coords.alt.rad)
        sun_azs = np.array(sun_coords.az.rad)

        return {
            "data_slice": data_slice,
            "moon_alts": moon_alts,
            "moon_azs": moon_azs,
            "mjds": mjds,
            "sun_alts": sun_alts,
            "sun_azs": sun_azs,
        }
