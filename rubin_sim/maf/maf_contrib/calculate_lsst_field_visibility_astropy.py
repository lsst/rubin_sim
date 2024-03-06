"""
Created on Tue Sep 18 13:35:41 2018

@author: rstreet
"""

__all__ = ("calculate_lsst_field_visibility", "calculate_lsst_field_visibility_fast")


import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time, TimeDelta
from rubin_scheduler.utils import Site, approx_ra_dec2_alt_az


def calculate_lsst_field_visibility(
    ra, dec, start_date, end_date, min_alt=30.0, sun_alt_limit=18.0, sample_rate=0.0007, verbose=False
):
    """Method to calculate the visibility of a given RA and Dec from LSST
    over the course of a year

    Adapted from an example in the Astropy docs.

    Parameters
    ----------
    ra : `float`
        RA in decimal degrees.
    dec : `float`
        Declination in decimal degrees
    start_date : `astropy.time.Time`
        Start date for calculations
    end_date : `astropy.time.Time`
        End date for calculations
    min_alt : `float`, optional
        Minimal altitude for field
    sun_alt_limit : `float`, optional
        Maximum sun altitude to consider for visibility
    sample_rate : `float`, optional
        Time spacing between visibility tests (days)
    verbose : `bool`, optional
        Output extra information, including debugging
    """
    field = SkyCoord(ra, dec, frame="icrs", unit=(u.deg, u.deg))

    lsst_site = Site("LSST")
    lsst = EarthLocation(
        lat=lsst_site.latitude * u.deg,
        lon=lsst_site.longitude * u.deg,
        height=lsst_site.height * u.m,
    )

    total_time_visible = 0.0

    dates = np.arange(start_date, end_date, TimeDelta(1, format="jd", scale="tai"))

    target_alts = []
    hrs_visible_per_night = []
    hrs_per_night = []

    for d in dates:
        intervals = np.arange(0.0, 1.0, sample_rate)

        dt = TimeDelta(intervals, format="jd", scale="tai")

        ts = d + dt

        frame = AltAz(obstime=ts, location=lsst)

        altaz = field.transform_to(frame)

        alts = np.array((altaz.alt * u.deg).value)

        idx = np.where(alts >= min_alt)[0]

        sun_altaz = get_sun(ts).transform_to(frame)

        sun_alts = np.array((sun_altaz.alt * u.deg).value)

        jdx = np.where(sun_alts < sun_alt_limit)[0]
        # Hours available in the sun
        hrs_per_night.append(sample_rate * len(sun_alts[jdx]) * 24.0)
        # The indexes where sun down and target above min_alt
        idx = list(set(idx).intersection(set(jdx)))
        # The highest altitude for the target, when the sun is down
        target_alts.append(alts[jdx].max())

        if len(idx) > 0:
            ts_vis = ts[idx]

            tvis = sample_rate * len(ts_vis)

            total_time_visible += tvis

            if verbose:
                print("Target visible from LSST for " + str(round(tvis * 24.0, 2)) + "hrs on " + d.isot)
            # Hours of visibility of the target on this night
            hrs_visible_per_night.append((tvis * 24.0))

        else:
            # target_alts.append(-1e5)
            hrs_visible_per_night.append(0.0)

            if verbose:
                print("Target not visible from LSST on " + d.isot)

    return np.array(total_time_visible), np.array(hrs_visible_per_night)


def calculate_lsst_field_visibility_fast(
    ra,
    dec,
    start_date,
    end_date,
    min_alt=30.0,
    sun_alt_limit=18.0,
    sample_rate=0.0007,
):
    """Method to calculate the visibility of a given RA and Dec from LSST
    over the course of a year

    Skips astropy calculation of alt/az which is slow and uses
    approximate transform from rubin_scheduler.

    Parameters
    ----------
    ra : `float`
        RA in decimal degrees.
    dec : `float`
        Declination in decimal degrees
    start_date : `astropy.time.Time`
        Start date for calculations
    end_date : `astropy.time.Time`
        End date for calculations
    min_alt : `float`, optional
        Minimal altitude for field
    sun_alt_limit : `float`, optional
        Maximum sun altitude to consider for visibility
    cadence : `float`, optional
        Time spacing between visibility tests (days)

    Returns
    -------
    tvisible : `float`
        Total time target is visible (days)
    dates_visible : `np.ndarray`, (N,)
        Dates that target is above min_alt and sun is below sun_alt_limit,
        within start_date to end_date.
    """
    lsst_site = Site("LSST")

    dates = np.arange(start_date.mjd, end_date.mjd + sample_rate / 2.0, sample_rate)

    alts, _ = approx_ra_dec2_alt_az(ra, dec, lsst_site.latitude, lsst_site.longitude, dates)
    # where is the target above the minimum altitude
    target_high = np.where(alts >= min_alt)[0]

    # when is the sun above the sun_alt_limit
    dates = dates[target_high]
    sun_locs = get_sun(Time(dates, format="mjd", scale="utc"))
    sun_alts, _ = approx_ra_dec2_alt_az(
        sun_locs.ra.deg, sun_locs.dec.deg, lsst_site.latitude, lsst_site.longitude, dates
    )
    sun_low = np.where(sun_alts <= sun_alt_limit)[0]

    dates = dates[sun_low]
    # total amount of time target is visible, in days
    tvisible = len(dates) * sample_rate

    return tvisible, dates
