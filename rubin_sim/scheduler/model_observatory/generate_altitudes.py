__all__ = ("generate_nights",)

import numpy as np
from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
from astropy.time import Time
from scipy.optimize import Bounds, minimize

from rubin_sim.utils import Site


def lin_interp(x, x0, x1, y0, y1):
    """
    Do a bunch of linear interpolations
    """
    y = y0 * (1.0 - (x - x0) / (x1 - x0)) + y1 * (x - x0) / (x1 - x0)
    return y


def alt_passing_interp(times, altitudes, goal_alt=0.0, rising=True):
    """find time when a body passses some altitude"""
    if rising:
        below = np.where((altitudes < goal_alt) & (np.roll(altitudes, -1) > goal_alt))[0]
        above = below + 1
    else:
        above = np.where((altitudes > goal_alt) & (np.roll(altitudes, -1) < goal_alt))[0]
        below = above + 1

    if (below.max() >= np.size(below)) | (above.max() >= np.size(above)):
        below = below[:-1]
        above = above[:-1]

    pass_times = lin_interp(goal_alt, altitudes[above], altitudes[below], times[above], times[below])
    return pass_times


def alt_sun_sum(in_mjds, location, offset):
    times = Time(in_mjds, format="mjd")
    sun = get_sun(times)
    aa = AltAz(location=location, obstime=times)
    sun = sun.transform_to(aa)
    result = np.sum(np.abs(sun.alt.deg + offset))
    return result


def alt_moon_sum(in_mjds, location, offset):
    times = Time(in_mjds, format="mjd")
    moon = get_body("moon", times)
    aa = AltAz(location=location, obstime=times)
    moon = moon.transform_to(aa)
    result = np.sum(np.abs(moon.alt.deg + offset))
    return result


def generate_nights(mjd_start, duration=3653.0, rough_step=2, verbose=False):
    """Generate the sunset and twilight times for a range of dates

    Parameters
    ----------
    mjd_start : float
        The starting mjd
    duration : float (3653)
        How long to compute times for (days)
    rough_step : float (2.)
        Time step for computing first pass rough sunrise times (hours)
    """

    # Let's find the nights first, find the times where the sun crosses the meridian.
    site = Site("LSST")
    location = EarthLocation(lat=site.latitude, lon=site.longitude, height=site.height)
    # go on 1/10th of a day steps
    t_step = rough_step / 24.0
    pad_around = 30.0 / 24.0
    t_sparse = Time(
        np.arange(mjd_start - pad_around, duration + mjd_start + pad_around + t_step, t_step),
        format="mjd",
        location=location,
    )
    sun = get_sun(t_sparse)
    aa = AltAz(location=location, obstime=t_sparse)
    sun_aa_sparse = sun.transform_to(aa)

    moon = get_body("moon", t_sparse)
    moon_aa_sparse = moon.transform_to(aa)

    # Indices right before sunset
    mjd_sunset_rough = alt_passing_interp(t_sparse.mjd, sun_aa_sparse.alt.deg, goal_alt=0.0, rising=False)

    names = [
        "night",
        "sunset",
        "sun_n12_setting",
        "sun_n18_setting",
        "sun_n18_rising",
        "sun_n12_rising",
        "sunrise",
        "moonrise",
        "moonset",
    ]
    types = [int]
    types.extend([float] * (len(names) - 1))
    alt_info_array = np.zeros(mjd_sunset_rough.size, dtype=list(zip(names, types)))
    alt_info_array["sunset"] = mjd_sunset_rough
    # label the nights
    alt_info_array["night"] = np.arange(mjd_sunset_rough.size)
    night_1_index = np.searchsorted(alt_info_array["sunset"], mjd_start)
    alt_info_array["night"] += 1 - alt_info_array["night"][night_1_index]

    # OK, now I have sunset times
    sunrises = alt_passing_interp(t_sparse.mjd, sun_aa_sparse.alt.deg, goal_alt=0.0, rising=True)
    # Make sure sunrise happens after sunset
    insert_indices = np.searchsorted(alt_info_array["sunset"], sunrises, side="left") - 1
    good_indices = np.where(insert_indices > 0)[0]
    alt_info_array["sunrise"][insert_indices[good_indices]] = sunrises[good_indices]

    # Should probably write the function to get rin of the copy-pasta
    point = alt_passing_interp(t_sparse.mjd, sun_aa_sparse.alt.deg, goal_alt=-12.0, rising=False)
    insert_indices = np.searchsorted(alt_info_array["sunset"], point, side="left") - 1
    good_indices = np.where(insert_indices > 0)[0]
    alt_info_array["sun_n12_setting"][insert_indices[good_indices]] = point[good_indices]

    point = alt_passing_interp(t_sparse.mjd, sun_aa_sparse.alt.deg, goal_alt=-12.0, rising=True)
    insert_indices = np.searchsorted(alt_info_array["sunset"], point, side="left") - 1
    good_indices = np.where(insert_indices > 0)[0]
    alt_info_array["sun_n12_rising"][insert_indices[good_indices]] = point[good_indices]

    point = alt_passing_interp(t_sparse.mjd, sun_aa_sparse.alt.deg, goal_alt=-18.0, rising=True)
    insert_indices = np.searchsorted(alt_info_array["sunset"], point, side="left") - 1
    good_indices = np.where(insert_indices > 0)[0]
    alt_info_array["sun_n18_rising"][insert_indices[good_indices]] = point[good_indices]

    point = alt_passing_interp(t_sparse.mjd, sun_aa_sparse.alt.deg, goal_alt=-18.0, rising=False)
    insert_indices = np.searchsorted(alt_info_array["sunset"], point, side="left") - 1
    good_indices = np.where(insert_indices > 0)[0]
    alt_info_array["sun_n18_setting"][insert_indices[good_indices]] = point[good_indices]

    point = alt_passing_interp(t_sparse.mjd, moon_aa_sparse.alt.deg, goal_alt=0.0, rising=True)
    insert_indices = np.searchsorted(alt_info_array["sunset"], point, side="left") - 1
    good_indices = np.where(insert_indices > 0)[0]
    alt_info_array["moonrise"][insert_indices[good_indices]] = point[good_indices]

    point = alt_passing_interp(t_sparse.mjd, moon_aa_sparse.alt.deg, goal_alt=0.0, rising=False)
    insert_indices = np.searchsorted(alt_info_array["sunset"], point, side="left") - 1
    good_indices = np.where(insert_indices > 0)[0]
    alt_info_array["moonset"][insert_indices[good_indices]] = point[good_indices]

    # Crop off some regions that might not have been filled
    good = np.where(alt_info_array["night"] > 0)[0]
    alt_info_array = alt_info_array[good[:-1]]

    # Put bounds so we don't go to far from rough fit
    refined_mjds = np.empty_like(alt_info_array)
    refined_mjds["night"] = alt_info_array["night"]

    names_dict = {
        "sunset": 0.0,
        "sun_n12_setting": 12.0,
        "sun_n18_setting": 18.0,
        "sun_n18_rising": 18.0,
        "sun_n12_rising": 12.0,
        "sunrise": 0,
    }

    # Need to keep the runtime reasonable
    options = {"maxiter": 5}
    for key in names_dict:
        if verbose:
            print(key)
        bounds = Bounds(
            alt_info_array[key] - rough_step / 10.0 / 24.0,
            alt_info_array[key] + rough_step / 10.0 / 24.0,
        )
        new_mjds = minimize(
            alt_sun_sum,
            alt_info_array[key],
            bounds=bounds,
            args=(location, names_dict[key]),
            options=options,
        )
        refined_mjds[key] = new_mjds.x

    for key in ["moonrise", "moonset"]:
        if verbose:
            print(key)
        bounds = Bounds(
            alt_info_array[key] - rough_step / 10.0 / 24.0,
            alt_info_array[key] + rough_step / 10.0 / 24.0,
        )
        new_mjds = minimize(
            alt_moon_sum,
            alt_info_array[key],
            bounds=bounds,
            args=(location, 0.0),
            options=options,
        )
        refined_mjds[key] = new_mjds.x

    # Note, there is the possibility that some moonrise/moonset times changed nights upon refinement. I suppose
    # I could do another seatchsorted pass here just to be extra sure nothing changed.

    return alt_info_array, refined_mjds


if __name__ == "__main__":
    # Let's use astropy to pre-compute the sunrise/sunset/twilight/moonrise/moonset times we're interested in.
    mjd_start = 59853.5
    #
    rough_times, refined_mjds = generate_nights(
        mjd_start - 365.25 * 2 - 40.0, duration=365.25 * 24 + 80, rough_step=2
    )
    # rough_times, refined_mjds = generate_nights(mjd_start, duration=50, rough_step=2)
    # Maybe just use pandas to dump it to a csv file?
    np.savez("night_info.npz", rough_times=rough_times, refined_mjds=refined_mjds)
