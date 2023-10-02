import sys

import astropy.units as u
import numpy as np
from astroplan import Observer
from astropy.time import Time

from rubin_sim.utils import Site

if __name__ == "__main__":
    # Trying out the astroplan sunrise/set code.
    # conda install -c astropy astroplan
    mjd_start = 59853.5 - 3.0 * 365.25
    duration = 50.0 * 365.25
    pad_around = 40
    t_step = 0.2

    site = Site("LSST")
    observer = Observer(
        longitude=site.longitude * u.deg,
        latitude=site.latitude * u.deg,
        elevation=site.height * u.m,
        name="LSST",
    )

    results = []

    mjd = mjd_start + 0
    counter = 0
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
    base_al = np.zeros(1, dtype=list(zip(names, types)))

    while mjd < (mjd_start + duration):
        times = Time(mjd, format="mjd")
        sunsets = observer.sun_set_time(times, which="next")
        times = sunsets

        almanac = base_al.copy()
        almanac["sunset"] = sunsets.mjd

        almanac["moonset"] = observer.moon_set_time(times, which="next").mjd
        almanac["moonrise"] = observer.moon_rise_time(times, which="next").mjd
        almanac["sun_n12_setting"] = observer.twilight_evening_nautical(times, which="next").mjd
        times = observer.twilight_evening_astronomical(times, which="next")
        almanac["sun_n18_setting"] = times.mjd
        almanac["sun_n18_rising"] = observer.twilight_morning_astronomical(times, which="next").mjd
        almanac["sun_n12_rising"] = observer.twilight_morning_nautical(times, which="next").mjd
        almanac["sunrise"] = observer.sun_rise_time(times, which="next").mjd
        results.append(almanac)
        mjd = almanac["sunrise"] + t_step

        progress = (mjd - mjd_start) / duration * 100
        text = "\rprogress = %.2f%%" % progress
        sys.stdout.write(text)
        sys.stdout.flush()

    almanac = np.concatenate(results)
    almanac["night"] = np.arange(almanac["night"].size)

    np.savez("sunsets.npz", almanac=almanac)

    # runs in real    193m21.246s
