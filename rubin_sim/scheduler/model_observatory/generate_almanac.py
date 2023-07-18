import astropy.units as u
import numpy as np
from astroplan import Observer
from astropy.time import Time

from rubin_sim.utils import Site

if __name__ == "__main__":
    # Trying out the astroplan sunrise/set code.
    # conda install -c astropy astroplan
    mjd_start = 59853.5 - 3.0 * 365.25
    duration = 25.0 * 365.25
    pad_around = 40
    t_step = 0.7

    mjds = np.arange(mjd_start - pad_around, duration + mjd_start + pad_around + t_step, t_step)

    site = Site("LSST")
    observer = Observer(
        longitude=site.longitude * u.deg,
        latitude=site.latitude * u.deg,
        elevation=site.height * u.m,
        name="LSST",
    )

    # This blows up if I try to do it all at once? 250 GB of memory?
    results = []

    mjds_list = np.array_split(mjds, 500)

    for i, mjds in enumerate(mjds_list):
        print("chunk %i of %i" % (i, len(mjds_list)))
        times = Time(mjds, format="mjd")
        print("getting sunsets")
        sunsets = observer.sun_set_time(times)

        sunsets = np.unique(np.round(sunsets.mjd, decimals=4))

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
        almanac = np.zeros(sunsets.size, dtype=list(zip(names, types)))
        almanac["sunset"] = sunsets

        times = Time(sunsets, format="mjd")
        print("evening twilight 1")
        almanac["sun_n12_setting"] = observer.twilight_evening_nautical(times).mjd
        almanac["sun_n18_setting"] = observer.twilight_evening_astronomical(times).mjd
        almanac["sun_n18_rising"] = observer.twilight_morning_astronomical(times).mjd
        almanac["sun_n12_rising"] = observer.twilight_morning_nautical(times).mjd
        almanac["sunrise"] = observer.sun_rise_time(times).mjd
        almanac["moonset"] = observer.moon_set_time(times).mjd
        print("moonrise")
        almanac["moonrise"] = observer.moon_rise_time(times).mjd
        results.append(almanac)

    almanac = np.concatenate(results)
    umjds, indx = np.unique(almanac["sunset"], return_index=True)
    almanac = almanac[indx]
    almanac["night"] = np.arange(almanac["night"].size)

    np.savez("sunsets.npz", almanac=almanac)
