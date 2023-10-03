import numpy as np
from astropy.coordinates import EarthLocation, get_body, solar_system_ephemeris
from astropy.time import Time

from rubin_sim.utils import Site

if __name__ == "__main__":
    mjd_start = 59853.5 - 3.0 * 365.25
    duration = 30.0 * 365.25
    pad_around = 40
    t_step = 1.0 / 24.0  # Start with 1-hour timesteps.

    mjds = np.arange(mjd_start - pad_around, duration + mjd_start + pad_around + t_step, t_step)

    planet_names = ["venus", "mars", "jupiter", "saturn"]

    names = ["mjd"]
    for pn in planet_names:
        names.append(pn + "_RA")
        names.append(pn + "_dec")

    types = [float] * len(names)

    planet_loc = np.zeros(mjds.size, dtype=list(zip(names, types)))
    planet_loc["mjd"] = np.arange(mjd_start - pad_around, duration + mjd_start + pad_around + t_step, t_step)
    site = Site("LSST")
    location = EarthLocation(lat=site.latitude, lon=site.longitude, height=site.height)
    t_sparse = Time(mjds, format="mjd", location=location)

    # Should get dowloaded and chached if needed. https://docs.astropy.org/en/stable/coordinates/solarsystem.html
    # might need to `pip install jplephem`
    solar_system_ephemeris.set("de432s")

    for pn in planet_names:
        print("computing locations of %s" % pn)
        body = get_body(pn, t_sparse, location)
        body_icrs = body.icrs
        planet_loc[pn + "_RA"] = body_icrs.ra.rad
        planet_loc[pn + "_dec"] = body_icrs.dec.rad

    # I could crop off the times when the sun is up, but meh, bits are cheap
    np.savez("planet_locations.npz", planet_loc=planet_loc)
