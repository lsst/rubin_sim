import os
import sys

import h5py
import healpy as hp
import numpy as np
import rubin_scheduler.utils as utils
from astropy.coordinates import AltAz, EarthLocation, get_sun
from astropy.time import Time

import rubin_sim.skybrightness as sb


def generate_sky(
    mjd0=59560.2,
    mjd_max=59565.2,
    timestep=5.0,
    timestep_max=15.0,
    outfile=None,
    outpath=None,
    nside=32,
    sunLimit=-11.5,
    airmass_overhead=1.5,
    dm=0.2,
    airmass_limit=2.5,
    alt_limit=86.5,
    verbose=True,
):
    """
    Pre-compute the sky brighntess for a series of mjd dates at the LSST site.

    Parameters
    ----------
    mjd0 : `float`
        The starting MJD time
    duration : `float`
        The length of time to generate sky maps for (years)
    timestep : `float`
        The timestep between sky maps (minutes)
    timestep_max : `float`
        The maximum allowable timestep (minutes)
    outfile : `str`
        The name of the output file to save the results in
    nside : `int`
        The nside to create the healpixel maps.
        Default of 32 matches expectation of rubin_scheduler.
    sunLimit : `float`
        The maximum altitude of the sun to try and generate maps for.
        MJDs with a higher sun altitude are dropped.
    fieldID : `bool`
        If True, computes sky magnitudes at OpSim field locations.
        If False computes at healpixel centers.
    airmass_overhead : `float`
        The airmass region to demand sky models are well matched before
        dropping and assuming the timestep can be interpolated.
    dm : `float`
        If a skymap can be interpolated from neighboring maps with
        precision dm, that mjd is dropped.
    airmass_limit : `float`
        Pixels with an airmass greater than airmass_limit are masked.
    moon_dist_limit : float
        Pixels (fields) closer than moon_dist_limit (degrees) are masked.
    planet_dist_limit : `float`
        Pixels (fields) closer than planet_dist_limit (degrees) to Venus,
        Mars, Jupiter, or Saturn are masked.
    alt_limit : `float`
        Altitude limit of the telescope (degrees).
        Altitudes higher than this are masked.

    Returns
    -------
    dict_of_lists : `dict`
        includes key-value pairs:
        mjds : the MJD at every computation. Not necessarily evenly spaced.
        airmass : the airmass maps for each MJD
        masks : The `bool` mask map for each MJD
        (True means the pixel should be masked)
        sunAlts : The sun altitude at each MJD
    sky_brightness : `dict`
        Has keys for each u,g,r,i,z,y filter.
        Each one is a 2-d array with dimensions of healpix ID and
        mjd (matched to the mjd list above).
    """

    sunLimit_rad = np.radians(sunLimit)

    # Set the time steps
    timestep = timestep / 60.0 / 24.0  # Convert to days
    timestep_max = timestep_max / 60.0 / 24.0  # Convert to days
    # Switch the indexing to opsim field ID if requested

    # Look at the mjds and toss ones where the sun is up
    mjds = np.arange(mjd0, mjd_max + timestep, timestep)
    sunAlts = np.zeros(mjds.size, dtype=float)

    if outfile is None:
        outfile = "%i_%i.h5" % (mjd0, mjd_max)
    if outpath is not None:
        outfile = os.path.join(outpath, outfile)

    site = utils.Site("LSST")

    location = EarthLocation(lat=site.latitude, lon=site.longitude, height=site.height)
    t_sparse = Time(mjds, format="mjd", location=location)

    sun = get_sun(t_sparse)
    aa = AltAz(location=location, obstime=t_sparse)
    sunAlts = sun.transform_to(aa)

    mjds = mjds[np.where(sunAlts.alt.rad <= sunLimit_rad)]
    required_mjds = mjds[::3]

    hpindx = np.arange(hp.nside2npix(nside))
    ra, dec = utils.hpid2_ra_dec(nside, hpindx)

    if verbose:
        print("using %i points on the sky" % ra.size)
        print("using %i mjds" % mjds.size)

    # Set up the sky brightness model
    sm = sb.SkyModel(mags=True, airmass_limit=airmass_limit)

    filter_names = ["u", "g", "r", "i", "z", "y"]

    # Initialize the relevant lists
    dict_of_lists = {
        "mjds": [],
    }
    sky_brightness = {}
    for filter_name in filter_names:
        sky_brightness[filter_name] = []

    length = mjds[-1] - mjds[0]
    last_5_mags = []
    last_5_mjds = []
    for mjd in mjds:
        progress = (mjd - mjd0) / length * 100
        text = "\rprogress = %.1f%%" % progress
        sys.stdout.write(text)
        sys.stdout.flush()
        sm.set_ra_dec_mjd(ra, dec, mjd, degrees=True)
        if sm.sun_alt <= sunLimit_rad:
            mags = sm.return_mags()
            for key in filter_names:
                sky_brightness[key].append(mags[key])
            dict_of_lists["mjds"].append(mjd)
            last_5_mjds.append(mjd)
            last_5_mags.append(mags)
            if len(last_5_mjds) > 5:
                del last_5_mjds[0]
                del last_5_mags[0]

            if np.size(dict_of_lists["mjds"]) > 3:
                if dict_of_lists["mjds"][-2] not in required_mjds:
                    # Check if we can interpolate the second to
                    # last sky brightnesses
                    if dict_of_lists["mjds"][-1] - dict_of_lists["mjds"][-3] < timestep_max:
                        can_interp = True
                        for mjd2 in last_5_mjds:
                            mjd1 = dict_of_lists["mjds"][-3]
                            mjd3 = dict_of_lists["mjds"][-1]
                            if (mjd2 > mjd1) & (mjd2 < mjd3):
                                indx = np.min(np.where(last_5_mjds == mjd2)[0])
                                # Linear interpolation weights
                                wterm = (mjd2 - mjd1) / (mjd3 - mjd1)
                                w1 = 1.0 - wterm
                                w2 = wterm
                                for filter_name in filter_names:
                                    interp_sky = w1 * sky_brightness[filter_name][-3]
                                    interp_sky += w2 * sky_brightness[filter_name][-1]
                                    diff = np.abs(last_5_mags[indx][filter_name] - interp_sky)
                                    if np.size(diff[~np.isnan(diff)]) > 0:
                                        if np.max(diff[~np.isnan(diff)]) > dm:
                                            can_interp = False
                        if can_interp:
                            for key in dict_of_lists:
                                del dict_of_lists[key][-2]
                            for key in sky_brightness:
                                del sky_brightness[key][-2]
    print("")

    final_mjds = np.array(dict_of_lists["mjds"])
    final_sky_mags = np.zeros(
        (final_mjds.size, sky_brightness["r"][0].size),
        dtype=list(zip(filter_names, ["float"] * 6)),
    )
    for key in sky_brightness:
        final_sky_mags[key] = sky_brightness[key]

    import rubin_sim

    version = rubin_sim.version.__version__
    fingerprint = version
    # Generate a header to save all the kwarg info for this run
    if outpath is None:
        outpath = ""
    header = {
        "mjd0": mjd0,
        "mjd_max": mjd_max,
        "timestep": timestep,
        "timestep_max": timestep_max,
        "outfile": outfile,
        "outpath": outpath,
        "nside": nside,
        "verbose": verbose,
        "version": version,
        "fingerprint": fingerprint,
    }

    # Save mjd and sky brightness arrays to an hdf5 file
    hf = h5py.File(outfile, "w")
    hf.create_dataset("mjds", data=final_mjds)
    hf.create_dataset("sky_mags", data=final_sky_mags, compression="gzip")
    hf.create_dataset("timestep_max", data=timestep_max)
    hf.attrs.update(header)
    hf.close()


if __name__ == "__main__":
    # Make a quick small one for speed loading
    print("generating small file")
    m0 = utils.SURVEY_START_MJD
    generate_sky(mjd0=m0 - 1, mjd_max=m0 + 31)

    nyears = 25.0  # 20  # 13
    day_pad = 30

    # Full year
    # mjds = np.arange(59560, 59560+365.25*nyears+day_pad+366, 366)
    # 6-months
    mjds = np.arange(59560, 59560 + 366 * nyears + 366 / 2.0, 366 / 2.0)
    # mjds = [59560, 59563.5]
    # mjds = [60218, 60226]
    # Add some time ahead of time for ComCam
    # nyears = 3
    # mjds = np.arange(58462, 58462+366*nyears+366/2., 366/2.)
    count = 0
    for mjd1, mjd2 in zip(mjds[:-1], mjds[1:]):
        print("Generating file %i" % count)
        # generate_sky(mjd0=mjd1, mjd_max=mjd2+day_pad,
        # outpath='opsimFields', fieldID=True)
        generate_sky(mjd0=mjd1, mjd_max=mjd2 + day_pad)
        count += 1
