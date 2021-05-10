import numpy as np
import lsst.sims.skybrightness as sb
import lsst.sims.utils as utils
import healpy as hp
import sys
import os
import ephem
from lsst.sims.skybrightness.utils import mjd2djd

# For generating big pre-computed blocks of sky

def generate_sky(mjd0=59560.2, mjd_max=59565.2, timestep=5., timestep_max=15.,
                 outfile=None, outpath=None, nside=32,
                 sunLimit=-8., airmass_overhead=1.5, dm=0.2,
                 airmass_limit=3.,
                 requireStride=3, verbose=True, alt_limit=None):
    """
    Pre-compute the sky brighntess for a series of mjd dates at the LSST site.

    Parameters
    ----------
    mjd0 : float (9560.2)
        The starting MJD time
    duration : float
        The length of time to generate sky maps for (years)
    timestep : float (5.)
        The timestep between sky maps (minutes)
    timestep_max : float (20.)
        The maximum alowable timestep (minutes)
    outfile : str
        The name of the output file to save the results in
    nside : in (32)
        The nside to run the healpixel map at
    sunLimit : float (-12)
        The maximum altitude of the sun to try and generate maps for. MJDs with a higher
        sun altitude are dropped
    fieldID : bool (False)
        If True, computes sky magnitudes at OpSim field locations. If False
        computes at healpixel centers.
    airmass_overhead : float
        The airmass region to demand sky models are well matched before dropping
        and assuming the timestep can be interpolated
    dm : float
        If a skymap can be interpolated from neighboring maps with precision dm (mag/arcsec^2),
        that mjd is dropped.
    airmass_limit : float
        Pixels with an airmass greater than airmass_limit are masked
    moon_dist_limit : float
        Pixels (fields) closer than moon_dist_limit (degrees) are masked
    planet_dist_limit : float (2.)
        Pixels (fields) closer than planet_dist_limit (degrees) to Venus, Mars, Jupiter, or Saturn are masked
    requireStride : int (3)
        Require every nth mjd. Makes it possible to easily select an evenly spaced number states of a pixel.

    Returns
    -------
    dict_of_lists : dict
        includes key-value pairs:
        mjds : the MJD at every computation. Not evenly spaced as no computations.
        airmass : the airmass maps for each MJD
        masks : The boolean mask map for each MJD (True means the pixel should be masked)
        sunAlts : The sun altitude at each MJD
    sky_brightness : dict
        Has keys for each u,g,r,i,z,y filter. Each one is a 2-d array with dimensions of healpix ID and
        mjd (matched to the mjd list above).
    """

    sunLimit_rad = np.radians(sunLimit)

    if alt_limit is None:
        alt_limit = np.degrees(np.pi/2. - np.arccos(1./airmass_limit))
        print('setting alt limit to %f degrees' % alt_limit)

    # Set the time steps
    timestep = timestep / 60. / 24.  # Convert to days
    timestep_max = timestep_max / 60. / 24.  # Convert to days
    # Switch the indexing to opsim field ID if requested

    # Look at the mjds and toss ones where the sun is up
    mjds = np.arange(mjd0, mjd_max+timestep, timestep)
    sunAlts = np.zeros(mjds.size, dtype=float)

    if outfile is None:
        outfile = '%i_%i.npz' % (mjd0, mjd_max)
    if outpath is not None:
        outfile = os.path.join(outpath, outfile)

    telescope = utils.Site('LSST')
    Observatory = ephem.Observer()
    Observatory.lat = telescope.latitude_rad
    Observatory.lon = telescope.longitude_rad
    Observatory.elevation = telescope.height

    sun = ephem.Sun()

    # Compute the sun altitude for all the possible MJDs
    for i, mjd in enumerate(mjds):
        Observatory.date = mjd2djd(mjd)
        sun.compute(Observatory)
        sunAlts[i] = sun.alt

    mjds = mjds[np.where(sunAlts <= sunLimit_rad)]
    required_mjds = mjds[::3]

    hpindx = np.arange(hp.nside2npix(nside))
    az, alt = utils.hpid2RaDec(nside, hpindx)
    above_alt_limit = np.where(alt > alt_limit)[0]

    if verbose:
        print('using %i points on the sky' % az.size)
        print('using %i mjds' % mjds.size)

    # Set up the sky brightness model
    sm = sb.SkyModel(mags=True)

    filter_names = ['u', 'g', 'r', 'i', 'z', 'y']

    # Initialize the relevant lists
    dict_of_lists = {'airmass': [], 'sunAlts': [], 'mjds': [],
                     'moonAlts': [], 'moonRAs': [], 'moonDecs': [], 'sunRAs': [],
                     'sunDecs': [], 'moonSunSep': []}
    sky_brightness = {}
    for filter_name in filter_names:
        sky_brightness[filter_name] = []

    length = mjds[-1] - mjds[0]
    last_5_mags = []
    last_5_mjds = []
    for mjd in mjds:
        progress = (mjd-mjd0)/length*100
        text = "\rprogress = %.1f%%"%progress
        sys.stdout.write(text)
        sys.stdout.flush()
        sm.setRaDecMjd(az[above_alt_limit], alt[above_alt_limit], mjd, degrees=True, azAlt=True)
        if sm.sunAlt <= sunLimit_rad:
            mags_dict = sm.returnMags()
            mags = {}
            for key in mags_dict:
                mags[key] = np.zeros(az.size, dtype=float)
                mags[key][above_alt_limit] += mags_dict[key]
            for key in filter_names:
                sky_brightness[key].append(mags[key])
            dict_of_lists['airmass'].append(sm.airmass)
            dict_of_lists['sunAlts'].append(sm.sunAlt)
            dict_of_lists['mjds'].append(mjd)
            dict_of_lists['sunRAs'].append(sm.sunRA)
            dict_of_lists['sunDecs'].append(sm.sunDec)
            dict_of_lists['moonRAs'].append(sm.moonRA)
            dict_of_lists['moonDecs'].append(sm.moonDec)
            dict_of_lists['moonSunSep'].append(sm.moonSunSep)
            dict_of_lists['moonAlts'].append(sm.moonAlt)
            last_5_mjds.append(mjd)
            last_5_mags.append(mags)
            if len(last_5_mjds) > 5:
                del last_5_mjds[0]
                del last_5_mags[0]

            if len(dict_of_lists['airmass']) > 3:
                if dict_of_lists['mjds'][-2] not in required_mjds:
                    # Check if we can interpolate the second to last sky brightnesses
                    overhead = np.where((dict_of_lists['airmass'][-1] <= airmass_overhead) &
                                        (dict_of_lists['airmass'][-2] <= airmass_overhead))

                    if (np.size(overhead[0]) > 0) & (dict_of_lists['mjds'][-1] -
                                                     dict_of_lists['mjds'][-3] < timestep_max):
                        can_interp = True
                        for mjd2 in last_5_mjds:
                            mjd1 = dict_of_lists['mjds'][-3]
                            mjd3 = dict_of_lists['mjds'][-1]
                            if (mjd2 > mjd1) & (mjd2 < mjd3):
                                indx = np.where(last_5_mjds == mjd2)[0]
                                # Linear interpolation weights
                                wterm = (mjd2 - mjd1) / (mjd3-mjd1)
                                w1 = 1. - wterm
                                w2 = wterm
                                for filter_name in filter_names:
                                    interp_sky = w1 * sky_brightness[filter_name][-3][overhead]
                                    interp_sky += w2 * sky_brightness[filter_name][-1][overhead]
                                    diff = np.abs(last_5_mags[int(indx)][filter_name][overhead]-interp_sky)
                                    if np.size(diff[~np.isnan(diff)]) > 0:
                                        if np.max(diff[~np.isnan(diff)]) > dm:
                                            can_interp = False
                        if can_interp:
                            for key in dict_of_lists:
                                del dict_of_lists[key][-2]
                            for key in sky_brightness:
                                del sky_brightness[key][-2]
    print('')

    for key in dict_of_lists:
        dict_of_lists[key] = np.array(dict_of_lists[key])
    for key in sky_brightness:
        sky_brightness[key] = np.array(sky_brightness[key])

    version = sb.version.__version__
    fingerprint = sb.version.__fingerprint__
    # Generate a header to save all the kwarg info for how this run was computed
    header = {'mjd0': mjd0, 'mjd_max': mjd_max, 'timestep': timestep, 'timestep_max': timestep_max,
              'outfile': outfile, 'outpath': outpath, 'nside': nside, 'sunLimit': sunLimit,
              'airmas_overhead': airmass_overhead, 'dm': dm,
              'airmass_limit': airmass_limit,
              'alt': alt, 'az': az, 'verbose': verbose, 'required_mjds': required_mjds,
              'version': version, 'fingerprint': fingerprint}

    np.savez(outfile, dict_of_lists=dict_of_lists, header=header)
    # Convert sky_brightness to a true array so it's easier to save
    types = [float]*len(sky_brightness.keys())
    result = np.zeros(sky_brightness[list(sky_brightness.keys())[0]].shape,
                      dtype=list(zip(sky_brightness.keys(), types)))
    for key in sky_brightness.keys():
        result[key] = sky_brightness[key]
    np.save(outfile[:-3]+'npy', result)


if __name__ == "__main__":

    # Make a quick small one for speed loading
    #generate_sky(mjd0=59579, mjd_max=59579+10., outpath='healpix', outfile='small_example.npz_small')
    #generate_sky(mjd0=59579, mjd_max=59579+10., outpath='opsimFields', fieldID=True)

    nyears = 15. #20  # 13
    day_pad = 3
    # Full year
    # mjds = np.arange(59560, 59560+365.25*nyears+day_pad+366, 366)
    # 6-months
    #mjds = np.arange(59560, 59560+366*nyears+366/2., 366/2.)
    # Add some time ahead of time for ComCam
    #nyears = 3
    #mjds = np.arange(58462, 58462+366*nyears+366/2., 366/2.)
    mjds = [58462, 58462+2]
    count = 0
    for mjd1, mjd2 in zip(mjds[:-1], mjds[1:]):
        print('Generating file %i' % count)
        #generate_sky(mjd0=mjd1, mjd_max=mjd2+day_pad, outpath='opsimFields', fieldID=True)
        generate_sky(mjd0=mjd1, mjd_max=mjd2+day_pad, outpath='temp')
        count += 1

