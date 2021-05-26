import numpy as np
import pandas as pd
from rubin_sim.skybrightness import SkyModel
import rubin_sim.skybrightness_pre as sb
from rubin_sim.utils import raDec2Hpid, m5_flat_sed, Site, _approx_RaDec2AltAz
import healpy as hp
import sqlite3
import sys

__all__ = ['obs2sqlite']


def obs2sqlite(observations_in, location='LSST', outfile='observations.sqlite', slewtime_limit=5.,
               full_sky=False, radians=True):
    """
    Utility to take an array of observations and dump it to a sqlite file, filling in useful columns along the way.

    observations_in: numpy array with at least columns of
        ra : RA in degrees
        dec : dec in degrees
        mjd : MJD in day
        filter : string with the filter name
        exptime : the exposure time in seconds
    slewtime_limit : float
        Consider all slewtimes larger than this to be closed-dome time not part of a slew.
    """

    # Set the location to be LSST
    if location == 'LSST':
        telescope = Site('LSST')

    # Check that we have the columns we need
    needed_cols = ['ra', 'dec', 'mjd', 'filter']
    in_cols = observations_in.dtype.names
    for col in needed_cols:
        if needed_cols not in in_cols:
            ValueError('%s column not found in observtion array' % col)

    n_obs = observations_in.size
    sm = None
    # make sure they are in order by MJD
    observations_in.sort(order='mjd')

    # Take all the columns that are in the input and add any missing
    names = ['filter', 'ra', 'dec', 'mjd', 'exptime', 'alt', 'az', 'skybrightness',
             'seeing', 'night', 'slewtime', 'fivesigmadepth', 'airmass', 'sunAlt', 'moonAlt']
    types = ['|S1']
    types.extend([float]*(len(names)-1))

    observations = np.zeros(n_obs, dtype=list(zip(names, types)))

    # copy over the ones we have
    for col in in_cols:
        observations[col] = observations_in[col]

    # convert output to be in degrees like expected
    if radians:
        observations['ra'] = np.degrees(observations['ra'])
        observations['dec'] = np.degrees(observations['dec'])

    if 'exptime' not in in_cols:
        observations['exptime'] = 30.

    # Fill in the slewtime. Note that filterchange time gets included in slewtimes
    if 'slewtime' not in in_cols:
        # Assume MJD is midpoint of exposures
        mjd_sec = observations_in['mjd']*24.*3600.
        observations['slewtime'][1:] = mjd_sec[1:]-mjd_sec[0:-1] - observations['exptime'][0:-1]*0.5 - observations['exptime'][1:]*0.5
        closed = np.where(observations['slewtime'] > slewtime_limit*60.)
        observations['slewtime'][closed] = 0.

    # Let's just use the stupid-fast to get alt-az
    if 'alt' not in in_cols:
        alt, az = _approx_RaDec2AltAz(np.radians(observations['ra']), np.radians(observations['dec']),
                                      telescope.latitude_rad, telescope.longitude_rad, observations['mjd'])
        observations['alt'] = np.degrees(alt)
        observations['az'] = np.degrees(az)

    # Fill in the airmass
    if 'airmass' not in in_cols:
        observations['airmass'] = 1./np.cos(np.pi/2. - np.radians(observations['alt']))

    # Fill in the seeing
    if 'seeing' not in in_cols:
        # XXX just fill in a dummy val
        observations['seeing'] = 0.8

    # Sky Brightness
    if 'skybrightness' not in in_cols:
        if full_sky:
            sm = SkyModel(mags=True)
            for i, obs in enumerate(observations):
                sm.setRaDecMjd(obs['ra'], obs['dec'], obs['mjd'], degrees=True)
                observations['skybrightness'][i] = sm.returnMags()[obs['filter']]
        else:
            # Let's try using the pre-computed sky brighntesses
            sm = sb.SkyModelPre(preload=False)
            full = sm.returnMags(observations['mjd'][0])
            nside = hp.npix2nside(full['r'].size)
            imax = float(np.size(observations))
            for i, obs in enumerate(observations):
                indx = raDec2Hpid(nside, obs['ra'], obs['dec'])
                observations['skybrightness'][i] = sm.returnMags(obs['mjd'], indx=[indx])[obs['filter']]
                sunMoon = sm.returnSunMoon(obs['mjd'])
                observations['sunAlt'][i] = sunMoon['sunAlt']
                observations['moonAlt'][i] = sunMoon['moonAlt']
                progress = i/imax*100
                text = "\rprogress = %.2f%%"%progress
                sys.stdout.write(text)
                sys.stdout.flush()
            observations['sunAlt'] = np.degrees(observations['sunAlt'])
            observations['moonAlt'] = np.degrees(observations['moonAlt'])

    # 5-sigma depth
    for fn in np.unique(observations['filter']):
        good = np.where(observations['filter'] == fn)
        observations['fivesigmadepth'][good] = m5_flat_sed(fn, observations['skybrightness'][good],
                                                           observations['seeing'][good],
                                                           observations['exptime'][good],
                                                           observations['airmass'][good])

    conn = sqlite3.connect(outfile)
    df = pd.DataFrame(observations)
    df.to_sql('observations', conn)
