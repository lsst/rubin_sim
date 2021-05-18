import numpy as np
import astropy.units as u
from astropy.time import Time
from lsst.sims.utils import Site
from astropy.coordinates import get_sun, get_moon, EarthLocation, AltAz
from lsst.sims.utils import _angularSeparation

# want to generate moon and sun position information.

mjd_start = 59853.5 - 3.*365.25
duration = 25.*365.25
pad_around = 40
t_step = 1./24.  # Start with 1-hour timesteps.


mjds = np.arange(mjd_start-pad_around, duration+mjd_start+pad_around+t_step, t_step)

names = ['mjd', 'sun_RA', 'sun_dec', 'sun_alt', 'sun_az', 'moon_RA',
         'moon_dec', 'moon_alt', 'moon_az', 'moon_phase']
types = [float]*len(names)

sun_moon_info = np.zeros(mjds.size, dtype=list(zip(names, types)))
sun_moon_info['mjd'] = np.arange(mjd_start-pad_around, duration+mjd_start+pad_around+t_step, t_step)

site = Site('LSST')
location = EarthLocation(lat=site.latitude, lon=site.longitude, height=site.height)
t_sparse = Time(mjds, format='mjd', location=location)

sun = get_sun(t_sparse)
aa = AltAz(location=location, obstime=t_sparse)
sun_aa = sun.transform_to(aa)

moon = get_moon(t_sparse)
moon_aa = moon.transform_to(aa)

sun_moon_info['sun_RA'] = sun.ra.rad
sun_moon_info['sun_dec'] = sun.dec.rad

sun_moon_info['sun_alt'] = sun_aa.alt.rad
sun_moon_info['sun_az'] = sun_aa.az.rad

sun_moon_info['moon_RA'] = moon.ra.rad
sun_moon_info['moon_dec'] = moon.dec.rad

sun_moon_info['moon_alt'] = moon_aa.alt.rad
sun_moon_info['moon_az'] = moon_aa.az.rad

sun_moon_sep = _angularSeparation(sun.ra.rad, sun.dec.rad,
                                  moon.ra.rad, moon.dec.rad)
sun_moon_info['moon_phase'] = sun_moon_sep/np.pi*100.

# Let's cut down a bit, no need to save info when the sun is high in the sky
good_suns = np.where(sun_moon_info['sun_alt'] < np.radians(10.))
sun_moon_info = sun_moon_info[good_suns]

np.savez('sun_moon.npz', sun_moon_info=sun_moon_info)

# Takes 6.25 mintues for 221,072.  In a sim, I need 5e6, so 141 min--over 2 hours.