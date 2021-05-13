import numpy as np
from rubin_sim.scheduler.detailers import Base_detailer
from rubin_sim.utils import _approx_RaDec2AltAz, _approx_altaz2pa


__all__ = ["Dither_detailer", "Camera_rot_detailer", "Euclid_dither_detailer"]


def gnomonic_project_toxy(ra, dec, raCen, decCen):
    """Calculate x/y projection of RA1/Dec1 in system with center at RAcen, Deccenp.
    Input radians. Returns x/y."""
    # also used in Global Telescope Network website
    if (len(ra) != len(dec)):
        raise Exception("Expect RA and Dec arrays input to gnomonic projection to be same length.")
    cosc = np.sin(decCen) * np.sin(dec) + np.cos(decCen) * np.cos(dec) * np.cos(ra-raCen)
    x = np.cos(dec) * np.sin(ra-raCen) / cosc
    y = (np.cos(decCen)*np.sin(dec) - np.sin(decCen)*np.cos(dec)*np.cos(ra-raCen)) / cosc
    return x, y


def gnomonic_project_tosky(x, y, raCen, decCen):
    """Calculate RA/Dec on sky of object with x/y and RA/Cen of field of view.
    Returns Ra/Dec in radians."""
    denom = np.cos(decCen) - y * np.sin(decCen)
    ra = raCen + np.arctan2(x, denom)
    dec = np.arctan2(np.sin(decCen) + y * np.cos(decCen), np.sqrt(x*x + denom*denom))
    return ra, dec


class Dither_detailer(Base_detailer):
    """
    make a uniform dither pattern. Offset by a maximum radius in a random direction.
    Mostly intended for DDF pointings, the BaseMarkovDF_survey class includes dithering
    for large areas.

    Parameters
    ----------
    max_dither : float (0.7)
        The maximum dither size to use (degrees).
    per_night : bool (True)
        If true, us the same dither offset for an entire night


    """
    def __init__(self, max_dither=0.7, seed=42, per_night=True):
        self.survey_features = {}

        self.current_night = -1
        self.max_dither = np.radians(max_dither)
        self.per_night = per_night
        np.random.seed(seed=seed)
        self.offset = None

    def _generate_offsets(self, n_offsets, night):
        if self.per_night:
            if night != self.current_night:
                self.current_night = night
                self.offset = (np.random.random((1, 2))-0.5) * 2.*self.max_dither
                angle = np.random.random(1)*2*np.pi
                radius = self.max_dither * np.sqrt(np.random.random(1))
                self.offset = np.array([radius*np.cos(angle), radius*np.sin(angle)])
            offsets = np.tile(self.offset, (n_offsets, 1))
        else:
            angle = np.random.random(n_offsets)*2*np.pi
            radius = self.max_dither * np.sqrt(np.random.random(n_offsets))
            offsets = np.array([radius*np.cos(angle), radius*np.sin(angle)])

        return offsets

    def __call__(self, observation_list, conditions):

        # Generate offsets in RA and Dec
        offsets = self._generate_offsets(len(observation_list), conditions.night)

        obs_array = np.concatenate(observation_list)
        newRA, newDec = gnomonic_project_tosky(offsets[0, :], offsets[1, :], obs_array['RA'], obs_array['dec'])
        for i, obs in enumerate(observation_list):
            observation_list[i]['RA'] = newRA[i]
            observation_list[i]['dec'] = newDec[i]
        return observation_list


def bearing(lon1, lat1, lon2, lat2):
    """Bearing between two points
    """

    delta_l = lon2 - lon1
    X = np.cos(lat2) * np.sin(delta_l)
    Y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(delta_l)
    theta = np.arctan2(X, Y)

    return theta


def dest(dist, bearing, lat1, lon1):

    lat2 = np.arcsin(np.sin(lat1)*np.cos(dist)+np.cos(lat1)*np.sin(dist)*np.cos(bearing))
    lon2 = lon1 + np.arctan2(np.sin(bearing)*np.sin(dist)*np.cos(lat1), np.cos(dist)-np.sin(lat1)*np.sin(lat2))
    return lat2, lon2


class Euclid_dither_detailer(Base_detailer):
    """Directional dithering for Euclid DDFs

    Parameters
    ----------
    XXX--fill in docstring

    """
    def __init__(self, dither_bearing_dir=[-0.25, 1], dither_bearing_perp=[-0.25, 0.25],
                 seed=42, per_night=True, ra_a=58.90,
                 dec_a=-49.315, ra_b=63.6, dec_b=-47.60):
        self.survey_features = {}

        self.ra_a = np.radians(ra_a)
        self.ra_b = np.radians(ra_b)
        self.dec_a = np.radians(dec_a)
        self.dec_b = np.radians(dec_b)

        self.dither_bearing_dir = np.radians(dither_bearing_dir)
        self.dither_bearing_perp = np.radians(dither_bearing_perp)

        self.bearing_atob = bearing(self.ra_a, self.dec_a, self.ra_b, self.dec_b)
        self.bearing_btoa = bearing(self.ra_b, self.dec_b, self.ra_a, self.dec_a)

        self.current_night = -1

        self.per_night = per_night
        np.random.seed(seed=seed)
        self.shifted_ra_a = None
        self.shifted_dec_a = None
        self.shifted_ra_b = None
        self.shifted_dec_b = None

    def _generate_offsets(self, n_offsets, night):
        if self.per_night:
            if night != self.current_night:
                self.current_night = night
                bearing_mag = np.random.uniform(low=self.dither_bearing_dir.min(), high=self.dither_bearing_dir.max())
                perp_mag = np.random.uniform(low=self.dither_bearing_perp.min(), high=self.dither_bearing_perp.max())
                # Move point a along the bearings
                self.shifted_dec_a, self.shifted_ra_a = dest(bearing_mag, self.bearing_atob, self.dec_a, self.ra_a)
                self.shifted_dec_a, self.shifted_ra_a = dest(perp_mag, self.bearing_atob+np.pi/2.,
                                                             self.shifted_dec_a, self.shifted_ra_a)

                # Shift the second position
                bearing_mag = np.random.uniform(low=self.dither_bearing_dir.min(), high=self.dither_bearing_dir.max())
                perp_mag = np.random.uniform(low=self.dither_bearing_perp.min(), high=self.dither_bearing_perp.max())

                self.shifted_dec_b, self.shifted_ra_b = dest(bearing_mag, self.bearing_btoa, self.dec_b, self.ra_b)
                self.shifted_dec_b, self.shifted_ra_b = dest(perp_mag, self.bearing_btoa+np.pi/2.,
                                                             self.shifted_dec_b, self.shifted_ra_b)
        else:
            # XXX--not implamented
            ValueError('not implamented')

        return self.shifted_ra_a, self.shifted_dec_a, self.shifted_ra_b, self.shifted_dec_b

    def __call__(self, observation_list, conditions):
        # Generate offsets in RA and Dec
        ra_a, dec_a, ra_b, dec_b = self._generate_offsets(len(observation_list), conditions.night)

        for i, obs in enumerate(observation_list):
            if obs[0]['note'][-1] == 'a':
                observation_list[i]['RA'] = ra_a
                observation_list[i]['dec'] = dec_a
            elif obs[0]['note'][-1] == 'b':
                observation_list[i]['RA'] = ra_b
                observation_list[i]['dec'] = dec_b
            else:
                ValueError('observation note does not end in a or b.')
        return observation_list


class Camera_rot_detailer(Base_detailer):
    """
    Randomly set the camera rotation, either for each exposure, or per night.

    Parameters
    ----------
    max_rot : float (90.)
        The maximum amount to offset the camera (degrees)
    min_rot : float (90)
        The minimum to offset the camera (degrees)
    per_night : bool (True)
        If True, only set a new offset per night. If False, randomly rotates every observation.
    """
    def __init__(self, max_rot=90., min_rot=-90., per_night=True, seed=42):
        self.survey_features = {}

        self.current_night = -1
        self.max_rot = np.radians(max_rot)
        self.min_rot = np.radians(min_rot)
        self.range = self.max_rot - self.min_rot
        self.per_night = per_night
        np.random.seed(seed=seed)
        self.offset = None

    def _generate_offsets(self, n_offsets, night):
        if self.per_night:
            if night != self.current_night:
                self.current_night = night
                self.offset = np.random.random(1) * self.range + self.min_rot
            offsets = np.ones(n_offsets) * self.offset
        else:
            offsets = np.random.random(n_offsets) * self.range + self.min_rot

        offsets = offsets % (2.*np.pi)

        return offsets

    def __call__(self, observation_list, conditions):

        # Generate offsets in camamera rotator
        offsets = self._generate_offsets(len(observation_list), conditions.night)

        for i, obs in enumerate(observation_list):
            alt, az = _approx_RaDec2AltAz(obs['RA'], obs['dec'], conditions.site.latitude_rad,
                                          conditions.site.longitude_rad, conditions.mjd)
            obs_pa = _approx_altaz2pa(alt, az, conditions.site.latitude_rad)
            obs['rotSkyPos'] = (offsets[i] - obs_pa) % (2.*np.pi)
            obs['rotTelPos'] = offsets[i]

        return observation_list
