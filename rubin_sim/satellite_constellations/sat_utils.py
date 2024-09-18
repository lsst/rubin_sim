__all__ = (
    "create_constellation",
    "starlink_tles_v1",
    "starlink_tles_v2",
    "oneweb_tles",
    "Constellation",
    "sun_alt_limits",
)

import numpy as np
from astropy import constants as const
from astropy import units as u
from rubin_scheduler.utils import SURVEY_START_MJD, Site, gnomonic_project_toxy, point_to_line_distance
from shapely.geometry import LineString, Point
from skyfield.api import EarthSatellite, load, wgs84

MJDOFFSET = 2400000.5
mjd0 = SURVEY_START_MJD


def sun_alt_limits():
    """Return sun altitude limits (degrees) at which zero illuminated
    satellites above 20 degrees altitude result.

    Different constellations have different limits at which zero illumination
    above 20 degrees occurs.

    Returns
    -------
    sun_alt_limits : `dict` [`str`: `float`]
        Dict with satellite constellation name keys,
        altitude limits values (degrees).
    """
    # Estimated in sun_alts_limits.ipynb
    result = {"slv1": -36.0, "slv2": -36.0, "oneweb": -53.0}
    return result


def satellite_mean_motion(altitude, mu=const.GM_earth, r_earth=const.R_earth):
    """Calculate mean motion of satellites at a given altitude in Earth's
    gravitational field.
    See https://en.wikipedia.org/wiki/Mean_motion#Formulae

    Parameters
    ----------
    altitude : `float`
        Altitude of the satellite.
        Should be a float with astropy units attached.

    Returns
    -------
    mean_motion : `float`
    """
    no = np.sqrt(4.0 * np.pi**2 * (altitude + r_earth) ** 3 / mu).to(u.day)
    return 1 / no


def tle_from_orbital_parameters(sat_name, sat_nr, epoch, inclination, raan, mean_anomaly, mean_motion):
    """Generate TLE strings from orbital parameters.

    Parameters
    ----------
    sat_name : `str`
        Satellite name
    sat_nr : `float`
        Satellite nr
    epoch : `float`
        Epoch
    inclination : `float`
        Inclination, probably with astropy units attached
    raan : `float`
        RA of the ascending node. Assumes astropy units.
    mean_anomaly : `float`
        Mean anomaly.
    mean_motion : `float`
       Mean motion.

    Returns
    -------
    tle : `str`

    Notes
    -----
    epoch has the format: first two digits are the year,
    next three digits are the day from beginning of year,
    then fraction of a day is given, e.g.
    20180.25 would be 2020, day 180, 6 hours (UT?)
    """

    # Note: RAAN = right ascention (or longitude) of ascending node
    # I suspect this is filling in 0 eccentricity everywhere.

    def checksum(line):
        s = 0
        for c in line[:-1]:
            if c.isdigit():
                s += int(c)
            if c == "-":
                s += 1
        return "{:s}{:1d}".format(line[:-1], s % 10)

    tle0 = sat_name
    tle1 = checksum(
        "1 {:05d}U 20001A   {:14.8f}  .00000000  00000-0  50000-4 " "0    0X".format(sat_nr, epoch)
    )
    tle2 = checksum(
        "2 {:05d} {:8.4f} {:8.4f} 0001000   0.0000 {:8.4f} "
        "{:11.8f}    0X".format(
            sat_nr,
            inclination.to_value(u.deg),
            raan.to_value(u.deg),
            mean_anomaly.to_value(u.deg),
            mean_motion.to_value(1 / u.day),
        )
    )

    return "\n".join([tle0, tle1, tle2])


def create_constellation(
    altitudes,
    inclinations,
    nplanes,
    sats_per_plane,
    epoch=23274.0,
    name="Test",
    seed=42,
):
    """Create a set of orbital elements for a satellite constellation then
    convert them to TLEs.

    Parameters
    ----------
    altitudes : `np.ndarray`, (N,)
        Altitudes (degrees).
    inclinations : `np.ndarray`, (N,)
        Inclinations (degrees).
    nplanes : `np.ndarray`, (N,)
        Number of satellite planes.
    sats_per_plane : `np.ndarray`, (N,)
        Number of satellites per orbital plane.
    epoch : `float`
        Epoch.
    name : `str`
        Satellite name.
    seed : `float`
        Random number seed.

    Returns
    -------
    my_sat_tles : `list` of `str`
    """

    rng = np.random.default_rng(seed)

    my_sat_tles = []
    sat_nr = 8000
    for alt, inc, n, s in zip(altitudes, inclinations, nplanes, sats_per_plane):
        if s == 1:
            # random placement for lower orbits
            mas = rng.uniform(0, 360, n) * u.deg
            raans = rng.uniform(0, 360, n) * u.deg
        else:
            mas = np.linspace(0.0, 360.0, s, endpoint=False) * u.deg
            mas += rng.uniform(0, 360, 1) * u.deg
            raans = np.linspace(0.0, 360.0, n, endpoint=False) * u.deg
            mas, raans = np.meshgrid(mas, raans)
            mas, raans = mas.flatten(), raans.flatten()

        mm = satellite_mean_motion(alt)
        for ma, raan in zip(mas, raans):
            my_sat_tles.append(
                tle_from_orbital_parameters(name + " {:d}".format(sat_nr), sat_nr, epoch, inc, raan, ma, mm)
            )
            sat_nr += 1

    return my_sat_tles


def starlink_tles_v1():
    """Create a list of satellite TLE's, appropriate for
    Starlink v1 (as of July 2022). Should create 4,408 orbits

    Returns
    -------
    my_sat_tles : `list` [`str`]
    """
    altitudes = np.array([550, 540, 570, 560, 560]) * u.km
    inclinations = np.array([53, 53.2, 70, 97.6, 97.6]) * u.deg
    nplanes = np.array([72, 72, 36, 6, 4])
    sats_per_plane = np.array([22, 22, 20, 58, 43])

    my_sat_tles = create_constellation(altitudes, inclinations, nplanes, sats_per_plane, name="starV1")

    return my_sat_tles


def starlink_tles_v2():
    """Create a list of satellite TLE's appropriate for
    Starlink v2 (as of July 2022). Should create 29,988 orbits

    Returns
    -------
    my_sat_tles : `list` [`str`]
    """
    altitudes = np.array([340, 345, 350, 360, 525, 530, 535, 604, 614]) * u.km
    inclinations = np.array([53, 46, 38, 96.9, 53, 43, 33, 148, 115.7]) * u.deg
    nplanes = np.array([48, 48, 48, 30, 28, 28, 28, 12, 18])
    sats_per_plane = np.array([110, 110, 110, 120, 120, 120, 120, 12, 18])

    my_sat_tles = create_constellation(altitudes, inclinations, nplanes, sats_per_plane, name="starV2")

    return my_sat_tles


def oneweb_tles():
    """Create a list of satellite TLE's appropriate
    for OneWeb plans (as of July 2022). Should create 6,372 orbits

    Returns
    -------
    my_sat_tles : `list` [`str`]
    """
    altitudes = np.array([1200, 1200, 1200]) * u.km
    inclinations = np.array([87.9, 40, 55]) * u.deg
    nplanes = np.array([49, 72, 72])
    sats_per_plane = np.array([36, 32, 32])

    my_sat_tles = create_constellation(altitudes, inclinations, nplanes, sats_per_plane, name="oneWe")

    return my_sat_tles


class Constellation:
    """Holds the constellation TLEs and calculates their appearance
    in a series of observations.

    Parameters
    ----------
    sat_tle_list : `list` [`str`]
        A list of satellite TLEs to be used
    alt_limit : `float`
        Altitude limit below which satellites can be ignored (degrees)
    fov : `float`
        The field of view diameter (degrees)
    """

    def __init__(self, sat_tle_list, alt_limit=20.0, fov=3.5):
        self.alt_limit_rad = np.radians(alt_limit)
        self.fov_radius_rad = np.radians(fov / 2.0)

        # Load ephemeris for sun position
        self.eph = load("de421.bsp")

        self.sat_list = []
        self.ts = load.timescale()
        for tle in sat_tle_list:
            name, line1, line2 = tle.split("\n")
            self.sat_list.append(EarthSatellite(line1, line2, name, self.ts))

        self._make_location()

    def _make_location(self):
        telescope = Site(name="LSST")

        self.observatory_site = wgs84.latlon(telescope.latitude, telescope.longitude, telescope.height)

    def update_mjd(self, mjd):
        """Calculate and record the alt/az position and illumination status
        for all the satellites at a given time.

        Parameters
        ----------
        mjd : `float`
            New MJD.
        """
        jd = mjd + MJDOFFSET
        t = self.ts.ut1_jd(jd)

        self.altitudes_rad = []
        self.azimuth_rad = []
        self.illum = []
        for sat in self.sat_list:
            current_sat = sat.at(t)
            illum = current_sat.is_sunlit(self.eph)
            self.illum.append(illum.copy())
            if illum:
                topo = current_sat - self.observatory_site.at(t)
                # this returns an Angle object
                alt, az, dist = topo.altaz()
                self.altitudes_rad.append(alt.radians + 0)
                self.azimuth_rad.append(az.radians + 0)
            else:
                self.altitudes_rad.append(np.nan)
                self.azimuth_rad.append(np.nan)

        self.altitudes_rad = np.array(self.altitudes_rad)
        self.azimuth_rad = np.array(self.azimuth_rad)
        self.illum = np.array(self.illum)
        # Keep track of the ones that are up and illuminated
        self.visible = np.where((self.altitudes_rad >= self.alt_limit_rad) & (self.illum == True))[0]

    def paths_array(self, mjds):
        """Calculate and return the RA/Dec/Alt and illumination status
        for all the satellites at an array of times.

        Parameters
        ----------
        mjds : `np.ndarray`, (N,)
            Modified Julian Dates.

        Returns
        -------
        ras : `np.ndarray`, (N,)
            RAs at each MJD
        decs : `np.ndarray`, (N,)
            Decs at each MJD
        alts : `np.ndarray`, (N,)
            Altitudes at each MJD
        illums : `np.ndarray`, (N,)
            Array of bools for if satellite is illuminated
        """

        jd = mjds + MJDOFFSET
        t = self.ts.ut1_jd(jd)

        ras = []
        decs = []
        alts = []
        illums = []
        for sat in self.sat_list:
            current_sat = sat.at(t)
            illum = current_sat.is_sunlit(self.eph)
            illums.append(illum.copy())
            topo = current_sat - self.observatory_site.at(t)
            ra, dec, distance = topo.radec()
            alt, az, dist = topo.altaz()
            ras.append(ra.radians)
            decs.append(dec.radians)
            alts.append(alt.radians)
        return np.vstack(ras), np.vstack(decs), np.vstack(alts), np.vstack(illums)

    def check_pointings(
        self,
        pointing_ras,
        pointing_decs,
        mjds,
        visit_time,
        fov_radius=1.75,
        test_radius=10.0,
        dt=2.0,
    ):
        """Calculate streak length and number of streaks in a set of visits.

        Parameters
        ----------
        pointing_ras : `np.ndarray`, (N,)
            The RA for each pointing (degrees).
        pointing_decs : `np.ndarray`, (N,)
            The dec for each pointing (degrees).
        mjds : `np.ndarray`, (N,)
            The MJD for the (start) of each pointing (days).
        visit_time : `np.ndarray`, (N,)
            The start to end time for a visit (seconds).
        fov_radius : `float`
            The radius of the science field of view (degrees)
        test_radius : `float`
            The radius to use to see if a streak gets close (degrees).
            Should be large, because satellites can be moving at ~1 deg/s.
        dt : `float`
            The timestep to use for high resolution checking
            if a satellite crossed (seconds).

        Returns
        -------
        streak length : `float`
            The total length of satellite streaks in the FoV (degrees)
        n_streak : `int`
            The number of streaks that were in the FoV.
        """
        test_radius_rad = np.radians(test_radius)
        dt = dt / 3600 / 24  # to days
        visit_time = visit_time / 3600.0 / 24.0

        # Arrays to hold results
        lengths_rad = np.zeros(pointing_ras.size, dtype=float)
        n_streaks = np.zeros(pointing_ras.size, dtype=int)

        input_id_indx_oned = np.arange(pointing_ras.size, dtype=int)

        # Convert everything to radians for internal computations
        pointing_ras_rad = np.radians(pointing_ras)
        pointing_decs_rad = np.radians(pointing_decs)
        fov_radius_rad = np.radians(fov_radius)

        # Note self.paths_array should return an array that is
        # N_sats x N_mjds in shape
        # And all angles in radians.
        sat_ra_1, sat_dec_1, sat_alt_1, sat_illum_1 = self.paths_array(mjds)
        mjd_end = mjds + visit_time
        sat_ra_2, sat_dec_2, sat_alt_2, sat_illum_2 = self.paths_array(mjd_end)

        # broadcast the pointings to be the same shape as the satellite arrays.
        pointing_ras_rad = np.broadcast_to(pointing_ras_rad, sat_ra_1.shape)
        pointing_decs_rad = np.broadcast_to(pointing_decs_rad, sat_ra_1.shape)
        input_id_indx = np.broadcast_to(input_id_indx_oned, sat_ra_1.shape)

        # Which satellites are above the altitude limit and illuminated
        # np.where confuses me when used on a 2d array.
        above_illum_indx = np.where(
            ((sat_alt_1 > self.alt_limit_rad) | (sat_alt_2 > self.alt_limit_rad))
            & ((sat_illum_1 == True) | (sat_illum_2 == True))
        )

        # point_to_line_distance can take arrays,
        # but they all need to be the same shape,
        # thus why we broadcast pointing ra and dec above.
        distances = point_to_line_distance(
            sat_ra_1[above_illum_indx],
            sat_dec_1[above_illum_indx],
            sat_ra_2[above_illum_indx],
            sat_dec_2[above_illum_indx],
            pointing_ras_rad[above_illum_indx],
            pointing_decs_rad[above_illum_indx],
        )

        close = np.where(distances < test_radius_rad)[0]

        # Numpy broadcasting is such a dark art
        sat_indx = np.arange(len(self.sat_list), dtype=int)[np.newaxis]
        sat_indx = np.broadcast_to(sat_indx.T, sat_ra_1.shape)

        mjd_broad = np.broadcast_to(mjds, sat_ra_1.shape)[above_illum_indx][close]
        visit_broad = np.broadcast_to(visit_time, sat_ra_1.shape)[above_illum_indx][close]

        # ok, this is pretty ugly, but should get the job done
        # Loop over all the potential collisions we have found
        for p_ra, p_dec, ob_indx, mjd, vt, sat_in in zip(
            pointing_ras_rad[above_illum_indx][close],
            pointing_decs_rad[above_illum_indx][close],
            input_id_indx[above_illum_indx][close],
            mjd_broad,
            visit_broad,
            sat_indx[above_illum_indx][close],
        ):
            mjd = np.linspace(mjd, mjd + vt, num=np.round(vt / dt).astype(int))
            jd = mjd + MJDOFFSET
            t = self.ts.ut1_jd(jd)
            sat = self.sat_list[sat_in]
            current_sat = sat.at(t)
            topo = current_sat - self.observatory_site.at(t)
            sat_ra, sat_dec, _distance = topo.radec()

            length = _streak_length(sat_ra.radians, sat_dec.radians, p_ra, p_dec, fov_radius_rad)
            if length > 0:
                lengths_rad[ob_indx] += length
                n_streaks[ob_indx] += 1
        return np.degrees(lengths_rad), n_streaks


def _streak_length(sat_ras, sat_decs, pointing_ra, pointing_dec, radius):
    """Calculate streak lengths for satellites in a given (circular) pointing.

    Parameters
    ----------
    sat_ras : `np.ndarray`, (N,)
        RA for each satellite (radians).
    sat_decs : `np.ndarray`, (N,)
        Decs for the satelltes (radians).
    pointing_ra : `float`
        RA of the pointing (radians).
    pointing_dec : `float`
        Dec of the pointing (radians).
    radius : `float`
        Radius of the field of view (radians).

    Returns
    -------
    length : `float`
        The length of the streak in radians
    """
    # Hopefully this broadcasts properly
    x, y = gnomonic_project_toxy(sat_ras, sat_decs, pointing_ra, pointing_dec)
    ls = LineString(zip(x, y))
    p = Point(0, 0)
    circle_buffer = p.buffer(radius)
    length = circle_buffer.intersection(ls).length
    return length
