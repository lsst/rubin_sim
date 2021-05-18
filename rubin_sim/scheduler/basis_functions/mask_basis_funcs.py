import numpy as np
import healpy as hp
from rubin_sim.utils import _hpid2RaDec, Site, _angularSeparation, _xyz_from_ra_dec
import matplotlib.pylab as plt
from rubin_sim.scheduler.basis_functions import Base_basis_function
from rubin_sim.scheduler.utils import hp_in_lsst_fov, int_rounded


__all__ = ['Zenith_mask_basis_function', 'Zenith_shadow_mask_basis_function',
           'Moon_avoidance_basis_function', 'Map_cloud_basis_function',
           'Planet_mask_basis_function', 'Mask_azimuth_basis_function',
           'Solar_elongation_mask_basis_function', 'Area_check_mask_basis_function']


class Area_check_mask_basis_function(Base_basis_function):
    """Take a list of other mask basis functions, and do an additional check for area available
    """
    def __init__(self, bf_list, nside=32, min_area=1000.):
        super(Area_check_mask_basis_function, self).__init__(nside=nside)
        self.bf_list = bf_list
        self.result = np.zeros(hp.nside2npix(self.nside), dtype=float)
        self.min_area = min_area

    def check_feasibility(self, conditions):
        result = True
        for bf in self.bf_list:
            if not bf.check_feasibility(conditions):
                return False

        area_map = self.result.copy()
        for bf in self.bf_list:
            area_map *= bf(conditions)

        good_pix = np.where(area_map == 0)[0]
        if hp.nside2pixarea(self.nside, degrees=True)*good_pix.size < self.min_area:
            result = False
        return result

    def _calc_value(self, conditions, **kwargs):
        result = self.result.copy()
        for bf in self.bf_list:
            result *= bf(conditions)
        return result


class Solar_elongation_mask_basis_function(Base_basis_function):
    """Mask things at various solar elongations

    Parameters
    ----------
    min_elong : float (0)
        The minimum solar elongation to consider (degrees).
    max_elong : float (60.)
        The maximum solar elongation to consider (degrees).
    """

    def __init__(self, min_elong=0., max_elong=60., nside=None, penalty=np.nan):
        super(Solar_elongation_mask_basis_function, self).__init__(nside=nside)
        self.min_elong = np.radians(min_elong)
        self.max_elong = np.radians(max_elong)
        self.penalty = penalty
        self.result = np.empty(hp.nside2npix(self.nside), dtype=float)
        self.result.fill(self.penalty)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        in_range = np.where((int_rounded(conditions.solar_elongation) >= int_rounded(self.min_elong)) &
                            (int_rounded(conditions.solar_elongation) <= int_rounded(self.max_elong)))[0]
        result[in_range] = 1
        return result


class Zenith_mask_basis_function(Base_basis_function):
    """Just remove the area near zenith.

    Parameters
    ----------
    min_alt : float (20.)
        The minimum possible altitude (degrees)
    max_alt : float (82.)
        The maximum allowed altitude (degrees)
    """
    def __init__(self, min_alt=20., max_alt=82., nside=None):
        super(Zenith_mask_basis_function, self).__init__(nside=nside)
        self.update_on_newobs = False
        self.min_alt = np.radians(min_alt)
        self.max_alt = np.radians(max_alt)
        self.result = np.empty(hp.nside2npix(self.nside), dtype=float).fill(self.penalty)

    def _calc_value(self, conditions, indx=None):

        result = self.result.copy()
        alt_limit = np.where((int_rounded(conditions.alt) > int_rounded(self.min_alt)) &
                             (int_rounded(conditions.alt) < int_rounded(self.max_alt)))[0]
        result[alt_limit] = 1
        return result


class Planet_mask_basis_function(Base_basis_function):
    """Mask the bright planets

    Parameters
    ----------
    mask_radius : float (3.5)
        The radius to mask around a planet (degrees).
    planets : list of str (None)
        A list of planet names to mask. Defaults to ['venus', 'mars', 'jupiter']. Not including
        Saturn because it moves really slow and has average apparent mag of ~0.4, so fainter than Vega.

    """
    def __init__(self, mask_radius=3.5, planets=None, nside=None, scale=1e5):
        super(Planet_mask_basis_function, self).__init__(nside=nside)
        if planets is None:
            planets = ['venus', 'mars', 'jupiter']
        self.planets = planets
        self.mask_radius = np.radians(mask_radius)
        self.result = np.zeros(hp.nside2npix(nside))
        # set up a kdtree. Could maybe use healpy.query_disc instead.
        self.in_fov = hp_in_lsst_fov(nside=nside, fov_radius=mask_radius, scale=scale)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()
        for pn in self.planets:
            indices = self.in_fov(conditions.planet_positions[pn+'_RA'], conditions.planet_positions[pn+'_dec'])
            result[indices] = np.nan

        return result


class Zenith_shadow_mask_basis_function(Base_basis_function):
    """Mask the zenith, and things that will soon pass near zenith. Useful for making sure
    observations will not be too close to zenith when they need to be observed again (e.g. for a pair).

    Parameters
    ----------
    min_alt : float (20.)
        The minimum alititude to alow. Everything lower is masked. (degrees)
    max_alt : float (82.)
        The maximum altitude to alow. Everything higher is masked. (degrees)
    shadow_minutes : float (40.)
        Mask anything that will pass through the max alt in the next shadow_minutes time. (minutes)
    """
    def __init__(self, nside=None, min_alt=20., max_alt=82.,
                 shadow_minutes=40., penalty=np.nan, site='LSST'):
        super(Zenith_shadow_mask_basis_function, self).__init__(nside=nside)
        self.update_on_newobs = False

        self.penalty = penalty

        self.min_alt = np.radians(min_alt)
        self.max_alt = np.radians(max_alt)
        self.ra, self.dec = _hpid2RaDec(nside, np.arange(hp.nside2npix(nside)))
        self.shadow_minutes = np.radians(shadow_minutes/60. * 360./24.)
        # Compute the declination band where things could drift into zenith
        self.decband = np.zeros(self.dec.size, dtype=float)
        self.zenith_radius = np.radians(90.-max_alt)/2.
        site = Site(name=site)
        self.lat_rad = site.latitude_rad
        self.lon_rad = site.longitude_rad
        self.decband[np.where((int_rounded(self.dec) < int_rounded(self.lat_rad+self.zenith_radius)) &
                              (int_rounded(self.dec) > int_rounded(self.lat_rad-self.zenith_radius)))] = 1

        self.result = np.empty(hp.nside2npix(self.nside), dtype=float)
        self.result.fill(self.penalty)

    def _calc_value(self, conditions, indx=None):

        result = self.result.copy()
        alt_limit = np.where((int_rounded(conditions.alt) > int_rounded(self.min_alt)) &
                             (int_rounded(conditions.alt) < int_rounded(self.max_alt)))[0]
        result[alt_limit] = 1
        to_mask = np.where((int_rounded(conditions.HA) > int_rounded(2.*np.pi-self.shadow_minutes-self.zenith_radius)) &
                           (self.decband == 1))
        result[to_mask] = np.nan
        return result


class Moon_avoidance_basis_function(Base_basis_function):
    """Avoid looking too close to the moon.

    Parameters
    ----------
    moon_distance: float (30.)
        Minimum allowed moon distance. (degrees)

    XXX--TODO:  This could be a more complicated function of filter and moon phase.
    """
    def __init__(self, nside=None, moon_distance=30.):
        super(Moon_avoidance_basis_function, self).__init__(nside=nside)
        self.update_on_newobs = False

        self.moon_distance = int_rounded(np.radians(moon_distance))
        self.result = np.ones(hp.nside2npix(self.nside), dtype=float)

    def _calc_value(self, conditions, indx=None):
        result = self.result.copy()

        angular_distance = _angularSeparation(conditions.az, conditions.alt,
                                              conditions.moonAz,
                                              conditions.moonAlt)

        result[int_rounded(angular_distance) < self.moon_distance] = np.nan

        return result


class Bulk_cloud_basis_function(Base_basis_function):
    """Mark healpixels on a map if their cloud values are greater than
    the same healpixels on a maximum cloud map.

    Parameters
    ----------
    nside: int (default_nside)
        The healpix resolution.
    max_cloud_map : numpy array (None)
        A healpix map showing the maximum allowed cloud values for all points on the sky
    out_of_bounds_val : float (10.)
        Point value to give regions where there are no observations requested
    """

    def __init__(self, nside=None, max_cloud_map=None, max_val=0.7,
                 out_of_bounds_val=np.nan):
        super(Bulk_cloud_basis_function, self).__init__(nside=nside)
        self.update_on_newobs = False

        if max_cloud_map is None:
            self.max_cloud_map = np.zeros(hp.nside2npix(nside), dtype=float) + max_val
        else:
            self.max_cloud_map = max_cloud_map
        self.out_of_bounds_area = np.where(self.max_cloud_map > 1.)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.ones(hp.nside2npix(self.nside))

    def _calc_value(self, conditions, indx=None):
        """
        Parameters
        ----------
        indx : list (None)
            Index values to compute, if None, full map is computed
        Returns
        -------
        Healpix map where pixels with a cloud value greater than the max_cloud_map
        value are marked as unseen.
        """

        result = self.result.copy()

        clouded = np.where(self.max_cloud_map <= conditions.bulk_cloud)
        result[clouded] = self.out_of_bounds_val

        return result


class Map_cloud_basis_function(Base_basis_function):
    """Mark healpixels on a map if their cloud values are greater than
    the same healpixels on a maximum cloud map. Currently a placeholder for
    when the telemetry stream can include a full sky cloud map.

    Parameters
    ----------
    nside: int (default_nside)
        The healpix resolution.
    max_cloud_map : numpy array (None)
        A healpix map showing the maximum allowed cloud values for all points on the sky
    out_of_bounds_val : float (10.)
        Point value to give regions where there are no observations requested
    """

    def __init__(self, nside=None, max_cloud_map=None, max_val=0.7,
                 out_of_bounds_val=np.nan):
        super(Bulk_cloud_basis_function, self).__init__(nside=nside)
        self.update_on_newobs = False

        if max_cloud_map is None:
            self.max_cloud_map = np.zeros(hp.nside2npix(nside), dtype=float) + max_val
        else:
            self.max_cloud_map = max_cloud_map
        self.out_of_bounds_area = np.where(self.max_cloud_map > 1.)[0]
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.ones(hp.nside2npix(self.nside))

    def _calc_value(self, conditions, indx=None):
        """
        Parameters
        ----------
        indx : list (None)
            Index values to compute, if None, full map is computed
        Returns
        -------
        Healpix map where pixels with a cloud value greater than the max_cloud_map
        value are marked as unseen.
        """

        result = self.result.copy()

        clouded = np.where(self.max_cloud_map <= conditions.bulk_cloud)
        result[clouded] = self.out_of_bounds_val

        return result


class Mask_azimuth_basis_function(Base_basis_function):
    """Mask pixels based on azimuth
    """
    def __init__(self, nside=None, out_of_bounds_val=np.nan, az_min=0., az_max=180.):
        super(Mask_azimuth_basis_function, self).__init__(nside=nside)
        self.az_min = int_rounded(np.radians(az_min))
        self.az_max = int_rounded(np.radians(az_max))
        self.out_of_bounds_val = out_of_bounds_val
        self.result = np.ones(hp.nside2npix(self.nside))

    def _calc_value(self, conditions, indx=None):
        to_mask = np.where((int_rounded(conditions.az) > self.az_min) & (int_rounded(conditions.az) < self.az_max))[0]
        result = self.result.copy()
        result[to_mask] = self.out_of_bounds_val

        return result
