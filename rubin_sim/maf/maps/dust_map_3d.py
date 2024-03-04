__all__ = ("DustMap3D",)

import warnings

import numpy as np

from rubin_sim.maf.maps import BaseMap
from rubin_sim.phot_utils import DustValues

from .ebv_3d_hp import ebv_3d_hp, get_x_at_nearest_y


class DustMap3D(BaseMap):
    """Add 3-d E(B-V) values to the slice points.

    See "notes" below for a discussion of the content of the map keys,
    and functionality that can be accessed by calling
    `DustMap3d.distance_at_mag` with the key values at a given slice point.

    Parameters
    ----------
    nside : `int`
        Healpixel resolution (2^x) to read from disk.
    map_file : `str`, opt
        Path to dust map file.
    interp : `bool`, opt
        Should returned values be interpolated (True)
        or just nearest neighbor (False).
        Default True, but is ignored if 'pixels' is provided.
    filtername : 'str', opt
        Name of the filter (to match the lsst filter names in
        rubin_sim.photUtils.DustValues) in which to calculate dust
        extinction magnitudes
    dist_pc : `float`, opt
        Distance at which to precalculate the nearest ebv value (pc)
    d_mag : `float`, opt
        Calculate the maximum distance which matches this `d_mag`
        d_mag == m-mO (dust extinction + distance modulus)
    r_x : `dict` {`str`: `float`}, opt
        Per-filter dust extinction curve coefficients.
        Calculated by rubin_sim.photUtils.DustValues if "None".

    Notes
    -----
    The slice point dictionary keys are expanded with the following keys:
    ebv3d_dists -
    the distances from the 3d dust map at each slice_point (in pc)
    `ebv3d_ebvs` -
    the E(B-V) values corresponding to each distance at each slice_point
    `ebv3d_ebv_at_<dist_pc>` -
    the (single) ebv value at the nearest distance to dist_pc
    `ebv3d_dist_at_<d_mag>` -
    the (single) distance value corresponding to where extinction and
    distance modulus combine to create a m-Mo value of d_mag, for the filter
    specified in filtername (in pc).
    Note that <dist_pc> and <d_mag> will be formatted with a
    single decimal place.

    The additional method 'distance_at_mag' can be called either with the
    distances and ebv values for the entire map or with the values from a
    single slice_point, in order to calculate the distance at which
    extinction and distance modulus combine to create a m-Mo value closest
    to 'dmag' in any filter. This is the same value as would be reported in
    ebv3d_dist_at_<d_mag>, but can be calculated on the fly,
    allowing variable filters and dmag values.
    """

    def __init__(
        self,
        nside=64,
        map_file=None,
        interp=True,
        filtername="r",
        dist_pc=3000,
        d_mag=15.2,
        r_x=None,
    ):
        self.nside = nside
        self.interp = interp
        self.map_file = map_file
        self.filtername = filtername
        self.dist_pc = dist_pc
        self.d_mag = d_mag
        # r_x is the extinction coefficient (A_v = R_v * E(B-V) ..
        # A_x = r_x * E(B-V)) per filter
        # This is equivalent to calculating A_x
        # (using rubin_sim.photUtils.Sed.addDust) in each
        # filter and setting E(B-V) to 1 [so similar to the values
        # calculated in DustValues.
        if r_x is None:
            self.r_x = DustValues().r_x.copy()
        else:
            self.r_x = r_x
        # The values that will be added to the slicepoints
        self.keynames = [
            "ebv3d_ebvs",
            "ebv3d_dists",
            f"ebv3d_ebv_at_{self.dist_pc:.1f}",
            f"ebv3d_dist_at_{self.d_mag:.1f}",
        ]

    def run(self, slice_points):
        # If the slicer has nside,
        # it's a healpix slicer so we can read the map directly
        if "nside" in slice_points:
            if slice_points["nside"] != self.nside:
                warnings.warn(
                    f"Slicer value of nside {slice_points['nside']} different "
                    f"from map value {self.nside}, will use correct slicer value here."
                )
            dists, ebvs = ebv_3d_hp(
                slice_points["nside"],
                pixels=slice_points["sid"],
                map_file=self.map_file,
            )
        # Not a healpix slicer,
        # look up values based on RA,dec with possible interpolation
        else:
            dists, ebvs = ebv_3d_hp(
                self.nside,
                ra=slice_points["ra"],
                dec=slice_points["dec"],
                interp=self.interp,
                map_file=self.map_file,
            )

        # Calculate the map ebv and dist values at the initialized distance
        dist_closest, ebv_at_dist = get_x_at_nearest_y(dists, ebvs, self.dist_pc)

        # Calculate the distances at which m_minus_Mo values
        # of 'dmag' are reached
        dist_dmag = self.distance_at_dmag(self.d_mag, dists, ebvs, self.filtername)

        slice_points["ebv3d_dists"] = dists
        slice_points["ebv3d_ebvs"] = ebvs
        slice_points[f"ebv3d_ebv_at_{self.dist_pc:.1f}"] = ebv_at_dist
        slice_points[f"ebv3d_dist_at_{self.d_mag:.1f}"] = dist_dmag

        return slice_points

    def distance_at_dmag(self, dmag, dists, ebvs, filtername=None):
        """Calculate the distance at which a given change of magnitude would
        occur (including distance modulus and dust extinction).

        Parameters
        ----------
        dmag : `float`
            The magnitude change expected.
        dists : `np.ndarray`, (N,)
            The distances corresponding to the ebv values.
        ebvs : `np.ndarray`, (N,)
            The ebv values at each distance.
        filtername : `str` or None
            The filter in which to evaluate the magnitude change.
            If None, uses the default filter for the map.
            The filter translates ebv into magnitudes of extinction.

        Returns
        -------
        dist_dmag : `float`
            The distance at which the specified dmag occurs.
        """
        # Provide this as a method which could be used for a single
        # slice_point as well as for whole map
        # (single slice_point means you could calculate this for any
        # arbitrary magnitude or filter if needed)
        # This method is here because some metrics require it.
        if filtername is None:
            filtername = self.filtername
        # calculate distance modulus for each distance
        dmods = 5.0 * np.log10(dists) - 5.0
        # calculate dust extinction at each distance, for the filtername
        a_x = self.r_x[filtername] * ebvs
        # calculate the (m-Mo) = distmod + a_x -- combination of extinction
        # due to distance and dust
        m_minus__mo = dmods + a_x

        # Maximum distance for the given m-Mo (dmag) value
        # first do the 'within the map' closest distance
        m_minus__mo_at_mag, dist_closest = get_x_at_nearest_y(m_minus__mo, dists, dmag)
        # calculate distance modulus for an object with the maximum dust
        # extinction (and then the distance)
        if a_x.ndim == 2:
            dist_mods_far = dmag - a_x[:, -1]
        else:
            dist_mods_far = dmag - a_x.max()
        dists_far = 10.0 ** (0.2 * dist_mods_far + 1.0)
        # Use furthest of the two
        dist_dmag = np.where(dists_far > dist_closest, dists_far, dist_closest)
        return dist_dmag
