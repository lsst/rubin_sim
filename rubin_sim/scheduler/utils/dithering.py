__all__ = ("wrap_ra_dec", "rotate_ra_dec", "Pointings2hp", "HpmapCross")

import healpy as hp
import numpy as np
from scipy.optimize import minimize

from rubin_sim.site_models import _read_fields
from rubin_sim.utils import _hpid2_ra_dec, _xyz_angular_radius, _xyz_from_ra_dec

from .utils import hp_kd_tree, set_default_nside

default_nside = set_default_nside()


def wrap_ra_dec(ra, dec):
    # XXX--from MAF, should put in general utils
    """
    Wrap RA into 0-2pi and Dec into +/0 pi/2.

    Parameters
    ----------
    ra : numpy.ndarray
        RA in radians
    dec : numpy.ndarray
        Dec in radians

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Wrapped RA/Dec values, in radians.
    """
    # Wrap dec.
    low = np.where(dec < -np.pi / 2.0)[0]
    dec[low] = -1 * (np.pi + dec[low])
    ra[low] = ra[low] - np.pi
    high = np.where(dec > np.pi / 2.0)[0]
    dec[high] = np.pi - dec[high]
    ra[high] = ra[high] - np.pi
    # Wrap RA.
    ra = ra % (2.0 * np.pi)
    return ra, dec


def rotate_ra_dec(ra, dec, ra_target, dec_target, init_rotate=0.0):
    """
    Rotate ra and dec coordinates to be centered on a new dec.

    Rotates around the x-axis 1st, then to the dec, then ra.

    Parameters
    ----------
    ra : float or np.array
        RA coordinate(s) to be rotated in radians
    dec : float or np.array
        Dec coordinate(s) to be rotated in radians
    ra_rotation : float
        RA distance to rotate in radians
    dec_target : float
        Dec distance to rotate in radians
    init_rotate : float (0.)
        The amount to rotate the points around the x-axis first (radians).
    """
    # point (ra,dec) = (0,0) is at x,y,z = 1,0,0

    x, y, z = _xyz_from_ra_dec(ra, dec)

    # Rotate around the x axis to start
    xp = x
    if init_rotate != 0.0:
        c_i = np.cos(init_rotate)
        s_i = np.sin(init_rotate)
        yp = c_i * y - s_i * z
        zp = s_i * y + c_i * z
    else:
        yp = y
        zp = z

    theta_y = dec_target
    c_ty = np.cos(theta_y)
    s_ty = np.sin(theta_y)

    # Rotate about y
    xp2 = c_ty * xp + s_ty * zp
    zp2 = -s_ty * xp + c_ty * zp

    # Convert back to RA, Dec
    ra_p = np.arctan2(yp, xp2)
    dec_p = -np.arcsin(zp2)

    # Rotate to the correct RA
    ra_p += ra_target

    ra_p, dec_p = wrap_ra_dec(ra_p, dec_p)

    return ra_p, dec_p


class Pointings2hp:
    """
    Convert a list of telescope pointings and convert them to a pointing map
    """

    def __init__(self, nside, radius=1.75):
        """"""
        # hmm, not sure what the leafsize should be? Kernel can crash if too low.
        self.tree = hp_kd_tree(nside=nside, leafsize=300)
        self.nside = nside
        self.rad = _xyz_angular_radius(radius)
        self.bins = np.arange(hp.nside2npix(nside) + 1) - 0.5

    def __call__(self, ra, dec, stack=True):
        """
        similar to utils.hp_in_lsst_fov, but can take a arrays of ra,dec.

        Parameters
        ----------
        ra : array_like
            RA in radians
        dec : array_like
            Dec in radians

        Returns
        -------
        result : healpy map
            The number of times each healpxel is observed by the given pointings
        """
        xs, ys, zs = _xyz_from_ra_dec(ra, dec)
        coords = np.array((xs, ys, zs)).T
        indx = self.tree.query_ball_point(coords, self.rad)
        # Convert array of lists to single array
        if stack:
            indx = np.hstack(indx)
            result, bins = np.histogram(indx, bins=self.bins)
        else:
            result = indx

        return result


class HpmapCross:
    """
    Find the cross-correlation of a healpix map and a bunch of rotated pointings
    """

    # XXX--just a very random radius search
    def __init__(self, nside=default_nside, radius=1.75, radius_search=1.75):
        """"""
        self.nside = nside
        # XXX -- should I shrink the radius slightly to get rid of overlap? That would be clever!
        self.p2hp_search = Pointings2hp(nside=nside, radius=radius_search)
        self.p2hp = Pointings2hp(nside=nside, radius=radius)
        # Load up a list of pointings, chop them down to a small block
        ra, dec = _read_fields()
        fields = np.empty(ra.size, dtype=list(zip(["RA", "dec"], [float, float])))
        fields["RA"] = ra
        fields["dec"] = dec
        good = np.where((fields["RA"] > np.radians(360.0 - 15.0)) | (fields["RA"] < np.radians(15.0)))
        fields = fields[good]
        good = np.where(np.abs(fields["dec"]) < np.radians(15.0))
        fields = fields[good]
        self.ra = fields["RA"]
        self.dec = fields["dec"]

        # Healpixel ra and dec
        self.hp_ra, self.hp_dec = _hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))

    def set_map(self, inmap):
        """
        Set the map that will be cross correlated
        """
        self.inmap = inmap

    def __call__(self, x, return_pointings_map=False):
        """
        Parameters
        ----------
        x[0], ra_rot : float
            Amount to rotate the fields in RA (radians)
        x[1], dec_rot : float
            Amount to rotate the fields in Dec (radians)
        x[2], im_rot : float
            Initial rotation to apply to fields (radians)
        return_pointings_map : bool (False)
            If set, return the overlapping fields and the resulting observing helpix map

        Returns
        -------
        cross_corr : float
            If return_pointings_map is False, return the sum of the pointing map multipled
            with the
        """
        # XXX-check the nside

        # Unpack the x variable
        ra_rot = x[0]
        dec_rot = x[1]
        im_rot = x[2]

        # Rotate pointings to desired position
        final_ra, final_dec = rotate_ra_dec(self.ra, self.dec, ra_rot, dec_rot, init_rotate=im_rot)
        # Find the number of observations at each healpixel
        obs_map = self.p2hp_search(final_ra, final_dec)
        good = np.where(self.inmap != hp.UNSEEN)[0]

        if return_pointings_map:
            obs_indx = self.p2hp_search(final_ra, final_dec, stack=False)
            good_pointings = np.array(
                [True if np.intersect1d(indxes, good).size > 0 else False for indxes in obs_indx]
            )
            if True not in good_pointings:
                raise ValueError("No pointings overlap requested pixels")
            obs_map = self.p2hp(final_ra[good_pointings], final_dec[good_pointings])
            return final_ra[good_pointings], final_dec[good_pointings], obs_map
        else:
            # If some requested pixels are not observed
            if np.min(obs_map[good]) == 0:
                return np.inf
            else:
                result = np.sum(self.inmap[good] * obs_map[good]) / float(
                    np.sum(self.inmap[good] + obs_map[good])
                )
                return result

    def minimize(self, ra_delta=1.0, dec_delta=1.0, rot_delta=30.0):
        """
        Let's find the minimum of the cross correlation.
        """

        reward_max = np.where(self.inmap == self.inmap.max())[0]
        ra_guess = np.median(self.hp_ra[reward_max])
        dec_guess = np.median(self.hp_dec[reward_max])

        x0 = np.array([ra_guess, dec_guess, 0.0])

        ra_delta = np.radians(ra_delta)
        dec_delta = np.radians(dec_delta)
        rot_delta = np.radians(rot_delta)

        # rots = np.arange(-np.pi/2., np.pi/2.+rot_delta, rot_delta)
        rots = [np.radians(0.0)]
        # Make sure the initial simplex is large enough
        # XXX--might need to update scipy latest version to actually use this.
        deltas = np.array(
            [
                [ra_delta, 0, 0],
                [0, dec_delta, rot_delta],
                [-ra_delta, 0, -rot_delta],
                [ra_delta, -dec_delta, 2.0 * rot_delta],
            ]
        )
        init_simplex = deltas + x0
        minimum = None
        for rot in rots:
            x0[-1] = rot
            min_result = minimize(
                self,
                x0,
                method="Nelder-Mead",
                options={"initial_simplex": init_simplex},
            )
            if minimum is None:
                minimum = min_result.fun
                result = min_result
            if min_result.fun < minimum:
                minimum = min_result.fun
                result = min_result
        return result.x
