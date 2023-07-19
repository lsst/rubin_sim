__all__ = ("SkyModelPre", "interp_angle")

import glob
import os
import warnings

import h5py
import healpy as hp
import numpy as np

from rubin_sim.data import get_data_dir
from rubin_sim.utils import _angular_separation, _hpid2_ra_dec, survey_start_mjd


def short_angle_dist(a0, a1):
    """
    from https://gist.github.com/shaunlebron/8832585
    """
    max_angle = 2.0 * np.pi
    da = (a1 - a0) % max_angle
    return 2.0 * da % max_angle - da


def interp_angle(x_out, xp, anglep, degrees=False):
    """
    Interpolate angle values (handle wrap around properly). Does nearest neighbor
    interpolation if values out of range.

    Parameters
    ----------
    x_out : `float` or array
        The points to interpolate to.
    xp : array
        Points to interpolate between (must be sorted)
    anglep : array
        The angles ascociated with xp
    degrees : `bool` (False)
        Set if anglep is degrees (True) or radidian (False)
    """

    # Where are the interpolation points
    x = np.atleast_1d(x_out)
    left = np.searchsorted(xp, x) - 1
    right = left + 1

    # If we are out of bounds, just use the edges
    right[np.where(right >= xp.size)] -= 1
    left[np.where(left < 0)] += 1
    baseline = xp[right] - xp[left]

    wterm = (x - xp[left]) / baseline
    wterm[np.where(baseline == 0)] = 0
    if degrees:
        result = (
            np.radians(anglep[left])
            + short_angle_dist(np.radians(anglep[left]), np.radians(anglep[right])) * wterm
        )
        result = result % (2.0 * np.pi)
        result = np.degrees(result)
    else:
        result = anglep[left] + short_angle_dist(anglep[left], anglep[right]) * wterm
        result = result % (2.0 * np.pi)
    return result


class SkyModelPre:
    """
    Load pre-computed sky brighntess maps for the LSST site and use them to interpolate to
    arbitrary dates.
    """

    def __init__(
        self,
        data_path=None,
        init_load_length=10,
        load_length=365,
        verbose=False,
        mjd0=None,
    ):
        """
        Parameters
        ----------
        data_path : `str`, opt
            path to the numpy save files. Looks in standard SIMS_SKYBRIGHTNESS_DATA or RUBIN_SIM_DATA_DIR
            if set to default (None).
        init_load_length : `int` (10)
            The length of time (days) to load from disk initially. Set to something small for fast reads.
        load_length : `int` (365)
            The number of days to load after the initial load.
        mjd0 : `float` (None)
            The starting MJD to load on initilization (days). Uses util to lookup default if None.
        """

        self.info = None
        self.sb = None
        self.header = None
        self.filter_names = None
        self.verbose = verbose

        # Look in default location for .npz files to load
        if "SIMS_SKYBRIGHTNESS_DATA" in os.environ:
            data_path = os.environ["SIMS_SKYBRIGHTNESS_DATA"]
        else:
            data_path = os.path.join(get_data_dir(), "skybrightness_pre")

        # Expect filenames of the form mjd1_mjd2.npz, e.g., 59632.155_59633.2.npz
        self.files = glob.glob(os.path.join(data_path, "*.h5"))
        if len(self.files) == 0:
            errmssg = "Failed to find pre-computed .h5 files. "
            errmssg += "Copy data from NCSA with sims_skybrightness_pre/data/data_down.sh \n"
            errmssg += "or build by running sims_skybrightness_pre/data/generate_hdf5.py"
            warnings.warn(errmssg)
        self.filesizes = np.array([os.path.getsize(filename) for filename in self.files])
        mjd_left = []
        mjd_right = []
        # glob does not always order things I guess?
        self.files.sort()
        for filename in self.files:
            temp = os.path.split(filename)[-1].replace(".h5", "").split("_")
            mjd_left.append(float(temp[0]))
            mjd_right.append(float(temp[1]))

        self.mjd_left = np.array(mjd_left)
        self.mjd_right = np.array(mjd_right)

        # Set that nothing is loaded at this point
        self.loaded_range = np.array([-1])
        self.timestep_max = -1

        if mjd0 is None:
            mjd0 = survey_start_mjd()

        # Do a quick initial load if set
        if init_load_length is not None:
            self.load_length = init_load_length
            self._load_data(mjd0)
        # swap back to the longer load length
        self.load_length = load_length
        self.nside = 32
        hpid = np.arange(hp.nside2npix(self.nside))
        self.ra, self.dec = _hpid2_ra_dec(self.nside, hpid)

    def _load_data(self, mjd, filename=None, npyfile=None):
        """
        Load up the .npz file to interpolate things. After python 3 upgrade, numpy.savez refused
        to write large .npz files, so data is split between .npz and .npy files.

        Parameters
        ----------
        mjd : `float`
            The Modified Julian Date that we want to load
        filename : `str` (None)
            The filename to restore. If None, it checks the filenames on disk to find one that
            should have the requested MJD
        npyfile : `str` (None)
            If sky brightness data not in npz file, checks the .npy file with same root name.
        """
        del self.sb
        del self.filter_names
        del self.timestep_max

        if filename is None:
            # Figure out which file to load.
            file_indx = np.where((mjd >= self.mjd_left) & (mjd <= self.mjd_right))[0]
            if np.size(file_indx) == 0:
                raise ValueError(
                    "MJD = %f is out of range for the files found (%f-%f)"
                    % (mjd, self.mjd_left.min(), self.mjd_right.max())
                )
            # Just take the later one, assuming we're probably simulating forward in time
            file_indx = np.max(file_indx)

            filename = self.files[file_indx]
        else:
            self.loaded_range = None

        if self.verbose:
            print("Loading file %s" % filename)
        h5 = h5py.File(filename, "r")
        mjds = h5["mjds"][:]
        indxs = np.where((mjds >= mjd) & (mjds <= (mjd + self.load_length)))
        indxs = [np.min(indxs), np.max(indxs)]
        if indxs[0] > 0:
            indxs[0] -= 1
        self.loaded_range = np.array([mjds[indxs[0]], mjds[indxs[1]]])
        self.mjds = mjds[indxs[0] : indxs[1]]
        _timestep_max = np.empty(1, dtype=float)
        h5["timestep_max"].read_direct(_timestep_max)
        self.timestep_max = np.max(_timestep_max)

        self.sb = h5["sky_mags"][indxs[0] : indxs[1]]
        self.filter_names = self.sb.dtype.names
        h5.close()

        if self.verbose:
            print("%s loaded" % os.path.split(filename)[1])

        self.nside = hp.npix2nside(self.sb[self.filter_names[0]][0, :].size)

    def return_mags(
        self,
        mjd,
        indx=None,
        badval=hp.UNSEEN,
        filters=["u", "g", "r", "i", "z", "y"],
        extrapolate=False,
    ):
        """
        Return a full sky map or individual pixels for the input mjd

        Parameters
        ----------
        mjd : `float`
            Modified Julian Date to interpolate to
        indx : `List` of `int` (None)
            indices to interpolate the sky values at. Returns full sky if None. If the class was
            instatiated with opsimFields, indx is the field ID, otherwise it is the healpix ID.
        airmass_mask : `bool` (True)
            Set high (>2.5) airmass pixels to badval.
        planet_mask : `bool` (True)
            Set sky maps to badval near (2 degrees) bright planets.
        moon_mask : `bool` (True)
            Set sky maps near (10 degrees) the moon to badval.
        zenith_mask : `bool` (True)
            Set sky maps at high altitude (>86.5) to badval.
        badval : `float` (-1.6375e30)
            Mask value. Defaults to the healpy mask value.
        filters : `list`, opt
            List of strings for the filters that should be returned.
            Default returns ugrizy.
        extrapolate : `bool` (False)
            In indx is set, extrapolate any masked pixels to be the same as the nearest non-masked
            value from the full sky map.

        Returns
        -------
        sbs : `dict`
            A dictionary with filter names as keys and np.arrays as values which
            hold the sky brightness maps in mag/sq arcsec.
        """
        if mjd < self.loaded_range.min() or (mjd > self.loaded_range.max()):
            self._load_data(mjd)

        left = np.searchsorted(self.mjds, mjd) - 1
        right = left + 1

        # Do full sky by default
        if indx is None:
            indx = np.arange(self.sb["r"].shape[1])
            full_sky = True
        else:
            full_sky = False

        # If we are out of bounds
        if right >= self.mjds.size:
            right -= 1
            baseline = 1.0
        elif left < 0:
            left += 1
            baseline = 1.0
        else:
            baseline = self.mjds[right] - self.mjds[left]

        # Check if we are between sunrise/set
        if baseline > self.timestep_max + 1e-6:
            warnings.warn("Requested MJD between sunrise and sunset, returning closest maps")
            diff = np.abs(self.mjds[left.max() : right.max() + 1] - mjd)
            closest_indx = np.array([left, right])[np.where(diff == np.min(diff))].min()
            sbs = {}
            for filter_name in filters:
                sbs[filter_name] = self.sb[filter_name][closest_indx, indx]
                sbs[filter_name][np.isinf(sbs[filter_name])] = badval
                sbs[filter_name][np.where(sbs[filter_name] == hp.UNSEEN)] = badval
        else:
            wterm = (mjd - self.mjds[left]) / baseline
            w1 = 1.0 - wterm
            w2 = wterm
            sbs = {}
            for filter_name in filters:
                sbs[filter_name] = (
                    self.sb[filter_name][left, indx] * w1 + self.sb[filter_name][right, indx] * w2
                )
        # If requested a certain pixel(s), and want to extrapolate.
        if (not full_sky) & extrapolate:
            masked_pix = False
            for filter_name in filters:
                if (badval in sbs[filter_name]) | (True in np.isnan(sbs[filter_name])):
                    masked_pix = True
            if masked_pix:
                # We have pixels that are masked that we want reasonable values for
                full_sky_sb = self.return_mags(
                    mjd,
                    filters=filters,
                )
                good = np.where((full_sky_sb[filters[0]] != badval) & ~np.isnan(full_sky_sb[filters[0]]))[0]
                ra_full = self.ra[good]  # np.radians(self.header["ra"][good])
                dec_full = self.dec[good]  # np.radians(self.header["dec"][good])
                for filtername in filters:
                    full_sky_sb[filtername] = full_sky_sb[filtername][good]
                # Going to assume the masked pixels are the same in all filters
                masked_indx = np.where(
                    (sbs[filters[0]].ravel() == badval) | np.isnan(sbs[filters[0]].ravel())
                )[0]
                for i, mi in enumerate(masked_indx):
                    # Note, this is going to be really slow for many pixels, should use a kdtree
                    dist = _angular_separation(
                        self.ra[indx[i]],
                        self.dec[indx[i]],
                        ra_full,
                        dec_full,
                    )
                    closest = np.where(dist == dist.min())[0]
                    for filtername in filters:
                        sbs[filtername].ravel()[mi] = np.min(full_sky_sb[filtername][closest])

        return sbs
