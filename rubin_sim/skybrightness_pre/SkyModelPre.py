import numpy as np
import glob
import os
import healpy as hp
import warnings
from rubin_sim.utils import _angularSeparation
from rubin_sim.data import get_data_dir

__all__ = ['SkyModelPre', 'interp_angle']


def shortAngleDist(a0, a1):
    """
    from https://gist.github.com/shaunlebron/8832585
    """
    max_angle = 2.*np.pi
    da = (a1 - a0) % max_angle
    return 2.*da % max_angle - da


def interp_angle(x_out, xp, anglep, degrees=False):
    """
    Interpolate angle values (handle wrap around properly). Does nearest neighbor
    interpolation if values out of range.

    Parameters
    ----------
    x_out : float (or array)
        The points to interpolate to.
    xp : array
        Points to interpolate between (must be sorted)
    anglep : array
        The angles ascociated with xp
    degrees : bool (False)
        Set if anglep is degrees (True) or radidian (False)
    """

    # Where are the interpolation points
    x = np.atleast_1d(x_out)
    left = np.searchsorted(xp, x)-1
    right = left+1

    # If we are out of bounds, just use the edges
    right[np.where(right >= xp.size)] -= 1
    left[np.where(left < 0)] += 1
    baseline = xp[right] - xp[left]

    wterm = (x - xp[left])/baseline
    wterm[np.where(baseline == 0)] = 0
    if degrees:
        result = np.radians(anglep[left]) + shortAngleDist(np.radians(anglep[left]), np.radians(anglep[right]))*wterm
        result = result % (2.*np.pi)
        result = np.degrees(result)
    else:
        result = anglep[left] + shortAngleDist(anglep[left], anglep[right])*wterm
        result = result % (2.*np.pi)
    return result


class SkyModelPre(object):
    """
    Load pre-computed sky brighntess maps for the LSST site and use them to interpolate to
    arbitrary dates.
    """

    def __init__(self, data_path=None, speedLoad=True, verbose=False):
        """
        Parameters
        ----------
        data_path : str (None)
            path to the numpy save files. Looks in standard plances if set to None.
        speedLoad : bool (True)
            If True, use the small 3-day file to load found in the usual spot.
        """

        self.info = None
        self.sb = None
        self.header = None
        self.filter_names = None
        self.verbose = verbose

        # Look in default location for .npz files to load
        if 'SIMS_SKYBRIGHTNESS_DATA' in os.environ:
            data_path = os.environ['SIMS_SKYBRIGHTNESS_DATA']
        else:
            data_path = os.path.join(get_data_dir(), 'skybrightness_pre')

        # Expect filenames of the form mjd1_mjd2.npz, e.g., 59632.155_59633.2.npz
        self.files = glob.glob(os.path.join(data_path, '*.npz'))
        if len(self.files) == 0:
            errmssg = 'Failed to find pre-computed .npz files. '
            errmssg += 'Copy data from NCSA with sims_skybrightness_pre/data/data_down.sh \n'
            errmssg += 'or build by running sims_skybrightness_pre/data/generate_sky.py'
            warnings.warn(errmssg)
        self.filesizes = np.array([os.path.getsize(filename) for filename in self.files])
        mjd_left = []
        mjd_right = []
        # glob does not always order things I guess?
        self.files.sort()
        for filename in self.files:
            temp = os.path.split(filename)[-1].replace('.npz', '').split('_')
            mjd_left.append(float(temp[0]))
            mjd_right.append(float(temp[1]))

        self.mjd_left = np.array(mjd_left)
        self.mjd_right = np.array(mjd_right)

        # Set that nothing is loaded at this point
        self.loaded_range = np.array([-1])

        # Go ahead and load the small one in the repo by default
        if speedLoad:
            self._load_data(59853.,
                            filename=os.path.join(data_path, '59853_59856.npz'),
                            npyfile=os.path.join(data_path, '59853_59856.npy'))

    def _load_data(self, mjd, filename=None, npyfile=None):
        """
        Load up the .npz file to interpolate things. After python 3 upgrade, numpy.savez refused
        to write large .npz files, so data is split between .npz and .npy files.

        Parameters
        ----------
        mjd : float
            The Modified Julian Date that we want to load
        filename : str (None)
            The filename to restore. If None, it checks the filenames on disk to find one that
            should have the requested MJD
        npyfile : str (None)
            If sky brightness data not in npz file, checks the .npy file with same root name.
        """
        del self.info
        del self.sb
        del self.header
        del self.filter_names

        if filename is None:
            # Figure out which file to load.
            file_indx = np.where((mjd >= self.mjd_left) & (mjd <= self.mjd_right))[0]
            if np.size(file_indx) == 0:
                raise ValueError('MJD = %f is out of range for the files found (%f-%f)' % (mjd,
                                                                                           self.mjd_left.min(),
                                                                                           self.mjd_right.max()))
            # Just take the later one, assuming we're probably simulating forward in time
            file_indx = np.max(file_indx)

            filename = self.files[file_indx]

            self.loaded_range = np.array([self.mjd_left[file_indx], self.mjd_right[file_indx]])
        else:
            self.loaded_range = None

        if self.verbose:
            print('Loading file %s' % filename)
        # Add encoding kwarg to restore Python 2.7 generated files
        data = np.load(filename, encoding='bytes', allow_pickle=True)
        self.info = data['dict_of_lists'][()]
        self.header = data['header'][()]
        if 'sky_brightness' in data.keys():
            self.sb = data['sky_brightness'][()]
            data.close()
        else:
            # the sky brightness had to go in it's own npy file
            data.close()
            if npyfile is None:
                npyfile = filename[:-3]+'npy'
            self.sb = np.load(npyfile)
            if self.verbose:
                print('also loading %s' % npyfile)

        # Step to make sure keys are strings not bytes
        all_dicts = [self.info, self.sb, self.header]
        all_dicts = [single_dict for single_dict in all_dicts if hasattr(single_dict, 'keys')]
        for selfDict in all_dicts:
            for key in list(selfDict.keys()):
                if type(key) != str:
                    selfDict[key.decode("utf-8")] = selfDict.pop(key)

        # Ugh, different versions of the save files could have dicts or np.array.
        # Let's hope someone fits some Fourier components to the sky brightnesses and gets rid
        # of the giant save files for good.
        if hasattr(self.sb, 'keys'):
            self.filter_names = list(self.sb.keys())
        else:
            self.filter_names = self.sb.dtype.names

        if self.verbose:
            print('%s loaded' % os.path.split(filename)[1])

        self.nside = hp.npix2nside(self.sb[self.filter_names[0]][0, :].size)

        if self.loaded_range is None:
            self.loaded_range = np.array([self.info['mjds'].min(), self.info['mjds'].max()])

    def returnSunMoon(self, mjd):
        """
        Parameters
        ----------
        mjd : float
           Modified Julian Date(s) to interpolate to

        Returns
        -------
        sunMoon : dict
            Dict with keys for the sun and moon RA and Dec and the
            mooon-sun separation. All values in radians, except for moonSunSep
            that is in degrees for some reason (that reason is probably because I'm sloppy).
        """

        #warnings.warn('Method returnSunMoon to be depreciated. Interpolating angles is bad!')

        keys = ['sunAlts', 'moonAlts', 'moonRAs', 'moonDecs', 'sunRAs',
                'sunDecs', 'moonSunSep']

        degrees = [False, False, False, False, False, False, True]

        if (mjd < self.loaded_range.min() or (mjd > self.loaded_range.max())):
            self._load_data(mjd)

        result = {}
        for key, degree in zip(keys, degrees):
            if key[-1] == 's':
                newkey = key[:-1]
            else:
                newkey = key
            if 'RA' in key:
                result[newkey] = interp_angle(mjd, self.info['mjds'], self.info[key], degrees=degree)
                # Return a scalar if only doing 1 date.
                if np.size(result[newkey]) == 1:
                    result[newkey] = np.max(result[newkey])
            else:
                result[newkey] = np.interp(mjd, self.info['mjds'], self.info[key])
        return result

    def returnAirmass(self, mjd, maxAM=10., indx=None, badval=hp.UNSEEN):
        """

        Parameters
        ----------
        mjd : float
            Modified Julian Date to interpolate to
        indx : List of int(s) (None)
            indices to interpolate the sky values at. Returns full sky if None. The indx is the healpix ID.
        maxAM : float (10)
            The maximum airmass to return, everything above this airmass will be set to badval

        Returns
        -------
        airmass : np.array
            Array of airmass values. If the MJD is between sunrise and sunset, all values are masked.
        """
        if (mjd < self.loaded_range.min() or (mjd > self.loaded_range.max())):
            self._load_data(mjd)

        left = np.searchsorted(self.info['mjds'], mjd)-1
        right = left+1

        # If we are out of bounds
        if right >= self.info['mjds'].size:
            right -= 1
            baseline = 1.
        elif left < 0:
            left += 1
            baseline = 1.
        else:
            baseline = self.info['mjds'][right] - self.info['mjds'][left]

        if indx is None:
            result_size = self.sb[self.filter_names[0]][left, :].size
            indx = np.arange(result_size)
        else:
            result_size = len(indx)
        # Check if we are between sunrise/set
        if baseline > self.header['timestep_max']:
            warnings.warn('Requested MJD between sunrise and sunset, returning closest maps')
            diff = np.abs(self.info['mjds'][left.max():right.max()+1]-mjd)
            closest_indx = np.array([left, right])[np.where(diff == np.min(diff))]
            airmass = self.info['airmass'][closest_indx, indx]
            mask = np.where((self.info['airmass'][closest_indx, indx].ravel() < 1.) |
                            (self.info['airmass'][closest_indx, indx].ravel() > maxAM))
            airmass = airmass.ravel()

        else:
            wterm = (mjd - self.info['mjds'][left])/baseline
            w1 = (1. - wterm)
            w2 = wterm
            airmass = self.info['airmass'][left, indx] * w1 + self.info['airmass'][right, indx] * w2
            mask = np.where((self.info['airmass'][left, indx] < 1.) |
                            (self.info['airmass'][left, indx] > maxAM) |
                            (self.info['airmass'][right, indx] < 1.) |
                            (self.info['airmass'][right, indx] > maxAM))
        airmass[mask] = badval

        return airmass

    def returnMags(self, mjd, indx=None, airmass_mask=True, planet_mask=True,
                   moon_mask=True, zenith_mask=True, badval=hp.UNSEEN,
                   filters=['u', 'g', 'r', 'i', 'z', 'y'], extrapolate=False):
        """
        Return a full sky map or individual pixels for the input mjd

        Parameters
        ----------
        mjd : float
            Modified Julian Date to interpolate to
        indx : List of int(s) (None)
            indices to interpolate the sky values at. Returns full sky if None. If the class was
            instatiated with opsimFields, indx is the field ID, otherwise it is the healpix ID.
        airmass_mask : bool (True)
            Set high (>2.5) airmass pixels to badval.
        planet_mask : bool (True)
            Set sky maps to badval near (2 degrees) bright planets.
        moon_mask : bool (True)
            Set sky maps near (10 degrees) the moon to badval.
        zenith_mask : bool (True)
            Set sky maps at high altitude (>86.5) to badval.
        badval : float (-1.6375e30)
            Mask value. Defaults to the healpy mask value.
        filters : list
            List of strings for the filters that should be returned.
        extrapolate : bool (False)
            In indx is set, extrapolate any masked pixels to be the same as the nearest non-masked
            value from the full sky map.

        Returns
        -------
        sbs : dict
            A dictionary with filter names as keys and np.arrays as values which
            hold the sky brightness maps in mag/sq arcsec.
        """
        if (mjd < self.loaded_range.min() or (mjd > self.loaded_range.max())):
            self._load_data(mjd)

        mask_rules = {'airmass': airmass_mask, 'planet': planet_mask,
                      'moon': moon_mask, 'zenith': zenith_mask}

        left = np.searchsorted(self.info['mjds'], mjd)-1
        right = left+1

        # Do full sky by default
        if indx is None:
            indx = np.arange(self.sb['r'].shape[1])
            full_sky = True
        else:
            full_sky = False

        # If we are out of bounds
        if right >= self.info['mjds'].size:
            right -= 1
            baseline = 1.
        elif left < 0:
            left += 1
            baseline = 1.
        else:
            baseline = self.info['mjds'][right] - self.info['mjds'][left]

        # Check if we are between sunrise/set
        if baseline > self.header['timestep_max']:
            warnings.warn('Requested MJD between sunrise and sunset, returning closest maps')
            diff = np.abs(self.info['mjds'][left.max():right.max()+1]-mjd)
            closest_indx = np.array([left, right])[np.where(diff == np.min(diff))].min()
            sbs = {}
            for filter_name in filters:
                sbs[filter_name] = self.sb[filter_name][closest_indx, indx]
                for mask_name in mask_rules:
                    if mask_rules[mask_name]:
                        toMask = np.where(self.info[mask_name+'_masks'][closest_indx, indx])
                        sbs[filter_name][toMask] = badval
                sbs[filter_name][np.isinf(sbs[filter_name])] = badval
                sbs[filter_name][np.where(sbs[filter_name] == hp.UNSEEN)] = badval
        else:
            wterm = (mjd - self.info['mjds'][left])/baseline
            w1 = (1. - wterm)
            w2 = wterm
            sbs = {}
            for filter_name in filters:
                sbs[filter_name] = self.sb[filter_name][left, indx] * w1 + \
                    self.sb[filter_name][right, indx] * w2
                for mask_name in mask_rules:
                    if mask_rules[mask_name]:
                        toMask = np.where(self.info[mask_name+'_masks'][left, indx] |
                                          self.info[mask_name+'_masks'][right, indx] |
                                          np.isinf(sbs[filter_name]))
                        sbs[filter_name][toMask] = badval
                sbs[filter_name][np.where(sbs[filter_name] == hp.UNSEEN)] = badval
                sbs[filter_name][np.where(sbs[filter_name] == hp.UNSEEN)] = badval

        # If requested a certain pixel(s), and want to extrapolate.
        if (not full_sky) & extrapolate:
            masked_pix = False
            for filter_name in filters:
                if (badval in sbs[filter_name]) | (True in np.isnan(sbs[filter_name])):
                    masked_pix = True
            if masked_pix:
                # We have pixels that are masked that we want reasonable values for
                full_sky_sb = self.returnMags(mjd, airmass_mask=False, planet_mask=False, moon_mask=False,
                                              zenith_mask=False, filters=filters)
                good = np.where((full_sky_sb[filters[0]] != badval) & ~np.isnan(full_sky_sb[filters[0]]))[0]
                ra_full = np.radians(self.header['ra'][good])
                dec_full = np.radians(self.header['dec'][good])
                for filtername in filters:
                    full_sky_sb[filtername] = full_sky_sb[filtername][good]
                # Going to assume the masked pixels are the same in all filters
                masked_indx = np.where((sbs[filters[0]].ravel() == badval) |
                                       np.isnan(sbs[filters[0]].ravel()))[0]
                for i, mi in enumerate(masked_indx):
                    # Note, this is going to be really slow for many pixels, should use a kdtree
                    dist = _angularSeparation(np.radians(self.header['ra'][indx[i]]),
                                              np.radians(self.header['dec'][indx[i]]),
                                              ra_full, dec_full)
                    closest = np.where(dist == dist.min())[0]
                    for filtername in filters:
                        sbs[filtername].ravel()[mi] = np.min(full_sky_sb[filtername][closest])

        return sbs
