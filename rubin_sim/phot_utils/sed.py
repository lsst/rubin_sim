#
# LSST Data Management System
# Copyright 2008, 2009, 2010, 2011, 2012 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#

"""
Sed -

Class data:
wavelen (nm)
flambda (ergs/cm^2/s/nm)
fnu (Jansky)
zp  (translates to units of fnu = -8.9 (if Janskys) or 48.6 (ergs/cm^2/s/hz))
the name of the sed file

It is important to note the units are NANOMETERS, not ANGSTROMS.

Methods:
Because of how these methods will be applied for catalog generation,
(taking one base SED and then applying various dust extinctions and redshifts),
many of the methods will either work on, and update self, OR they can be given
a set of lambda/flambda arrays and then will return new versions of
these arrays. In general, the methods will not explicitly set flambda or fnu to
something you (the user) did not specify. So, for example, when calculating
magnitudes (which depend on a wavelength/fnu gridded to match the given
bandpass) the wavelength and fnu used are temporary copies and the object
itself is not changed.

In general, the philosophy of Sed.py is to not define the wavelength
grid for the object until necessary (so, not until needed for the
magnitude calculation or resample_sed is called). At that time the min/max/step
wavelengths or the bandpass wavelengths are used to define a new wavelength
grid for the sed object.

When considering whether to use the internal wavelen/flambda (self) values,
versus input values:
For consistency, anytime self.wavelen/flambda is used, it will be updated
if the values are changed (except in the special case of calculating
magnitudes), and if self.wavelen/flambda is updated, self.fnu will be set to
None. This is because many operations are typically chained together
which alter flambda -- so it is more efficient to wait and recalculate fnu at
the end, plus it avoids possible de-synchronization errors
(flambda reflecting the addition of dust while fnu does
not, for example). If arrays are passed into a method, they will not be
altered and the arrays which are returned will be allocated new memory.

Another general philosophy for Sed.py is use separate methods for items which
only need to be generated once for several objects (such as the dust A_x, b_x
arrays). This allows the user to optimize their code for faster operation,
depending on what their requirements are (see example_SedBandpass_star.py and
exampleSedBandpass_galaxy for examples).
"""

__all__ = ("Sed", "cache_lsst_seds", "read_close__kurucz")

import gzip
import os
import pickle
import sys
import time
import warnings

import numpy

try:
    from numpy import trapezoid as trapezoid
except ImportError:
    from numpy import trapz as trapezoid
from rubin_scheduler.data import get_data_dir

from .physical_parameters import PhysicalParameters
from .spectral_resampling import spectres

_global_lsst_sed_cache = None

# a cache for ASCII files read-in by the user
_global_misc_sed_cache = None


class SedCacheError(Exception):
    pass


class SedUnpickler(pickle.Unpickler):
    _allowed_obj = (
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
    )

    def find_class(self, module, name):
        allowed = False
        for _module, _name in self._allowed_obj:
            if module == _module and name == _name:
                allowed = True
                break

        if not allowed:
            raise RuntimeError(
                "Cannot call find_class() on %s, %s with SedUnpickler " % (module, name)
                + "this is for security reasons\n"
                + "https://docs.python.org/3.1/library/pickle.html#pickle-restrict"
            )

        if module == "numpy":
            if name == "ndarray":
                return getattr(numpy, name)
            elif name == "dtype":
                return getattr(numpy, name)
            else:
                raise RuntimeError("SedUnpickler not meant to load numpy.%s" % name)
        elif module == "numpy.core.multiarray":
            return getattr(numpy.core.multiarray, name)
        else:
            raise RuntimeError("SedUnpickler cannot handle module %s" % module)


def _validate_sed_cache():
    """
    Verifies that the pickled SED cache exists, is a dict, and contains
    an entry for every SED in starSED/ and galaxySED.  Does nothing if so,
    raises a RuntimeError if false.

    We are doing this here so that sims_sed_library does not have to depend
    on any lsst testing software (in which case, users would have to get
    a new copy of sims_sed_library every time the upstream software changed).

    We are doing this through a method (rather than giving users access to
    _global_lsst_sed_cache) so that users do not accidentally ruin
    _global_lsst_sed_cache.
    """
    global _global_lsst_sed_cache
    if _global_lsst_sed_cache is None:
        raise SedCacheError("_global_lsst_sed_cache does not exist")
    if not isinstance(_global_lsst_sed_cache, dict):
        raise SedCacheError("_global_lsst_sed_cache is a %s; not a dict" % str(type(_global_lsst_sed_cache)))
    sed_dir = os.path.join(get_data_dir(), "sims_sed_library")
    sub_dir_list = ["galaxySED", "starSED"]
    file_ct = 0
    for sub_dir in sub_dir_list:
        tree = os.walk(os.path.join(sed_dir, sub_dir))
        for entry in tree:
            local_dir = entry[0]
            file_list = entry[2]
            for file_name in file_list:
                if file_name.endswith(".gz"):
                    full_name = os.path.join(sed_dir, sub_dir, local_dir, file_name)
                    if full_name not in _global_lsst_sed_cache:
                        raise SedCacheError("%s is not in _global_lsst_sed_cache" % full_name)
                    file_ct += 1
    if file_ct == 0:
        raise SedCacheError("There were not files in _global_lsst_sed_cache")

    return


def _compare_cached_versus_uncached():
    """
    Verify that loading an SED from the pickled cache give identical
    results to loading the same SED from ASCII
    """
    sed_dir = os.path.join(get_data_dir(), "sims_sed_library", "starSED", "kurucz")

    dtype = numpy.dtype([("wavelen", float), ("flambda", float)])

    sed_name_list = os.listdir(sed_dir)
    msg = (
        "An SED loaded from the pickled cache is not "
        "identical to the same SED loaded from ASCII; "
        "it is possible that the pickled cache was incorrectly "
        "created in sims_sed_library\n\n"
        "Try removing the cache file (the name should hav been printed "
        "to stdout above) and re-running sims_phot_utils.cache_LSST_seds()"
    )
    for ix in range(5):
        full_name = os.path.join(sed_dir, sed_name_list[ix])
        from_np = numpy.genfromtxt(full_name, dtype=dtype)
        ss_cache = Sed()
        ss_cache.read_sed_flambda(full_name)
        ss_uncache = Sed(wavelen=from_np["wavelen"], flambda=from_np["flambda"], name=full_name)

        if not ss_cache == ss_uncache:
            raise SedCacheError(msg)


def _generate_sed_cache(cache_dir, cache_name):
    """
    Read all of the SEDs from sims_sed_library into a dict.
    Pickle the dict and store it in
    sims_phot_utils/cacheDir/lsst_sed_cache.p

    Parameters
    ----------
    cache_dir : `str`
        The directory where the cache will be created.
    cache_name : `str`
        The name of the cache to be created.

    Returns
    -------
    cache : `dict` [`str`, `Sed`]
        The dict of SEDs (keyed to their full file name).
    """
    sed_root = os.path.join(get_data_dir(), "sims_sed_library")
    dtype = numpy.dtype([("wavelen", float), ("flambda", float)])

    sub_dir_list = ["agnSED", "flatSED", "ssmSED", "starSED", "galaxySED"]

    cache = {}

    total_files = 0
    for sub_dir in sub_dir_list:
        dir_tree = os.walk(os.path.join(sed_root, sub_dir))
        for sub_tree in dir_tree:
            total_files += len([name for name in sub_tree[2] if name.endswith(".gz")])

    t_start = time.time()
    print("This could take about 15 minutes.")
    print("Note: not all SED files are the same size. ")
    print("Do not expect the loading rate to be uniform.\n")

    for sub_dir in sub_dir_list:
        dir_tree = os.walk(os.path.join(sed_root, sub_dir))
        for sub_tree in dir_tree:
            dir_name = sub_tree[0]
            file_list = sub_tree[2]

            for file_name in file_list:
                if file_name.endswith(".gz"):
                    try:
                        full_name = os.path.join(dir_name, file_name)
                        data = numpy.genfromtxt(full_name, dtype=dtype)
                        cache[full_name] = (data["wavelen"], data["flambda"])
                        if len(cache) % (total_files // 20) == 0:
                            if len(cache) > total_files // 20:
                                sys.stdout.write("\r")
                            sys.stdout.write(
                                "loaded %d of %d files in about %.2f seconds"
                                % (len(cache), total_files, time.time() - t_start)
                            )
                            sys.stdout.flush()
                    except Exception:
                        pass

    print("\n")

    with open(os.path.join(cache_dir, cache_name), "wb") as file_handle:
        pickle.dump(cache, file_handle)

    print("LSST SED cache saved to:\n")
    print("%s" % os.path.join(cache_dir, cache_name))

    # record the specific sims_sed_library directory being cached so that
    # a new cache will be generated if sims_sed_library gets updated
    with open(os.path.join(cache_dir, "cache_version_%d.txt" % sys.version_info.major), "w") as file_handle:
        file_handle.write("%s %s" % (sed_root, cache_name))

    return cache


def cache_lsst_seds(wavelen_min=None, wavelen_max=None, cache_dir=None):
    """
    Read all of the SEDs in sims_sed_library into a dict.  Pickle the dict
    and store it in phot_utils/cacheDir/lsst_sed_cache.p for future use.

    After the file has initially been created,
    the next time you run this script, it will just use the pickle.

    Once the dict is loaded, Sed.read_sed_flambda() will be able to read any
    LSST-shipped SED directly from memory, rather than using I/O to read it
    from an ASCII file stored on disk.

    Note: the dict of cached SEDs will take up about 5GB on disk.  Once loaded,
    the cache will take up about 1.5GB of memory.

    Parameters
    -----------
    wavelen_min : `float`
        Wavelength minimum value to store for each Sed.
    wavelen_max : `float`
        Wavelength maximum value to store for each Sed.
    cache_dir : `str`
        The directory to place the cache pickle.

    If either of wavelen_min or wavelen_max are not None,
    then every SED in the cache will be
    truncated to only include the wavelength range (in nm) between
    wavelen_min and wavelen_max
    """

    global _global_lsst_sed_cache

    sed_cache_name = os.path.join("lsst_sed_cache_%d.p" % sys.version_info.major)
    sed_dir = os.path.join(get_data_dir(), "sims_sed_library")
    if cache_dir is None:
        cache_dir = os.path.join(get_data_dir(), "sims_sed_library", "lsst_sed_cache_dir")

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    must_generate = False
    if not os.path.exists(os.path.join(cache_dir, sed_cache_name)):
        must_generate = True
    if not os.path.exists(os.path.join(cache_dir, "cache_version_%d.txt" % sys.version_info.major)):
        must_generate = True
    else:
        with open(
            os.path.join(cache_dir, "cache_version_%d.txt" % sys.version_info.major),
            "r",
        ) as input_file:
            lines = input_file.readlines()
            if len(lines) != 1:
                must_generate = True
            else:
                info = lines[0].split()
                if len(info) != 2:
                    must_generate = True
                elif info[0] != sed_dir:
                    must_generate = True
                elif info[1] != sed_cache_name:
                    must_generate = True

    if must_generate:
        print("\nCreating cache of LSST SEDs in:\n%s" % os.path.join(cache_dir, sed_cache_name))
        cache = _generate_sed_cache(cache_dir, sed_cache_name)
        _global_lsst_sed_cache = cache
    else:
        print("\nOpening cache of LSST SEDs in:\n%s" % os.path.join(cache_dir, sed_cache_name))
        with open(os.path.join(cache_dir, sed_cache_name), "rb") as input_file:
            _global_lsst_sed_cache = SedUnpickler(input_file).load()

    # Now that we have generated/loaded the cache, we must run tests
    # to make sure that the cache is correctly constructed.  If these
    # fail, _global_lsst_sed_cache will be set to 'None' and the code will
    # continue running.
    try:
        _validate_sed_cache()
        _compare_cached_versus_uncached()
    except SedCacheError as ee:
        print(ee.message)
        print("Cannot use cache of LSST SEDs")
        _global_lsst_sed_cache = None
        pass

    if wavelen_min is not None or wavelen_max is not None:
        if wavelen_min is None:
            wavelen_min = 0.0
        if wavelen_max is None:
            wavelen_max = numpy.inf

        new_cache = {}
        list_of_sed_names = list(_global_lsst_sed_cache.keys())
        for file_name in list_of_sed_names:
            wav, fl = _global_lsst_sed_cache.pop(file_name)
            valid_dexes = numpy.where(numpy.logical_and(wav >= wavelen_min, wav <= wavelen_max))
            new_cache[file_name] = (wav[valid_dexes], fl[valid_dexes])

        _global_lsst_sed_cache = new_cache

    return


class Sed:
    """ "Hold and use spectral energy distributions (SEDs)"""

    def __init__(self, wavelen=None, flambda=None, fnu=None, badval=numpy.nan, name=None):
        """
        Initialize sed object by giving filename or lambda/flambda array.

        Note that this does *not* regrid flambda and leaves fnu undefined.
        """
        self.fnu = None
        self.wavelen = None
        self.flambda = None
        # self.zp = -8.9  # default units, Jansky.
        self.zp = -2.5 * numpy.log10(3631)
        self.name = name
        self.badval = badval

        self._phys_params = PhysicalParameters()

        # If init was given data to initialize class, use it.
        if (wavelen is not None) and ((flambda is not None) or (fnu is not None)):
            if name is None:
                name = "FromArray"
            self.set_sed(wavelen, flambda=flambda, fnu=fnu, name=name)
        return

    def __eq__(self, other):
        if self.name != other.name:
            return False
        if self.zp != other.zp:
            return False
        if not numpy.isnan(self.badval):
            if self.badval != other.badval:
                return False
        else:
            if not numpy.isnan(other.badval):
                return False
        if self.fnu is not None and other.fnu is None:
            return False
        if self.fnu is None and other.fnu is not None:
            return False
        if self.fnu is not None:
            try:
                numpy.testing.assert_array_equal(self.fnu, other.fnu)
            except AssertionError:
                return False

        if self.flambda is None and other.flambda is not None:
            return False
        if other.flambda is not None and self.flambda is None:
            return False
        if self.flambda is not None:
            try:
                numpy.testing.assert_array_equal(self.flambda, other.flambda)
            except AssertionError:
                return False

        if self.wavelen is None and other.wavelen is not None:
            return False
        if self.wavelen is not None and other.wavelen is None:
            return False
        if self.wavelen is not None:
            try:
                numpy.testing.assert_array_equal(self.wavelen, other.wavelen)
            except AssertionError:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # Methods for getters and setters.

    def set_sed(self, wavelen, flambda=None, fnu=None, name="FromArray"):
        """
        Populate wavelen/flambda fields in sed by giving lambda/flambda
        or lambda/fnu array.

        If flambda present, this overrides fnu.
        Method sets fnu=None unless only fnu is given.
        """
        # Check wavelen array for type matches.
        if isinstance(wavelen, numpy.ndarray) is False:
            raise ValueError("Wavelength must be a numpy array")
        # Wavelen type ok - make new copy of data for self.
        self.wavelen = numpy.copy(wavelen)
        self.flambda = None
        self.fnu = None
        # Check if given flambda or fnu.
        if flambda is not None:
            # Check flambda data type and length.
            if (isinstance(flambda, numpy.ndarray) is False) or (len(flambda) != len(self.wavelen)):
                raise ValueError("Flambda must be a numpy array of same length as Wavelen.")
            # Flambda ok, make a new copy of data for self.
            self.flambda = numpy.copy(flambda)
        else:
            # Were passed fnu instead : check fnu data type and length.
            if fnu is None:
                raise ValueError("Both fnu and flambda are 'None', cannot set the SED.")
            elif (isinstance(fnu, numpy.ndarray) is False) or (len(fnu) != len(self.wavelen)):
                raise ValueError("(No Flambda) - Fnu must be numpy array of same length as Wavelen.")
            # Convert fnu to flambda.
            self.wavelen, self.flambda = self.fnu_toflambda(wavelen, fnu)
            self.fnu = fnu
        self.name = name
        return

    def set_flat_sed(self, wavelen_min=300.0, wavelen_max=1150.0, wavelen_step=0.1, name="Flat"):
        """
        Populate the wavelength/flambda/fnu fields in sed according to a
        flat fnu source.
        """

        self.wavelen = numpy.arange(wavelen_min, wavelen_max + wavelen_step, wavelen_step, dtype="float")
        self.fnu = numpy.ones(len(self.wavelen), dtype="float") * 3631  # jansky
        self.fnu_toflambda()
        self.name = name
        return

    def read_sed_flambda(self, filename, name=None, cache_sed=True):
        """
        Read a file containing [lambda Flambda]
        (lambda in nm) (Flambda erg/cm^2/s/nm).

        Does not resample wavelen/flambda onto grid; leave fnu=None.
        """
        global _global_lsst_sed_cache
        global _global_misc_sed_cache

        # Try to open data file.
        # ASSUME that if filename ends with '.gz' that the file is gzipped.
        # Otherwise, regular file.
        if filename.endswith(".gz"):
            gzipped_filename = filename
            unzipped_filename = filename[:-3]
        else:
            gzipped_filename = filename + ".gz"
            unzipped_filename = filename

        cached_source = None
        if _global_lsst_sed_cache is not None:
            if gzipped_filename in _global_lsst_sed_cache:
                cached_source = _global_lsst_sed_cache[gzipped_filename]
            elif unzipped_filename in _global_lsst_sed_cache:
                cached_source = _global_lsst_sed_cache[unzipped_filename]

        if cached_source is None and _global_misc_sed_cache is not None:
            if gzipped_filename in _global_misc_sed_cache:
                cached_source = _global_misc_sed_cache[gzipped_filename]
            if unzipped_filename in _global_misc_sed_cache:
                cached_source = _global_misc_sed_cache[unzipped_filename]

        if cached_source is not None:
            sourcewavelen = numpy.copy(cached_source[0])
            sourceflambda = numpy.copy(cached_source[1])

        if cached_source is None:
            # Read source SED from file - lambda, flambda should be first
            # two columns in the file.
            # lambda should be in nm and flambda should be in ergs/cm2/s/nm
            dtype = numpy.dtype([("wavelen", float), ("flambda", float)])
            try:
                data = numpy.genfromtxt(gzipped_filename, dtype=dtype)
            except IOError:
                try:
                    data = numpy.genfromtxt(unzipped_filename, dtype=dtype)
                except Exception as err:
                    new_args = [
                        err.args[0] + "\n\nError reading sed file %s; " % filename + "it may not exist."
                    ]
                    for aa in err.args[1:]:
                        new_args.append(aa)
                    err.args = tuple(new_args)
                    raise

            sourcewavelen = data["wavelen"]
            sourceflambda = data["flambda"]

            if cache_sed:
                if _global_misc_sed_cache is None:
                    _global_misc_sed_cache = {}
                _global_misc_sed_cache[filename] = (
                    numpy.copy(sourcewavelen),
                    numpy.copy(sourceflambda),
                )

        self.wavelen = sourcewavelen
        self.flambda = sourceflambda
        self.fnu = None
        if name is None:
            self.name = filename
        else:
            self.name = name
        return

    def read_sed_fnu(self, filename, name=None):
        """
        Read a file containing [lambda Fnu]
        (lambda in nm) (Fnu in Jansky).

        Does not resample wavelen/fnu/flambda onto a grid; leaves fnu set.
        """
        # Try to open the data file.
        try:
            if filename.endswith(".gz"):
                f = gzip.open(filename, "rt")
            else:
                f = open(filename, "r")
        # if the above fails, look for the file with and without the gz
        except IOError:
            try:
                if filename.endswith(".gz"):
                    f = open(filename[:-3], "r")
                else:
                    f = gzip.open(filename + ".gz", "rt")
            except IOError:
                raise IOError("The throughput file %s does not exist" % (filename))
        # Read source SED from file - lambda, fnu should be
        # the first two columns in the file.
        # lambda should be in nm and fnu should be in Jansky.
        sourcewavelen = []
        sourcefnu = []
        for line in f:
            if line.startswith("#"):
                continue
            values = line.split()
            sourcewavelen.append(float(values[0]))
            sourcefnu.append(float(values[1]))
        f.close()
        # Convert to numpy arrays and set members
        self.wavelen = numpy.array(sourcewavelen)
        self.fnu = numpy.array(sourcefnu)
        # convert fnu to flambda
        self.fnu_toflambda()
        if name is None:
            self.name = filename
        else:
            self.name = name
        return

    def get_sed_flambda(self):
        """
        Return copy of wavelen/flambda.
        """
        # Get new memory copies of the arrays.
        wavelen = numpy.copy(self.wavelen)
        flambda = numpy.copy(self.flambda)
        return wavelen, flambda

    def get_sed_fnu(self):
        """
        Return copy of wavelen/fnu, without altering self.
        """
        wavelen = numpy.copy(self.wavelen)
        # Check if fnu currently set.
        if self.fnu is not None:
            # Get new memory copy of fnu.
            fnu = numpy.copy(self.fnu)
        else:
            # Fnu was not set .. grab copy fnu without changing self.
            wavelen, fnu = self.flambda_tofnu(self.wavelen, self.flambda)
            # Now wavelen/fnu (new mem) are gridded evenly,
            # but self.wavelen/flambda/fnu remain unchanged.
        return wavelen, fnu

    # Methods that update or change self.

    def clear_sed(self):
        """
        Reset all data in sed to None.
        """
        self.wavelen = None
        self.fnu = None
        self.flambda = None
        self.zp = -8.9
        self.name = None
        return

    def synchronize_sed(self, wavelen_min=None, wavelen_max=None, wavelen_step=None):
        """
        Set all wavelen/flambda/fnu values, potentially on min/max/step grid.

        Uses flambda to recalculate fnu.
        If wavelen min/max/step are given, resamples
        wavelength/flambda/fnu onto an even grid with these values.
        """
        # Grid wavelength/flambda/fnu if desired.
        if (wavelen_min is not None) and (wavelen_max is not None) and (wavelen_step is not None):
            self.resample_sed(
                wavelen_min=wavelen_min,
                wavelen_max=wavelen_max,
                wavelen_step=wavelen_step,
            )
        # Reset or set fnu.
        self.flambda_tofnu()
        return

    # Utilities common to several later methods.

    def _check_use_self(self, wavelen, flux):
        """
        Simple utility to check if should be using self's data or
        passed arrays.

        Also does data integrity check on wavelen/flux if not self.
        """
        update_self = False
        if (wavelen is None) or (flux is None):
            # Then one of the arrays was not passed -
            # check if this is true for both arrays.
            if (wavelen is not None) or (flux is not None):
                # Then one of the arrays was passed - raise exception.
                raise ValueError("Must either pass *both* wavelen/flux pair, or use defaults.")
            update_self = True
        else:
            # Both of the arrays were passed in - check their validity.
            if (isinstance(wavelen, numpy.ndarray) is False) or (isinstance(flux, numpy.ndarray) is False):
                raise ValueError("Must pass wavelen/flux as numpy arrays.")
            if len(wavelen) != len(flux):
                raise ValueError("Must pass equal length wavelen/flux arrays.")
        return update_self

    def _need_resample(
        self,
        wavelen_match=None,
        wavelen=None,
        wavelen_min=None,
        wavelen_max=None,
        wavelen_step=None,
    ):
        """
        Check if wavelen or self.wavelen matches wavelen
        or wavelen_min/max/step grid.
        """
        # Check if should use self or passed wavelen.
        if wavelen is None:
            wavelen = self.wavelen
        # Check if wavelength arrays are equal, if wavelen_match passed.
        if wavelen_match is not None:
            if numpy.shape(wavelen_match) != numpy.shape(wavelen):
                need_regrid = True
            else:
                # check the elements to see if any vary
                need_regrid = numpy.any(abs(wavelen_match - wavelen) > 1e-10)
        else:
            need_regrid = True
            # Check if wavelen_min/max/step are set -
            # if ==None, then return (no regridding).
            # It's possible (writeSED) to call this routine,
            # even with no final grid in mind.
            if (wavelen_min is None) and (wavelen_max is None) and (wavelen_step is None):
                need_regrid = False
            else:
                # Okay, now look at comparison of wavelen to the grid.
                wavelen_max_in = wavelen[len(wavelen) - 1]
                wavelen_min_in = wavelen[0]
                # First check match to minimum/maximum :
                if (wavelen_min_in == wavelen_min) and (wavelen_max_in == wavelen_max):
                    # Then check on step size in wavelength array.
                    stepsize = numpy.unique(numpy.diff(wavelen))
                    if (len(stepsize) == 1) and (stepsize[0] == wavelen_step):
                        need_regrid = False
        # At this point, need_grid=True unless it's proven to be False,
        # so return value.
        return need_regrid

    def resample_sed(
        self,
        wavelen=None,
        flux=None,
        wavelen_match=None,
        wavelen_min=None,
        wavelen_max=None,
        wavelen_step=None,
        force=False,
        fill=numpy.nan,
    ):
        """
        Resample flux onto grid defined by min/max/step OR
        another wavelength array.

        Give method wavelen/flux OR default to self.wavelen/self.flambda.
        Method either returns wavelen/flambda (if given those arrays) or
        updates wavelen/flambda in self.
        If updating self, resets fnu to None.
        Method will first check if resampling needs to be done or not,
        unless 'force' is True.
        """
        # Check if need resampling:
        if force or (
            self._need_resample(
                wavelen_match=wavelen_match,
                wavelen=wavelen,
                wavelen_min=wavelen_min,
                wavelen_max=wavelen_max,
                wavelen_step=wavelen_step,
            )
        ):
            # Is method acting on self.wavelen/flambda or
            # passed in wavelen/flux arrays?
            update_self = self._check_use_self(wavelen, flux)
            if update_self:
                wavelen = self.wavelen
                flux = self.flambda
                self.fnu = None
            # Now, on with the resampling.
            # Set up gridded wavelength or copy of wavelen array to match.
            if wavelen_match is None:
                if (wavelen_min is None) and (wavelen_max is None) and (wavelen_step is None):
                    raise ValueError("Must set either wavelen_match or wavelen_min/max/step.")
                wavelen_grid = numpy.arange(
                    wavelen_min, wavelen_max + wavelen_step, wavelen_step, dtype="float"
                )
            else:
                wavelen_grid = numpy.copy(wavelen_match)
            # Check if the wavelength range desired and the wavelength
            # range of the object overlap.
            # If there is any non-overlap, raise warning.
            if (wavelen.max() < wavelen_grid.max()) or (wavelen.min() > wavelen_grid.min()):
                warnings.warn(
                    "There is an area of non-overlap between desired wavelength range "
                    + " (%.2f to %.2f)" % (wavelen_grid.min(), wavelen_grid.max())
                    + "and sed %s (%.2f to %.2f)" % (self.name, wavelen.min(), wavelen.max())
                )
            # rebin the spectra. Fill with NaNs if there's non-overlap regions.
            flux_grid = spectres(wavelen_grid, wavelen, flux, fill=fill, verbose=False)

            # Update self values if necessary.
            if update_self:
                self.wavelen = wavelen_grid
                self.flambda = flux_grid
                return
            return wavelen_grid, flux_grid
        else:  # wavelength grids already match.
            update_self = self._check_use_self(wavelen, flux)
            if update_self:
                return
            return wavelen, flux

    def flambda_tofnu(self, wavelen=None, flambda=None):
        """
        Convert flambda into fnu.

        This routine assumes that flambda is in ergs/cm^s/s/nm and
        produces fnu in Jansky.
        Can act on self or user can provide wavelen/flambda and
        get back wavelen/fnu.
        """
        # Change Flamda to Fnu by multiplying Flambda * lambda^2 = Fv
        # Fv dv = Fl dl .. Fv = Fl dl / dv = Fl dl / (dl*c/l/l) = Fl*l*l/c
        # Check - Is the method acting on self.wavelen/flambda/fnu
        # or passed wavelen/flambda arrays?
        update_self = self._check_use_self(wavelen, flambda)
        if update_self:
            wavelen = self.wavelen
            flambda = self.flambda
            self.fnu = None
        # Now on with the calculation.
        # Calculate fnu.
        fnu = flambda * wavelen * wavelen * self._phys_params.nm2m / self._phys_params.lightspeed
        fnu = fnu * self._phys_params.ergsetc2jansky
        # If are using/updating self, then *all* wavelen/flambda/fnu
        # will be gridded.
        # This is so wavelen/fnu AND wavelen/flambda can be kept in sync.
        if update_self:
            self.wavelen = wavelen
            self.flambda = flambda
            self.fnu = fnu
            return
        # Return wavelen, fnu, unless updating self (then does not return).
        return wavelen, fnu

    def fnu_toflambda(self, wavelen=None, fnu=None):
        """
        Convert fnu into flambda.

        Assumes fnu in units of Jansky and flambda in ergs/cm^s/s/nm.
        Can act on self or user can give wavelen/fnu and
        get wavelen/flambda returned.
        """
        # Fv dv = Fl dl .. Fv = Fl dl / dv = Fl dl / (dl*c/l/l) = Fl*l*l/c
        # Is method acting on self or passed arrays?
        update_self = self._check_use_self(wavelen, fnu)
        if update_self:
            wavelen = self.wavelen
            fnu = self.fnu
        # On with the calculation.
        # Calculate flambda.
        flambda = fnu / wavelen / wavelen * self._phys_params.lightspeed / self._phys_params.nm2m
        flambda = flambda / self._phys_params.ergsetc2jansky
        # If updating self, then *all of wavelen/fnu/flambda will be updated.
        # This is so wavelen/fnu AND wavelen/flambda can be kept in sync.
        if update_self:
            self.wavelen = wavelen
            self.flambda = flambda
            self.fnu = fnu
            return
        # Return wavelen/flambda.
        return wavelen, flambda

    # methods to alter the sed

    def redshift_sed(self, redshift, dimming=False, wavelen=None, flambda=None):
        """
        Redshift an SED, optionally adding cosmological dimming.

        Pass wavelen/flambda or redshift/update self.wavelen/flambda
        (unsets fnu).
        """
        # Updating self or passed arrays?
        update_self = self._check_use_self(wavelen, flambda)
        if update_self:
            wavelen = self.wavelen
            flambda = self.flambda
            self.fnu = None
        else:
            # Make a copy of input data, because will change its values.
            wavelen = numpy.copy(wavelen)
            flambda = numpy.copy(flambda)
        # Okay, move onto redshifting the wavelen/flambda pair.
        # Or blueshift, as the case may be.
        if redshift < 0:
            wavelen = wavelen / (1.0 - redshift)
        else:
            wavelen = wavelen * (1.0 + redshift)
        # Flambda now just has different wavelength for each value.
        # Add cosmological dimming if required.
        if dimming:
            if redshift < 0:
                flambda = flambda * (1.0 - redshift)
            else:
                flambda = flambda / (1.0 + redshift)
        # Update self, if required - but just flambda (still no grid required).
        if update_self:
            self.wavelen = wavelen
            self.flambda = flambda
            return
        return wavelen, flambda

    def setup_cc_mab(self, wavelen=None):
        """
        Calculate a(x) and b(x) for CCM dust model. (x=1/wavelen).

        If wavelen not specified, calculates a and b on the own object's
        wavelength grid.
        Returns a(x) and b(x) can be common to many seds, wavelen is the same.

        This method sets up extinction due to the model of
        Cardelli, Clayton and Mathis 1989 (ApJ 345, 245)
        """
        warnings.warn(
            "Sed.setup_cc_mab is now deprecated in favor of Sed.setup_ccm_ab",
            DeprecationWarning,
        )

        return self.setup_ccm_ab(wavelen=wavelen)

    def setup_ccm_ab(self, wavelen=None):
        """
        Calculate a(x) and b(x) for CCM dust model. (x=1/wavelen).

        If wavelen not specified, calculates a and b on the own object's
        wavelength grid.
        Returns a(x) and b(x) can be common to many seds, wavelen is the same.

        This method sets up extinction due to the model of
        Cardelli, Clayton and Mathis 1989 (ApJ 345, 245)
        """
        # This extinction law taken from Cardelli, Clayton and Mathis ApJ 1989.
        # The general form is A_l / A(V) = a(x) + b(x)/R_V
        # (where x=1/lambda in microns),
        # then different values for a(x) and b(x) depending on wavelength.
        # Also, the extinction is parametrized as R_v = a_v / E(B-V).
        # Magnitudes of extinction (A_l) translates to flux by
        # a_l = -2.5log(f_red / f_nonred).
        if wavelen is None:
            wavelen = numpy.copy(self.wavelen)
        a_x = numpy.zeros(len(wavelen), dtype="float")
        b_x = numpy.zeros(len(wavelen), dtype="float")
        # Convert wavelength to x (in inverse microns).
        x = numpy.empty(len(wavelen), dtype=float)
        nm_to_micron = 1 / 1000.0
        x = 1.0 / (wavelen * nm_to_micron)
        # Dust in infrared 0.3 /mu < x < 1.1 /mu (inverse microns).
        condition = (x >= 0.3) & (x <= 1.1)
        if len(a_x[condition]) > 0:
            y = x[condition]
            a_x[condition] = 0.574 * y**1.61
            b_x[condition] = -0.527 * y**1.61
        # Dust in optical/NIR 1.1 /mu < x < 3.3 /mu region.
        condition = (x >= 1.1) & (x <= 3.3)
        if len(a_x[condition]) > 0:
            y = x[condition] - 1.82
            a_x[condition] = 1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 + 0.72085 * y**4
            a_x[condition] = a_x[condition] + 0.01979 * y**5 - 0.77530 * y**6 + 0.32999 * y**7
            b_x[condition] = 1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 - 5.38434 * y**4
            b_x[condition] = b_x[condition] - 0.62251 * y**5 + 5.30260 * y**6 - 2.09002 * y**7
        # Dust in ultraviolet and UV (if needed for high-z) 3.3 /mu< x< 8 /mu.
        condition = (x >= 3.3) & (x < 5.9)
        if len(a_x[condition]) > 0:
            y = x[condition]
            a_x[condition] = 1.752 - 0.316 * y - 0.104 / ((y - 4.67) ** 2 + 0.341)
            b_x[condition] = -3.090 + 1.825 * y + 1.206 / ((y - 4.62) ** 2 + 0.263)
        condition = (x > 5.9) & (x < 8)
        if len(a_x[condition]) > 0:
            y = x[condition]
            fa_x = numpy.empty(len(a_x[condition]), dtype=float)
            fb_x = numpy.empty(len(a_x[condition]), dtype=float)
            fa_x = -0.04473 * (y - 5.9) ** 2 - 0.009779 * (y - 5.9) ** 3
            fb_x = 0.2130 * (y - 5.9) ** 2 + 0.1207 * (y - 5.9) ** 3
            a_x[condition] = 1.752 - 0.316 * y - 0.104 / ((y - 4.67) ** 2 + 0.341) + fa_x
            b_x[condition] = -3.090 + 1.825 * y + 1.206 / ((y - 4.62) ** 2 + 0.263) + fb_x
        # Dust in far UV (if needed for high-z) 8 /mu < x < 10 /mu region.
        condition = (x >= 8) & (x <= 11.0)
        if len(a_x[condition]) > 0:
            y = x[condition] - 8.0
            a_x[condition] = -1.073 - 0.628 * (y) + 0.137 * (y) ** 2 - 0.070 * (y) ** 3
            b_x[condition] = 13.670 + 4.257 * (y) - 0.420 * (y) ** 2 + 0.374 * (y) ** 3
        return a_x, b_x

    def setup_o_donnell_ab(self, wavelen=None):
        """
        Calculate a(x) and b(x) for O'Donnell dust model. (x=1/wavelen).

        If wavelen not specified, calculates a and b on the own object's
        wavelength grid.
        Returns a(x) and b(x) can be common to many seds, wavelen is the same.

        This method sets up the extinction parameters from the model of
        O'Donnel 1994 (ApJ 422, 158)
        """
        # The general form is A_l / A(V) = a(x) + b(x)/R_V
        # (where x=1/lambda in microns),
        # then different values for a(x) and b(x) depending on wavelength.
        # Also, the extinction is parametrized as R_v = a_v / E(B-V).
        # Magnitudes of extinction (A_l) translates to flux by
        # a_l = -2.5log(f_red / f_nonred).
        if wavelen is None:
            wavelen = numpy.copy(self.wavelen)
        a_x = numpy.zeros(len(wavelen), dtype="float")
        b_x = numpy.zeros(len(wavelen), dtype="float")
        # Convert wavelength to x (in inverse microns).
        x = numpy.empty(len(wavelen), dtype=float)
        nm_to_micron = 1 / 1000.0
        x = 1.0 / (wavelen * nm_to_micron)
        # Dust in infrared 0.3 /mu < x < 1.1 /mu (inverse microns).
        condition = (x >= 0.3) & (x <= 1.1)
        if len(a_x[condition]) > 0:
            y = x[condition]
            a_x[condition] = 0.574 * y**1.61
            b_x[condition] = -0.527 * y**1.61
        # Dust in optical/NIR 1.1 /mu < x < 3.3 /mu region.
        condition = (x >= 1.1) & (x <= 3.3)
        if len(a_x[condition]) > 0:
            y = x[condition] - 1.82
            a_x[condition] = 1 + 0.104 * y - 0.609 * y**2 + 0.701 * y**3 + 1.137 * y**4
            a_x[condition] = a_x[condition] - 1.718 * y**5 - 0.827 * y**6 + 1.647 * y**7 - 0.505 * y**8
            b_x[condition] = 1.952 * y + 2.908 * y**2 - 3.989 * y**3 - 7.985 * y**4
            b_x[condition] = b_x[condition] + 11.102 * y**5 + 5.491 * y**6 - 10.805 * y**7 + 3.347 * y**8
        # Dust in ultraviolet and UV (if needed for high-z) 3.3 /mu< x< 8 /mu.
        condition = (x >= 3.3) & (x < 5.9)
        if len(a_x[condition]) > 0:
            y = x[condition]
            a_x[condition] = 1.752 - 0.316 * y - 0.104 / ((y - 4.67) ** 2 + 0.341)
            b_x[condition] = -3.090 + 1.825 * y + 1.206 / ((y - 4.62) ** 2 + 0.263)
        condition = (x > 5.9) & (x < 8)
        if len(a_x[condition]) > 0:
            y = x[condition]
            fa_x = numpy.empty(len(a_x[condition]), dtype=float)
            fb_x = numpy.empty(len(a_x[condition]), dtype=float)
            fa_x = -0.04473 * (y - 5.9) ** 2 - 0.009779 * (y - 5.9) ** 3
            fb_x = 0.2130 * (y - 5.9) ** 2 + 0.1207 * (y - 5.9) ** 3
            a_x[condition] = 1.752 - 0.316 * y - 0.104 / ((y - 4.67) ** 2 + 0.341) + fa_x
            b_x[condition] = -3.090 + 1.825 * y + 1.206 / ((y - 4.62) ** 2 + 0.263) + fb_x
        # Dust in far UV (if needed for high-z) 8 /mu < x < 10 /mu region.
        condition = (x >= 8) & (x <= 11.0)
        if len(a_x[condition]) > 0:
            y = x[condition] - 8.0
            a_x[condition] = -1.073 - 0.628 * (y) + 0.137 * (y) ** 2 - 0.070 * (y) ** 3
            b_x[condition] = 13.670 + 4.257 * (y) - 0.420 * (y) ** 2 + 0.374 * (y) ** 3
        return a_x, b_x

    def add_ccm_dust(self, a_x, b_x, a_v=None, ebv=None, r_v=3.1, wavelen=None, flambda=None):
        """
        Add dust model extinction to the SED, modifying flambda and fnu.

        Get a_x and b_x either from setupCCMab or setupODonnell_ab

        Specify any two of A_V, E(B-V) or R_V (=3.1 default).
        """
        warnings.warn(
            "Sed.add_ccm_dust is now deprecated in favor of Sed.add_dust",
            DeprecationWarning,
        )
        return self.add_dust(a_x, b_x, a_v=a_v, ebv=ebv, r_v=r_v, wavelen=wavelen, flambda=flambda)

    def add_dust(self, a_x, b_x, a_v=None, ebv=None, r_v=3.1, wavelen=None, flambda=None):
        """
        Add dust model extinction to the SED, modifying flambda and fnu.

        Get a_x and b_x either from setupCCMab or setupODonnell_ab

        Specify any two of A_V, E(B-V) or R_V (=3.1 default).
        """
        if not hasattr(self, "_ln10_04"):
            self._ln10_04 = 0.4 * numpy.log(10.0)

        # The extinction law taken from Cardelli, Clayton and Mathis ApJ 1989.
        # The general form is A_l / A(V) = a(x) + b(x)/R_V
        # (where x=1/lambda in microns).
        # Then, different values for a(x) and b(x) depending on wavelength
        # regime.
        # Also, the extinction is parametrized as r_v = a_v / E(B-V).
        # The magnitudes of extinction (A_l) translates to flux by
        # a_l = -2.5log(f_red / f_nonred).
        #
        # Figure out if updating self or passed arrays.
        update_self = self._check_use_self(wavelen, flambda)
        if update_self:
            wavelen = self.wavelen
            flambda = self.flambda
            self.fnu = None
        else:
            wavelen = numpy.copy(wavelen)
            flambda = numpy.copy(flambda)
        # Input parameters for reddening can include any of 3 parameters;
        # only 2 are independent.
        # Figure out what parameters were given, and see if self-consistent.
        if r_v == 3.1:
            if a_v is None:
                a_v = r_v * ebv
            elif (a_v is not None) and (ebv is not None):
                # Specified a_v and ebv, so r_v should be nondefault.
                r_v = a_v / ebv
        if r_v != 3.1:
            if (a_v is not None) and (ebv is not None):
                calc_rv = a_v / ebv
                if calc_rv != r_v:
                    raise ValueError(
                        "CCM parametrization expects r_v = a_v / E(B-V);",
                        "Please check input values, because values are inconsistent.",
                    )
            elif a_v is None:
                a_v = r_v * ebv
        # r_v and a_v values are specified or calculated.

        a_lambda = (a_x + b_x / r_v) * a_v
        # dmag_red(dust) = -2.5 log10 (f_red / f_nored) :
        # (f_red / f_nored) = 10**-0.4*dmag_red
        dust = numpy.exp(-a_lambda * self._ln10_04)
        flambda *= dust
        # Update self if required.
        if update_self:
            self.flambda = flambda
            return
        return wavelen, flambda

    def multiply_sed(self, other_sed, wavelen_step=None):
        """
        Multiply two SEDs together - flambda * flambda -
        and return a new sed object.

        Unless the two wavelength arrays are equal, returns a SED
        gridded with stepsize wavelen_step
        over intersecting wavelength region. Does not alter self or other_sed.
        """

        if wavelen_step is None:
            wavelen_step = self._phys_params.wavelenstep

        # Check if the wavelength arrays are equal
        # (in which case do not resample)
        if numpy.all(self.wavelen == other_sed.wavelen):
            flambda = self.flambda * other_sed.flambda
            new_sed = Sed(self.wavelen, flambda=flambda)
        else:
            # Find overlapping wavelength region.
            wavelen_max = min(self.wavelen.max(), other_sed.wavelen.max())
            wavelen_min = max(self.wavelen.min(), other_sed.wavelen.min())
            if wavelen_max < wavelen_min:
                raise Exception("The two SEDS do not overlap in wavelength space.")
            # Set up wavelen/flambda of first object, on grid.
            wavelen_1, flambda_1 = self.resample_sed(
                self.wavelen,
                self.flambda,
                wavelen_min=wavelen_min,
                wavelen_max=wavelen_max,
                wavelen_step=wavelen_step,
            )
            # Set up wavelen/flambda of second object, on grid.
            wavelen_2, flambda_2 = self.resample_sed(
                wavelen=other_sed.wavelen,
                flux=other_sed.flambda,
                wavelen_min=wavelen_min,
                wavelen_max=wavelen_max,
                wavelen_step=wavelen_step,
            )
            # Multiply the two flambda together.
            flambda = flambda_1 * flambda_2
            # Instantiate new sed object.
            # wavelen_1 == wavelen_2 as both are on grid.
            new_sed = Sed(wavelen_1, flambda)
        return new_sed

    # routines related to magnitudes and fluxes

    def calc_adu(self, bandpass, phot_params, wavelen=None, fnu=None):
        """
        Calculate the number of adu from camera, using sb and fnu.

        Given wavelen/fnu arrays or use self.
        Self or passed wavelen/fnu arrays will be unchanged.
        Calculating the AB mag requires the wavelen/fnu pair to be
        on the same grid as bandpass;
        (temporary values of these are used).

        Parameters
        ----------
        bandpass : `rubin_sim.phot_utils.Bandpass`
        phot_params : `rubin_sim.phot_utils.PhotometricParameters`
        wavelen : `np.ndarray`, optional
            wavelength grid in nm
        fnu : `np.ndarray`, optional
            flux in Janskys

        If wavelen and fnu are not specified, this will just use self.wavelen
        and self.fnu
        """

        use_self = self._check_use_self(wavelen, fnu)
        # Use self values if desired, otherwise use values passed to function.
        if use_self:
            # Calculate fnu if required.
            if self.fnu is None:
                # If fnu not present, calculate. (does not regrid).
                self.flambda_tofnu()
            wavelen = self.wavelen
            fnu = self.fnu
        # Make sure wavelen/fnu are on the same wavelength grid as bandpass.
        wavelen, fnu = self.resample_sed(wavelen, fnu, wavelen_match=bandpass.wavelen)
        # Calculate the number of photons.
        dlambda = wavelen[1] - wavelen[0]
        # Nphoton in units of 10^-23 ergs/cm^s/nm.
        nphoton = (fnu / wavelen * bandpass.sb).sum()
        adu = (
            nphoton
            * (phot_params.exptime * phot_params.nexp * phot_params.effarea / phot_params.gain)
            * (1 / self._phys_params.ergsetc2jansky)
            * (1 / self._phys_params.planck)
            * dlambda
        )
        return adu

    def flux_from_mag(self, mag):
        """
        Convert a magnitude back into a flux (implies knowledge of the
        zeropoint, which is stored in this class)
        """

        return numpy.power(10.0, -0.4 * (mag + self.zp))

    def mag_from_flux(self, flux):
        """
        Convert a flux into a magnitude (implies knowledge of the
        zeropoint, which is stored in this class)
        """

        return -2.5 * numpy.log10(flux) - self.zp

    def calc_ergs(self, bandpass, fill=numpy.nan):
        r"""
        Integrate the SED over a bandpass directly.  If self.flambda
        is in ergs/s/cm^2/nm and bandpass.sb is the unitless probability
        that a photon of a given wavelength will pass through the system,
        this method will return the ergs/s/cm^2 of the source observed
        through that bandpass (i.e. it will return the integral

        \int self.flambda(lambda) * bandpass.sb(lambda) * dlambda

        This is to be contrasted with self.calc_flux(), which returns
        the integral of the source's specific flux density over the
        normalized response function of bandpass, giving a flux in
        Janskys (10^-23 erg/cm^2/s/Hz), which should be though of as
        a weighted average of the specific flux density of the source
        over the normalized response function, as detailed in Section
        4.1 of the LSST design document LSE-180.

        Parameters
        ----------
        bandpass is an instantiation of the Bandpass class

        Returns
        -------
        The flux of the current SED through the bandpass in ergs/s/cm^2
        """
        wavelen, flambda = self.resample_sed(
            wavelen=self.wavelen, flux=self.flambda, wavelen_match=bandpass.wavelen, fill=fill
        )

        dlambda = wavelen[1] - wavelen[0]

        # use the trapezoid rule
        energy = (0.5 * (flambda[1:] * bandpass.sb[1:] + flambda[:-1] * bandpass.sb[:-1]) * dlambda).sum()
        return energy

    def calc_flux(self, bandpass, wavelen=None, fnu=None, fill=numpy.nan):
        """
        Integrate the specific flux density of the object over the normalized
        response curve of a bandpass, giving a flux in Janskys
        (10^-23 ergs/s/cm^2/Hz) through the normalized response curve, as
        detailed in Section 4.1 of the LSST design document LSE-180 and
        Section 2.6 of the LSST Science Book
        (http://ww.lsst.org/scientists/scibook).
        This flux in Janskys (which is usually thought of as a unit of
        specific flux density), should be considered a weighted average of
        the specific flux density over the normalized response curve of the
        bandpass.  Because we are using the normalized response curve
        (phi in LSE-180), this quantity will depend only on the shape of the
        response curve, not its absolute normalization.

        Note: the way that the normalized response curve has been defined
        (see equation 5 of LSE-180) is appropriate for photon-counting
        detectors, not calorimeters.

        Passed wavelen/fnu arrays will be unchanged, but if uses self will
        check if fnu is set.

        Calculating the AB mag requires the wavelen/fnu pair to be on the
        same grid as bandpass;
        (temporary values of these are used).
        """
        use_self = self._check_use_self(wavelen, fnu)
        # Use self values if desired, otherwise use values passed to function.
        if use_self:
            # Calculate fnu if required.
            if self.fnu is None:
                self.flambda_tofnu()
            wavelen = self.wavelen
            fnu = self.fnu
        # Go on with magnitude calculation.
        wavelen, fnu = self.resample_sed(wavelen, fnu, wavelen_match=bandpass.wavelen, fill=fill)
        # Calculate bandpass phi value if required.
        if bandpass.phi is None:
            bandpass.sb_tophi()
        # Calculate flux in bandpass and return this value.
        flux = trapezoid(fnu * bandpass.phi, x=wavelen)
        return flux

    def calc_mag(self, bandpass, wavelen=None, fnu=None, fill=numpy.nan):
        """
        Calculate the AB magnitude of an object using the normalized system
        response (phi from Section 4.1 of the LSST design document LSE-180).

        Can pass wavelen/fnu arrays or use self. Self or passed wavelen/fnu
        arrays will be unchanged. Calculating the AB mag requires the
        wavelen/fnu pair to be on the same grid as bandpass;
        (but only temporary values of these are used).
        """
        flux = self.calc_flux(bandpass, wavelen=wavelen, fnu=fnu, fill=fill)
        if flux < 1e-300:
            raise ValueError("This SED has no flux within this bandpass.")
        mag = self.mag_from_flux(flux)
        return mag

    def calc_flux_norm(self, magmatch, bandpass, wavelen=None, fnu=None, fill=numpy.nan):
        """
        Calculate the fluxNorm (SED normalization value for a given mag)
        for a sed.

        Equivalent to adjusting a particular f_nu to Jansky's appropriate
        for the desired mag. Can pass wavelen/fnu or apply to self.
        """
        use_self = self._check_use_self(wavelen, fnu)
        if use_self:
            # Check possibility that fnu is not calculated yet.
            if self.fnu is None:
                self.flambda_tofnu()
            wavelen = self.wavelen
            fnu = self.fnu
        # Fluxnorm gets applied to f_nu
        # (fluxnorm * SED(f_nu) * PHI = mag - 8.9 (AB zeropoint).
        # FluxNorm * SED => correct magnitudes for this object.
        # Calculate fluxnorm.
        curmag = self.calc_mag(bandpass, wavelen, fnu, fill=fill)
        if curmag == self.badval:
            return self.badval
        dmag = magmatch - curmag
        fluxnorm = numpy.power(10, (-0.4 * dmag))
        return fluxnorm

    def multiply_flux_norm(self, flux_norm, wavelen=None, fnu=None):
        """
        Multiply wavelen/fnu (or self.wavelen/fnu) by fluxnorm.

        Returns wavelen/fnu arrays (or updates self).
        Note that multiply_flux_norm does not regrid self.wavelen/flambda/fnu
        at all.
        """
        # Note that flux_norm is intended to be applied to f_nu,
        # so that fluxnorm*fnu*phi = mag (expected magnitude).
        update_self = self._check_use_self(wavelen, fnu)
        if update_self:
            # Make sure fnu is defined.
            if self.fnu is None:
                self.flambda_tofnu()
            wavelen = self.wavelen
            fnu = self.fnu
        else:
            # Require new copy of the data for multiply.
            wavelen = numpy.copy(wavelen)
            fnu = numpy.copy(fnu)
        # Apply fluxnorm.
        fnu = fnu * flux_norm
        # Update self.
        if update_self:
            self.wavelen = wavelen
            self.fnu = fnu
            # Update flambda as well.
            self.fnu_toflambda()
            return
        # Else return new wavelen/fnu pairs.
        return wavelen, fnu

    def renormalize_sed(
        self,
        wavelen=None,
        flambda=None,
        fnu=None,
        lambdanorm=500,
        normvalue=1,
        gap=0,
        normflux="flambda",
        wavelen_step=None,
    ):
        """
        Renormalize sed in flambda to have normflux=normvalue @ lambdanorm
        averaged over gap.

        Can normalized in flambda or fnu values. wavelen_step specifies
        the wavelength spacing when using 'gap'.

        Either returns wavelen/flambda values or updates self.
        """
        # Normalizes the fnu/flambda SED at one wavelength or average value
        # over small range (gap).
        # This is useful for generating SED catalogs, mostly, to make them
        # match schema.
        # Do not use this for calculating specific magnitudes -- use
        # calcfluxNorm and multiply_flux_norm.
        # Start normalizing wavelen/flambda.

        if wavelen_step is None:
            wavelen_step = self._phys_params.wavelenstep

        if normflux == "flambda":
            update_self = self._check_use_self(wavelen, flambda)
            if update_self:
                wavelen = self.wavelen
                flambda = self.flambda
            else:
                # Make a copy of the input data.
                wavelen = numpy.copy(wavelen)
                # Look for either flambda or fnu in input data.
                if flambda is None:
                    if fnu is None:
                        raise Exception("If passing wavelength, must also pass fnu or flambda.")
                    # If not given flambda, must calculate from fnu.
                    wavelen, flambda = self.fnu_toflambda(wavelen, fnu)
                # Make a copy of the input data.
                else:
                    flambda = numpy.copy(flambda)
            # Calculate renormalization values.
            # Check that flambda is defined at the wavelength want to use for
            # renormalization.
            if (lambdanorm > wavelen.max()) or (lambdanorm < wavelen.min()):
                raise Exception(
                    "Desired wavelength for renormalization, %f, " % (lambdanorm)
                    + "is outside defined wavelength range."
                )
            # "standard" schema have flambda = 1 at 500 nm.
            if gap == 0:
                flambda_atpt = numpy.interp(lambdanorm, wavelen, flambda, left=None, right=None)
                gapval = flambda_atpt
            else:
                lambdapt = numpy.arange(lambdanorm - gap, lambdanorm + gap, wavelen_step, dtype=float)
                flambda_atpt = numpy.zeros(len(lambdapt), dtype="float")
                flambda_atpt = numpy.interp(lambdapt, wavelen, flambda, left=None, right=None)
                gapval = flambda_atpt.sum() / len(lambdapt)
            # Now renormalize fnu and flambda, in the case of normalizing
            # flambda.
            if gapval == 0:
                raise Exception(
                    "Original flambda is 0 at the desired point of normalization. " "Cannot renormalize."
                )
            konst = normvalue / gapval
            flambda = flambda * konst
            wavelen, fnu = self.flambda_tofnu(wavelen, flambda)
        elif normflux == "fnu":
            update_self = self._check_use_self(wavelen, fnu)
            if update_self:
                wavelen = self.wavelen
                if self.fnu is None:
                    self.flambda_tofnu()
                fnu = self.fnu
            else:
                # Make a copy of the input data.
                wavelen = numpy.copy(wavelen)
                # Look for either flambda or fnu in input data.
                if fnu is None:
                    if flambda is None:
                        raise Exception("If passing wavelength, must also pass fnu or flambda.")
                    wavelen, fnu = self.flambda_tofnu(wavelen, fnu)
                # Make a copy of the input data.
                else:
                    fnu = numpy.copy(fnu)
            # Calculate renormalization values.
            # Check that flambda is defined at the wavelength want to use
            # for renormalization.
            if (lambdanorm > wavelen.max()) or (lambdanorm < wavelen.min()):
                raise Exception(
                    "Desired wavelength for renormalization, %f, " % (lambdanorm)
                    + "is outside defined wavelength range."
                )
            if gap == 0:
                fnu_atpt = numpy.interp(lambdanorm, wavelen, flambda, left=None, right=None)
                gapval = fnu_atpt
            else:
                lambdapt = numpy.arange(lambdanorm - gap, lambdanorm + gap, wavelen_step, dtype=float)
                fnu_atpt = numpy.zeros(len(lambdapt), dtype="float")
                fnu_atpt = numpy.interp(lambdapt, wavelen, fnu, left=None, right=None)
                gapval = fnu_atpt.sum() / len(lambdapt)
            # Now renormalize fnu and flambda in the case of normalizing fnu.
            if gapval == 0:
                raise Exception(
                    "Original fnu is 0 at the desired point of normalization. " "Cannot renormalize."
                )
            konst = normvalue / gapval
            fnu = fnu * konst
            wavelen, flambda = self.fnutoflambda(wavelen, fnu)
        if update_self:
            self.wavelen = wavelen
            self.flambda = flambda
            self.fnu = fnu
            return
        new_sed = Sed(wavelen=wavelen, flambda=flambda)
        return new_sed

    def write_sed(
        self,
        filename,
        print_header=None,
        print_fnu=False,
        wavelen_min=None,
        wavelen_max=None,
        wavelen_step=None,
    ):
        """
        Write SED (wavelen, flambda, optional fnu) out to file.

        Option of adding a header line (such as version info) to output file.
        Does not alter self, regardless of grid or presence/absence of fnu.
        """
        # This can be useful for debugging or recording an SED.
        f = open(filename, "w")
        wavelen = self.wavelen
        flambda = self.flambda
        wavelen, flambda = self.resample_sed(
            wavelen,
            flambda,
            wavelen_min=wavelen_min,
            wavelen_max=wavelen_max,
            wavelen_step=wavelen_step,
        )
        # Then just use this gridded wavelen/flambda to calculate fnu.
        # Print header.
        if print_header is not None:
            if not print_header.startswith("#"):
                print_header = "# " + print_header
            f.write(print_header)
        # Print standard header info.
        if print_fnu:
            wavelen, fnu = self.flambda_tofnu(wavelen, flambda)
            print("# Wavelength(nm)  Flambda(ergs/cm^s/s/nm)   Fnu(Jansky)", file=f)
        else:
            print("# Wavelength(nm)  Flambda(ergs/cm^s/s/nm)", file=f)
        for i in range(0, len(wavelen), 1):
            if print_fnu:
                fnu = self.flambda_tofnu(wavelen=wavelen, flambda=flambda)
                print(wavelen[i], flambda[i], fnu[i], file=f)
            else:
                print("%.2f %.7g" % (wavelen[i], flambda[i]), file=f)
        # Done writing, close file.
        f.close()
        return

    # Bonus, functions for many-magnitude calculation for many SEDs with
    # a single bandpass

    def setup_phi_array(self, bandpasslist):
        """
        Sets up a 2-d numpy phi array from bandpasslist suitable for input
        to Sed's many_mag_calc.

        This is intended to be used once, most likely before using Sed's
        many_mag_calc many times on many SEDs.
        Returns 2-d phi array and the wavelen_step (dlambda) appropriate for
        that array.
        """
        # Calculate dlambda for phi array.
        wavelen_step = bandpasslist[0].wavelen[1] - bandpasslist[0].wavelen[0]
        wavelen_min = numpy.min([bandpass.wavelen[0] for bandpass in bandpasslist])
        wavelen_max = numpy.max([bandpass.wavelen[-1] for bandpass in bandpasslist])
        # Set up
        phiarray = numpy.empty(
            (
                len(bandpasslist),
                numpy.size(numpy.arange(wavelen_min, wavelen_max + wavelen_step, wavelen_step)),
            ),
            dtype="float",
        )
        # Check phis calculated and on same wavelength grid.
        i = 0
        for bp in bandpasslist:
            # Be sure bandpasses on same grid and calculate phi.
            bp.resample_bandpass(
                wavelen_min=wavelen_min,
                wavelen_max=wavelen_max,
                wavelen_step=wavelen_step,
            )
            bp.sb_tophi()
            phiarray[i] = bp.phi
            i = i + 1
        return phiarray, wavelen_step

    def many_flux_calc(self, phiarray, wavelen_step, observed_bandpass_ind=None):
        """
        Calculate fluxes of a single sed for which fnu has been evaluated in a
        set of bandpasses for which phiarray has been set up to have the same
        wavelength grid as the SED in units of ergs/cm^2/sec. It is assumed
        that `self.fnu` is set before calling this method, and that phiArray
        has the same wavelength grid as the Sed.


        Parameters
        ----------
        phiarray : `np.ndarray`
            phiarray corresponding to the list of bandpasses in which the band
            fluxes need to be calculated, in the same wavelength grid as Sed
        wavelen_step : `float`
            the uniform grid size of the SED
        observed_bandpass_ind : `list` [`int`], optional
            list of indices of phiarray corresponding to observed bandpasses,
            if None, the original phiarray is returned

        Returns
        -------
        `np.ndarray` with size equal to number of bandpass filters  band flux
        values in units of ergs/cm^2/sec

        .. note: Sed.many_flux_calc `assumes` phiArray has the same wavelength
        grid as the Sed and that `sed.fnu` has been calculated for the sed,
        perhaps using `sed.flambda_tofnu()`. This requires calling
        `sed.setupPhiArray()` first. These assumptions are to avoid error
        checking within this function (for speed), but could lead to errors if
        method is used incorrectly.

        Note on units: Fluxes calculated this way will be the flux density
        integrated over the weighted response curve of the bandpass.
        See equaiton 2.1 of the LSST Science Book

        http://www.lsst.org/scientists/scibook
        """

        if observed_bandpass_ind is not None:
            phiarray = phiarray[observed_bandpass_ind]
        flux = numpy.sum(phiarray * self.fnu, axis=1) * wavelen_step
        return flux

    def many_mag_calc(self, phiarray, wavelen_step, observed_bandpass_ind=None):
        """
        Calculate many magnitudes for many bandpasses using a single sed.

        This method assumes that there will be flux within a particular
        bandpass
        (could return '-Inf' for a magnitude if there is none).
        Use setupPhiArray first, and note that Sed.many_mag_calc *assumes*
        phiArray has the same wavelength grid as the Sed, and that fnu has
        already been calculated for Sed.
        These assumptions are to avoid error checking within this function
        (for speed), but could lead to errors if method is used incorrectly.

        Parameters
        ----------
        phiarray : `np.ndarray`, mandatory
            phiarray corresponding to the list of bandpasses in which the band
            fluxes need to be calculated, in the same wavelength grid as SED
        wavelen_step : `float`, mandatory
            the uniform grid size of the SED
        observed_bandpass_ind : `list` [`int`], optional
            list of indices of phiarray corresponding to observed bandpasses,
            if None, the original phiarray is returned

        """
        fluxes = self.many_flux_calc(phiarray, wavelen_step, observed_bandpass_ind)
        mags = -2.5 * numpy.log10(fluxes) - self.zp
        return mags


def read_close__kurucz(teff, fe_h, logg):
    """
    Check the cached Kurucz models and load the model closest to the
    input stellar parameters.
    Parameters are matched in order of Teff, fe_h, and logg.

    Parameters
    ----------
    teff : `float`
        Effective temperature of the stellar template.
        Reasonable range is 3830-11,100 K.
    fe_h : `float`
        Metallicity [Fe/H] of stellar template. Values in range -5 to 1.
    logg : `float`
       Log of the surface gravity for the stellar template.
       Values in range 0. to 50.

    Returns
    -------
    sed : `rubin_sim.phot_utils.Sed`
        The SED of the closest matching stellar template
    paramDict : `dict`
        Dictionary of the teff, fe_h, logg that were actually loaded

    """
    global _global_lsst_sed_cache

    # Load the cache if it hasn't been done
    if _global_lsst_sed_cache is None:
        cache_lsst_seds()
    # Build an array with all the files in the cache
    if not hasattr(read_close__kurucz, "param_combos"):
        kurucz_files = [
            filename
            for filename in _global_lsst_sed_cache
            if ("kurucz" in filename) & ("_g" in os.path.basename(filename))
        ]
        kurucz_files = list(set(kurucz_files))
        read_close__kurucz.param_combos = numpy.zeros(
            len(kurucz_files),
            dtype=[
                ("filename", ("|U200")),
                ("teff", float),
                ("fe_h", float),
                ("logg", float),
            ],
        )
        for i, filename in enumerate(kurucz_files):
            read_close__kurucz.param_combos["filename"][i] = filename
            filename = os.path.basename(filename)
            if filename[1] == "m":
                sign = -1
            else:
                sign = 1
            logz = sign * float(filename.split("_")[0][2:]) / 10.0
            read_close__kurucz.param_combos["fe_h"][i] = logz
            logg_temp = float(filename.split("g")[1].split("_")[0])
            read_close__kurucz.param_combos["logg"][i] = logg_temp
            teff_temp = float(filename.split("_")[-1].split(".")[0])
            read_close__kurucz.param_combos["teff"][i] = teff_temp
        read_close__kurucz.param_combos = numpy.sort(
            read_close__kurucz.param_combos, order=["teff", "fe_h", "logg"]
        )

    # Lookup the closest match. Prob a faster way to do this.
    teff_diff = numpy.abs(read_close__kurucz.param_combos["teff"] - teff)
    g1 = numpy.where(teff_diff == teff_diff.min())[0]
    fe_h_diff = numpy.abs(read_close__kurucz.param_combos["fe_h"][g1] - fe_h)
    g2 = numpy.where(fe_h_diff == fe_h_diff.min())[0]
    logg_diff = numpy.abs(read_close__kurucz.param_combos["logg"][g1][g2] - logg)
    g3 = numpy.where(logg_diff == logg_diff.min())[0]
    file_match = read_close__kurucz.param_combos["filename"][g1][g2][g3]
    if numpy.size(file_match > 1):
        warnings.warn("Multiple close files")
        file_match = file_match[0]

    # Record what Parameters were actually loaded
    teff = read_close__kurucz.param_combos["teff"][g1][g2][g3][0]
    fe_h = read_close__kurucz.param_combos["fe_h"][g1][g2][g3][0]
    logg = read_close__kurucz.param_combos["logg"][g1][g2][g3][0]

    # Read in the matching file
    sed = Sed()
    sed.read_sed_flambda(file_match)
    return sed, {"teff": teff, "fe_h": fe_h, "logg": logg}
