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
bandpass -

Class data:
wavelen (nm)
sb  (Transmission, 0-1)
phi (Normalized system response)
phi will be None until specifically needed; any updates to wavelen/sb within class will reset phi to None.
the name of the bandpass file


Methods:
* __init__ : pass wavelen/sb arrays and set values  OR set data to None's
* setWavelenLimits / getWavelenLimits: set or get the wavelength limits of bandpass
* setBandpass: set bandpass using wavelen/sb input values
* getBandpass: return copies of wavelen/sb values
* imsim_bandpass : set up a bandpass which is 0 everywhere but one wavelength
(this can be useful for imsim magnitudes)
* read_throughput : set up a bandpass by reading data from a single file
* readThroughtputList : set up a bandpass by reading data from many files and multiplying
the individual throughputs
* resample_bandpass : use linear interpolation to resample wavelen/sb arrays onto a regular grid
(grid is specified by min/max/step size)
* sb_tophi : calculate phi from sb - needed for calculating magnitudes
* multiply_throughputs : multiply self.wavelen/sb by given wavelen/sb and return
new wavelen/sb arrays (wavelength sampled like self)
* calc_zp_t : calculate instrumental zeropoint for this bandpass
* calc_eff_wavelen: calculate the effective wavelength (using both Sb and Phi) for this bandpass
* writeThroughput : utility to write bandpass information to file

"""
import gzip
import os
import warnings

import numpy
import scipy.interpolate as interpolate

from .physical_parameters import PhysicalParameters
from .sed import Sed  # For ZP_t and M5 calculations. And for 'fast mags' calculation.

__all__ = ["Bandpass"]


class Bandpass:
    """
    Hold and use telescope throughput curves.
    """

    def __init__(self, wavelen=None, sb=None, sampling_warning=0.2):
        """
        Initialize bandpass object, with option to pass wavelen/sb arrays in directly.

        Parameters
        ----------
        wavelen : np.array (None)
            Wavelength array in nm
        sb : np.array (None)
            Throughput array (fraction, 0-1)
        sampling_warning : float (0.2)
            If wavelength sampling lower than this, throw a warning because it might not
            work well with Sed (nm).
        """

        self._phys_params = PhysicalParameters()
        self.sampling_warning = sampling_warning
        self.wavelen = None
        self.sb = None
        self.phi = None
        self.bandpassname = None
        if (wavelen is not None) and (sb is not None):
            self.set_bandpass(wavelen, sb)
        return

    def _check_wavelength_sampling(self):
        """Check that the wavelength sampling is above some threshold"""
        if self.wavelen is not None:
            dif = numpy.diff(self.wavelen)
            if numpy.max(dif) > self.sampling_warning:
                warnings.warn(
                    "Wavelength sampling of %.1f nm is > %.1f nm" % (numpy.max(dif), self.sampling_warning)
                    + ", this may not work well"
                    " with a Sed object. Consider resampling with resample_bandpass method."
                )

    def set_bandpass(self, wavelen, sb):
        """
        Populate bandpass data with wavelen/sb arrays.

        Phi set to None.
        """
        # Check data type.
        if (isinstance(wavelen, numpy.ndarray) == False) or (isinstance(sb, numpy.ndarray) == False):
            raise ValueError("Wavelen and sb arrays must be numpy arrays.")
        # Check data matches in length.
        if len(wavelen) != len(sb):
            raise ValueError("Wavelen and sb arrays must have the same length.")

        self.wavelen = numpy.copy(wavelen)
        self.phi = None
        self.sb = numpy.copy(sb)
        self.bandpassname = "FromArrays"
        self._check_wavelength_sampling()

    def imsim_bandpass(self, imsimwavelen=500.0, wavelen_min=300, wavelen_max=1150, wavelen_step=0.1):
        """
        Populate bandpass data with sb=0 everywhere except sb=1 at imsimwavelen.

        Sets wavelen/sb, with grid min/max/step as Parameters. Does NOT set phi.
        """
        # Set up arrays.
        self.wavelen = numpy.arange(
            wavelen_min,
            wavelen_max + wavelen_step,
            wavelen_step,
            dtype="float",
        )
        self.phi = None
        # Set sb.
        self.sb = numpy.zeros(len(self.wavelen), dtype="float")
        self.sb[abs(self.wavelen - imsimwavelen) < wavelen_step / 2.0] = 1.0
        self.bandpassname = "IMSIM"
        self._check_wavelength_sampling()

    def read_throughput(self, filename):
        """
        Populate bandpass data with data (wavelen/sb) read from file.

        Sets wavelen/sb. Does NOT set phi.
        """
        # Set self values to None in case of file read error.
        self.wavelen = None
        self.phi = None
        self.sb = None
        # Check for filename error.
        # If given list of filenames, pass to (and return from) read_throughputList.
        if isinstance(filename, list):
            warnings.warn(
                "Was given list of files, instead of a single file. Using read_throughputList instead"
            )
            self.read_throughput_list(component_list=filename)
        # Filename is single file, now try to open file and read data.
        try:
            if filename.endswith(".gz"):
                f = gzip.open(filename, "rt")
            else:
                f = open(filename, "r")
        except IOError:
            try:
                if filename.endswith(".gz"):
                    f = open(filename[:-3], "r")
                else:
                    f = gzip.open(filename + ".gz", "rt")
            except IOError:
                raise IOError("The throughput file %s does not exist" % (filename))
        # The throughput file should have wavelength(A), throughput(Sb) as first two columns.
        wavelen = []
        sb = []
        for line in f:
            if line.startswith("#") or line.startswith("$") or line.startswith("!"):
                continue
            values = line.split()
            if len(values) < 2:
                continue
            if (values[0] == "$") or (values[0] == "#") or (values[0] == "!"):
                continue
            wavelen.append(float(values[0]))
            sb.append(float(values[1]))
        f.close()
        self.bandpassname = filename
        # Set up wavelen/sb.
        self.wavelen = numpy.array(wavelen, dtype="float")
        self.sb = numpy.array(sb, dtype="float")
        # Check that wavelength is monotonic increasing and non-repeating in wavelength. (Sort on wavelength).
        if len(self.wavelen) != len(numpy.unique(self.wavelen)):
            raise ValueError("The wavelength values in file %s are non-unique." % (filename))
        # Sort values.
        p = self.wavelen.argsort()
        self.wavelen = self.wavelen[p]
        self.sb = self.sb[p]
        self._check_wavelength_sampling()

    def read_throughput_list(
        self,
        component_list=[
            "detector.dat",
            "lens1.dat",
            "lens2.dat",
            "lens3.dat",
            "m1.dat",
            "m2.dat",
            "m3.dat",
            "atmos_std.dat",
        ],
        root_dir=".",
        wavelen_min=300,
        wavelen_max=1150,
        wavelen_step=0.1,
    ):
        """
        Populate bandpass data by reading from a series of files with wavelen/Sb data.

        Multiplies throughputs (sb) from each file to give a final bandpass throughput.
        Sets wavelen/sb, with grid min/max/step as Parameters.  Does NOT set phi.
        """
        # ComponentList = names of files in that directory.
        # A typical component list of all files to build final component list, including filter, might be:
        #   component_list=['detector.dat', 'lens1.dat', 'lens2.dat', 'lens3.dat',
        #                 'm1.dat', 'm2.dat', 'm3.dat', 'atmos_std.dat', 'ideal_g.dat']
        #
        # Set up wavelen/sb on grid.
        self.wavelen = numpy.arange(
            wavelen_min,
            wavelen_max + wavelen_step / 2.0,
            wavelen_step,
            dtype="float",
        )
        self.phi = None
        self.sb = numpy.ones(len(self.wavelen), dtype="float")
        # Set up a temporary bandpass object to hold data from each file.
        tempbandpass = Bandpass()
        for component in component_list:
            # Read data from file.
            tempbandpass.read_throughput(os.path.join(root_dir, component))
            tempbandpass.resample_bandpass(
                wavelen_min=wavelen_min,
                wavelen_max=wavelen_max,
                wavelen_step=wavelen_step,
            )
            # Multiply self by new sb values.
            self.sb = self.sb * tempbandpass.sb
        self.bandpassname = "".join(component_list)
        self._check_wavelength_sampling()

    def get_bandpass(self):
        wavelen = numpy.copy(self.wavelen)
        sb = numpy.copy(self.sb)
        return wavelen, sb

    ## utilities

    def check_use_self(self, wavelen, sb):
        """
        Simple utility to check if should be using self.wavelen/sb or passed arrays.

        Useful for other methods in this class.
        Also does data integrity check on wavelen/sb if not self.
        """
        update_self = False
        if (wavelen is None) or (sb is None):
            # Then one of the arrays was not passed - check if true for both.
            if (wavelen is not None) or (sb is not None):
                # Then only one of the arrays was passed - raise exception.
                raise ValueError("Must either pass *both* wavelen/sb pair, or use self defaults")
            # Okay, neither wavelen or sb was passed in - using self only.
            update_self = True
        else:
            # Both of the arrays were passed in - check their validity.
            if (isinstance(wavelen, numpy.ndarray) == False) or (isinstance(sb, numpy.ndarray) == False):
                raise ValueError("Must pass wavelen/sb as numpy arrays")
            if len(wavelen) != len(sb):
                raise ValueError("Must pass equal length wavelen/sb arrays")
        return update_self

    def resample_bandpass(
        self,
        wavelen=None,
        sb=None,
        wavelen_min=300,
        wavelen_max=1150,
        wavelen_step=0.1,
    ):
        """
        Resamples wavelen/sb (or self.wavelen/sb) onto grid defined by min/max/step.

        Either returns wavelen/sb (if given those arrays) or updates wavelen / Sb in self.
        If updating self, resets phi to None.
        """
        # Check wavelength limits.
        wavelen_min = wavelen_min
        wavelen_max = wavelen_max
        wavelen_step = wavelen_step
        # Is method acting on self.wavelen/sb or passed in wavelen/sb? Sort it out.
        update_self = (wavelen is None) & (sb is None)
        if update_self:
            wavelen = self.wavelen
            sb = self.sb
        # Now, on with the resampling.
        if (wavelen.min() > wavelen_max) or (wavelen.max() < wavelen_min):
            raise Exception("No overlap between known wavelength range and desired wavelength range.")
        # Set up gridded wavelength.
        wavelen_grid = numpy.arange(
            wavelen_min, wavelen_max + wavelen_step / 2.0, wavelen_step, dtype="float"
        )
        # Do the interpolation of wavelen/sb onto the grid. (note wavelen/sb type failures will die here).
        f = interpolate.interp1d(wavelen, sb, fill_value=0, bounds_error=False)
        sb_grid = f(wavelen_grid)
        # Update self values if necessary.
        if update_self:
            self.phi = None
            self.wavelen = wavelen_grid
            self.sb = sb_grid
            return
        self._check_wavelength_sampling()
        return wavelen_grid, sb_grid

    ## more complicated bandpass functions

    def sb_tophi(self):
        """
        Calculate and set phi - the normalized system response.

        This function only updates self.phi.
        """
        # The definition of phi = (Sb/wavelength)/\int(Sb/wavelength)dlambda.
        self.phi = self.sb / self.wavelen
        # Normalize phi so that the integral of phi is 1.
        norm = numpy.trapz(self.phi, x=self.wavelen)
        self.phi = self.phi / norm
        return

    def multiply_throughputs(self, wavelen_other, sb_other):
        """
        Multiply self.sb by another wavelen/sb pair, return wavelen/sb arrays.

        The returned arrays will be gridded like this bandpass.
        This method does not affect self.
        """
        # Resample wavelen_other/sb_other to match this bandpass.
        if not numpy.all(self.wavelen == wavelen_other):
            wavelen_other, sb_other = self.resample_bandpass(
                wavelen=wavelen_other,
                sb=sb_other,
                wavelen_min=self.wavelen.min(),
                wavelen_max=self.wavelen.max(),
                wavelen_step=self.wavelen[1] - self.wavelen[0],
            )
        # Make new memory copy of wavelen.
        wavelen_new = numpy.copy(self.wavelen)
        # Calculate new transmission - this is also new memory.
        sb_new = self.sb * sb_other
        return wavelen_new, sb_new

    def calc_zp_t(self, photometric_parameters):
        """
        Calculate the instrumental zeropoint for a bandpass.

        @param [in] photometric_parameters is an instantiation of the
        PhotometricParameters class that carries details about the
        photometric response of the telescope.  Defaults to LSST values.
        """
        # ZP_t is the magnitude of a (F_nu flat) source which produced 1 count per second.
        # This is often also known as the 'instrumental zeropoint'.
        # Set gain to 1 if want to explore photo-electrons rather than adu.
        # The typical LSST exposure time is 15s and this is default here, but typical zp_t definition is for 1s.
        # SED class uses flambda in ergs/cm^2/s/nm, so need effarea in cm^2.
        #
        # Check dlambda value for integral.
        dlambda = self.wavelen[1] - self.wavelen[0]
        # Set up flat source of arbitrary brightness,
        #   but where the units of fnu are Jansky (for AB mag zeropoint = -8.9).
        flatsource = Sed()
        flatsource.set_flat_sed()
        adu = flatsource.calc_adu(self, phot_params=photometric_parameters)
        # Scale fnu so that adu is 1 count/expTime.
        flatsource.fnu = flatsource.fnu * (1 / adu)
        # Now need to calculate AB magnitude of the source with this fnu.
        if self.phi is None:
            self.sb_tophi()
        zp_t = flatsource.calc_mag(self)
        return zp_t

    def calc_eff_wavelen(self):
        """
        Calculate effective wavelengths for filters.
        """
        # This is useful for summary numbers for filters.
        # Calculate effective wavelength of filters.
        if self.phi is None:
            self.sb_tophi()
        effwavelenphi = (self.wavelen * self.phi).sum() / self.phi.sum()
        effwavelensb = (self.wavelen * self.sb).sum() / self.sb.sum()
        return effwavelenphi, effwavelensb

    def write_throughput(self, filename, print_header=None, write_phi=False):
        """
        Write throughput to a file.
        """
        # Useful if you build a throughput up from components and need to record the combined value.
        f = open(filename, "w")
        # Print header.
        if print_header is not None:
            if not print_header.startswith("#"):
                print_header = "#" + print_header
            f.write(print_header)
        if write_phi:
            if self.phi is None:
                self.sb_tophi()
            print("# Wavelength(nm)  Throughput(0-1)   Phi", file=f)
        else:
            print("# Wavelength(nm)  Throughput(0-1)", file=f)
        # Loop through data, printing out to file.
        for i in range(0, len(self.wavelen), 1):
            if write_phi:
                print(self.wavelen[i], self.sb[i], self.phi[i], file=f)
            else:
                print(self.wavelen[i], self.sb[i], file=f)
        f.close()
