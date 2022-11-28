import os
import copy
from .sed import Sed
from rubin_sim.phot_utils import get_imsim_flux_norm

__all__ = ["SedList"]


class SedList(object):
    """
    This class will read in a list of Seds from disk and store them.

    It also has the ability to renormalize, redden (according to the
    O'Donnell 94, ApJ 422 158 dust model), and redshift the Seds.

    As it reads in the Seds, it will keep track of each unique file it reads
    in.  If two Seds are based on the same file (before normalization, reddening,
    etc.), it will refer back to its own memory, rather than reading the
    file from disk a second time.

    The method load_seds_from_list allows the user to add Seds to the list
    after the constructor has been called.
    """

    def __init__(
        self,
        sed_name_list,
        mag_norm_list,
        normalizing_bandpass=None,
        spec_map=None,
        file_dir="",
        wavelen_match=None,
        redshift_list=None,
        galactic_av_list=None,
        internal_av_list=None,
        cosmological_dimming=True,
    ):

        """
        @param [in] sed_name_list is a list of SED file names.

        @param [in] mag_norm_list is a list of magnitude normalizations
        (in the normalizing_bandpass) for each of the Seds.

        @param[in] normalizing_bandpass is an instantiation of the Bandpass
        class defining the bandpass in which the magNorms from mag_norm_list
        are calculated.  This defaults to the Bandpass().imsim_bandpass(),
        which is essentially a delta function at 500 nm.

        @param [in] file_dir is the base directory where the Sed files are stored
        (defaults to current working directory).

        @param [in] spec_map is a spec_map (defined in sims_utils/../fileMaps.py)
        that maps the names in sed_name_list to paths of the files relative to
        file_dir (defaults to None; a defaultSpecMap targeted at
        sims_sed_library is defined in sims_utils)

        @param [in] wavelen_match is an optional numpy array representing
        the wavelength grid to which all Seds will be re-mapped.

        @param [in] redshift_list is an optional list of redshifts for the Sed

        @param [in] internal_av_list is an optional list of A(V) due to internal
        dust (for spectra of galaxies).

        @param [in] galactic_av_list is an optional list of A(V) due to
        Milky Way Dust.

        @param [in] cosmological_dimming is a `bool` indicating whether cosmological
        dimming (the extray (1+z)^-1 factor in flux) should be applied to spectra
        when they are redshifted (defaults to True)

        Note: once wavelen_match and cosmological_dimming have been set in
        the constructor, they cannot be un-set.

        Similarly: if you construct a SedList without a galactic_av_list,
        internal_av_list, or redshift_list, you cannot later add spectra with
        whichever of those features were left out.
        """

        self._initialized = False
        self._spec_map = spec_map
        self._wavelen_match = copy.deepcopy(wavelen_match)
        self._file_dir = file_dir
        self._cosmological_dimming = cosmological_dimming

        self._normalizing_bandpass = normalizing_bandpass

        self._sed_list = []
        self._redshift_list = None
        self._galactic_av_list = None
        self._internal_av_list = None

        self._a_int = None
        self._b_int = None
        self._av_int_wavelen = None

        self._a_gal = None
        self._b_gal = None
        self._av_gal_wavelen = None

        self.load_seds_from_list(
            sed_name_list,
            mag_norm_list,
            internal_av_list=internal_av_list,
            galactic_av_list=galactic_av_list,
            redshift_list=redshift_list,
        )

    def __len__(self):
        return len(self._sed_list)

    def __getitem__(self, index):
        return self._sed_list[index]

    def __iter__(self):
        for val in self._sed_list:
            yield val

    # Handy routines for handling Sed/Bandpass routines with sets of dictionaries.
    def load_seds_from_list(
        self,
        sed_name_list,
        mag_norm_list,
        internal_av_list=None,
        galactic_av_list=None,
        redshift_list=None,
    ):
        """
        Load the Seds specified by sed_name_list, applying the specified normalization,
        extinction, and redshift.

        @param [in] sedList is a list of file names containing Seds

        @param [in] magNorm is the magnitude normalization

        @param [in] internal_av_list is an optional list of A(V) due to internal
        dust

        @param [in] galactic_av_list is an optional list of A(V) due to
        Milky Way dust

        @param [in] redshift_list is an optional list of redshifts for the
        input Sed

        Seds are read in and stored to this object's internal list of Seds.

        Note: if you constructed this SedList object without internal_av_list,
        you cannot load Seds with internal_av_list now.  Likewise for galacticAvlist
        and redshift_list.
        """

        if not self._initialized:
            if internal_av_list is not None:
                self._internal_av_list = copy.deepcopy(list(internal_av_list))
            else:
                self._internal_av_list = None

            if galactic_av_list is not None:
                self._galactic_av_list = copy.deepcopy(list(galactic_av_list))
            else:
                self._galactic_av_list = None

            if redshift_list is not None:
                self._redshift_list = copy.deepcopy(list(redshift_list))
            else:
                self._redshift_list = None

        else:
            if self._internal_av_list is None and internal_av_list is not None:
                raise RuntimeError("This SedList does not contain internal_av_list")
            elif self._internal_av_list is not None:
                if internal_av_list is None:
                    self._internal_av_list += [None] * len(sed_name_list)
                else:
                    self._internal_av_list += list(internal_av_list)

            if self._galactic_av_list is None and galactic_av_list is not None:
                raise RuntimeError("This SedList does not contain galactic_av_list")
            elif self._galactic_av_list is not None:
                if galactic_av_list is None:
                    self._galactic_av_list += [None] * len(sed_name_list)
                else:
                    self._galactic_av_list += list(galactic_av_list)

            if self._redshift_list is None and redshift_list is not None:
                raise RuntimeError("This SedList does not contain redshift_list")
            elif self._redshift_list is not None:
                if redshift_list is None:
                    self._redshift_list += [None] * len(sed_name_list)
                else:
                    self._redshift_list += list(redshift_list)

        temp_sed_list = []
        for sed_name, magNorm in zip(sed_name_list, mag_norm_list):
            sed = Sed()

            if sed_name != "None":
                if self._spec_map is not None:
                    sed.read_sed_flambda(
                        os.path.join(self._file_dir, self._spec_map[sed_name])
                    )
                else:
                    sed.read_sed_flambda(os.path.join(self._file_dir, sed_name))

                if self._normalizing_bandpass is not None:
                    f_norm = sed.calc_flux_norm(magNorm, self._normalizing_bandpass)
                else:
                    f_norm = get_imsim_flux_norm(sed, magNorm)

                sed.multiply_flux_norm(f_norm)

            temp_sed_list.append(sed)

        if internal_av_list is not None:
            self._av_int_wavelen, self._a_int, self._b_int = self.apply_av(
                temp_sed_list,
                internal_av_list,
                self._av_int_wavelen,
                self._a_int,
                self._b_int,
            )

        if redshift_list is not None:
            self.apply_redshift(temp_sed_list, redshift_list)

        if self._wavelen_match is not None:
            for sed_obj in temp_sed_list:
                if sed_obj.wavelen is not None:
                    sed_obj.resample_sed(wavelen_match=self._wavelen_match)

        if galactic_av_list is not None:
            self._av_gal_wavelen, self._a_gal, self._b_gal = self.apply_av(
                temp_sed_list,
                galactic_av_list,
                self._av_gal_wavelen,
                self._a_gal,
                self._b_gal,
            )

        self._sed_list += temp_sed_list

        self._initialized = True

    def apply_av(self, sed_list, av_list, dust_wavelen, a_coeffs, b_coeffs):
        """
        Take the array of Sed objects sed_list and apply extinction due to dust.

        This method makes the necessary changes to the Seds in SedList in situ.
        It returns the wavelength grid and corresponding dust coefficients so that
        they an be reused on Seds with identical wavelength grids.

        @param [in] sed_list is a list of Sed objects

        @param [in] av_list is a list of Av extinction values internal to each object

        @param [in] dust_wavelen is the wavelength grid corresponding to the
        dust model coefficients.  If this differs from the wavelength grid
        of any of the Seds in sed_list, the dust model coefficients will be
        re-generated.

        @param [in] a_coeffs are the 'a' dust model coefficients (see O'Donnell 1994
        ApJ 422 158)

        @param [in] b_coeffs are the 'b' dust model coefficients from O'Donnell.

        @param [out] dust_wavelen as generated/used by this method

        @param [out] a_coeffs as generated/used by this method

        @param [out] b_coeffs as generated/used by this method

        a_coeffs and b_coeffs are re-generated as needed
        """

        for sedobj, av in zip(sed_list, av_list):
            if sedobj.wavelen is not None and av is not None:
                # setup_ccm_ab only depends on the wavelen array
                # because this is supposed to be the same for every
                # SED object in sed_list, it is only called once for
                # each invocation of applyAv

                if (
                    dust_wavelen is None
                    or len(sedobj.wavelen) != len(dust_wavelen)
                    or (sedobj.wavelen != dust_wavelen).any()
                ):
                    a_coeffs, b_coeffs = sedobj.setup_ccm_ab()
                    dust_wavelen = sedobj.wavelen

                sedobj.add_dust(a_coeffs, b_coeffs, a_v=av)

        return dust_wavelen, a_coeffs, b_coeffs

    def apply_redshift(self, sed_list, redshift_list):
        """
        Take the array of SED objects sed_list and apply the arrays of extinction and redshift
        (internalAV and redshift)

        This method does not return anything.  It makes the necessary changes
        to the Seds in SedList in situ.

        @param [in] sed_list is a list of Sed objects

        @param [in] redshift_list is a list of redshift values

        This method will redshift each Sed object in sed_list
        """

        if redshift_list is None:
            return

        for sedobj, redshift in zip(sed_list, redshift_list):
            if sedobj.wavelen is not None and redshift is not None:
                sedobj.redshift_sed(redshift, dimming=self._cosmological_dimming)
                sedobj.name = sedobj.name + "_Z" + "%.2f" % (redshift)

    def flush(self):
        """
        Delete all SEDs stored in this SedList.
        """
        self._initialized = False
        self._sed_list = []
        self._internal_av_list = None
        self._galactic_av_list = None
        self._redshift_list = None

    @property
    def cosmological_dimming(self):
        """
        `bool` determining whether cosmological dimming (the extra
        (1+z)^-1 factor in flux) is applied to Seds when they are
        redshifte by this SedList.
        """
        return self._cosmological_dimming

    @property
    def wavelen_match(self):
        """
        Wavelength grid against which to match Seds stored in this
        SedList.
        """
        return self._wavelen_match

    @property
    def redshift_list(self):
        """
        List of redshifts applied to the Seds stored in this
        SedList.
        """
        return self._redshift_list

    @property
    def internal_av_list(self):
        """
        A(V) due to internal dust applied to the Seds stored in
        this SedList.
        """
        return self._internal_av_list

    @property
    def galactic_av_list(self):
        """
        List of A(V) due to Milky Way dust applied to the Seds
        stored in this SedList
        """
        return self._galactic_av_list
