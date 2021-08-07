import os
import copy
from .Sed import Sed
from rubin_sim.photUtils import getImsimFluxNorm

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

    The method loadSedsFromList allows the user to add Seds to the list
    after the constructor has been called.
    """

    def __init__(self, sedNameList, magNormList,
                 normalizingBandpass=None,
                 specMap=None,
                 fileDir = '',
                 wavelenMatch = None,
                 redshiftList = None,
                 galacticAvList = None,
                 internalAvList = None,
                 cosmologicalDimming = True):

        """
        @param [in] sedNameList is a list of SED file names.

        @param [in] magNormList is a list of magnitude normalizations
        (in the normalizingBandpass) for each of the Seds.

        @param[in] normalizingBandpass is an instantiation of the Bandpass
        class defining the bandpass in which the magNorms from magNormList
        are calculated.  This defaults to the Bandpass().imsimBandpass(),
        which is essentially a delta function at 500 nm.

        @param [in] fileDir is the base directory where the Sed files are stored
        (defaults to current working directory).

        @param [in] specMap is a specMap (defined in sims_utils/../fileMaps.py)
        that maps the names in sedNameList to paths of the files relative to
        fileDir (defaults to None; a defaultSpecMap targeted at
        sims_sed_library is defined in sims_utils)

        @param [in] wavelenMatch is an optional numpy array representing
        the wavelength grid to which all Seds will be re-mapped.

        @param [in] redshiftList is an optional list of redshifts for the Sed

        @param [in] internalAvList is an optional list of A(V) due to internal
        dust (for spectra of galaxies).

        @param [in] galacticAvList is an optional list of A(V) due to
        Milky Way Dust.

        @param [in] cosmologicalDimming is a `bool` indicating whether cosmological
        dimming (the extray (1+z)^-1 factor in flux) should be applied to spectra
        when they are redshifted (defaults to True)

        Note: once wavelenMatch and cosmologicalDimming have been set in
        the constructor, they cannot be un-set.

        Similarly: if you construct a SedList without a galacticAvList,
        internalAvList, or redshiftList, you cannot later add spectra with
        whichever of those features were left out.
        """


        self._initialized = False
        self._spec_map = specMap
        self._wavelen_match = copy.deepcopy(wavelenMatch)
        self._file_dir = fileDir
        self._cosmological_dimming = cosmologicalDimming

        self._normalizing_bandpass = normalizingBandpass

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

        self.loadSedsFromList(sedNameList, magNormList,
                              internalAvList = internalAvList,
                              galacticAvList = galacticAvList,
                              redshiftList = redshiftList)

    def __len__(self):
        return len(self._sed_list)

    def __getitem__(self, index):
        return self._sed_list[index]

    def __iter__(self):
        for val in self._sed_list:
            yield val

    # Handy routines for handling Sed/Bandpass routines with sets of dictionaries.
    def loadSedsFromList(self, sedNameList, magNormList, \
                         internalAvList=None, galacticAvList=None, redshiftList=None):
        """
        Load the Seds specified by sedNameList, applying the specified normalization,
        extinction, and redshift.

        @param [in] sedList is a list of file names containing Seds

        @param [in] magNorm is the magnitude normalization

        @param [in] internalAvList is an optional list of A(V) due to internal
        dust

        @param [in] galacticAvList is an optional list of A(V) due to
        Milky Way dust

        @param [in] redshiftList is an optional list of redshifts for the
        input Sed

        Seds are read in and stored to this object's internal list of Seds.

        Note: if you constructed this SedList object without internalAvList,
        you cannot load Seds with internalAvList now.  Likewise for galacticAvlist
        and redshiftList.
        """

        if not self._initialized:
            if internalAvList is not None:
                self._internal_av_list = copy.deepcopy(list(internalAvList))
            else:
                self._internal_av_list = None

            if galacticAvList is not None:
                self._galactic_av_list = copy.deepcopy(list(galacticAvList))
            else:
                self._galactic_av_list = None

            if redshiftList is not None:
                self._redshift_list = copy.deepcopy(list(redshiftList))
            else:
                self._redshift_list = None

        else:
            if self._internal_av_list is None and internalAvList is not None:
                raise RuntimeError("This SedList does not contain internalAvList")
            elif self._internal_av_list is not None:
                if internalAvList is None:
                    self._internal_av_list += [None] * len(sedNameList)
                else:
                    self._internal_av_list += list(internalAvList)

            if self._galactic_av_list is None and galacticAvList is not None:
                raise RuntimeError("This SedList does not contain galacticAvList")
            elif self._galactic_av_list is not None:
                if galacticAvList is None:
                    self._galactic_av_list += [None] * len(sedNameList)
                else:
                    self._galactic_av_list += list(galacticAvList)

            if self._redshift_list is None and redshiftList is not None:
                raise RuntimeError("This SedList does not contain redshiftList")
            elif self._redshift_list is not None:
                if redshiftList is None:
                    self._redshift_list += [None] * len(sedNameList)
                else:
                    self._redshift_list += list(redshiftList)

        temp_sed_list = []
        for sedName, magNorm in zip(sedNameList, magNormList):
            sed = Sed()

            if sedName != "None":
                if self._spec_map is not None:
                    sed.readSED_flambda(os.path.join(self._file_dir, self._spec_map[sedName]))
                else:
                    sed.readSED_flambda(os.path.join(self._file_dir, sedName))

                if self._normalizing_bandpass is not None:
                    fNorm = sed.calcFluxNorm(magNorm, self._normalizing_bandpass)
                else:
                    fNorm = getImsimFluxNorm(sed, magNorm)

                sed.multiplyFluxNorm(fNorm)

            temp_sed_list.append(sed)


        if internalAvList is not None:
            self._av_int_wavelen, \
            self._a_int, \
            self._b_int = self.applyAv(temp_sed_list, internalAvList,
                                       self._av_int_wavelen, self._a_int, self._b_int)

        if redshiftList is not None:
            self.applyRedshift(temp_sed_list, redshiftList)

        if self._wavelen_match is not None:
            for sedObj in temp_sed_list:
                if sedObj.wavelen is not None:
                    sedObj.resampleSED(wavelen_match=self._wavelen_match)

        if galacticAvList is not None:
            self._av_gal_wavelen, \
            self._a_gal, \
            self._b_gal = self.applyAv(temp_sed_list, galacticAvList,
                                       self._av_gal_wavelen, self._a_gal, self._b_gal)

        self._sed_list += temp_sed_list

        self._initialized = True



    def applyAv(self, sedList, avList, dustWavelen, aCoeffs, bCoeffs):
        """
        Take the array of Sed objects sedList and apply extinction due to dust.

        This method makes the necessary changes to the Seds in SedList in situ.
        It returns the wavelength grid and corresponding dust coefficients so that
        they an be reused on Seds with identical wavelength grids.

        @param [in] sedList is a list of Sed objects

        @param [in] avList is a list of Av extinction values internal to each object

        @param [in] dustWavelen is the wavelength grid corresponding to the
        dust model coefficients.  If this differs from the wavelength grid
        of any of the Seds in sedList, the dust model coefficients will be
        re-generated.

        @param [in] aCoeffs are the 'a' dust model coefficients (see O'Donnell 1994
        ApJ 422 158)

        @param [in] bCoeffs are the 'b' dust model coefficients from O'Donnell.

        @param [out] dustWavelen as generated/used by this method

        @param [out] aCoeffs as generated/used by this method

        @param [out] bCoeffs as generated/used by this method

        aCoeffs and bCoeffs are re-generated as needed
        """

        for sedobj, av in zip(sedList, avList):
            if sedobj.wavelen is not None and av is not None:
                #setupCCM_ab only depends on the wavelen array
                #because this is supposed to be the same for every
                #SED object in sedList, it is only called once for
                #each invocation of applyAv

                if dustWavelen is None or len(sedobj.wavelen)!=len(dustWavelen) \
                or (sedobj.wavelen!=dustWavelen).any():
                    aCoeffs, bCoeffs = sedobj.setupCCM_ab()
                    dustWavelen = sedobj.wavelen

                sedobj.addDust(aCoeffs, bCoeffs, A_v=av)


        return dustWavelen, aCoeffs, bCoeffs


    def applyRedshift(self, sedList, redshiftList):
        """
        Take the array of SED objects sedList and apply the arrays of extinction and redshift
        (internalAV and redshift)

        This method does not return anything.  It makes the necessary changes
        to the Seds in SedList in situ.

        @param [in] sedList is a list of Sed objects

        @param [in] redshiftList is a list of redshift values

        This method will redshift each Sed object in sedList
        """

        if redshiftList is None:
            return

        for sedobj, redshift in zip(sedList, redshiftList):
            if sedobj.wavelen is not None and redshift is not None:
                sedobj.redshiftSED(redshift, dimming=self._cosmological_dimming)
                sedobj.name = sedobj.name + '_Z' + '%.2f' %(redshift)


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
    def cosmologicalDimming(self):
        """
        `bool` determining whether cosmological dimming (the extra
        (1+z)^-1 factor in flux) is applied to Seds when they are
        redshifte by this SedList.
        """
        return self._cosmological_dimming


    @property
    def wavelenMatch(self):
        """
        Wavelength grid against which to match Seds stored in this
        SedList.
        """
        return self._wavelen_match


    @property
    def redshiftList(self):
        """
        List of redshifts applied to the Seds stored in this
        SedList.
        """
        return self._redshift_list


    @property
    def internalAvList(self):
        """
        A(V) due to internal dust applied to the Seds stored in
        this SedList.
        """
        return self._internal_av_list


    @property
    def galacticAvList(self):
        """
        List of A(V) due to Milky Way dust applied to the Seds
        stored in this SedList
        """
        return self._galactic_av_list
