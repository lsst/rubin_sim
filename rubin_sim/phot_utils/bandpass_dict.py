import copy
import numpy
import os
from rubin_sim.data import get_data_dir
from collections import OrderedDict
from .bandpass import Bandpass
from .sed import Sed

__all__ = ["BandpassDict"]


class BandpassDict(object):
    """
    This class will wrap an OrderedDict of Bandpass instantiations.

    Upon instantiation, this class's constructor will resample
    the input Bandpasses to be on the same wavelength grid (defined
    by the first input Bandpass).  The constructor will then calculate
    the 2-D phi_array for quick calculation of magnitudes in all
    Bandpasses simultaneously (see the member methods mag_list_for_sed,
    mag_list_for_sed_list, flux_list_for_sed, flux_list_for_sed_list).

    Note: when re-sampling the wavelength grid, it is assumed that
    the first bandpass is sampled on a uniform grid (i.e. all bandpasses
    are resampled to a grid with wavlen_min, wavelen_max determined by
    the bounds of the first bandpasses grid and with wavelen_step defined
    to be the difference between the 0th and 1st element of the first
    bandpass' wavelength grid).

    The class methods load_bandpasses_from_files and load_total_bandpasses_from_files
    can be used to easily read throughput files in from disk and conver them
    into BandpassDict objects.
    """

    def __init__(self, bandpass_list, bandpass_name_list):
        """
        @param [in] bandpass_list is a list of Bandpass instantiations

        @param [in] bandpass_name_list is a list of tags to be associated
        with those Bandpasses.  These will be used as keys for the BandpassDict.
        """
        self._bandpass_dict = OrderedDict()
        self._wavelen_match = None
        for bandpass_name, bandpass in zip(bandpass_name_list, bandpass_list):
            if bandpass_name in self._bandpass_dict:
                raise RuntimeError(
                    "The bandpass %s occurs twice in your input " % bandpass_name
                    + "to BandpassDict"
                )

            self._bandpass_dict[bandpass_name] = copy.deepcopy(bandpass)
            if self._wavelen_match is None:
                self._wavelen_match = self._bandpass_dict[bandpass_name].wavelen

        dummy_sed = Sed()
        self._phi_array, self._wavelen_step = dummy_sed.setup_phi_array(
            list(self._bandpass_dict.values())
        )

    def __getitem__(self, bandpass):
        return self._bandpass_dict[bandpass]

    def __len__(self):
        return len(self._bandpass_dict)

    def __iter__(self):
        for val in self._bandpass_dict:
            yield val

    def values(self):
        """
        Returns a list of the BandpassDict's values.
        """
        return list(self._bandpass_dict.values())

    def keys(self):
        """
        Returns a list of the BandpassDict's keys.
        """
        return list(self._bandpass_dict.keys())

    @classmethod
    def load_bandpasses_from_files(
        cls,
        bandpass_names=["u", "g", "r", "i", "z", "y"],
        filedir=os.path.join(get_data_dir(), "throughputs", "baseline"),
        bandpass_root="filter_",
        component_list=[
            "detector.dat",
            "m1.dat",
            "m2.dat",
            "m3.dat",
            "lens1.dat",
            "lens2.dat",
            "lens3.dat",
        ],
        atmo_transmission=os.path.join(
            get_data_dir(), "throughputs", "baseline", "atmos_std.dat"
        ),
    ):
        """
        Load bandpass information from files into BandpassDicts.
        This method will separate the bandpasses into contributions due to instrumentations
        and contributions due to the atmosphere.

        @param [in] bandpass_names is a list of strings labeling the bandpasses
        (e.g. ['u', 'g', 'r', 'i', 'z', 'y'])

        @param [in] filedir is a string indicating the name of the directory containing the
        bandpass files

        @param [in] bandpass_root is the root of the names of the files associated with the
        bandpasses.  This method assumes that bandpasses are stored in
        filedir/bandpassRoot_bandpass_names[i].dat

        @param [in] component_list lists the files associated with bandpasses representing
        hardware components shared by all filters
        (defaults to ['detector.dat', 'm1.dat', 'm2.dat', 'm3.dat', 'lens1.dat',
        'lens2.dat', 'lens3.dat'] for LSST).  These files are also expected to be stored in filedir

        @param [in] atmo_transmission is the absolute path to the file representing the
        transmissivity of the atmosphere (defaults to baseline/atmos_std.dat in the LSST
        'throughputs' package).

        @param [out] bandpass_dict is a BandpassDict containing the total
        throughput (instrumentation + atmosphere)

        @param [out] hardware_bandpass_dict is a BandpassDict containing
        the throughput due to instrumentation only
        """

        common_components = []
        for cc in component_list:
            common_components.append(os.path.join(filedir, cc))

        bandpass_list = []
        hardware_bandpass_list = []

        for w in bandpass_names:
            components = common_components + [
                os.path.join(filedir, "%s.dat" % (bandpass_root + w))
            ]
            bandpass_dummy = Bandpass()
            bandpass_dummy.read_throughput_list(components)
            hardware_bandpass_list.append(bandpass_dummy)

            components += [atmo_transmission]
            bandpass_dummy = Bandpass()
            bandpass_dummy.read_throughput_list(components)
            bandpass_list.append(bandpass_dummy)

        bandpass_dict = cls(bandpass_list, bandpass_names)
        hardware_bandpass_dict = cls(hardware_bandpass_list, bandpass_names)

        return bandpass_dict, hardware_bandpass_dict

    @classmethod
    def load_total_bandpasses_from_files(
        cls,
        bandpass_names=["u", "g", "r", "i", "z", "y"],
        bandpass_dir=os.path.join(get_data_dir(), "throughputs", "baseline"),
        bandpass_root="total_",
    ):
        """
        This will take the list of band passes named by bandpass_names and load them into
        a BandpassDict

        The bandpasses loaded this way are total bandpasses: they account for instrumental
        and atmospheric transmission.

        @param [in] bandpass_names is a list of names identifying each filter.
        Defaults to ['u', 'g', 'r', 'i', 'z', 'y']

        @param [in] bandpass_dir is the name of the directory where the bandpass files are stored

        @param [in] bandpass_root contains the first part of the bandpass file name, i.e., it is assumed
        that the bandpasses are stored in files of the type

        bandpass_dir/bandpassRoot_bandpass_names[i].dat

        if we want to load bandpasses for a telescope other than LSST, we would do so
        by altering bandpass_dir and bandpass_root

        @param [out] bandpassDict is a BandpassDict containing the loaded throughputs
        """

        bandpass_list = []

        for w in bandpass_names:
            bandpass_dummy = Bandpass()
            bandpass_dummy.read_throughput(
                os.path.join(bandpass_dir, "%s.dat" % (bandpass_root + w))
            )
            bandpass_list.append(bandpass_dummy)

        return cls(bandpass_list, bandpass_names)

    def _mag_list_for_sed(self, sedobj, indices=None):
        """
        This is a private method which will take an sedobj which has already
        been resampled to self._wavelen_match and calculate the magnitudes
        of that object in each of the bandpasses stored in this Dict.

        The results are returned as a list.
        """

        if sedobj.wavelen is None:
            return [numpy.NaN] * len(self._bandpass_dict)
        else:
            # for some reason, moving this call to flambda_tofnu()
            # to a point earlier in the
            # process results in some SEDs having 'None' for fnu.
            #
            # I looked more carefully at the documentation in Sed.py
            # Any time you update flambda in any way, fnu gets set to 'None'
            # This is to prevent the two arrays from getting out synch
            # (e.g. renormalizing flambda but forgettint to renormalize fnu)
            #
            sedobj.flambda_tofnu()

            if indices is not None:
                output_list = [numpy.NaN] * len(self._bandpass_dict)
                mag_list = sedobj.many_mag_calc(
                    self._phi_array, self._wavelen_step, observed_bandpass_ind=indices
                )
                for i, ix in enumerate(indices):
                    output_list[ix] = mag_list[i]
            else:
                output_list = sedobj.many_mag_calc(self._phi_array, self._wavelen_step)

            return output_list

    def mag_list_for_sed(self, sedobj, indices=None):
        """
        Return a list of magnitudes for a single Sed object.

        @param [in] sedobj is an Sed object.  Its wavelength grid can be arbitrary.  If necessary,
        a copy will be created and resampled onto the wavelength grid of the Bandpasses before
        magnitudes are calculated.  The original Sed will be unchanged.

        @param [in] indices is an optional list of indices indicating which bandpasses to actually
        calculate magnitudes for.  Other magnitudes will be listed as numpy.NaN (i.e. this method will
        return as many magnitudes as were loaded with the loadBandpassesFromFiles methods; it will
        just return numpy.NaN for magnitudes you did not actually ask for)

        @param [out] magList is a list of magnitudes in the bandpasses stored in this BandpassDict
        """

        if sedobj.wavelen is not None:
            # If the Sed's wavelength grid agrees with self._wavelen_match to one part in
            # 10^6, just use the Sed as-is.  Otherwise, copy it and resample it onto
            # self._wavelen_match
            if sedobj._need_resample(wavelen_match=self._wavelen_match):
                dummy_sed = Sed(wavelen=sedobj.wavelen, flambda=sedobj.flambda)
                dummy_sed.resample_sed(force=True, wavelen_match=self._wavelen_match)
            else:
                dummy_sed = sedobj

            return numpy.array(self._mag_list_for_sed(dummy_sed, indices=indices))

        else:
            return numpy.array([numpy.NaN] * len(self._bandpass_dict))

    def mag_dict_for_sed(self, sedobj, indices=None):
        """
        Return an OrderedDict of magnitudes for a single Sed object.

        The OrderedDict will be keyed off of the keys to this BandpassDict

        @param [in] sedobj is an Sed object.  Its wavelength grid can be arbitrary.  If necessary,
        a copy will be created and resampled onto the wavelength grid of the Bandpasses before
        magnitudes are calculated.  The original Sed will be unchanged.

        @param [in] indices is an optional list of indices indicating which bandpasses to actually
        calculate magnitudes for.  Other magnitudes will be listed as numpy.NaN (i.e. this method will
        return as many magnitudes as were loaded with the loadBandpassesFromFiles methods; it will
        just return numpy.NaN for magnitudes you did not actually ask for)

        @param [out] magDict is an OrderedDict of magnitudes in the bandpasses stored in this BandpassDict
        """

        mag_list = self.mag_list_for_sed(sedobj, indices=indices)

        output_dict = OrderedDict()

        for ix, bp in enumerate(self._bandpass_dict.keys()):
            output_dict[bp] = mag_list[ix]

        return output_dict

    def mag_list_for_sed_list(self, sed_list, indices=None):
        """
        Return a 2-D array of magnitudes from a SedList.
        Each row will correspond to a different Sed, each column
        will correspond to a different bandpass, i.e. in the case of

        mag = myBandpassDict.magListForSedList(mySedList)

        mag[0][0] will be the magnitude of the 0th Sed in the 0th bandpass
        mag[0][1] will be the magnitude of the 0th Sed in the 1st bandpass
        mag[1][1] will be the magnitude of the 1st Sed in the 1st bandpass
        etc.

        For maximum efficiency, use the wavelen_match keyword when loading
        SEDs into your SedList and make sure that wavelen_match = myBandpassDict.wavelen_match.
        That way, this method will not have to waste time resampling the Seds
        onto the wavelength grid of the BandpassDict.

        @param [in] sed_list is a SedList containing the Seds
        whose magnitudes are desired.

        @param [in] indices is an optional list of indices indicating which bandpasses to actually
        calculate magnitudes for.  Other magnitudes will be listed as numpy.NaN (i.e. this method will
        return as many magnitudes as were loaded with the loadBandpassesFromFiles methods; it will
        just return numpy.NaN for magnitudes you did not actually ask for)

        @param [out] output_list is a 2-D numpy array containing the magnitudes
        of each Sed (the rows) in each bandpass contained in this BandpassDict
        (the columns)
        """

        one_at_a_time = False
        if sed_list.wavelen_match is None:
            one_at_a_time = True
        elif sed_list[0]._need_resample(wavelen_match=self._wavelen_match):
            one_at_a_time = True

        output_list = []
        if one_at_a_time:
            for sed_obj in sed_list:
                sub_list = self.mag_list_for_sed(sed_obj, indices=indices)
                output_list.append(sub_list)
        else:
            # the difference between this block and the block above is that the block
            # above performs the additional check of making sure that sed_obj.wavelen
            # is equivalent to self._wavelen_match
            for sed_obj in sed_list:
                sub_list = self._mag_list_for_sed(sed_obj, indices=indices)
                output_list.append(sub_list)

        return numpy.array(output_list)

    def mag_array_for_sed_list(self, sed_list, indices=None):
        """
        Return a dtyped numpy array of magnitudes from a SedList.
        The array will be keyed to the keys of this BandpassDict,
        i.e. in the case of

        mag = myBandpassDict.magArrayForSedList(mySedList)

        mag['u'][0] will be the magnitude of the 0th Sed in the 'u' bandpass
        mag['u'][1] will be the magnitude of the 1st Sed in the 'u' bandpass
        mag['z'] will be a numpy array of every Sed's magnitude in the 'z' bandpass
        etc.

        For maximum efficiency, use the wavelen_match keyword when loading
        SEDs into your SedList and make sure that wavelen_match = myBandpassDict.wavelen_match.
        That way, this method will not have to waste time resampling the Seds
        onto the wavelength grid of the BandpassDict.

        @param [in] sed_list is a SedList containing the Seds
        whose magnitudes are desired.

        @param [in] indices is an optional list of indices indicating which bandpasses to actually
        calculate magnitudes for.  Other magnitudes will be listed as numpy.NaN (i.e. this method will
        return as many magnitudes as were loaded with the loadBandpassesFromFiles methods; it will
        just return numpy.NaN for magnitudes you did not actually ask for)

        @param [out] output_array is a dtyped numpy array of magnitudes (see above).
        """

        mag_list = self.mag_list_for_sed_list(sed_list, indices=indices)

        dtype = numpy.dtype([(bp, float) for bp in self._bandpass_dict.keys()])

        output_array = numpy.array([tuple(row) for row in mag_list], dtype=dtype)

        return output_array

    def _flux_list_for_sed(self, sedobj, indices=None):
        """
        This is a private method which will take an sedobj which has already
        been resampled to self._wavelen_match and calculate the fluxes
        of that object in each of the bandpasses stored in this Dict.

        The results are returned as a list.
        """

        if sedobj.wavelen is None:
            return [numpy.NaN] * len(self._bandpass_dict)
        else:
            # for some reason, moving this call to flambda_tofnu()
            # to a point earlier in the
            # process results in some SEDs having 'None' for fnu.
            #
            # I looked more carefully at the documentation in Sed.py
            # Any time you update flambda in any way, fnu gets set to 'None'
            # This is to prevent the two arrays from getting out synch
            # (e.g. renormalizing flambda but forgettint to renormalize fnu)
            #
            sedobj.flambda_tofnu()

            if indices is not None:
                output_list = [numpy.NaN] * len(self._bandpass_dict)
                mag_list = sedobj.many_flux_calc(
                    self._phi_array, self._wavelen_step, observed_bandpass_ind=indices
                )
                for i, ix in enumerate(indices):
                    output_list[ix] = mag_list[i]
            else:
                output_list = sedobj.many_flux_calc(self._phi_array, self._wavelen_step)

            return output_list

    def flux_list_for_sed(self, sedobj, indices=None):
        """
        Return a list of Fluxes for a single Sed object.

        @param [in] sedobj is an Sed object.   Its wavelength grid can be arbitrary. If necessary,
        a copy will be created and resampled onto the wavelength grid of the Bandpasses before
        fluxes are calculated.  The original Sed will be unchanged.

        @param [in] indices is an optional list of indices indicating which bandpasses to actually
        calculate fluxes for.  Other fluxes will be listed as numpy.NaN (i.e. this method will
        return as many fluxes as were loaded with the loadBandpassesFromFiles methods; it will
        just return numpy.NaN for fluxes you did not actually ask for)

        @param [out] fluxList is a list of fluxes in the bandpasses stored in this BandpassDict

        Note on units: Fluxes calculated this way will be the flux density integrated over the
        weighted response curve of the bandpass.  See equaiton 2.1 of the LSST Science Book

        http://www.lsst.org/scientists/scibook
        """

        if sedobj.wavelen is not None:
            # If the Sed's wavelength grid agrees with self._wavelen_match to one part in
            # 10^6, just use the Sed as-is.  Otherwise, copy it and resample it onto
            # self._wavelen_match
            if sedobj._need_resample(wavelen_match=self._wavelen_match):
                dummy_sed = Sed(wavelen=sedobj.wavelen, flambda=sedobj.flambda)
                dummy_sed.resample_sed(force=True, wavelen_match=self._wavelen_match)
            else:
                dummy_sed = sedobj

            return numpy.array(self._flux_list_for_sed(dummy_sed, indices=indices))

        else:
            return numpy.array([numpy.NaN] * len(self._bandpass_dict))

    def flux_dict_for_sed(self, sedobj, indices=None):
        """
        Return an OrderedDict of fluxes for a single Sed object.

        The OrderedDict will be keyed off of the keys for this BandpassDict

        @param [in] sedobj is an Sed object.   Its wavelength grid can be arbitrary. If necessary,
        a copy will be created and resampled onto the wavelength grid of the Bandpasses before
        fluxes are calculated.  The original Sed will be unchanged.

        @param [in] indices is an optional list of indices indicating which bandpasses to actually
        calculate fluxes for.  Other fluxes will be listed as numpy.NaN (i.e. this method will
        return as many fluxes as were loaded with the loadBandpassesFromFiles methods; it will
        just return numpy.NaN for fluxes you did not actually ask for)

        @param [out] flux_list is a list of fluxes in the bandpasses stored in this BandpassDict

        Note on units: Fluxes calculated this way will be the flux density integrated over the
        weighted response curve of the bandpass.  See equaiton 2.1 of the LSST Science Book

        http://www.lsst.org/scientists/scibook
        """
        flux_list = self.flux_list_for_sed(sedobj, indices=indices)

        output_dict = OrderedDict()

        for ix, bp in enumerate(self._bandpass_dict.keys()):
            output_dict[bp] = flux_list[ix]

        return output_dict

    def flux_list_for_sed_list(self, sed_list, indices=None):
        """
        Return a 2-D array of fluxes from a SedList.
        Each row will correspond to a different Sed, each column
        will correspond to a different bandpass, i.e. in the case of

        flux = myBandpassDict.fluxListForSedList(mySedList)

        flux[0][0] will be the flux of the 0th Sed in the 0th bandpass
        flux[0][1] will be the flux of the 0th Sed in the 1st bandpass
        flux[1][1] will be the flux of the 1st Sed in the 1st bandpass
        etc.

        For maximum efficiency, use the wavelen_match keyword when loading
        SEDs into your SedList and make sure that wavelen_match = myBandpassDict.wavelen_match.
        That way, this method will not have to waste time resampling the Seds
        onto the wavelength grid of the BandpassDict.

        @param [in] sed_list is a SedList containing the Seds
        whose fluxes are desired.

        @param [in] indices is an optional list of indices indicating which bandpasses to actually
        calculate fluxes for.  Other fluxes will be listed as numpy.NaN (i.e. this method will
        return as many fluxes as were loaded with the loadBandpassesFromFiles methods; it will
        just return numpy.NaN for fluxes you did not actually ask for)

        @param [out] output_list is a 2-D numpy array containing the fluxes
        of each Sed (the rows) in each bandpass contained in this BandpassDict
        (the columns)

        Note on units: Fluxes calculated this way will be the flux density integrated over the
        weighted response curve of the bandpass.  See equaiton 2.1 of the LSST Science Book

        http://www.lsst.org/scientists/scibook
        """

        one_at_a_time = False
        if sed_list.wavelen_match is None:
            one_at_a_time = True
        elif sed_list[0]._need_resample(wavelen_match=self._wavelen_match):
            one_at_a_time = True

        output_list = []
        if one_at_a_time:
            for sed_obj in sed_list:
                sub_list = self.flux_list_for_sed(sed_obj, indices=indices)
                output_list.append(sub_list)
        else:
            # the difference between this block and the block above is that the block
            # above performs the additional check of making sure that sed_obj.wavelen
            # is equivalent to self._wavelen_match
            for sed_obj in sed_list:
                sub_list = self._flux_list_for_sed(sed_obj, indices=indices)
                output_list.append(sub_list)

        return numpy.array(output_list)

    def flux_array_for_sed_list(self, sed_list, indices=None):
        """
        Return a dtyped numpy array of fluxes from a SedList.
        The array will be keyed to the keys of this BandpassDict,
        i.e. in the case of

        flux = myBandpassDict.fluxArrayForSedList(mySedList)

        flux['u'][0] will be the flux of the 0th Sed in the 'u' bandpass
        flux['u'][1] will be the flux of the 1st Sed in the 'u' bandpass
        flux['z'] will be a numpy array of every Sed's flux in the 'z' bandpass
        etc.

        For maximum efficiency, use the wavelen_match keyword when loading
        SEDs into your SedList and make sure that wavelen_match = myBandpassDict.wavelen_match.
        That way, this method will not have to waste time resampling the Seds
        onto the wavelength grid of the BandpassDict.

        @param [in] sed_list is a SedList containing the Seds
        whose fluxes are desired.

        @param [in] indices is an optional list of indices indicating which bandpasses to actually
        calculate fluxes for.  Other fluxes will be listed as numpy.NaN (i.e. this method will
        return as many fluxes as were loaded with the loadBandpassesFromFiles methods; it will
        just return numpy.NaN for fluxes you did not actually ask for)

        @param [out] output_list is a 2-D numpy array containing the fluxes
        of each Sed (the rows) in each bandpass contained in this BandpassDict
        (the columns)

        Note on units: Fluxes calculated this way will be the flux density integrated over the
        weighted response curve of the bandpass.  See equaiton 2.1 of the LSST Science Book

        http://www.lsst.org/scientists/scibook
        """

        flux_list = self.flux_list_for_sed_list(sed_list, indices=indices)

        dtype = numpy.dtype([(bp, float) for bp in self._bandpass_dict.keys()])

        output_array = numpy.array([tuple(row) for row in flux_list], dtype=dtype)

        return output_array

    @property
    def phi_array(self):
        """
        A 2-D numpy array storing the values of phi (see eqn 2.3 of the science
        book) for all of the bandpasses in this dict.
        """
        return self._phi_array

    @property
    def wavelen_step(self):
        """
        The step size of the wavelength grid for all of the bandpasses
        stored in this dict.
        """
        return self._wavelen_step

    @property
    def wavelen_match(self):
        """
        The wavelength grid (in nm) on which all of the bandpass
        throughputs have been sampled.
        """
        return self._wavelen_match
