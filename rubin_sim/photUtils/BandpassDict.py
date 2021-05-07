import copy
import numpy
import os
from rubin_sim.data import get_data_dir
from collections import OrderedDict
from .Bandpass import Bandpass
from .Sed import Sed

__all__ = ["BandpassDict"]

class BandpassDict(object):
    """
    This class will wrap an OrderedDict of Bandpass instantiations.

    Upon instantiation, this class's constructor will resample
    the input Bandpasses to be on the same wavelength grid (defined
    by the first input Bandpass).  The constructor will then calculate
    the 2-D phiArray for quick calculation of magnitudes in all
    Bandpasses simultaneously (see the member methods magListForSed,
    magListForSedList, fluxListForSed, fluxListForSedList).

    Note: when re-sampling the wavelength grid, it is assumed that
    the first bandpass is sampled on a uniform grid (i.e. all bandpasses
    are resampled to a grid with wavlen_min, wavelen_max determined by
    the bounds of the first bandpasses grid and with wavelen_step defined
    to be the difference between the 0th and 1st element of the first
    bandpass' wavelength grid).

    The class methods loadBandpassesFromFiles and loadTotalBandpassesFromFiles
    can be used to easily read throughput files in from disk and conver them
    into BandpassDict objects.
    """

    def __init__(self, bandpassList, bandpassNameList):
        """
        @param [in] bandpassList is a list of Bandpass instantiations

        @param [in] bandpassNameList is a list of tags to be associated
        with those Bandpasses.  These will be used as keys for the BandpassDict.
        """
        self._bandpassDict = OrderedDict()
        self._wavelen_match = None
        for bandpassName, bandpass in zip(bandpassNameList, bandpassList):

            if bandpassName in self._bandpassDict:
                raise RuntimeError("The bandpass %s occurs twice in your input " % bandpassName \
                                   + "to BandpassDict")

            self._bandpassDict[bandpassName] = copy.deepcopy(bandpass)
            if self._wavelen_match is None:
                self._wavelen_match = self._bandpassDict[bandpassName].wavelen

        dummySed = Sed()
        self._phiArray, self._wavelenStep = dummySed.setupPhiArray(list(self._bandpassDict.values()))

    def __getitem__(self, bandpass):
        return self._bandpassDict[bandpass]

    def __len__(self):
        return len(self._bandpassDict)

    def __iter__(self):
        for val in self._bandpassDict:
            yield val

    def values(self):
        """
        Returns a list of the BandpassDict's values.
        """
        return list(self._bandpassDict.values())

    def keys(self):
        """
        Returns a list of the BandpassDict's keys.
        """
        return list(self._bandpassDict.keys())

    @classmethod
    def loadBandpassesFromFiles(cls,
                                bandpassNames=['u', 'g', 'r', 'i', 'z', 'y'],
                                filedir=os.path.join(get_data_dir(), 'throughputs', 'baseline'),
                                bandpassRoot='filter_',
                                componentList=['detector.dat', 'm1.dat', 'm2.dat', 'm3.dat',
                                               'lens1.dat', 'lens2.dat', 'lens3.dat'],
                                atmoTransmission=os.path.join(get_data_dir(), 'throughputs',
                                                              'baseline', 'atmos_std.dat')):
        """
        Load bandpass information from files into BandpassDicts.
        This method will separate the bandpasses into contributions due to instrumentations
        and contributions due to the atmosphere.

        @param [in] bandpassNames is a list of strings labeling the bandpasses
        (e.g. ['u', 'g', 'r', 'i', 'z', 'y'])

        @param [in] filedir is a string indicating the name of the directory containing the
        bandpass files

        @param [in] bandpassRoot is the root of the names of the files associated with the
        bandpasses.  This method assumes that bandpasses are stored in
        filedir/bandpassRoot_bandpassNames[i].dat

        @param [in] componentList lists the files associated with bandpasses representing
        hardware components shared by all filters
        (defaults to ['detector.dat', 'm1.dat', 'm2.dat', 'm3.dat', 'lens1.dat',
                      'lense2.dat', 'lenst3.dat']
        for LSST).  These files are also expected to be stored in filedir

        @param [in] atmoTransmission is the absolute path to the file representing the
        transmissivity of the atmosphere (defaults to baseline/atmos_std.dat in the LSST
        'throughputs' package).

        @param [out] bandpassDict is a BandpassDict containing the total
        throughput (instrumentation + atmosphere)

        @param [out] hardwareBandpassDict is a BandpassDict containing
        the throughput due to instrumentation only
        """

        commonComponents = []
        for cc in componentList:
            commonComponents.append(os.path.join(filedir, cc))

        bandpassList = []
        hardwareBandpassList = []

        for w in bandpassNames:
            components = commonComponents + [os.path.join(filedir, "%s.dat" % (bandpassRoot +w))]
            bandpassDummy = Bandpass()
            bandpassDummy.readThroughputList(components)
            hardwareBandpassList.append(bandpassDummy)

            components += [atmoTransmission]
            bandpassDummy = Bandpass()
            bandpassDummy.readThroughputList(components)
            bandpassList.append(bandpassDummy)

        bandpassDict = cls(bandpassList, bandpassNames)
        hardwareBandpassDict = cls(hardwareBandpassList, bandpassNames)

        return bandpassDict, hardwareBandpassDict

    @classmethod
    def loadTotalBandpassesFromFiles(cls,
                                     bandpassNames=['u', 'g', 'r', 'i', 'z', 'y'],
                                     bandpassDir=os.path.join(get_data_dir(), 'throughputs', 'baseline'),
                                     bandpassRoot='total_'):
        """
        This will take the list of band passes named by bandpassNames and load them into
        a BandpassDict

        The bandpasses loaded this way are total bandpasses: they account for instrumental
        and atmospheric transmission.

        @param [in] bandpassNames is a list of names identifying each filter.
        Defaults to ['u', 'g', 'r', 'i', 'z', 'y']

        @param [in] bandpassDir is the name of the directory where the bandpass files are stored

        @param [in] bandpassRoot contains the first part of the bandpass file name, i.e., it is assumed
        that the bandpasses are stored in files of the type

        bandpassDir/bandpassRoot_bandpassNames[i].dat

        if we want to load bandpasses for a telescope other than LSST, we would do so
        by altering bandpassDir and bandpassRoot

        @param [out] bandpassDict is a BandpassDict containing the loaded throughputs
        """

        bandpassList = []

        for w in bandpassNames:
            bandpassDummy = Bandpass()
            bandpassDummy.readThroughput(os.path.join(bandpassDir, "%s.dat" % (bandpassRoot + w)))
            bandpassList.append(bandpassDummy)

        return cls(bandpassList, bandpassNames)

    def _magListForSed(self, sedobj, indices=None):
        """
        This is a private method which will take an sedobj which has already
        been resampled to self._wavelen_match and calculate the magnitudes
        of that object in each of the bandpasses stored in this Dict.

        The results are returned as a list.
        """

        if sedobj.wavelen is None:
            return [numpy.NaN]*len(self._bandpassDict)
        else:

            #for some reason, moving this call to flambdaTofnu()
            #to a point earlier in the
            #process results in some SEDs having 'None' for fnu.
            #
            #I looked more carefully at the documentation in Sed.py
            #Any time you update flambda in any way, fnu gets set to 'None'
            #This is to prevent the two arrays from getting out synch
            #(e.g. renormalizing flambda but forgettint to renormalize fnu)
            #
            sedobj.flambdaTofnu()

            if indices is not None:
                outputList = [numpy.NaN] * len(self._bandpassDict)
                magList = sedobj.manyMagCalc(self._phiArray, self._wavelenStep, observedBandpassInd=indices)
                for i, ix in enumerate(indices):
                    outputList[ix] = magList[i]
            else:
                outputList = sedobj.manyMagCalc(self._phiArray, self._wavelenStep)

            return outputList

    def magListForSed(self, sedobj, indices=None):
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
            if sedobj._needResample(wavelen_match=self._wavelen_match):
                dummySed = Sed(wavelen=sedobj.wavelen, flambda=sedobj.flambda)
                dummySed.resampleSED(force=True, wavelen_match=self._wavelen_match)
            else:
                dummySed = sedobj

            return numpy.array(self._magListForSed(dummySed, indices=indices))

        else:
            return numpy.array([numpy.NaN]*len(self._bandpassDict))

    def magDictForSed(self, sedobj, indices=None):
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

        magList = self.magListForSed(sedobj, indices=indices)

        outputDict = OrderedDict()

        for ix, bp in enumerate(self._bandpassDict.keys()):
            outputDict[bp] = magList[ix]

        return outputDict

    def magListForSedList(self, sedList, indices=None):
        """
        Return a 2-D array of magnitudes from a SedList.
        Each row will correspond to a different Sed, each column
        will correspond to a different bandpass, i.e. in the case of

        mag = myBandpassDict.magListForSedList(mySedList)

        mag[0][0] will be the magnitude of the 0th Sed in the 0th bandpass
        mag[0][1] will be the magnitude of the 0th Sed in the 1st bandpass
        mag[1][1] will be the magnitude of the 1st Sed in the 1st bandpass
        etc.

        For maximum efficiency, use the wavelenMatch keyword when loading
        SEDs into your SedList and make sure that wavelenMatch = myBandpassDict.wavelenMatch.
        That way, this method will not have to waste time resampling the Seds
        onto the wavelength grid of the BandpassDict.

        @param [in] sedList is a SedList containing the Seds
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
        if sedList.wavelenMatch is None:
            one_at_a_time = True
        elif sedList[0]._needResample(wavelen_match=self._wavelen_match):
            one_at_a_time = True

        output_list = []
        if one_at_a_time:
            for sed_obj in sedList:
                sub_list = self.magListForSed(sed_obj, indices=indices)
                output_list.append(sub_list)
        else:
            # the difference between this block and the block above is that the block
            # above performs the additional check of making sure that sed_obj.wavelen
            # is equivalent to self._wavelen_match
            for sed_obj in sedList:
                sub_list = self._magListForSed(sed_obj, indices=indices)
                output_list.append(sub_list)

        return numpy.array(output_list)

    def magArrayForSedList(self, sedList, indices=None):
        """
        Return a dtyped numpy array of magnitudes from a SedList.
        The array will be keyed to the keys of this BandpassDict,
        i.e. in the case of

        mag = myBandpassDict.magArrayForSedList(mySedList)

        mag['u'][0] will be the magnitude of the 0th Sed in the 'u' bandpass
        mag['u'][1] will be the magnitude of the 1st Sed in the 'u' bandpass
        mag['z'] will be a numpy array of every Sed's magnitude in the 'z' bandpass
        etc.

        For maximum efficiency, use the wavelenMatch keyword when loading
        SEDs into your SedList and make sure that wavelenMatch = myBandpassDict.wavelenMatch.
        That way, this method will not have to waste time resampling the Seds
        onto the wavelength grid of the BandpassDict.

        @param [in] sedList is a SedList containing the Seds
        whose magnitudes are desired.

        @param [in] indices is an optional list of indices indicating which bandpasses to actually
        calculate magnitudes for.  Other magnitudes will be listed as numpy.NaN (i.e. this method will
        return as many magnitudes as were loaded with the loadBandpassesFromFiles methods; it will
        just return numpy.NaN for magnitudes you did not actually ask for)

        @param [out] output_array is a dtyped numpy array of magnitudes (see above).
        """

        magList = self.magListForSedList(sedList, indices=indices)

        dtype = numpy.dtype([(bp, float) for bp in self._bandpassDict.keys()])

        outputArray = numpy.array([tuple(row) for row in magList], dtype=dtype)

        return outputArray

    def _fluxListForSed(self, sedobj, indices=None):
        """
        This is a private method which will take an sedobj which has already
        been resampled to self._wavelen_match and calculate the fluxes
        of that object in each of the bandpasses stored in this Dict.

        The results are returned as a list.
        """

        if sedobj.wavelen is None:
            return [numpy.NaN]*len(self._bandpassDict)
        else:

            #for some reason, moving this call to flambdaTofnu()
            #to a point earlier in the
            #process results in some SEDs having 'None' for fnu.
            #
            #I looked more carefully at the documentation in Sed.py
            #Any time you update flambda in any way, fnu gets set to 'None'
            #This is to prevent the two arrays from getting out synch
            #(e.g. renormalizing flambda but forgettint to renormalize fnu)
            #
            sedobj.flambdaTofnu()

            if indices is not None:
                outputList = [numpy.NaN] * len(self._bandpassDict)
                magList = sedobj.manyFluxCalc(self._phiArray, self._wavelenStep, observedBandpassInd=indices)
                for i, ix in enumerate(indices):
                    outputList[ix] = magList[i]
            else:
                outputList = sedobj.manyFluxCalc(self._phiArray, self._wavelenStep)

            return outputList

    def fluxListForSed(self, sedobj, indices=None):
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
            if sedobj._needResample(wavelen_match=self._wavelen_match):
                dummySed = Sed(wavelen=sedobj.wavelen, flambda=sedobj.flambda)
                dummySed.resampleSED(force=True, wavelen_match=self._wavelen_match)
            else:
                dummySed = sedobj

            return numpy.array(self._fluxListForSed(dummySed, indices=indices))

        else:
            return numpy.array([numpy.NaN]*len(self._bandpassDict))

    def fluxDictForSed(self, sedobj, indices=None):
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

        @param [out] fluxList is a list of fluxes in the bandpasses stored in this BandpassDict

        Note on units: Fluxes calculated this way will be the flux density integrated over the
        weighted response curve of the bandpass.  See equaiton 2.1 of the LSST Science Book

        http://www.lsst.org/scientists/scibook
        """
        fluxList = self.fluxListForSed(sedobj, indices=indices)

        outputDict = OrderedDict()

        for ix, bp in enumerate(self._bandpassDict.keys()):
            outputDict[bp] = fluxList[ix]

        return outputDict

    def fluxListForSedList(self, sedList, indices=None):
        """
        Return a 2-D array of fluxes from a SedList.
        Each row will correspond to a different Sed, each column
        will correspond to a different bandpass, i.e. in the case of

        flux = myBandpassDict.fluxListForSedList(mySedList)

        flux[0][0] will be the flux of the 0th Sed in the 0th bandpass
        flux[0][1] will be the flux of the 0th Sed in the 1st bandpass
        flux[1][1] will be the flux of the 1st Sed in the 1st bandpass
        etc.

        For maximum efficiency, use the wavelenMatch keyword when loading
        SEDs into your SedList and make sure that wavelenMatch = myBandpassDict.wavelenMatch.
        That way, this method will not have to waste time resampling the Seds
        onto the wavelength grid of the BandpassDict.

        @param [in] sedList is a SedList containing the Seds
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
        if sedList.wavelenMatch is None:
            one_at_a_time = True
        elif sedList[0]._needResample(wavelen_match=self._wavelen_match):
            one_at_a_time = True

        output_list = []
        if one_at_a_time:
            for sed_obj in sedList:
                sub_list = self.fluxListForSed(sed_obj, indices=indices)
                output_list.append(sub_list)
        else:
            # the difference between this block and the block above is that the block
            # above performs the additional check of making sure that sed_obj.wavelen
            # is equivalent to self._wavelen_match
            for sed_obj in sedList:
                sub_list = self._fluxListForSed(sed_obj, indices=indices)
                output_list.append(sub_list)

        return numpy.array(output_list)

    def fluxArrayForSedList(self, sedList, indices=None):
        """
        Return a dtyped numpy array of fluxes from a SedList.
        The array will be keyed to the keys of this BandpassDict,
        i.e. in the case of

        flux = myBandpassDict.fluxArrayForSedList(mySedList)

        flux['u'][0] will be the flux of the 0th Sed in the 'u' bandpass
        flux['u'][1] will be the flux of the 1st Sed in the 'u' bandpass
        flux['z'] will be a numpy array of every Sed's flux in the 'z' bandpass
        etc.

        For maximum efficiency, use the wavelenMatch keyword when loading
        SEDs into your SedList and make sure that wavelenMatch = myBandpassDict.wavelenMatch.
        That way, this method will not have to waste time resampling the Seds
        onto the wavelength grid of the BandpassDict.

        @param [in] sedList is a SedList containing the Seds
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

        fluxList = self.fluxListForSedList(sedList, indices=indices)

        dtype = numpy.dtype([(bp, float) for bp in self._bandpassDict.keys()])

        outputArray = numpy.array([tuple(row) for row in fluxList], dtype=dtype)

        return outputArray

    @property
    def phiArray(self):
        """
        A 2-D numpy array storing the values of phi (see eqn 2.3 of the science
        book) for all of the bandpasses in this dict.
        """
        return self._phiArray

    @property
    def wavelenStep(self):
        """
        The step size of the wavelength grid for all of the bandpasses
        stored in this dict.
        """
        return self._wavelenStep

    @property
    def wavelenMatch(self):
        """
        The wavelength grid (in nm) on which all of the bandpass
        throughputs have been sampled.
        """
        return self._wavelen_match
