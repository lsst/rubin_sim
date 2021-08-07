import numpy as np
import numbers
from .SpatialBounds import SpatialBounds
from rubin_sim.utils import ModifiedJulianDate
from rubin_sim.utils import Site

__all__ = ["ObservationMetaData"]


class ObservationMetaData(object):
    """Observation Metadata

    This class contains any metadata for a query which is associated with
    a particular telescope pointing, including bounds in RA and DEC, and
    the time of the observation.

    **Parameters**

        All parameters are optional.  It is possible to instantiate an
        ObservationMetaData that is empty of data.

        * pointing[RA,Dec] float
          The coordinates of the pointing (in degrees; in the International
          Celestial Reference System)

        * boundType characterizes the shape of the field of view.  Current options
          are 'box, and 'circle'

        * boundLength is the characteristic length scale of the field of view in degrees.

          If boundType is 'box', boundLength can be a float(in which case boundLength is
          half the length of the side of each box) or boundLength can be a numpy array
          in which case the first argument is half the width of the RA side of the box
          and the second argument is half the Dec side of the box.

          If boundType is 'circle,' this will be the radius of the circle.

          The bound will be centered on the point (pointingRA, pointingDec), however,
          because objects are stored at their mean RA, Dec in the LSST databases
          (i.e. they are stored at values of RA, Dec which neglect proper motion), the
          bounds applied to database queries will be made slightly larger so that queries
          can be reasonably expected to return all of the objects within the desired field
          of view once those corrections have been applied.

        * mjd :
          Either a float (in which case, it will be assumed to be in International
          Atomic Time), or an instantiation of the ModifiedJulianDate class representing
          the date of the observation

        * bandpassName : a char (e.g. 'u') or list (e.g. ['u', 'g', 'z'])
          denoting the bandpasses used for this particular observation

        * site: an instantiation of the rubin_sim.utils.Site class characterizing
          the site of the observatory.

        * m5: float or list
          this should be the 5-sigma limiting magnitude in the bandpass or
          bandpasses specified in bandpassName.  Ultimately, m5 will be stored
          in a dict keyed to the bandpassName (or Names) you passed in, i.e.
          you will be able to access m5 from outside of this class using, for
          example:

          myObservationMetaData.m5['u']

        * skyBrightness: float the magnitude of the sky in the
          filter specified by bandpassName

        * seeing float or list
          Analogous to m5, corresponds to the seeing in arcseconds in the bandpasses in
          bandpassName

        * rotSkyPos float
          The orientation of the telescope in degrees.
          This is used by the Astrometry mixins in sims_coordUtils.

          The convention for rotSkyPos is as follows:

          rotSkyPos = 0 means north is in the +y direction on the focal plane and east is +x

          rotSkyPos = 90 means north is +x and east is -y

          rotSkyPos = -90 means north is -x and east is +y

          rotSkyPos = 180 means north is -y and east is -x

          This should be consistent with PhoSim conventions.

    **Examples**::
        >>> data = ObservationMetaData(boundType='box', pointingRA=5.0, pointingDec=15.0,
                    boundLength=5.0)

    """
    def __init__(self, boundType=None, boundLength=None,
                 mjd=None, pointingRA=None, pointingDec=None, rotSkyPos=None,
                 bandpassName=None, site=Site(name='LSST'), m5=None, skyBrightness=None,
                 seeing=None):

        self._bounds = None
        self._boundType = boundType
        self._bandpass = bandpassName
        self._skyBrightness = skyBrightness
        self._site = site
        self._OpsimMetaData = None

        if mjd is not None:
            if isinstance(mjd, numbers.Number):
                self._mjd = ModifiedJulianDate(TAI=mjd)
            elif isinstance(mjd, ModifiedJulianDate):
                self._mjd = mjd
            else:
                raise RuntimeError("You must pass either a float or a ModifiedJulianDate "
                                   "as the kwarg mjd to ObservationMetaData")
        else:
            self._mjd = None

        if rotSkyPos is not None:
            self._rotSkyPos = np.radians(rotSkyPos)
        else:
            self._rotSkyPos = None

        if pointingRA is not None:
            self._pointingRA = np.radians(pointingRA)
        else:
            self._pointingRA = None

        if pointingDec is not None:
            self._pointingDec = np.radians(pointingDec)
        else:
            self._pointingDec = None

        if boundLength is not None:
            self._boundLength = np.radians(boundLength)
        else:
            self._boundLength = None

        self._m5 = self._assignDictKeyedToBandpass(m5, 'm5')

        self._seeing = self._assignDictKeyedToBandpass(seeing, 'seeing')

        if self._bounds is None:
            self._buildBounds()

    @property
    def summary(self):
        mydict = {}
        mydict['site'] = self.site

        mydict['boundType'] = self.boundType
        mydict['boundLength'] = self.boundLength
        mydict['pointingRA'] = self.pointingRA
        mydict['pointingDec'] = self.pointingDec
        mydict['rotSkyPos'] = self.rotSkyPos

        if self.mjd is None:
            mydict['mjd'] = None
        else:
            mydict['mjd'] = self.mjd.TAI

        mydict['bandpass'] = self.bandpass
        mydict['skyBrightness'] = self.skyBrightness
        # mydict['m5'] = self.m5

        mydict['OpsimMetaData'] = self._OpsimMetaData

        return mydict

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):

        if self.bounds != other.bounds:
            return False

        if self.pointingRA != other.pointingRA:
            return False

        if self.pointingDec != other.pointingDec:
            return False

        if self.rotSkyPos != other.rotSkyPos:
            return False

        if self.bandpass != other.bandpass:
            return False

        if self.seeing != other.seeing:
            return False

        if self.m5 != other.m5:
            return False

        if self.site != other.site:
            return False

        if self.mjd != other.mjd:
            return False

        if self.skyBrightness != other.skyBrightness:
            return False

        if self.OpsimMetaData != other.OpsimMetaData:
            return False

        return True

    def _assignDictKeyedToBandpass(self, inputValue, inputName):
        """
        This method sets up a dict of either m5 or seeing values (or any other quantity
        keyed to bandpassName).  It reads in a list of values and associates them with
        the list of bandpass names in self._bandpass.

        Note: this method assumes that self._bandpass has already been set.
        It will raise an exception of self._bandpass is None.

        @param [in] inputValue is a single value or list of m5/seeing/etc. corresponding to
        the bandpasses stored in self._bandpass

        @param [in] inputName is the name of the Parameter stored in inputValue
        (for constructing helpful error message)

        @param [out] returns a dict of inputValue values keed to self._bandpass
        """

        if inputValue is None:
            return None
        else:
            bandpassIsList = False
            inputIsList = False

            if self._bandpass is None:
                raise RuntimeError('You cannot set %s if you have not set ' % inputName +
                                   'bandpass in ObservationMetaData')

            if hasattr(self._bandpass, '__iter__') and not isinstance(self._bandpass, str):
                bandpassIsList = True

            if hasattr(inputValue, '__iter__') and not isinstance(inputValue, str):
                inputIsList = True

            if bandpassIsList and not inputIsList:
                raise RuntimeError('You passed a list of bandpass names' +
                                   'but did not pass a list of %s to ObservationMetaData' % inputName)

            if inputIsList and not bandpassIsList:
                raise RuntimeError('You passed a list of %s ' % inputName +
                                   'but did not pass a list of bandpass names to ObservationMetaData')

            if inputIsList:
                if len(inputValue) != len(self._bandpass):
                    raise RuntimeError('The list of %s you passed to ObservationMetaData ' % inputName +
                                       'has a different length than the list of bandpass names you passed')

            # now build the dict
            if bandpassIsList:
                if len(inputValue) != len(self._bandpass):
                    raise RuntimeError('In ObservationMetaData you tried to assign bandpass ' +
                                       'and %s with lists of different length' % inputName)

                outputDict = {}
                for b, m in zip(self._bandpass, inputValue):
                    outputDict[b] = m
            else:
                outputDict = {self._bandpass: inputValue}

            return outputDict

    def _buildBounds(self):
        """
        Set up the member variable self._bounds.

        If self._boundType, self._boundLength, self._pointingRA, or
        self._pointingDec are None, nothing will happen.
        """

        if self._boundType is None:
            return

        if self._boundLength is None:
            return

        if self._pointingRA is None or self._pointingDec is None:
            return

        self._bounds = SpatialBounds.getSpatialBounds(self._boundType, self._pointingRA, self._pointingDec,
                                                      self._boundLength)

    @property
    def pointingRA(self):
        """
        The RA of the telescope pointing in degrees
        (in the International Celestial Reference System).
        """
        if self._pointingRA is not None:
            return np.degrees(self._pointingRA)
        else:
            return None

    @pointingRA.setter
    def pointingRA(self, value):
        self._pointingRA = np.radians(value)
        self._buildBounds()

    @property
    def pointingDec(self):
        """
        The Dec of the telescope pointing in degrees
        (in the International Celestial Reference System).
        """
        if self._pointingDec is not None:
            return np.degrees(self._pointingDec)
        else:
            return None

    @pointingDec.setter
    def pointingDec(self, value):
        self._pointingDec = np.radians(value)
        self._buildBounds()

    @property
    def boundLength(self):
        """
        Either a list or a float indicating the size of the field
        of view associated with this ObservationMetaData.

        See the documentation in the SpatialBounds class for more
        details (specifically, the 'length' Parameter).

        In degrees (Yes: the documentation in SpatialBounds says that
        the length should be in radians.  The present class converts
        from degrees to radians before passing to SpatialBounds).
        """
        if self._boundLength is None:
            return None

        return np.degrees(self._boundLength)

    @boundLength.setter
    def boundLength(self, value):
        self._boundLength = np.radians(value)
        self._buildBounds()

    @property
    def boundType(self):
        """
        Tag indicating what sub-class of SpatialBounds should
        be instantiated for this ObservationMetaData.
        """
        return self._boundType

    @boundType.setter
    def boundType(self, value):
        self._boundType = value
        self._buildBounds()

    @property
    def bounds(self):
        """
        Instantiation of a sub-class of SpatialBounds.  This
        is what actually construct the WHERE clause of the SQL
        query associated with this ObservationMetaData.
        """
        return self._bounds

    @property
    def rotSkyPos(self):
        """
        The rotation of the telescope with respect to the sky in degrees.
        It is a parameter you should get from OpSim.
        """
        if self._rotSkyPos is not None:
            return np.degrees(self._rotSkyPos)
        else:
            return None

    @rotSkyPos.setter
    def rotSkyPos(self, value):
        self._rotSkyPos = np.radians(value)

    @property
    def m5(self):
        """
        A dict of m5 (the 5-sigma limiting magnitude) values
        associated with the bandpasses represented by this
        ObservationMetaData.
        """
        return self._m5

    @m5.setter
    def m5(self, value):
        self._m5 = self._assignDictKeyedToBandpass(value, 'm5')

    @property
    def seeing(self):
        """
        A dict of seeing values in arcseconds associated
        with the bandpasses represented by this ObservationMetaData
        """
        return self._seeing

    @seeing.setter
    def seeing(self, value):
        self._seeing = self._assignDictKeyedToBandpass(value, 'seeing')

    @property
    def site(self):
        """
        An instantiation of the Site class containing information about
        the telescope site.
        """
        return self._site

    @site.setter
    def site(self, value):
        self._site = value

    @property
    def mjd(self):
        """
        The MJD of the observation associated with this ObservationMetaData.
        """
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        """
        Either a float or a ModifiedJulianDate.  If a float, this setter
        assumes that you are passing in International Atomic Time
        """
        if isinstance(value, float):
            self._mjd = ModifiedJulianDate(TAI=value)
        elif isinstance(value, ModifiedJulianDate):
            self._mjd = value
        else:
            raise RuntimeError("You can only set mjd to either a float or a ModifiedJulianDate")

    @property
    def bandpass(self):
        """
        The bandpass associated with this ObservationMetaData.
        Can be a list.
        """
        return self._bandpass

    def setBandpassM5andSeeing(self, bandpassName=None, m5=None, seeing=None):
        """
        Set the bandpasses and associated 5-sigma limiting magnitudes
        and seeing values for this ObservationMetaData.

        @param [in] bandpassName is either a char or a list of chars denoting
        the name of the bandpass associated with this ObservationMetaData.

        @param [in] m5 is the 5-sigma-limiting magnitude(s) associated
        with bandpassName

        @param [in] seeing is the seeing(s) in arcseconds associated
        with bandpassName

        Nothing is returned.  This method just sets member variables of
        this ObservationMetaData.
        """

        self._bandpass = bandpassName
        self._m5 = self._assignDictKeyedToBandpass(m5, 'm5')
        self._seeing = self._assignDictKeyedToBandpass(seeing, 'seeing')

    @property
    def skyBrightness(self):
        """
        The sky brightness in mags per square arcsecond associated
        with this ObservationMetaData.
        """
        return self._skyBrightness

    @skyBrightness.setter
    def skyBrightness(self, value):
        self._skyBrightness = value

    @property
    def OpsimMetaData(self):
        """
        A dict of all of the columns taken from OpSim when constructing this
        ObservationMetaData
        """
        return self._OpsimMetaData

    @OpsimMetaData.setter
    def OpsimMetaData(self, value):
        if not isinstance(value, dict):
            raise RuntimeError('OpsimMetaData must be a dict')
        self._OpsimMetaData = value
