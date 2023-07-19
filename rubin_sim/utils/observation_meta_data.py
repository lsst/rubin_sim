__all__ = ("ObservationMetaData",)

import numbers

import numpy as np

from .modified_julian_date import ModifiedJulianDate
from .site import Site
from .spatial_bounds import SpatialBounds


class ObservationMetaData:
    """Track metadata for a given telescope pointing.

    This class contains any metadata for a query which is associated with
    a particular telescope pointing, including bounds in RA and DEC, and
    the time of the observation.

    Parameters
    ----------
    pointing : `float`, opt
        [RA,Dec] float
        The coordinates of the pointing (in degrees; in the International
        Celestial Reference System)
    bound_type : `str`, opt
        characterizes the shape of the field of view.  Current options are 'box, and 'circle'
    bound_length : `float` or `np.ndarray`, opt
        is the characteristic length scale of the field of view in degrees.
        If bound_type is 'box', bound_length can be a float (in which case bound_length is
        half the length of the side of each box) or bound_length can be a numpy array
        in which case the first argument is half the width of the RA side of the box
        and the second argument is half the Dec side of the box.
        If bound_type is 'circle,' this will be the radius of the circle.
        The bound will be centered on the point (pointing_ra, pointing_dec), however,
        because objects are stored at their mean RA, Dec in the LSST databases
        (i.e. they are stored at values of RA, Dec which neglect proper motion), the
        bounds applied to database queries will be made slightly larger so that queries
        can be reasonably expected to return all of the objects within the desired field
        of view once those corrections have been applied.
    mjd : `float`, opt
        Either a float (in which case, it will be assumed to be in International
        Atomic Time), or an instantiation of the ModifiedJulianDate class representing
        the date of the observation
    bandpass_name : `str` or `list` of `str`, opt
        a char (e.g. 'u') or list (e.g. ['u', 'g', 'z']) denoting the bandpasses used
        for this particular observation
    site : `rubin_sim.utils.Site`, opt
        an instantiation of the rubin_sim.utils.Site class characterizing the site of the observatory.
    m5 : `float` or `list` of `float, opt
        this should be the 5-sigma limiting magnitude in the bandpass or
        bandpasses specified in bandpass_name.  Ultimately, m5 will be stored
        in a dict keyed to the bandpass_name (or Names) you passed in, i.e.
        you will be able to access m5 from outside of this class using, for
        example:
        myObservationMetaData.m5['u']
    sky_brightness : `float`, opt
        the magnitude of the sky in the filter specified by bandpass_name
    seeing : `float` or `list` of `float, opt
        Analogous to m5, corresponds to the seeing in arcseconds in the bandpasses in bandpass_name
    rot_sky_pos : `float`, opt
        The orientation of the telescope in degrees.
        The convention for rot_sky_pos is as follows:
        rot_sky_pos = 0 means north is in the +y direction on the focal plane and east is +x
        rot_sky_pos = 90 means north is +x and east is -y
        rot_sky_pos = -90 means north is -x and east is +y
        rot_sky_pos = 180 means north is -y and east is -x
        This should be consistent with PhoSim conventions.

    Examples
    --------
    ```>>> data = ObservationMetaData(bound_type='box', pointing_ra=5.0, pointing_dec=15.0, bound_length=5.0)```

    """

    def __init__(
        self,
        bound_type=None,
        bound_length=None,
        mjd=None,
        pointing_ra=None,
        pointing_dec=None,
        rot_sky_pos=None,
        bandpass_name=None,
        site=Site(name="LSST"),
        m5=None,
        sky_brightness=None,
        seeing=None,
    ):
        self._bounds = None
        self._bound_type = bound_type
        self._bandpass = bandpass_name
        self._sky_brightness = sky_brightness
        self._site = site
        self.__sim_meta_data = None

        if mjd is not None:
            if isinstance(mjd, numbers.Number):
                self._mjd = ModifiedJulianDate(TAI=mjd)
            elif isinstance(mjd, ModifiedJulianDate):
                self._mjd = mjd
            else:
                raise RuntimeError(
                    "You must pass either a float or a ModifiedJulianDate "
                    "as the kwarg mjd to ObservationMetaData"
                )
        else:
            self._mjd = None

        if rot_sky_pos is not None:
            self._rot_sky_pos = np.radians(rot_sky_pos)
        else:
            self._rot_sky_pos = None

        if pointing_ra is not None:
            self._pointing_ra = np.radians(pointing_ra)
        else:
            self._pointing_ra = None

        if pointing_dec is not None:
            self._pointing_dec = np.radians(pointing_dec)
        else:
            self._pointing_dec = None

        if bound_length is not None:
            self._bound_length = np.radians(bound_length)
        else:
            self._bound_length = None

        self._m5 = self._assign_dict_keyed_to_bandpass(m5, "m5")

        self._seeing = self._assign_dict_keyed_to_bandpass(seeing, "seeing")

        if self._bounds is None:
            self._build_bounds()

    @property
    def summary(self):
        mydict = {}
        mydict["site"] = self.site

        mydict["bound_type"] = self.bound_type
        mydict["bound_length"] = self.bound_length
        mydict["pointing_ra"] = self.pointing_ra
        mydict["pointing_dec"] = self.pointing_dec
        mydict["rot_sky_pos"] = self.rot_sky_pos

        if self.mjd is None:
            mydict["mjd"] = None
        else:
            mydict["mjd"] = self.mjd.TAI

        mydict["bandpass"] = self.bandpass
        mydict["sky_brightness"] = self.sky_brightness
        # mydict['m5'] = self.m5

        mydict["sim_meta_data"] = self.__sim_meta_data

        return mydict

    def __ne__(self, other):
        return not self.__eq__(other)

    def __eq__(self, other):
        if self.bounds != other.bounds:
            return False

        if self.pointing_ra != other.pointing_ra:
            return False

        if self.pointing_dec != other.pointing_dec:
            return False

        if self.rot_sky_pos != other.rot_sky_pos:
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

        if self.sky_brightness != other.sky_brightness:
            return False

        if self.sim_meta_data != other.sim_meta_data:
            return False

        return True

    def _assign_dict_keyed_to_bandpass(self, input_value, input_name):
        """
        This method sets up a dict of either m5 or seeing values (or any other quantity
        keyed to bandpass_name).  It reads in a list of values and associates them with
        the list of bandpass names in self._bandpass.

        Note: this method assumes that self._bandpass has already been set.
        It will raise an exception of self._bandpass is None.

        Parameters
        ----------
        input_value : `Unknown`
            is a single value or list of m5/seeing/etc. corresponding to
            the bandpasses stored in self._bandpass
        input_name : `Unknown`
            is the name of the Parameter stored in input_value
            (for constructing helpful error message)

        Returns
        -------
        returns : `Unknown`
            a dict of input_value values keed to self._bandpass
        """

        if input_value is None:
            return None
        else:
            bandpass_is_list = False
            input_is_list = False

            if self._bandpass is None:
                raise RuntimeError(
                    "You cannot set %s if you have not set " % input_name + "bandpass in ObservationMetaData"
                )

            if hasattr(self._bandpass, "__iter__") and not isinstance(self._bandpass, str):
                bandpass_is_list = True

            if hasattr(input_value, "__iter__") and not isinstance(input_value, str):
                input_is_list = True

            if bandpass_is_list and not input_is_list:
                raise RuntimeError(
                    "You passed a list of bandpass names"
                    + "but did not pass a list of %s to ObservationMetaData" % input_name
                )

            if input_is_list and not bandpass_is_list:
                raise RuntimeError(
                    "You passed a list of %s " % input_name
                    + "but did not pass a list of bandpass names to ObservationMetaData"
                )

            if input_is_list:
                if len(input_value) != len(self._bandpass):
                    raise RuntimeError(
                        "The list of %s you passed to ObservationMetaData " % input_name
                        + "has a different length than the list of bandpass names you passed"
                    )

            # now build the dict
            if bandpass_is_list:
                if len(input_value) != len(self._bandpass):
                    raise RuntimeError(
                        "In ObservationMetaData you tried to assign bandpass "
                        + "and %s with lists of different length" % input_name
                    )

                output_dict = {}
                for b, m in zip(self._bandpass, input_value):
                    output_dict[b] = m
            else:
                output_dict = {self._bandpass: input_value}

            return output_dict

    def _build_bounds(self):
        """
        Set up the member variable self._bounds.

        If self._bound_type, self._bound_length, self._pointing_ra, or
        self._pointing_dec are None, nothing will happen.
        """

        if self._bound_type is None:
            return

        if self._bound_length is None:
            return

        if self._pointing_ra is None or self._pointing_dec is None:
            return

        self._bounds = SpatialBounds.get_spatial_bounds(
            self._bound_type, self._pointing_ra, self._pointing_dec, self._bound_length
        )

    @property
    def pointing_ra(self):
        """
        The RA of the telescope pointing in degrees
        (in the International Celestial Reference System).
        """
        if self._pointing_ra is not None:
            return np.degrees(self._pointing_ra)
        else:
            return None

    @pointing_ra.setter
    def pointing_ra(self, value):
        self._pointing_ra = np.radians(value)
        self._build_bounds()

    @property
    def pointing_dec(self):
        """
        The Dec of the telescope pointing in degrees
        (in the International Celestial Reference System).
        """
        if self._pointing_dec is not None:
            return np.degrees(self._pointing_dec)
        else:
            return None

    @pointing_dec.setter
    def pointing_dec(self, value):
        self._pointing_dec = np.radians(value)
        self._build_bounds()

    @property
    def bound_length(self):
        """
        Either a list or a float indicating the size of the field
        of view associated with this ObservationMetaData.

        See the documentation in the SpatialBounds class for more
        details (specifically, the 'length' Parameter).

        In degrees (Yes: the documentation in SpatialBounds says that
        the length should be in radians.  The present class converts
        from degrees to radians before passing to SpatialBounds).
        """
        if self._bound_length is None:
            return None

        return np.degrees(self._bound_length)

    @bound_length.setter
    def bound_length(self, value):
        self._bound_length = np.radians(value)
        self._build_bounds()

    @property
    def bound_type(self):
        """
        Tag indicating what sub-class of SpatialBounds should
        be instantiated for this ObservationMetaData.
        """
        return self._bound_type

    @bound_type.setter
    def bound_type(self, value):
        self._bound_type = value
        self._build_bounds()

    @property
    def bounds(self):
        """
        Instantiation of a sub-class of SpatialBounds.  This
        is what actually construct the WHERE clause of the SQL
        query associated with this ObservationMetaData.
        """
        return self._bounds

    @property
    def rot_sky_pos(self):
        """
        The rotation of the telescope with respect to the sky in degrees.
        It is a parameter you should get from OpSim.
        """
        if self._rot_sky_pos is not None:
            return np.degrees(self._rot_sky_pos)
        else:
            return None

    @rot_sky_pos.setter
    def rot_sky_pos(self, value):
        self._rot_sky_pos = np.radians(value)

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
        self._m5 = self._assign_dict_keyed_to_bandpass(value, "m5")

    @property
    def seeing(self):
        """
        A dict of seeing values in arcseconds associated
        with the bandpasses represented by this ObservationMetaData
        """
        return self._seeing

    @seeing.setter
    def seeing(self, value):
        self._seeing = self._assign_dict_keyed_to_bandpass(value, "seeing")

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

    def set_bandpass_m5and_seeing(self, bandpass_name=None, m5=None, seeing=None):
        """
        Set the bandpasses and associated 5-sigma limiting magnitudes
        and seeing values for this ObservationMetaData.

        Parameters
        ----------
        bandpass_name : `Unknown`
            is either a char or a list of chars denoting
            the name of the bandpass associated with this ObservationMetaData.
        m5 : `Unknown`
            is the 5-sigma-limiting magnitude(s) associated
            with bandpass_name
        seeing : `Unknown`
            is the seeing(s) in arcseconds associated
            with bandpass_name
        """

        self._bandpass = bandpass_name
        self._m5 = self._assign_dict_keyed_to_bandpass(m5, "m5")
        self._seeing = self._assign_dict_keyed_to_bandpass(seeing, "seeing")

    @property
    def sky_brightness(self):
        """
        The sky brightness in mags per square arcsecond associated
        with this ObservationMetaData.
        """
        return self._sky_brightness

    @sky_brightness.setter
    def sky_brightness(self, value):
        self._sky_brightness = value

    @property
    def sim_meta_data(self):
        """
        A dict of all of the columns taken from OpSim when constructing this
        ObservationMetaData
        """
        return self.__sim_meta_data

    @sim_meta_data.setter
    def sim_meta_data(self, value):
        if not isinstance(value, dict):
            raise RuntimeError("sim_meta_data must be a dict")
        self.__sim_meta_data = value
