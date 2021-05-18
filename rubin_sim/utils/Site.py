""" Site Class

    Class defines the attributes of the site unless overridden
    ajc@astro 2/23/2010

    Restoring this so that the astrometry mixin in Astrometry.py
    can inherit the site information
    danielsf 1/27/2014

"""

import numpy as np
import warnings

__all__ = ["Site"]


class LSST_site_parameters(object):
    """
    This is a struct containing the LSST site parameters as defined in

    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-30

    (accessed on 4 January 2016)

    This class only exists for initializing Site with LSST parameter values.
    Users should not be accessing this class directly.
    """

    def __init__(self):
        self.longitude = -70.7494  # in degrees
        self.latitude = -30.2444  # in degrees
        self.height = 2650.0  # in meters
        self.temperature = 11.5  # in centigrade
        self.pressure = 750.0  # in millibars
        self.humidity = 0.4   # scale 0-1
        self.lapseRate = 0.0065  # in Kelvin per meter
        # the lapse rate was not specified by LSE-30;
        # 0.0065 K/m appears to be the "standard" value
        # see, for example http://mnras.oxfordjournals.org/content/365/4/1235.full


class Site (object):
    """
    This class will store site information for use in Catalog objects.

    Defaults values are LSST site values taken from the Observatory System Specification
    document

    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-30

    on 4 January 2016

    Attributes
    ----------
    longitude: in degrees

    longitude_rad: longitude in radians

    latitude: in degrees

    latitude_rad: latitude in radians

    height: in meters

    temperature: mean temperature in Centigrade

    temperature_kelvin: mean temperature in Kelvin

    pressure: in millibars

    humidity: relative humidity (range 0-1)

    lapseRate: change in temperature in Kelvins per meter

    name: name of the observatory.  If set to 'LSST' any unspecified
        values will default to LSST values as defined in

        https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-30

        i.e.
        longitude=-70.7494 degrees
        latitude=-30.2444 degrees
        height=2650.0 meters
        temperature=11.5 centigrade
        pressure=750.0 millibars
        humidity=0.4
        lapseRate=0.0065in Kelvin per meter
    """
    def __init__(self,
                 name=None,
                 longitude=None,
                 latitude=None,
                 height=None,
                 temperature=None,
                 pressure=None,
                 humidity=None,
                 lapseRate=None):
        """
        Parameters
        ----------
        name: a string denoting the name of the observator.  Set to 'LSST'
            for other parameters to default to LSST values.

            i.e.
            longitude=-70.7494 degrees
            latitude=-30.2444 degrees
            height=2650.0 meters
            temperature=11.5 centigrade
            pressure=750.0 millibars
            humidity=0.4
            lapseRate=0.0065 in Kelvin per meter

        longitude: in degrees

        latitude: in degrees

        height: in meters

        temperature: in Centigrade

        pressure: in millibars

        humidity: relative (range 0-1)

        lapseRate: in Kelvin per meter
        """

        default_params = None
        self._name = name
        if self._name == 'LSST':
            default_params = LSST_site_parameters()

        if default_params is not None:
            if longitude is None:
                longitude = default_params.longitude

            if latitude is None:
                latitude = default_params.latitude

            if height is None:
                height = default_params.height

            if temperature is None:
                temperature = default_params.temperature

            if pressure is None:
                pressure = default_params.pressure

            if humidity is None:
                humidity = default_params.humidity

            if lapseRate is None:
                lapseRate = default_params.lapseRate

        if longitude is not None:
            self._longitude_rad = np.radians(longitude)
        else:
            self._longitude_rad = None

        if latitude is not None:
            self._latitude_rad = np.radians(latitude)
        else:
            self._latitude_rad = None

        self._longitude_deg = longitude
        self._latitude_deg = latitude
        self._height = height
        self._pressure = pressure

        if temperature is not None:
            self._temperature_kelvin = temperature + 273.15  # in Kelvin
        else:
            self._temperature_kelvin = None

        self._temperature_centigrade = temperature
        self._humidity = humidity
        self._lapseRate = lapseRate

        # Go through all the attributes of this Site.
        # Raise a warning if any are None so that the user
        # is not surprised when some use of this Site fails
        # because something that should have beena a float
        # is NoneType
        list_of_nones = []
        if self.longitude is None or self.longitude_rad is None:
            if self.longitude_rad is not None:
                raise RuntimeError("in Site: longitude is None but longitude_rad is not")
            if self.longitude is not None:
                raise RuntimeError("in Site: longitude_rad is None but longitude is not")
            list_of_nones.append('longitude')

        if self.latitude is None or self.latitude_rad is None:
            if self.latitude_rad is not None:
                raise RuntimeError("in Site: latitude is None but latitude_rad is not")
            if self.latitude is not None:
                raise RuntimeError("in Site: latitude_rad is None but latitude is not")
            list_of_nones.append('latitude')

        if self.temperature is None or self.temperature_kelvin is None:
            if self.temperature is not None:
                raise RuntimeError("in Site: temperature_kelvin is None but temperature is not")
            if self.temperature_kelvin is not None:
                raise RuntimeError("in Site: temperature is None but temperature_kelvin is not")
            list_of_nones.append('temperature')

        if self.height is None:
            list_of_nones.append('height')

        if self.pressure is None:
            list_of_nones.append('pressure')

        if self.humidity is None:
            list_of_nones.append('humidity')

        if self.lapseRate is None:
            list_of_nones.append('lapseRate')

        if len(list_of_nones) != 0:
            msg = "The following attributes of your Site were None:\n"
            for name in list_of_nones:
                msg += "%s\n" % name
            msg += "If you want these to just default to LSST values,\n"
            msg += "instantiate your Site with name='LSST'"
            warnings.warn(msg)

    def __eq__(self, other):

        for param in self.__dict__:
            if param not in other.__dict__:
                return False
            if self.__dict__[param] != other.__dict__[param]:
                return False

        for param in other.__dict__:
            if param not in self.__dict__:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def name(self):
        """
        observatory name
        """
        return self._name

    @property
    def longitude_rad(self):
        """
        observatory longitude in radians
        """
        return self._longitude_rad

    @property
    def longitude(self):
        """
        observatory longitude in degrees
        """
        return self._longitude_deg

    @property
    def latitude_rad(self):
        """
        observatory latitude in radians
        """
        return self._latitude_rad

    @property
    def latitude(self):
        """
        observatory latitude in degrees
        """
        return self._latitude_deg

    @property
    def temperature(self):
        """
        mean temperature in centigrade
        """
        return self._temperature_centigrade

    @property
    def temperature_kelvin(self):
        """
        mean temperature in Kelvin
        """
        return self._temperature_kelvin

    @property
    def height(self):
        """
        height in meters
        """
        return self._height

    @property
    def pressure(self):
        """
        mean pressure in millibars
        """
        return self._pressure

    @property
    def humidity(self):
        """
        mean humidity in the range 0-1
        """
        return self._humidity

    @property
    def lapseRate(self):
        """
        temperature lapse rate (in Kelvin per meter)
        """
        return self._lapseRate
