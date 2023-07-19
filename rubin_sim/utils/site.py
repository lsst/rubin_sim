__all__ = ("Site",)

import warnings

import numpy as np


class LsstSiteParameters:
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
        self.humidity = 0.4  # scale 0-1
        self.lapse_rate = 0.0065  # in Kelvin per meter
        # the lapse rate was not specified by LSE-30;
        # 0.0065 K/m appears to be the "standard" value
        # see, for example http://mnras.oxfordjournals.org/content/365/4/1235.full


class Site:
    """
    This class will store site information for use in Catalog objects.

    Defaults values are LSST site values taken from the Observatory System Specification
    document
    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-30
    on 4 January 2016

    Parameters
    ----------
    name : `str`, opt
        The name of the observatory. Set to 'LSST' for other parameters to default to LSST values.
    longitude : `float`, opt
        Longitude of the site in degrees.
    latitude : `float`, opt
        Latitude of the site in degrees.
    height : `float`, opt
        Height of the site in meters.
    temperature : `float`, opt
        Mean temperature in Centigrade
    pressure : `float`, opt
        Pressure for the site in millibars.
    humidity : `float`, opt
        Relative humidity (range 0-1).
    lapse_rate : `float`, opt
        Change in temperature in Kelvins per meter
    """

    def __init__(
        self,
        name=None,
        longitude=None,
        latitude=None,
        height=None,
        temperature=None,
        pressure=None,
        humidity=None,
        lapse_rate=None,
    ):
        default_params = None
        self._name = name
        if self._name == "LSST":
            default_params = LsstSiteParameters()

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

            if lapse_rate is None:
                lapse_rate = default_params.lapse_rate

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
        self._lapse_rate = lapse_rate

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
            list_of_nones.append("longitude")

        if self.latitude is None or self.latitude_rad is None:
            if self.latitude_rad is not None:
                raise RuntimeError("in Site: latitude is None but latitude_rad is not")
            if self.latitude is not None:
                raise RuntimeError("in Site: latitude_rad is None but latitude is not")
            list_of_nones.append("latitude")

        if self.temperature is None or self.temperature_kelvin is None:
            if self.temperature is not None:
                raise RuntimeError("in Site: temperature_kelvin is None but temperature is not")
            if self.temperature_kelvin is not None:
                raise RuntimeError("in Site: temperature is None but temperature_kelvin is not")
            list_of_nones.append("temperature")

        if self.height is None:
            list_of_nones.append("height")

        if self.pressure is None:
            list_of_nones.append("pressure")

        if self.humidity is None:
            list_of_nones.append("humidity")

        if self.lapse_rate is None:
            list_of_nones.append("lapse_rate")

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
    def lapse_rate(self):
        """
        temperature lapse rate (in Kelvin per meter)
        """
        return self._lapse_rate
