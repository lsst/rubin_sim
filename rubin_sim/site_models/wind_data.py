__all__ = ("ConstantWindData",)

from collections import namedtuple
from dataclasses import dataclass

import astropy.time

WindConditions = namedtuple("WindConditions", ["speed", "direction"])


@dataclass
class ConstantWindData:
    """A source of constant wind values.

    Parameters
    ----------
    wind_speed : `float`
        Wind speed (m/s).
    wind_direction : `float`
        Direction from which the wind originates. A direction of 0.0 degrees
        means the wind originates from the north and 90.0 degrees from the
        east (radians).
    """

    wind_speed: float = 0.0
    wind_direction: float = 0.0

    def __call__(self, time: astropy.time.Time):
        """A constant wind conditions

        Parameters
        ----------
        time : `astropy.time.Time`
            It principle the time for which the wind is returned,
            in practice this argument is ignored, and included for
            compatibility.

        Returns
        -------
        wind_conditions : `tuple` (`float`, `float`)
            A named tuple with the wind speed (m/s) and originating
            direction (radians east of north)
        """
        conditions = WindConditions(self.wind_speed, self.wind_direction)
        return conditions
