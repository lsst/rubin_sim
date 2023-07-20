__all__ = ("CloudModel",)

import warnings

import numpy as np


class CloudModel:
    """LSST cloud calculations for cloud extinction.
    Currently this actually only returns the cloud coverage of the sky, exactly as reported in the
    cloud database (thus the sky coverage, in fractions of 8ths).


    Parameters
    ----------
    XXX--update docstring

    self.efd_requirements and self.map_requirements are also set.
    efd_requirements is a tuple: (list of str, float).
    This corresponds to the data columns required from the EFD and the amount of time history required.
    target_requirements is a list of str.
    This corresponds to the data columns required in the target map dictionary passed when calculating the
    processed telemetry values.
    """

    def __init__(self, cloud_column="cloud", altitude_column="altitude", azimuth_column="azimuth"):
        self.altcol = altitude_column
        self.azcol = azimuth_column
        self.cloudcol = cloud_column

    def configure(self, config=None):
        """"""
        warnings.warn("The configure method is deprecated.")

    def config_info(self):
        """"""
        warnings.warn("The config_info method is deprecated.")

    def __call__(self, cloud_value, altitude):
        """Calculate the sky coverage due to clouds.

        This is where we'd plug in Peter's cloud transparency maps and predictions.
        We could also try translating cloud transparency into a cloud extinction.
        For now, we're simply returning the cloud coverage that we already got from the database,
        but multiplied over the whole sky to provide a map.

        Parameters
        ----------
        cloud_value: float or efdData dict
            The value to give the clouds (XXX-units?).
        altitude:  float, np.array, or targetDict
            Altitude of the output (arbitrary).

        Returns
        -------
        dict of np.ndarray
            Cloud transparency map values.
        """
        if isinstance(cloud_value, dict):
            cloud_value = cloud_value[self.cloudcol]
        if isinstance(altitude, dict):
            altitude = altitude[self.altcol]

        model_cloud = np.zeros(len(altitude), float) + cloud_value
        return {"cloud": model_cloud}
