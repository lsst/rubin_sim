__all__ = ("Almanac",)

import os

import numpy as np
from scipy.interpolate import interp1d

from rubin_sim.data import get_data_dir


class Almanac:
    """Class to load and return pre-computed information about the LSST site."""

    def __init__(self, mjd_start=None, kind="quadratic"):
        # Load up the sunrise/sunset times
        data_dir = os.path.join(get_data_dir(), "site_models")

        temp = np.load(os.path.join(data_dir, "sunsets.npz"))
        self.sunsets = temp["almanac"].copy()
        temp.close()

        if mjd_start is not None:
            loc = np.searchsorted(self.sunsets["sunset"], mjd_start)
            # Set the start MJD to be night 1.
            self.sunsets["night"] -= self.sunsets["night"][loc - 1]

        # Load the sun and moon positions
        temp = np.load(os.path.join(data_dir, "sun_moon.npz"))
        self.sun_moon = temp["sun_moon_info"].copy()
        temp.close()

        self.interpolators = {
            "sun_alt": interp1d(self.sun_moon["mjd"], self.sun_moon["sun_alt"], kind=kind),
            "sun_az_x": interp1d(self.sun_moon["mjd"], np.cos(self.sun_moon["sun_az"]), kind=kind),
            "sun_az_y": interp1d(self.sun_moon["mjd"], np.sin(self.sun_moon["sun_az"]), kind=kind),
            "sun_RA_x": interp1d(self.sun_moon["mjd"], np.cos(self.sun_moon["sun_RA"]), kind=kind),
            "sun_RA_y": interp1d(self.sun_moon["mjd"], np.sin(self.sun_moon["sun_RA"]), kind=kind),
            "sun_dec": interp1d(self.sun_moon["mjd"], self.sun_moon["sun_dec"], kind=kind),
            "moon_alt": interp1d(self.sun_moon["mjd"], self.sun_moon["moon_alt"], kind=kind),
            "moon_az_x": interp1d(self.sun_moon["mjd"], np.cos(self.sun_moon["moon_az"]), kind=kind),
            "moon_az_y": interp1d(self.sun_moon["mjd"], np.sin(self.sun_moon["moon_az"]), kind=kind),
            "moon_RA_x": interp1d(self.sun_moon["mjd"], np.cos(self.sun_moon["moon_RA"]), kind=kind),
            "moon_RA_y": interp1d(self.sun_moon["mjd"], np.sin(self.sun_moon["moon_RA"]), kind=kind),
            "moon_dec": interp1d(self.sun_moon["mjd"], self.sun_moon["moon_dec"], kind=kind),
            "moon_phase": interp1d(self.sun_moon["mjd"], self.sun_moon["moon_phase"], kind=kind),
        }

        temp = np.load(os.path.join(data_dir, "planet_locations.npz"))
        self.planet_loc = temp["planet_loc"].copy()
        temp.close()

        self.planet_names = ["venus", "mars", "jupiter", "saturn"]
        self.planet_interpolators = {}
        for pn in self.planet_names:
            self.planet_interpolators[pn + "_RA_x"] = interp1d(
                self.planet_loc["mjd"], np.cos(self.planet_loc[pn + "_RA"]), kind=kind
            )
            self.planet_interpolators[pn + "_RA_y"] = interp1d(
                self.planet_loc["mjd"], np.sin(self.planet_loc[pn + "_RA"]), kind=kind
            )
            self.planet_interpolators[pn + "_dec"] = interp1d(
                self.planet_loc["mjd"], self.planet_loc[pn + "_dec"], kind=kind
            )

    def get_planet_positions(self, mjd):
        result = {}
        for pn in self.planet_names:
            result[pn + "_dec"] = self.planet_interpolators[pn + "_dec"](mjd)

            temp_result = np.array(
                [
                    np.arctan2(
                        self.planet_interpolators[pn + "_RA_y"](mjd),
                        self.planet_interpolators[pn + "_RA_x"](mjd),
                    )
                ]
            ).ravel()
            negative_angles = np.where(temp_result < 0.0)[0]
            temp_result[negative_angles] = 2.0 * np.pi + temp_result[negative_angles]
            result[pn + "_RA"] = temp_result
        return result

    def get_sunset_info(self, mjd):
        """
        Returns a numpy array with mjds for various events (sunset, moonrise, sun at -12 degrees alt, etc.).
        Also the integer night number.
        """
        indx = np.searchsorted(self.sunsets["sunset"], mjd) - 1
        return self.sunsets[indx]

    def mjd_indx(self, mjd):
        indx = np.searchsorted(self.sunsets["sunset"], mjd) - 1
        return indx

    def get_sun_moon_positions(self, mjd):
        """
        All angles in Radians. moonPhase between 0 and 100.
        """
        simple_calls = ["sun_alt", "sun_dec", "moon_alt", "moon_dec", "moon_phase"]
        result = {}
        for key in simple_calls:
            result[key] = self.interpolators[key](mjd)

        longitude_calls = ["sun_az", "moon_az", "sun_RA", "moon_RA"]
        for key in longitude_calls:
            # Need to wrap in case sent a scalar
            temp_result = np.array(
                [
                    np.arctan2(
                        self.interpolators[key + "_y"](mjd),
                        self.interpolators[key + "_x"](mjd),
                    )
                ]
            ).ravel()
            negative_angles = np.where(temp_result < 0.0)[0]
            temp_result[negative_angles] = 2.0 * np.pi + temp_result[negative_angles]
            result[key] = temp_result

        return result
