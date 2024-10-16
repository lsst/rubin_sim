__all__ = ("ModelObservatory",)

import numpy as np
from rubin_scheduler.scheduler.model_observatory import ModelObservatory as oMO
from rubin_scheduler.site_models import Almanac
from rubin_scheduler.utils import SURVEY_START_MJD, _healbin

# Take the model observatory from the scheduler and
# subclass to expand to include satellite constellations


class ModelObservatory(oMO):
    """A class to generate a realistic telemetry stream for the scheduler

    Parameters
    ----------
    nside : `int`
        The healpix nside resolution
    mjd_start : `float`
        The MJD to start the observatory up at.
        Uses util to lookup default if None.
    alt_min : `float`
        The minimum altitude to compute models at (degrees).
    lax_dome : `bool`
        Passed to observatory model. If true, allows dome creep.
    cloud_limit : `float`
        The limit to stop taking observations if the cloud model
        returns something equal or higher
    sim_to_o : `sim_targetoO`
        If one would like to inject simulated ToOs into the telemetry stream.
    seeing_db : `str`
        If one would like to use an alternate seeing database,
        filename of sqlite file
    park_after : `float`
        Park the telescope after a gap longer than park_after (minutes)
    init_load_length : `int`
        The length of pre-scheduled sky brighntess to load initially (days).
    alt_limit : `float`
        Altitude limit for considering satellite streaks (degrees).
    satellite_dt : `float`
        The time step to use for computing satellite positions (seconds).
    sat_nside : `int`
        The HEALpix nside to use for satellite streak maps.
    constellation : `rubin_sim.satellite_constellations.Constellation`
        The satellite constellation to use.
    """

    def __init__(
        self,
        nside=None,
        mjd_start=SURVEY_START_MJD,
        seed=42,
        alt_min=5.0,
        lax_dome=True,
        cloud_limit=0.3,
        sim_to_o=None,
        seeing_db=None,
        park_after=10.0,
        init_load_length=10,
        sat_nside=64,
        satellite_dt=10.0,
        constellation=None,
        alt_limit=20.0,
    ):
        # Add in the new satellite information
        self.alt_limit = np.radians(alt_limit)
        self.satelite_dt = satellite_dt / 3600.0 / 24.0  # Seconds to days
        self.sat_nside = sat_nside
        self.constellation = constellation

        # Need to do a little fiddle with the MJD since
        # self.mjd needs self.night set now.
        self.mjd_start = mjd_start

        self.almanac = Almanac(mjd_start=self.mjd_start)
        self.night = -1

        # Run the rest of the regular __init__ steps
        super().__init__(
            nside=None,
            mjd_start=self.mjd_start,
            seed=seed,
            alt_min=alt_min,
            lax_dome=lax_dome,
            cloud_limit=cloud_limit,
            sim_to_o=sim_to_o,
            seeing_db=seeing_db,
            park_after=park_after,
            init_load_length=init_load_length,
        )

    def return_conditions(self):
        """
        Returns
        -------
        conditions: `rubin_sim.scheduler.features.conditions`
            Current conditions as simulated by the ModelObservatory.
        """

        # Spot to put in satellite streak prediction maps
        self.conditions.satellite_mjds = self.sat_mjds
        self.conditions.satellite_maps = self.satellite_maps

        # Run the regular return conditions
        super().return_conditions()
        # I guess running super() means return statement gets skipped?
        return self.conditions

    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        self._mjd = value
        self.almanac_indx = self.almanac.mjd_indx(value)
        # Update night if needed
        if self.almanac.sunsets["night"][self.almanac_indx] != self.night:
            self.night = self.almanac.sunsets["night"][self.almanac_indx]
            # Update the satellite prediction map for the night
            self._update_satellite_maps()

    def _update_satellite_maps(self):
        """Make the satellite prediction maps for the night.

        Will set self.sat_mjds and self.satellite_maps that can then
        be attached to a conditions object in self.return_conditions
        """
        sunset = self.almanac.sunsets["sun_n12_setting"][self.almanac_indx]
        sunrise = self.almanac.sunsets["sun_n12_rising"][self.almanac_indx]

        self.sat_mjds = np.arange(sunset, sunrise, self.satelite_dt)

        # Compute RA and decs for when sun is down
        ras, decs, alts, illums = self.constellation.paths_array(self.sat_mjds)

        below_limit = np.where(alts < self.alt_limit)

        weights = np.zeros(ras.shape, dtype=int)
        weights[illums] = 1
        weights[below_limit] = 0

        satellite_maps = []
        for i, mjd in enumerate(self.sat_mjds):
            spot_map = _healbin(
                ras[:, i][illums[:, i]],
                decs[:, i][illums[:, i]],
                weights[:, i][illums[:, i]],
                self.sat_nside,
                reduce_func=np.sum,
                dtype=int,
                fill_val=0,
            )

            satellite_maps.append(spot_map)

        self.satellite_maps = np.vstack(satellite_maps)
