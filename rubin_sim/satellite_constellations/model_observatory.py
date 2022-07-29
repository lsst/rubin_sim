import numpy as np
from rubin_sim.scheduler.modelObservatory import Model_observatory as orig_model_observatory
from rubin_sim.utils import survey_start_mjd, _healbin
from rubin_sim.site_models import Almanac


__all__ = ["Model_observatory"]


# Take the model observatory from the scheduler and subclass and expand to include satellite constellations

class Model_observatory(orig_model_observatory):
    """A class to generate a realistic telemetry stream for the scheduler"""

    def __init__(
        self,
        nside=None,
        mjd_start=None,
        seed=42,
        alt_min=5.0,
        lax_dome=True,
        cloud_limit=0.3,
        sim_ToO=None,
        seeing_db=None,
        park_after=10.0,
        init_load_length=10,
        sat_nside=64,
        satellite_dt=10.,
        constellation=None,
        alt_limit=20.
    ):
        """
        Parameters
        ----------
        nside : int (None)
            The healpix nside resolution
        mjd_start : float (None)
            The MJD to start the observatory up at. Uses util to lookup default if None.
        alt_min : float (5.)
            The minimum altitude to compute models at (degrees).
        lax_dome : bool (True)
            Passed to observatory model. If true, allows dome creep.
        cloud_limit : float (0.3)
            The limit to stop taking observations if the cloud model returns something equal or higher
        sim_ToO : sim_targetoO object (None)
            If one would like to inject simulated ToOs into the telemetry stream.
        seeing_db : filename of the seeing data database (None)
            If one would like to use an alternate seeing database
        park_after : float (10)
            Park the telescope after a gap longer than park_after (minutes)
        init_load_length : int (10)
            The length of pre-scheduled sky brighntess to load initially (days).
        """
        # Add in the new satellite information
        self.alt_limit = np.radians(alt_limit)
        self.satelite_dt = satellite_dt/3600./24.  # Seconds to days
        self.sat_nside = sat_nside
        self.constellation = constellation

        # Need to do a little fiddle with the MJD since self.mjd needs self.night set now.
        self.mjd_start = survey_start_mjd() if mjd_start is None else mjd_start
        self.almanac = Almanac(mjd_start=self.mjd_start)
        self.night = -1
        
        # Run the rest of the regular __init__ steps
        super().__init__(nside=None, mjd_start=self.mjd_start, seed=42, alt_min=5.0, lax_dome=True,
                         cloud_limit=0.3, sim_ToO=None, seeing_db=None, park_after=10.0,
                         init_load_length=10)
   
    def return_conditions(self):
        """
        Returns
        -------
        rubin_sim.scheduler.features.conditions object
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
        """Make the satellite prediction maps for the night

        will set self.sat_mjds and self.satellite_maps that can then be attached to
        a conditions object in self.return_conditions
        """
        sunset = self.almanac.sunsets["sun_n12_setting"][
            self.almanac_indx
        ]
        sunrise = self.almanac.sunsets["sun_n12_rising"][
            self.almanac_indx
        ]

        self.sat_mjds = np.arange(sunset, sunrise, self.satelite_dt)

        # Compute RA and decs for when sun is down
        ras, decs, alts, illums = self.constellation.paths_array(self.sat_mjds)

        below_limit = np.where(alts < self.alt_limit)

        weights = np.zeros(ras.shape, dtype=int)
        weights[illums] = 1
        weights[below_limit] = 0

        satellite_maps = []
        for i, mjd in enumerate(self.sat_mjds):

            spot_map = _healbin(ras[:, i][illums[:, i]], decs[:, i][illums[:, i]],
                                weights[:, i][illums[:, i]], self.sat_nside, reduceFunc=np.sum, dtype=int,
                                fillVal=0)
                
            satellite_maps.append(spot_map)

        self.satellite_maps = np.vstack(satellite_maps)
