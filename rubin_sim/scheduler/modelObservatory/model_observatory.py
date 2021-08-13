import numpy as np
from rubin_sim.utils import (_hpid2RaDec, _raDec2Hpid, Site, calcLmstLast,
                             m5_flat_sed, _approx_RaDec2AltAz, _angularSeparation, _approx_altaz2pa)
import rubin_sim.skybrightness_pre as sb
import healpy as hp
from rubin_sim.site_models import (ScheduledDowntimeData, UnscheduledDowntimeData,
                                   SeeingData, SeeingModel, CloudData, Almanac)
from rubin_sim.scheduler.features import Conditions
from rubin_sim.scheduler.utils import set_default_nside, create_season_offset
from astropy.coordinates import EarthLocation
from astropy.time import Time
import warnings
import matplotlib.pylab as plt
from importlib import import_module
from rubin_sim.scheduler.modelObservatory import Kinem_model
from rubin_sim.data import data_versions

__all__ = ['Model_observatory']


class Model_observatory(object):
    """A class to generate a realistic telemetry stream for the scheduler
    """

    def __init__(self, nside=None, mjd_start=60218., seed=42, quickTest=True,
                 alt_min=5., lax_dome=True, cloud_limit=0.3, sim_ToO=None,
                 seeing_db=None, park_after=10.):
        """
        Parameters
        ----------
        nside : int (None)
            The healpix nside resolution
        mjd_start : float (60218)
            The MJD to start the observatory up at. 60218 = Oct 1, 2023.
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
        """

        if nside is None:
            nside = set_default_nside()
        self.nside = nside

        self.cloud_limit = cloud_limit

        self.alt_min = np.radians(alt_min)
        self.lax_dome = lax_dome

        self.mjd_start = mjd_start

        self.sim_ToO = sim_ToO

        self.park_after = park_after/60./24.  # To days

        # Create an astropy location
        self.site = Site('LSST')
        self.location = EarthLocation(lat=self.site.latitude, lon=self.site.longitude,
                                      height=self.site.height)

        # Load up all the models we need

        mjd_start_time = Time(self.mjd_start, format='mjd')
        # Downtime
        self.down_nights = []
        self.sched_downtime_data = ScheduledDowntimeData(mjd_start_time)
        self.unsched_downtime_data = UnscheduledDowntimeData(mjd_start_time)

        sched_downtimes = self.sched_downtime_data()
        unsched_downtimes = self.unsched_downtime_data()

        down_starts = []
        down_ends = []
        for dt in sched_downtimes:
            down_starts.append(dt['start'].mjd)
            down_ends.append(dt['end'].mjd)
        for dt in unsched_downtimes:
            down_starts.append(dt['start'].mjd)
            down_ends.append(dt['end'].mjd)

        self.downtimes = np.array(list(zip(down_starts, down_ends)), dtype=list(zip(['start', 'end'], [float, float])))
        self.downtimes.sort(order='start')

        # Make sure there aren't any overlapping downtimes
        diff = self.downtimes['start'][1:] - self.downtimes['end'][0:-1]
        while np.min(diff) < 0:
            # Should be able to do this wihtout a loop, but this works
            for i, dt in enumerate(self.downtimes[0:-1]):
                if self.downtimes['start'][i+1] < dt['end']:
                    new_end = np.max([dt['end'], self.downtimes['end'][i+1]])
                    self.downtimes[i]['end'] = new_end
                    self.downtimes[i+1]['end'] = new_end

            good = np.where(self.downtimes['end'] - np.roll(self.downtimes['end'], 1) != 0)
            self.downtimes = self.downtimes[good]
            diff = self.downtimes['start'][1:] - self.downtimes['end'][0:-1]

        self.seeing_data = SeeingData(mjd_start_time, seeing_db=seeing_db)
        self.seeing_model = SeeingModel()
        self.seeing_indx_dict = {}
        for i, filtername in enumerate(self.seeing_model.filter_list):
            self.seeing_indx_dict[filtername] = i

        self.cloud_data = CloudData(mjd_start_time, offset_year=0)

        self.sky_model = sb.SkyModelPre(speedLoad=quickTest)

        self.observatory = Kinem_model(mjd0=mjd_start)

        self.filterlist = ['u', 'g', 'r', 'i', 'z', 'y']
        self.seeing_FWHMeff = {}
        for key in self.filterlist:
            self.seeing_FWHMeff[key] = np.zeros(hp.nside2npix(self.nside), dtype=float)

        self.almanac = Almanac(mjd_start=mjd_start)

        # Let's make sure we're at an openable MJD
        good_mjd = False
        to_set_mjd = mjd_start
        while not good_mjd:
            good_mjd, to_set_mjd = self.check_mjd(to_set_mjd)
        self.mjd = to_set_mjd

        sun_moon_info = self.almanac.get_sun_moon_positions(self.mjd)
        season_offset = create_season_offset(self.nside, sun_moon_info['sun_RA'])
        self.sun_RA_start = sun_moon_info['sun_RA'] + 0
        # Conditions object to update and return on request
        self.conditions = Conditions(nside=self.nside, mjd_start=mjd_start,
                                     season_offset=season_offset, sun_RA_start=self.sun_RA_start)

        self.obsID_counter = 0

    def get_info(self):
        """
        Returns
        -------
        Array with model versions that were instantiated
        """

        # Could add in the data version
        result = []
        versions = data_versions()
        for key in versions:
            result.append([key, versions[key]])

        return result

    def return_conditions(self):
        """
        Returns
        -------
        rubin_sim.scheduler.features.conditions object
        """

        self.conditions.mjd = self.mjd

        self.conditions.night = self.night
        # Current time as astropy time
        current_time = Time(self.mjd, format='mjd')

        # Clouds. XXX--just the raw value
        self.conditions.bulk_cloud = self.cloud_data(current_time)

        # use conditions object itself to get aprox altitude of each healpx
        alts = self.conditions.alt
        azs = self.conditions.az

        good = np.where(alts > self.alt_min)

        # Compute the airmass at each heapix
        airmass = np.zeros(alts.size, dtype=float)
        airmass.fill(np.nan)
        airmass[good] = 1./np.cos(np.pi/2. - alts[good])
        self.conditions.airmass = airmass

        # reset the seeing
        for key in self.seeing_FWHMeff:
            self.seeing_FWHMeff[key].fill(np.nan)
        # Use the model to get the seeing at this time and airmasses.
        FWHM_500 = self.seeing_data(current_time)
        seeing_dict = self.seeing_model(FWHM_500, airmass[good])
        fwhm_eff = seeing_dict['fwhmEff']
        for i, key in enumerate(self.seeing_model.filter_list):
            self.seeing_FWHMeff[key][good] = fwhm_eff[i, :]
        self.conditions.FWHMeff = self.seeing_FWHMeff

        # sky brightness
        self.conditions.skybrightness = self.sky_model.returnMags(self.mjd, airmass_mask=False,
                                                                  planet_mask=False,
                                                                  moon_mask=False, zenith_mask=False)

        self.conditions.mounted_filters = self.observatory.mounted_filters
        self.conditions.current_filter = self.observatory.current_filter[0]

        # Compute the slewtimes
        slewtimes = np.empty(alts.size, dtype=float)
        slewtimes.fill(np.nan)
        # If there has been a gap, park the telescope
        gap = self.mjd - self.observatory.last_mjd
        if gap > self.park_after:
            self.observatory.park()
        slewtimes[good] = self.observatory.slew_times(0., 0., self.mjd, alt_rad=alts[good], az_rad=azs[good],
                                                      filtername=self.observatory.current_filter,
                                                      lax_dome=self.lax_dome, update_tracking=False)
        self.conditions.slewtime = slewtimes

        # Let's get the sun and moon
        sun_moon_info = self.almanac.get_sun_moon_positions(self.mjd)
        # convert these to scalars
        for key in sun_moon_info:
            sun_moon_info[key] = sun_moon_info[key].max()
        self.conditions.moonPhase = sun_moon_info['moon_phase']

        self.conditions.moonAlt = sun_moon_info['moon_alt']
        self.conditions.moonAz = sun_moon_info['moon_az']
        self.conditions.moonRA = sun_moon_info['moon_RA']
        self.conditions.moonDec = sun_moon_info['moon_dec']
        self.conditions.sunAlt = sun_moon_info['sun_alt']
        self.conditions.sunRA = sun_moon_info['sun_RA']
        self.conditions.sunDec = sun_moon_info['sun_dec']

        self.conditions.lmst, last = calcLmstLast(self.mjd, self.site.longitude_rad)

        self.conditions.telRA = self.observatory.current_RA_rad
        self.conditions.telDec = self.observatory.current_dec_rad
        self.conditions.telAlt = self.observatory.last_alt_rad
        self.conditions.telAz = self.observatory.last_az_rad

        self.conditions.rotTelPos = self.observatory.last_rot_tel_pos_rad
        self.conditions.cumulative_azimuth_rad = self.observatory.cumulative_azimuth_rad

        # Add in the almanac information
        self.conditions.night = self.night
        self.conditions.sunset = self.almanac.sunsets['sunset'][self.almanac_indx]
        self.conditions.sun_n12_setting = self.almanac.sunsets['sun_n12_setting'][self.almanac_indx]
        self.conditions.sun_n18_setting = self.almanac.sunsets['sun_n18_setting'][self.almanac_indx]
        self.conditions.sun_n18_rising = self.almanac.sunsets['sun_n18_rising'][self.almanac_indx]
        self.conditions.sun_n12_rising = self.almanac.sunsets['sun_n12_rising'][self.almanac_indx]
        self.conditions.sunrise = self.almanac.sunsets['sunrise'][self.almanac_indx]
        self.conditions.moonrise = self.almanac.sunsets['moonrise'][self.almanac_indx]
        self.conditions.moonset = self.almanac.sunsets['moonset'][self.almanac_indx]

        # Planet positions from almanac
        self.conditions.planet_positions = self.almanac.get_planet_positions(self.mjd)

        # See if there are any ToOs to include
        if self.sim_ToO is not None:
            toos = self.sim_ToO(self.mjd)
            if toos is not None:
                self.conditions.targets_of_opportunity = toos

        return self.conditions

    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        self._mjd = value
        self.almanac_indx = self.almanac.mjd_indx(value)
        self.night = self.almanac.sunsets['night'][self.almanac_indx]

    def observation_add_data(self, observation):
        """
        Fill in the metadata for a completed observation
        """
        current_time = Time(self.mjd, format='mjd')

        observation['clouds'] = self.cloud_data(current_time)
        observation['airmass'] = 1./np.cos(np.pi/2. - observation['alt'])
        # Seeing
        fwhm_500 = self.seeing_data(current_time)
        seeing_dict = self.seeing_model(fwhm_500, observation['airmass'])
        observation['FWHMeff'] = seeing_dict['fwhmEff'][self.seeing_indx_dict[observation['filter'][0]]]
        observation['FWHM_geometric'] = seeing_dict['fwhmGeom'][self.seeing_indx_dict[observation['filter'][0]]]
        observation['FWHM_500'] = fwhm_500

        observation['night'] = self.night
        observation['mjd'] = self.mjd

        hpid = _raDec2Hpid(self.sky_model.nside, observation['RA'], observation['dec'])
        observation['skybrightness'] = self.sky_model.returnMags(self.mjd,
                                                                 indx=[hpid],
                                                                 extrapolate=True)[observation['filter'][0]]

        observation['fivesigmadepth'] = m5_flat_sed(observation['filter'][0], observation['skybrightness'],
                                                    observation['FWHMeff'],
                                                    observation['exptime']/observation['nexp'],
                                                    observation['airmass'], nexp=observation['nexp'])

        lmst, last = calcLmstLast(self.mjd, self.site.longitude_rad)
        observation['lmst'] = lmst

        sun_moon_info = self.almanac.get_sun_moon_positions(self.mjd)
        observation['sunAlt'] = sun_moon_info['sun_alt']
        observation['sunAz'] = sun_moon_info['sun_az']
        observation['sunRA'] = sun_moon_info['sun_RA']
        observation['sunDec'] = sun_moon_info['sun_dec']
        observation['moonAlt'] = sun_moon_info['moon_alt']
        observation['moonAz'] = sun_moon_info['moon_az']
        observation['moonRA'] = sun_moon_info['moon_RA']
        observation['moonDec'] = sun_moon_info['moon_dec']
        observation['moonDist'] = _angularSeparation(observation['RA'], observation['dec'],
                                                     observation['moonRA'], observation['moonDec'])
        observation['solarElong'] = _angularSeparation(observation['RA'], observation['dec'],
                                                       observation['sunRA'], observation['sunDec'])
        observation['moonPhase'] = sun_moon_info['moon_phase']

        observation['ID'] = self.obsID_counter
        self.obsID_counter += 1

        return observation

    def check_up(self, mjd):
        """See if we are in downtime

        True if telescope is up
        False if in downtime
        """

        result = True
        indx = np.searchsorted(self.downtimes['start'], mjd, side='right')-1
        if indx >= 0:
            if mjd < self.downtimes['end'][indx]:
                result = False
        return result

    def check_mjd(self, mjd, cloud_skip=20.):
        """See if an mjd is ok to observe

        Parameters
        ----------
        cloud_skip : float (20)
            How much time to skip ahead if it's cloudy (minutes)

        Returns
        -------
        mjd_ok : `bool`
        mdj : `float`
            If True, the input mjd. If false, a good mjd to skip forward to.
        """
        passed = True
        new_mjd = mjd + 0

        # Maybe set this to a while loop to make sure we don't land on another cloudy time?
        # or just make this an entire recursive call?
        clouds = self.cloud_data(Time(mjd, format='mjd'))

        if clouds > self.cloud_limit:
            passed = False
            while clouds > self.cloud_limit:
                new_mjd = new_mjd + cloud_skip/60./24.
                clouds = self.cloud_data(Time(new_mjd, format='mjd'))
        alm_indx = np.searchsorted(self.almanac.sunsets['sunset'], mjd) - 1
        # at the end of the night, advance to the next setting twilight
        if mjd > self.almanac.sunsets['sun_n12_rising'][alm_indx]:
            passed = False
            new_mjd = self.almanac.sunsets['sun_n12_setting'][alm_indx+1]
        if mjd < self.almanac.sunsets['sun_n12_setting'][alm_indx]:
            passed = False
            new_mjd = self.almanac.sunsets['sun_n12_setting'][alm_indx+1]
        # We're in a down night, advance to next night
        if not self.check_up(mjd):
            passed = False
            new_mjd = self.almanac.sunsets['sun_n12_setting'][alm_indx+1]
        # recursive call to make sure we skip far enough ahead
        if not passed:
            while not passed:
                passed, new_mjd = self.check_mjd(new_mjd)
            return False, new_mjd
        else:
            return True, mjd

    def _update_rotSkyPos(self, observation):
        """If we have an undefined rotSkyPos, try to fill it out.
        """
        alt, az = _approx_RaDec2AltAz(observation['RA'], observation['dec'], self.site.latitude_rad,
                                      self.site.longitude_rad, self.mjd)
        obs_pa = _approx_altaz2pa(alt, az, self.site.latitude_rad)
        observation['rotSkyPos'] = (obs_pa + observation['rotTelPos']) % (2*np.pi)
        observation['rotTelPos'] = 0.

        return observation

    def observe(self, observation):
        """Try to make an observation

        Returns
        -------
        observation : observation object
            None if there was no observation taken. Completed observation with meta data filled in.
        new_night : bool
            Have we started a new night.
        """

        start_night = self.night.copy()

        if np.isnan(observation['rotSkyPos']):
            observation = self._update_rotSkyPos(observation)

        # If there has been a long gap, assume telescope stopped tracking and parked
        gap = self.mjd - self.observatory.last_mjd
        if gap > self.park_after:
            self.observatory.park()

        # Compute what alt,az we have tracked to (or are parked at)
        start_alt, start_az, start_rotTelPos = self.observatory.current_alt_az(self.mjd)
        # Slew to new position and execute observation. Use the requested rotTelPos position,
        # obsevation['rotSkyPos'] will be ignored.
        slewtime, visittime = self.observatory.observe(observation, self.mjd,
                                                       rotTelPos=observation['rotTelPos'],
                                                       lax_dome=self.lax_dome)

        # inf slewtime means the observation failed (probably outside alt limits)
        if ~np.all(np.isfinite(slewtime)):
            return None, False

        observation_worked, new_mjd = self.check_mjd(self.mjd + (slewtime + visittime)/24./3600.)

        if observation_worked:
            observation['visittime'] = visittime
            observation['slewtime'] = slewtime
            observation['slewdist'] = _angularSeparation(start_az, start_alt,
                                                         self.observatory.last_az_rad,
                                                         self.observatory.last_alt_rad)
            self.mjd = self.mjd + slewtime/24./3600.
            # Reach into the observatory model to pull out the relevant data it has calculated
            # Note, this might be after the observation has been completed.
            observation['alt'] = self.observatory.last_alt_rad
            observation['az'] = self.observatory.last_az_rad
            observation['pa'] = self.observatory.last_pa_rad
            observation['rotTelPos'] = self.observatory.last_rot_tel_pos_rad
            observation['rotSkyPos'] = self.observatory.current_rotSkyPos_rad
            observation['cummTelAz'] = self.observatory.cumulative_azimuth_rad

            # Metadata on observation is after slew and settle, so at start of exposure.
            result = self.observation_add_data(observation)
            self.mjd = self.mjd + visittime/24./3600.
            new_night = False
        else:
            result = None
            self.observatory.park()
            # Skip to next legitimate mjd
            self.mjd = new_mjd
            now_night = self.night
            if now_night == start_night:
                new_night = False
            else:
                new_night = True

        return result, new_night
