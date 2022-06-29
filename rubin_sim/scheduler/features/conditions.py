from io import StringIO
import numpy as np
import pandas as pd
from rubin_sim.utils import (
    _approx_RaDec2AltAz,
    Site,
    _hpid2RaDec,
    m5_flat_sed,
    _approx_altaz2pa,
    _angularSeparation,
)
import healpy as hp
from rubin_sim.scheduler.utils import (
    set_default_nside,
    match_hp_resolution,
    season_calc,
    smallest_signed_angle,
)

__all__ = ["Conditions"]


class Conditions(object):
    """
    Class to hold telemetry information

    If the incoming value is a healpix map, we use a setter to ensure the
    resolution matches.

    Unless otherwise noted, all values are assumed to be valid at the time
    given by self.mjd
    """

    def __init__(
        self,
        nside=None,
        site="LSST",
        exptime=30.0,
        mjd_start=59853.5,
        season_offset=None,
        sun_RA_start=None,
    ):
        """
        Parameters
        ----------
        nside : int
            The healpixel nside to set the resolution of attributes.
        site : str ('LSST')
            A site name used to create a sims.utils.Site object. For looking up
            observatory paramteres like latitude and longitude.
        expTime : float (30)
            The exposure time to assume when computing the 5-sigma limiting depth
        mjd_start : float (59853.5)
            The starting MJD of the survey.
        season_offset : np.array
            A HEALpix array that specifies the day offset when computing the season for each HEALpix.
        sun_RA_start : float (None)

        Attributes (Set on init)
        -----------
        nside : int
            Healpix resolution. All maps are set to this reslution.
        site : rubin_sim.Site object ('LSST')
            Contains static site-specific data (lat, lon, altitude, etc). Defaults to 'LSST'.
        ra : np.array
            A healpix array with the RA of each healpixel center (radians). Automatically
            generated.
        dec : np.array
            A healpix array with the Dec of each healpixel center (radians). Automatically generated.

        Attributes (to be set by user/telemetry stream/scheduler)
        -------------------------------------------
        mjd : float
            Modified Julian Date (days).
        bulk_cloud : float
            The fraction of sky covered by clouds. (In the future might update to transparency map)
        cloud_map : np.array
            XXX--to be done. HEALpix array with cloudy pixels set to NaN.
        slewtime : np.array
            Healpix showing the slewtime to each healpixel center (seconds)
        current_filter : str
            The name of the current filter. (expect one of u, g, r, i, z, y).
        mounted_filters : list of str
            The filters that are currently mounted and thu available (expect 5 of u, g, r, i, z, y)
        night : int
            The current night number (days). Probably starts at 1.
        skybrightness : dict of np.array
            Dictionary keyed by filtername. Values are healpix arrays with the sky brightness at each
            healpix center (mag/acsec^2)
        FWHMeff : dict of np.array
            Dictionary keyed by filtername. Values are the effective seeing FWHM at each healpix
            center (arcseconds)
        moonAlt : float
            The altitude of the Moon (radians)
        moonAz : float
            The Azimuth of the moon (radians)
        moonRA : float
            RA of the moon (radians)
        moonDec : float
            Declination of the moon (radians)
        moonPhase : float
            The Phase of the moon. (percent, 0=new moon, 100=full moon)
        sunAlt : float
            The altitude of the sun (radians).
        sunAz : float
            The Azimuth of the sun (radians).
        sunRA : float
            The RA of the sun (radians).
        sunDec : float
            The Dec of the sun (radians).
        telRA : float
            The current telescope RA pointing (radians).
        telDec : float
            The current telescope Declination (radians).
        telAlt : float
            The current telescope altitude (radians).
        telAz : float
            The current telescope azimuth (radians).
        cumulative_azimuth_rad : float
            The cummulative telescope azimuth (radians). For tracking cable wrap
        cloud_map : np.array
            A healpix map with the cloud coverage. XXX-expand, is this bool map? Transparency map?
        airmass : np.array
            A healpix map with the airmass value of each healpixel. (unitless)
        wind_speed : float
            Wind speed (m/s).
        wind_direction : float
            Direction from which the wind originates. A direction of 0.0 degrees
            means the wind originates from the north and 90.0 degrees from the
            east (radians).
        sunset : float
            The MJD of sunset that starts the current night. Note MJDs of sunset, moonset, twilight times, etc
            are from interpolations. This means the sun may actually be slightly above/below the horizon
            at the given sunset time.
        sun_n12_setting : float
            The MJD of when the sun is at -12 degees altitude and setting during the
            current night. From interpolation.
        sun_n18_setting : float
            The MJD when the sun is at -18 degrees altitude and setting during the current night.
            From interpolation.
        sun_n18_rising : float
            The MJD when the sun is at -18 degrees altitude and rising during the current night.
            From interpolation.
        sun_n12_rising : float
            The MJD when the sun is at -12 degrees altitude and rising during the current night.
            From interpolation.
        sunrise : float
            The MJD of sunrise during the current night. From interpolation
        moonrise : float
            The MJD of moonrise during the current night. From interpolation.
        moonset : float
            The MJD of moonset during the current night. From interpolation.
        targets_of_opportunity : list of rubin_sim.scheduler.targetoO object(s)
            targetoO objects.
        planet_positions : dict
            Dictionary of planet name and coordinate e.g., 'venus_RA', 'mars_dec'
        scheduled_observations : np.array
            A list of MJD times when there are scheduled observations. Defaults to empty array.

        Attributes (calculated on demand and cached)
        ------------------------------------------
        alt : np.array
            Altitude of each healpixel (radians). Recaclulated if mjd is updated. Uses fast
            approximate equation for converting RA,Dec to alt,az.
        az : np.array
            Azimuth of each healpixel (radians). Recaclulated if mjd is updated. Uses fast
            approximate equation for converting RA,Dec to alt,az.
        pa : np.array
            The parallactic angle of each healpixel (radians). Recaclulated if mjd is updated.
            Based on the fast approximate alt,az values.
        lmst : float
            The local mean sidearal time (hours). Updates is mjd is changed.
        M5Depth : dict of np.array
            the 5-sigma limiting depth healpix maps, keyed by filtername (mags). Will be recalculated
            if the skybrightness, seeing, or airmass are updated.
        HA : np.array
            Healpix map of the hour angle of each healpixel (radians). Runs from 0 to 2pi.
        az_to_sun : np.array
            Healpix map of the azimuthal distance to the sun for each healpixel (radians)
        az_to_anitsun : np.array
            Healpix map of the azimuthal distance to the anit-sun for each healpixel (radians)
        solar_elongation : np.array
            Healpix map of the solar elongation (angular distance to the sun) for each healpixel (radians)

        Attributes (set by the scheduler)
        -------------------------------
        queue : list of observation objects
            The current queue of observations core_scheduler is waiting to execute.

        """
        if nside is None:
            nside = set_default_nside()
        self.nside = nside
        self.site = Site(site)
        self.exptime = exptime
        self.mjd_start = mjd_start
        hpids = np.arange(hp.nside2npix(nside))
        self.season_offset = season_offset
        self.sun_RA_start = sun_RA_start
        # Generate an empty map so we can copy when we need a new map
        self.zeros_map = np.zeros(hp.nside2npix(nside), dtype=float)
        self.nan_map = np.zeros(hp.nside2npix(nside), dtype=float)
        self.nan_map.fill(np.nan)
        # The RA, Dec grid we are using
        self.ra, self.dec = _hpid2RaDec(nside, hpids)

        # Modified Julian Date (day)
        self._mjd = None
        # Altitude and azimuth. Dict with degrees and radians
        self._alt = None
        self._az = None
        self._pa = None
        # The cloud level. Fraction, but could upgrade to transparency map
        self.clouds = None
        self._slewtime = None
        self.current_filter = None
        self.mounted_filters = None
        self.night = None
        self._lmst = None
        # Should be a dict with filtername keys
        self._skybrightness = {}
        self._FWHMeff = {}
        self._M5Depth = None
        self._airmass = None

        self.wind_speed = None
        self.wind_direction = None

        # Upcomming scheduled observations
        self.scheduled_observations = np.array([], dtype=float)

        # Attribute to hold the current observing queue
        self.queue = None

        # Moon
        self.moonAlt = None
        self.moonAz = None
        self.moonRA = None
        self.moonDec = None
        self.moonPhase = None

        # Sun
        self.sunAlt = None
        self.sunAz = None
        self.sunRA = None
        self.sunDec = None

        # Almanac information
        self.sunset = None
        self.sun_n12_setting = None
        self.sun_n18_setting = None
        self.sun_n18_rising = None
        self.sun_n12_rising = None
        self.sunrise = None
        self.moonrise = None
        self.moonset = None

        self.planet_positions = None

        # Current telescope pointing
        self.telRA = None
        self.telDec = None
        self.telAlt = None
        self.telAz = None

        # Full sky cloud map
        self._cloud_map = None
        self._HA = None

        # XXX--document
        self.bulk_cloud = None

        self.rotTelPos = None

        self.targets_of_opportunity = None

        self._season = None

        self.season_modulo = None
        self.season_max_season = None
        self.season_length = 365.25
        self.season_floor = True

    @property
    def lmst(self):
        return self._lmst

    @lmst.setter
    def lmst(self, value):
        self._lmst = value
        self._HA = None

    @property
    def HA(self):
        if self._HA is None:
            self.calc_HA()
        return self._HA

    def calc_HA(self):
        self._HA = np.radians(self._lmst * 360.0 / 24.0) - self.ra
        self._HA[np.where(self._HA < 0)] += 2.0 * np.pi

    @property
    def cloud_map(self):
        return self._cloud_map

    @cloud_map.setter
    def cloud_map(self, value):
        self._cloud_map = match_hp_resolution(value, nside_out=self.nside)

    @property
    def slewtime(self):
        return self._slewtime

    @slewtime.setter
    def slewtime(self, value):
        # Using 0 for start of night
        if np.size(value) == 1:
            self._slewtime = value
        else:
            self._slewtime = match_hp_resolution(value, nside_out=self.nside)

    @property
    def airmass(self):
        return self._airmass

    @airmass.setter
    def airmass(self, value):
        self._airmass = match_hp_resolution(value, nside_out=self.nside)
        self._M5Depth = None

    @property
    def pa(self):
        if self._pa is None:
            self.calc_pa()
        return self._pa

    def calc_pa(self):
        self._pa = _approx_altaz2pa(self.alt, self.az, self.site.latitude_rad)

    @property
    def alt(self):
        if self._alt is None:
            self.calc_altAz()
        return self._alt

    @property
    def az(self):
        if self._az is None:
            self.calc_altAz()
        return self._az

    def calc_altAz(self):
        self._alt, self._az = _approx_RaDec2AltAz(
            self.ra,
            self.dec,
            self.site.latitude_rad,
            self.site.longitude_rad,
            self._mjd,
        )

    @property
    def mjd(self):
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        self._mjd = value
        # Set things that need to be recalculated to None
        self._az = None
        self._alt = None
        self._pa = None
        self._HA = None
        self._lmst = None
        self._az_to_sun = None
        self._az_to_antisun = None
        self._season = None
        self._solar_elongation = None

    @property
    def skybrightness(self):
        return self._skybrightness

    @skybrightness.setter
    def skybrightness(self, indict):
        for key in indict:

            self._skybrightness[key] = match_hp_resolution(
                indict[key], nside_out=self.nside
            )
        # If sky brightness changes, need to recalc M5 depth.
        self._M5Depth = None

    @property
    def FWHMeff(self):
        return self._FWHMeff

    @FWHMeff.setter
    def FWHMeff(self, indict):
        for key in indict:
            self._FWHMeff[key] = match_hp_resolution(indict[key], nside_out=self.nside)
        self._M5Depth = None

    @property
    def M5Depth(self):
        if self._M5Depth is None:
            self.calc_M5Depth()
        return self._M5Depth

    def calc_M5Depth(self):
        self._M5Depth = {}
        for filtername in self._skybrightness:
            good = ~np.isnan(self._skybrightness[filtername])
            self._M5Depth[filtername] = self.nan_map.copy()
            self._M5Depth[filtername][good] = m5_flat_sed(
                filtername,
                self._skybrightness[filtername][good],
                self._FWHMeff[filtername][good],
                self.exptime,
                self._airmass[good],
            )

    def calc_solar_elongation(self):
        self._solar_elongation = _angularSeparation(
            self.ra, self.dec, self.sunRA, self.sunDec
        )

    @property
    def solar_elongation(self):
        if self._solar_elongation is None:
            self.calc_solar_elongation()
        return self._solar_elongation

    def calc_az_to_sun(self):
        self._az_to_sun = smallest_signed_angle(self.ra, self.sunRA)

    def calc_az_to_antisun(self):
        self._az_to_antisun = smallest_signed_angle(self.ra + np.pi, self.sunRA)

    @property
    def az_to_sun(self):
        if self._az_to_sun is None:
            self.calc_az_to_sun()
        return self._az_to_sun

    @property
    def az_to_antisun(self):
        if self._az_to_antisun is None:
            self.calc_az_to_antisun()
        return self._az_to_antisun

    # XXX, there's probably an elegant decorator that could do this caching automatically
    def season(self, modulo=None, max_season=None, season_length=365.25, floor=True):
        if self.season_offset is not None:
            kwargs_match = (
                (modulo == self.season_modulo)
                & (max_season == self.season_max_season)
                & (season_length == self.season_length)
                & (floor == self.season_floor)
            )
            if ~kwargs_match:
                self.season_modulo = modulo
                self.season_max_season = max_season
                self.season_length = season_length
                self.season_floor = floor
            if (self._season is None) | (~kwargs_match):
                self._season = season_calc(
                    self.night,
                    offset=self.season_offset,
                    modulo=modulo,
                    max_season=max_season,
                    season_length=season_length,
                    floor=floor,
                )
        else:
            self._season = None

        return self._season

    def __repr__(self):
        return f"<{self.__class__.__name__} mjd_start='{self.mjd_start}' at {hex(id(self))}>"

    def __str__(self):
        # If dependencies of to_markdown are not installed, fall back on repr
        try:
            pd.DataFrame().to_markdown()
        except ImportError:
            return repr(self)

        output = StringIO()
        print(f"{self.__class__.__qualname__} at {hex(id(self))}", file=output)
        print("============================", file=output)
        print("nside: ", self.nside, "  ", file=output)
        print("site: ", self.site.name, "  ", file=output)
        print("exptime: ", self.exptime, "  ", file=output)
        print("lmst: ", self.lmst, "  ", file=output)
        print("season_offset: ", self.season_offset, "  ", file=output)
        print("sun_RA_start: ", self.sun_RA_start, "  ", file=output)
        print("clouds: ", self.clouds, "  ", file=output)
        print("current_filter: ", self.current_filter, "  ", file=output)
        print("mounted_filters: ", self.mounted_filters, "  ", file=output)
        print("night: ", self.night, "  ", file=output)
        print("wind_speed: ", self.wind_speed, "  ", file=output)
        print("wind_direction: ", self.wind_direction, "  ", file=output)
        print(
            "len(scheduled_observations): ",
            len(self.scheduled_observations),
            "  ",
            file=output,
        )
        print(
            "len(queue): ",
            None if self.queue is None else len(self.queue),
            "  ",
            file=output,
        )
        print("moonPhase: ", self.moonPhase, "  ", file=output)
        print("bulk_cloud: ", self.bulk_cloud, "  ", file=output)
        print(
            "targets_of_opportunity: ", self.targets_of_opportunity, "  ", file=output
        )
        print("season_modulo: ", self.season_modulo, "  ", file=output)
        print("season_max_season: ", self.season_max_season, "  ", file=output)
        print("season_length: ", self.season_length, "  ", file=output)
        print("season_floor: ", self.season_floor, "  ", file=output)
        print(
            "cumulative_azimuth_rad: ", self.cumulative_azimuth_rad, "  ", file=output
        )

        positions = [
            {
                "name": "sun",
                "alt": self.sunAlt,
                "az": self.sunAz,
                "RA": self.sunRA,
                "decl": self.sunDec,
            }
        ]
        positions.append(
            {
                "name": "moon",
                "alt": self.moonAlt,
                "az": self.moonAz,
                "RA": self.moonRA,
                "decl": self.moonDec,
            }
        )
        for planet_name in ("venus", "mars", "jupiter", "saturn"):
            positions.append(
                {
                    "name": planet_name,
                    "RA": np.asscalar(self.planet_positions[planet_name + "_RA"]),
                    "decl": np.asscalar(self.planet_positions[planet_name + "_dec"]),
                }
            )
        positions.append(
            {
                "name": "telescope",
                "alt": self.telAlt,
                "az": self.telAz,
                "RA": self.telRA,
                "decl": self.telDec,
                "rot": self.rotTelPos,
            }
        )
        positions = pd.DataFrame(positions).set_index("name")
        print(file=output)
        print("Positions (radians)", file=output)
        print("-------------------", file=output)
        print(positions.to_markdown(), file=output)

        positions_deg = np.degrees(positions)
        print(file=output)
        print("Positions (degrees)", file=output)
        print("-------------------", file=output)
        print(positions_deg.to_markdown(), file=output)

        events = (
            "mjd_start",
            "mjd",
            "sunset",
            "sun_n12_setting",
            "sun_n18_setting",
            "sun_n18_rising",
            "sun_n12_rising",
            "sunrise",
            "moonrise",
            "moonset",
            "sun_0_setting",
            "sun_0_rising",
        )
        event_rows = []
        for event in events:
            try:
                mjd = getattr(self, event)
                time = pd.to_datetime(mjd + 2400000.5, unit="D", origin="julian")
                event_rows.append({"event": event, "MJD": mjd, "date": time})
            except AttributeError:
                pass

        event_df = pd.DataFrame(event_rows).set_index("event").sort_values(by="MJD")
        print("", file=output)
        print("Events", file=output)
        print("------", file=output)
        print(event_df.to_markdown(), file=output)

        map_stats = []
        for map_name in ("ra", "dec", "slewtime", "airmass"):
            values = getattr(self, map_name)
            map_stats.append(
                {
                    "map": map_name,
                    "nside": hp.npix2nside(len(values)),
                    "min": np.nanmin(values),
                    "max": np.nanmax(values),
                    "median": np.nanmedian(values),
                }
            )

        for base_map_name in ("skybrightness", "FWHMeff"):
            for band in "ugrizy":
                values = getattr(self, base_map_name)[band]
                map_name = f"{base_map_name}_{band}"
                map_stats.append(
                    {
                        "map": map_name,
                        "nside": hp.npix2nside(len(values)),
                        "min": np.nanmin(values),
                        "max": np.nanmax(values),
                        "median": np.nanmedian(values),
                    }
                )
        maps_df = pd.DataFrame(map_stats).set_index("map")
        print("", file=output)
        print("Maps", file=output)
        print("----", file=output)
        print(maps_df.to_markdown(), file=output)

        result = output.getvalue()
        return result

    def _repr_markdown_(self):
        return str(self)
