__all__ = ("just_return", "SkyModel")

import warnings

import numpy as np
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body, get_sun
from astropy.time import Time
from rubin_scheduler.utils import Site, _approx_alt_az2_ra_dec, _approx_ra_dec2_alt_az, haversine

from rubin_sim.phot_utils import Sed

from .interp_components import (
    Airglow,
    LowerAtm,
    MergedSpec,
    MoonInterp,
    ScatteredStar,
    TwilightInterp,
    UpperAtm,
    ZodiacalInterp,
)
from .utils import wrap_ra


def just_return(inval):
    """
    Really, just return the input.

    Parameters
    ----------
    input : anything

    Returns
    -------
    input : anything
        Just return whatever you sent in.
    """
    return inval


def inrange(inval, minimum=-1.0, maximum=1.0):
    """
    Make sure values are within min/max
    """
    inval = np.array(inval)
    below = np.where(inval < minimum)
    inval[below] = minimum
    above = np.where(inval > maximum)
    inval[above] = maximum
    return inval


def calc_az_rel_moon(azs, moon_az):
    az_rel_moon = wrap_ra(azs - moon_az)
    if isinstance(azs, np.ndarray):
        over = np.where(az_rel_moon > np.pi)
        az_rel_moon[over] = 2.0 * np.pi - az_rel_moon[over]
    else:
        if az_rel_moon > np.pi:
            az_rel_moon = 2.0 * np.pi - az_rel_moon
    return az_rel_moon


class SkyModel:
    def __init__(
        self,
        observatory=None,
        twilight=True,
        zodiacal=True,
        moon=True,
        airglow=True,
        lower_atm=False,
        upper_atm=False,
        scattered_star=False,
        merged_spec=True,
        mags=False,
        precise_alt_az=False,
        airmass_limit=3.0,
    ):
        """A model of the sky, including all of the required
        template spectra or magnitudes needed to interpolate the
        sky spectrum or magnitudes during twilight or night time
        at any point on the sky.


        Parameters
        ----------
        observatory : `rubin_scheduler.site_models.Site`, optional
            Default of None loads the LSST site.
        twilight : `bool`, optional
            Include twilight component (True)
        zodiacal : `bool`, optional
            Include zodiacal light component (True)
        moon : `bool`, optional
            Include scattered moonlight component (True)
        airglow : `bool`, optional
            Include airglow component
        lower_atm : `bool`, optional
            Include lower atmosphere component.
            This component is part of `merged_spec`.
        upper_atm : `bool`, optional
            Include upper atmosphere component.
            This component is part of `merged_spec`.
        scattered_star : `bool`, optional
            Include scattered starlight component.
            This component is part of `merged_spec`.
        merged_spec : `bool`, optional
            Compute the lower_atm, upper_atm, and scattered_star
            simultaneously since they are all functions of only airmass.
        mags : `bool`, optional
            By default, the sky model computes a 17,001 element spectrum.
            If `mags` is True,
            the model will return the LSST ugrizy magnitudes (in that order).
        precise_alt_az : `bool`, optional
            If False, use the fast alt, az to ra, dec coordinate
            transformations that do not take aberation, diffraction, etc
            into account. Results in errors up to ~1.5 degrees,
            but an order of magnitude faster than the precise coordinate
            transformations available in rubin_scheduler.utils.
        airmass_limit : `float`, optional
            Most of the models are only accurate to airmass 3.0.
            If set higher, airmass values higher than 3.0 are set to 3.0.
        """

        self.moon = moon
        self.lower_atm = lower_atm
        self.twilight = twilight
        self.zodiacal = zodiacal
        self.upper_atm = upper_atm
        self.airglow = airglow
        self.scattered_star = scattered_star
        self.merged_spec = merged_spec
        self.mags = mags
        self.precise_alt_az = precise_alt_az

        # set this as a way to track if coords have been set
        self.azs = None

        # Airmass limit.
        self.airmass_limit = airmass_limit

        if self.mags:
            self.npix = 6
        else:
            self.npix = 11001

        self.components = {
            "moon": self.moon,
            "lower_atm": self.lower_atm,
            "twilight": self.twilight,
            "upper_atm": self.upper_atm,
            "airglow": self.airglow,
            "zodiacal": self.zodiacal,
            "scattered_star": self.scattered_star,
            "merged_spec": self.merged_spec,
        }

        # Check that the merged component isn't being run with other components
        merged_comps = [self.lower_atm, self.upper_atm, self.scattered_star]
        for comp in merged_comps:
            if comp & self.merged_spec:
                warnings.warn("Adding component multiple times to the final output spectra.")

        interpolators = {
            "scattered_star": ScatteredStar,
            "airglow": Airglow,
            "lower_atm": LowerAtm,
            "upper_atm": UpperAtm,
            "merged_spec": MergedSpec,
            "moon": MoonInterp,
            "zodiacal": ZodiacalInterp,
            "twilight": TwilightInterp,
        }

        # Load up the interpolation objects for each component
        self.interp_objs = {}
        for key in self.components:
            if self.components[key]:
                self.interp_objs[key] = interpolators[key](mags=self.mags)

        if observatory is None:
            self.telescope = Site("LSST")
        else:
            self.telescope = observatory
        self.location = EarthLocation(
            lat=self.telescope.latitude_rad * u.rad,
            lon=self.telescope.longitude_rad * u.rad,
            height=self.telescope.height * u.m,
        )

        # Note that observing conditions have not been set
        self.params_set = False

    def _init_points(self):
        """
        Set up an array for all the interpolation points
        """

        names = [
            "airmass",
            "nightTimes",
            "alt",
            "az",
            "azRelMoon",
            "moonSunSep",
            "moonAltitude",
            "altEclip",
            "azEclipRelSun",
            "sunAlt",
            "azRelSun",
            "solar_flux",
        ]
        types = [float] * len(names)
        self.points = np.zeros(self.npts, list(zip(names, types)))

    def set_ra_dec_mjd(
        self,
        lon,
        lat,
        mjd,
        degrees=False,
        az_alt=False,
        solar_flux=130.0,
        filter_names=["u", "g", "r", "i", "z", "y"],
    ):
        """
        Set the sky parameters by computing the sky conditions on a
        given MJD and sky location.


        Parameters
        ----------
        lon : `float` or `np.ndarray`, (N,)
            Longitude-like (RA or Azimuth).
            Can be single number, list, or numpy array
        lat: `float` or `np.ndarray`, (N,)
            Latitude-like (Dec or Altitude)
        mjd: `float`
            Modified Julian Date for the calculation. Must be single number.
        degrees: `bool`, optional
            If True, lon/lat are in degrees. If False, lon/lat in radians.
        az_alt: `bool`, optional
            Assume lon, lat are RA, Dec unless az_alt=True
        solar_flux: `float`
            Solar flux in SFU Between 50 and 310. Default=130. 1 SFU=10^4 Jy.
        filter_names: `list` [`str`]
            List of filter for which to return magnitudes
            (if initialized with mags=True).
        """
        self.filter_names = filter_names
        if self.mags:
            self.npix = len(self.filter_names)
        # Wrap in array just in case single points were passed
        if np.size(lon) == 1:
            lon = np.array([lon]).ravel()
            lat = np.array([lat]).ravel()
        else:
            lon = np.array(lon)
            lat = np.array(lat)
        if degrees:
            self.ra = np.radians(lon)
            self.dec = np.radians(lat)
        else:
            self.ra = lon
            self.dec = lat
        if np.size(mjd) > 1:
            raise ValueError("mjd must be single value.")
        self.mjd = mjd
        if az_alt:
            self.azs = self.ra.copy()
            self.alts = self.dec.copy()
            if self.precise_alt_az:
                raise ValueError("Need to swap in astropy")
                # self.ra, self.dec = _ra_dec_from_alt_az(
                #    self.alts,
                #    self.azs,
                #    ObservationMetaData(mjd=self.mjd, site=self.telescope),
                # )
            else:
                self.ra, self.dec = _approx_alt_az2_ra_dec(
                    self.alts,
                    self.azs,
                    self.telescope.latitude_rad,
                    self.telescope.longitude_rad,
                    mjd,
                )
        else:
            if self.precise_alt_az:
                raise ValueError("Need to swap in astropy")
                # self.alts, self.azs, pa = _alt_az_pa_from_ra_dec(
                #    self.ra,
                #    self.dec,
                #    ObservationMetaData(mjd=self.mjd, site=self.telescope),
                # )
            else:
                self.alts, self.azs = _approx_ra_dec2_alt_az(
                    self.ra,
                    self.dec,
                    self.telescope.latitude_rad,
                    self.telescope.longitude_rad,
                    mjd,
                )

        self.npts = self.ra.size
        self._init_points()

        self.solar_flux = solar_flux
        self.points["solar_flux"] = self.solar_flux

        self._setup_point_grid()

        self.params_set = True

        # Interpolate the templates to the set Parameters
        self.good_pix = np.where((self.airmass <= self.airmass_limit) & (self.airmass >= 1.0))[0]

        if self.good_pix.size <= 0:
            raise ValueError(
                "No valid points. Airmass limit=%.1f, min airmass of requested points=%.1f"
                % (self.airmass_limit, np.min(self.airmass))
            )
        else:
            self._interp_sky()

    def set_ra_dec_alt_az_mjd(
        self,
        ra,
        dec,
        alt,
        az,
        mjd,
        degrees=False,
        solar_flux=130.0,
        filter_names=["u", "g", "r", "i", "z", "y"],
    ):
        """
        Set the sky parameters by computing the sky conditions on a
        given MJD and sky location.

        Use if you already have alt az coordinates so you can skip the
        coordinate conversion.
        """
        self.filter_names = filter_names
        if self.mags:
            self.npix = len(self.filter_names)
        # Wrap in array just in case single points were passed
        if not type(ra).__module__ == np.__name__:
            if np.size(ra) == 1:
                ra = np.array([ra]).ravel()
                dec = np.array([dec]).ravel()
                alt = np.array(alt).ravel()
                az = np.array(az).ravel()
            else:
                ra = np.array(ra)
                dec = np.array(dec)
                alt = np.array(alt)
                az = np.array(az)
        if degrees:
            self.ra = np.radians(ra)
            self.dec = np.radians(dec)
            self.alts = np.radians(alt)
            self.azs = np.radians(az)
        else:
            self.ra = ra
            self.dec = dec
            self.azs = az
            self.alts = alt
        if np.size(mjd) > 1:
            raise ValueError("mjd must be single value.")
        self.mjd = mjd

        self.npts = self.ra.size
        self._init_points()

        self.solar_flux = solar_flux
        self.points["solar_flux"] = self.solar_flux

        self._setup_point_grid()

        self.params_set = True

        # Interpolate the templates to the set Parameters
        self.good_pix = np.where((self.airmass <= self.airmass_limit) & (self.airmass >= 1.0))[0]
        if self.good_pix.size <= 0:
            raise ValueError(
                "No valid points. Airmass limit=%.1f, min airmass of requested points=%.1f"
                % (self.airmass_limit, np.min(self.airmass))
            )
        else:
            self._interp_sky()

    def get_computed_vals(self):
        """
        Return the intermediate values that are caluculated by
        set_ra_dec_mjd and used for interpolation.
        All of these values are also accessible as class attributes, this is
        a convenience method to grab them all at once and document the formats.

        Returns
        -------
        out : `dict`
            Dictionary of all the intermediate calculated values that may
            be of use outside (the key:values in the output dict)
        ra : `np.ndarray`, (N,)
            RA of the interpolation points (radians)
        dec : `np.ndarray`, (N,)
            Dec of the interpolation points (radians)
        alts : `np.ndarray`, (N,)
            Altitude (radians)
        azs : `np.ndarray`, (N,)
            Azimuth of interpolation points (radians)
        airmass : `np.ndarray`, (N,)
            Airmass values for each point,
            computed via 1./np.cos(np.pi/2.-self.alts).
        solar_flux : `float`
            The solar flux used (SFU).
        sunAz : `float`
            Azimuth of the sun (radians)
        sunAlt : `float`
            Altitude of the sun (radians)
        sunRA : `float`
            RA of the sun (radians)
        sunDec : `float`
            Dec of the sun (radians)
        azRelSun : `np.ndarray`, (N,)
            Azimuth of each point relative to the sun
            (0=same direction as sun) (radians)
        moonAz : `float`
            Azimuth of the moon (radians)
        moonAlt : `float`
            Altitude of the moon (radians)
        moonRA : `float`
            RA of the moon (radians)
        moonDec : `float`
            Dec of the moon (radians).  Note, if you want distances
        moon_phase : `float`
            Phase of the moon (0-100)
        moonSunSep : `float`
            Seperation of moon and sun (radians)
        azRelMoon : `np.ndarray`, (N,)
            Azimuth of each point relative to teh moon
        eclipLon : `np.ndarray`, (N,)
            Ecliptic longitude (radians) of each point
        eclipLat : `np.ndarray`, (N,)
            Ecliptic latitude (radians) of each point
        sunEclipLon: `np.ndarray`, (N,)
            Ecliptic longitude (radians) of each point with the sun at
            longitude zero

        Note that since the alt and az can be calculated using the fast
        approximation, if one wants to compute the distance between the points
        and the sun or moon, it is probably better to use the ra,dec positions
        rather than the alt,az positions.
        """

        result = {}
        attributes = [
            "ra",
            "dec",
            "alts",
            "azs",
            "airmass",
            "solar_flux",
            "moon_phase",
            "moon_az",
            "moon_alt",
            "sun_alt",
            "sun_az",
            "az_rel_sun",
            "moon_sun_sep",
            "az_rel_moon",
            "eclip_lon",
            "eclip_lat",
            "moon_ra",
            "moon_dec",
            "sun_ra",
            "sun_dec",
            "sun_eclip_lon",
        ]

        for attribute in attributes:
            if hasattr(self, attribute):
                result[attribute] = getattr(self, attribute)
            else:
                result[attribute] = None

        return result

    def _setup_point_grid(self):
        """
        Setup the points for the interpolation functions.
        """

        time = Time(self.mjd, format="mjd")
        aa = AltAz(location=self.location, obstime=time)

        sun_coords = get_sun(time)
        self.sun_ra = sun_coords.ra.rad
        self.sun_dec = sun_coords.dec.rad

        sun_coords = sun_coords.transform_to(aa)
        self.sun_alt = sun_coords.alt.rad
        self.sun_az = sun_coords.az.rad

        # Compute airmass the same way as ESO model
        self.airmass = 1.0 / np.cos(np.pi / 2.0 - self.alts)

        self.points["airmass"] = self.airmass
        self.points["nightTimes"] = 0
        self.points["alt"] = self.alts
        self.points["az"] = self.azs

        if self.twilight:
            self.points["sunAlt"] = self.sun_alt
            self.az_rel_sun = wrap_ra(self.azs - self.sun_az)
            self.points["azRelSun"] = self.az_rel_sun

        if self.moon:
            moon_coords = get_body("moon", time)
            self.moon_ra = moon_coords.ra.rad
            self.moon_dec = moon_coords.dec.rad

            moon_coords = moon_coords.transform_to(aa)
            self.moon_alt = moon_coords.alt.rad
            self.moon_az = moon_coords.az.rad

            moon_coords = get_body("moon", time)
            sun_coords = get_sun(time)
            sep = sun_coords.separation(moon_coords)

            # looks like phase is 0-100
            self.moon_phase = sep.deg * 100 / 180.0

            # Calc azimuth relative to moon
            self.az_rel_moon = calc_az_rel_moon(self.azs, self.moon_az)
            self.moon_targ_sep = haversine(self.azs, self.alts, self.moon_az, self.moon_alt)
            # Oof, looks like some things were stored as degrees.
            self.points["moonAltitude"] += np.degrees(self.moon_alt)
            self.points["azRelMoon"] += self.az_rel_moon
            self.moon_sun_sep = sep.rad
            self.points["moonSunSep"] += np.degrees(self.moon_sun_sep)

        if self.zodiacal:
            self.eclip_lon = np.zeros(self.npts)
            self.eclip_lat = np.zeros(self.npts)

            coord = SkyCoord(ra=self.ra * u.rad, dec=self.dec * u.rad)
            coord_ecl = coord.geocentricmeanecliptic
            self.eclip_lon = coord_ecl.lon.rad
            self.eclip_lat = coord_ecl.lat.rad

            # Subtract off the sun ecliptic longitude
            sun_coords = get_sun(time)
            sun_eclip = sun_coords.geocentricmeanecliptic
            self.sun_eclip_lon = sun_eclip.lon.rad
            self.points["altEclip"] += self.eclip_lat
            self.points["azEclipRelSun"] += wrap_ra(self.eclip_lon - self.sun_eclip_lon)

        self.mask = np.where((self.airmass > self.airmass_limit) | (self.airmass < 1.0))[0]
        self.good_pix = np.where((self.airmass <= self.airmass_limit) & (self.airmass >= 1.0))[0]

    def set_params(
        self,
        airmass=1.0,
        azs=90.0,
        alts=None,
        moon_phase=31.67,
        moon_alt=45.0,
        moon_az=0.0,
        sun_alt=-12.0,
        sun_az=0.0,
        sun_eclip_lon=0.0,
        eclip_lon=135.0,
        eclip_lat=90.0,
        degrees=True,
        solar_flux=130.0,
        filter_names=["u", "g", "r", "i", "z", "y"],
    ):
        """
        Set parameters manually.
        Note, you can put in unphysical combinations of Parameters if you
        want to (e.g., put a full moon at zenith at sunset).
        If the alts kwarg is set it will override the airmass kwarg.
        MoonPhase is percent of moon illuminated (0-100)
        """

        # Convert all values to radians for internal use.
        self.filter_names = filter_names
        if self.mags:
            self.npix = len(self.filter_names)
        if degrees:
            convert_func = np.radians
        else:
            convert_func = just_return

        self.solar_flux = solar_flux
        self.sun_alt = convert_func(sun_alt)
        self.moon_phase = moon_phase
        self.moon_alt = convert_func(moon_alt)
        self.moon_az = convert_func(moon_az)
        self.eclip_lon = convert_func(eclip_lon)
        self.eclip_lat = convert_func(eclip_lat)
        self.sun_eclip_lon = convert_func(sun_eclip_lon)
        self.azs = convert_func(azs)
        if alts is not None:
            self.airmass = 1.0 / np.cos(np.pi / 2.0 - convert_func(alts))
            self.alts = convert_func(alts)
        else:
            self.airmass = airmass
            self.alts = np.pi / 2.0 - np.arccos(1.0 / airmass)
        self.moon_targ_sep = haversine(self.azs, self.alts, moon_az, self.moon_alt)
        self.npts = np.size(self.airmass)
        self._init_points()

        self.points["airmass"] = self.airmass
        self.points["nightTimes"] = 0
        self.points["alt"] = self.alts
        self.points["az"] = self.azs

        self.az_rel_moon = calc_az_rel_moon(self.azs, self.moon_az)
        self.points["moonAltitude"] += np.degrees(self.moon_alt)
        self.points["azRelMoon"] = self.az_rel_moon
        self.moon_sun_sep = self.moon_phase / 100.0 * np.pi
        self.points["moonSunSep"] += self.moon_phase / 100.0 * 180.0

        self.eclip_lon = convert_func(eclip_lon)
        self.eclip_lat = convert_func(eclip_lat)

        self.sun_eclip_lon = convert_func(sun_eclip_lon)
        self.points["altEclip"] += self.eclip_lat
        self.points["azEclipRelSun"] += wrap_ra(self.eclip_lon - self.sun_eclip_lon)

        self.sun_az = convert_func(sun_az)
        self.points["sunAlt"] = self.sun_alt
        self.points["azRelSun"] = wrap_ra(self.azs - self.sun_az)
        self.points["solar_flux"] = solar_flux

        self.params_set = True

        self.mask = np.where((self.airmass > self.airmass_limit) | (self.airmass < 1.0))[0]
        self.good_pix = np.where((self.airmass <= self.airmass_limit) & (self.airmass >= 1.0))[0]
        # Interpolate the templates to the set Parameters
        if self.good_pix.size > 0:
            self._interp_sky()
        else:
            warnings.warn("No points in interpolation range")

    def _interp_sky(self):
        """
        Interpolate the template spectra to the set RA, Dec and MJD.

        the results are stored as attributes of the class:
        .wave = the wavelength in nm
        .spec = array of spectra with units of ergs/s/cm^2/nm
        """

        if not self.params_set:
            raise ValueError(
                "No parameters have been set. Must run set_ra_dec_mjd or setParams before running interpSky."
            )

        # set up array to hold the resulting spectra for each ra, dec point.
        self.spec = np.zeros((self.npts, self.npix), dtype=float)

        # Rebuild the components dict so things can be turned on/off
        self.components = {
            "moon": self.moon,
            "lower_atm": self.lower_atm,
            "twilight": self.twilight,
            "upper_atm": self.upper_atm,
            "airglow": self.airglow,
            "zodiacal": self.zodiacal,
            "scattered_star": self.scattered_star,
            "merged_spec": self.merged_spec,
        }

        # Loop over each component and add it to the result.
        mask = np.ones(self.npts)
        for key in self.components:
            if self.components[key]:
                result = self.interp_objs[key](self.points[self.good_pix], filter_names=self.filter_names)
                # Make sure the component has something
                if np.size(result["spec"]) == 0:
                    self.spec[self.mask, :] = np.nan
                    return
                if np.max(result["spec"]) > 0:
                    mask[np.where(np.sum(result["spec"], axis=1) == 0)] = 0
                self.spec[self.good_pix] += result["spec"]
                if not hasattr(self, "wave"):
                    self.wave = result["wave"]
                else:
                    if not np.allclose(result["wave"], self.wave, rtol=1e-4, atol=1e-4):
                        warnings.warn("Wavelength arrays of components do not match.")
        if self.airmass_limit <= 2.5:
            self.spec[np.where(mask == 0), :] = 0
        self.spec[self.mask, :] = np.nan

    def return_wave_spec(self):
        """
        Return the wavelength and spectra.
        Wavelenth in nm
        spectra is flambda in ergs/cm^2/s/nm
        """
        if self.azs is None:
            raise ValueError(
                "No coordinates set. Use set_ra_dec_mjd, setRaDecAltAzMjd, or "
                "setParams methods before calling returnWaveSpec."
            )
        if self.mags:
            raise ValueError("SkyModel set to interpolate magnitudes. Initialize object with mags=False")
        # Mask out high airmass points
        # self.spec[self.mask] *= 0
        return self.wave.copy(), self.spec.copy()

    def return_mags(self, bandpasses=None):
        """Return the skybrightness in magnitudes.

        Convert the computed spectra to a magnitude using the
        supplied bandpass, or, if self.mags=True, return the mags in the
        LSST filters.

        Parameters
        ----------
        bandpasses : `dict` [`str`, `rubin_sim.phot_utils.Bandpass`], optional
            Dictionary with bandpass name as keys and `Bandpass` objects
            as values.

        If mags=True when initialized, return mags returns a structured array
        with dtype names u,g,r,i,z,y; the default LSST bandpasses are used.

        Returns
        -------
        mags : `np.ndarray`, (N,)
            Sky brightness in AB mags/sq arcsec
        """
        if self.azs is None:
            raise ValueError(
                "No coordinates set. Use set_ra_dec_mjd, setRaDecAltAzMjd, or "
                "setParams methods before calling return_mags."
            )

        if self.mags:
            if bandpasses:
                warnings.warn("Ignoring set bandpasses and returning LSST ugrizy.")
            mags = -2.5 * np.log10(self.spec) + np.log10(3631.0)
            # Mask out high airmass
            mags[self.mask] *= np.nan
            mags = mags.swapaxes(0, 1)
            mags_back = {}
            for i, f in enumerate(self.filter_names):
                mags_back[f] = mags[i]
        else:
            mags_back = {}
            for key in bandpasses:
                mags = np.zeros(self.npts, dtype=float) - 666
                temp_sed = Sed()
                is_through = np.where(bandpasses[key].sb > 0)
                min_wave = bandpasses[key].wavelen[is_through].min()
                max_wave = bandpasses[key].wavelen[is_through].max()
                in_band = np.where((self.wave >= min_wave) & (self.wave <= max_wave))
                for i, ra in enumerate(self.ra):
                    # Check that there is flux in the band,
                    # otherwise calc_mag fails
                    if np.max(self.spec[i, in_band]) > 0:
                        temp_sed.set_sed(self.wave, flambda=self.spec[i, :])
                        mags[i] = temp_sed.calc_mag(bandpasses[key])
                # Mask out high airmass
                mags[self.mask] *= np.nan
                mags_back[key] = mags
        return mags_back
