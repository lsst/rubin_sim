__all__ = (
    "NormAirmassStacker",
    "ParallaxFactorStacker",
    "HourAngleStacker",
    "ZenithDistStacker",
    "ParallacticAngleStacker",
    "DcrStacker",
    "FiveSigmaStacker",
    "SaturationStacker",
    "OverheadStacker",
)

import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import GCRS, SkyCoord
from astropy.time import Time
from rubin_scheduler.utils import Site, m5_flat_sed

from rubin_sim.maf.utils import load_inst_zeropoints

from .base_stacker import BaseStacker
from .date_stackers import DayObsMJDStacker


class SaturationStacker(BaseStacker):
    """Adds a calculated point-source saturation limit for each visit.

    Assumes Gaussian PSF.

    Parameters
    ----------
    pixscale : `float`, optional
        Arcsec per pixel.
    saturation_e : `float`, optional
        The saturation level in electrons.
    zeropoints : dict-like, optional
        The zeropoints for the telescope.
        Keys should be str with filter names, values in mags.
        Default of None, will use Rubin calculated zeropoints.
    km : dict-like, optional
        Atmospheric extinction values.
        Keys should be str with filter names.
        If None, will use Rubin calculated atmospheric extinction values.
    """

    cols_added = ["saturation_mag"]

    def __init__(
        self,
        seeing_col="seeingFwhmEff",
        skybrightness_col="skyBrightness",
        exptime_col="visitExposureTime",
        nexp_col="numExposures",
        filter_col="filter",
        airmass_col="airmass",
        saturation_e=150e3,
        zeropoints=None,
        km=None,
        pixscale=0.2,
    ):
        self.units = ["mag"]
        self.cols_req = [
            seeing_col,
            skybrightness_col,
            exptime_col,
            nexp_col,
            filter_col,
            airmass_col,
        ]
        self.seeing_col = seeing_col
        self.skybrightness_col = skybrightness_col
        self.exptime_col = exptime_col
        self.nexp_col = nexp_col
        self.filter_col = filter_col
        self.airmass_col = airmass_col
        self.saturation_e = saturation_e
        self.pixscale = pixscale
        self.zeropoints = zeropoints
        self.km = km

    def _run(self, sim_data, cols_present=False):
        if self.zeropoints is None:
            zp_inst, k_atm = load_inst_zeropoints()
            self.zeropoints = zp_inst
        if self.km is None:
            self.km = k_atm
        for filtername in np.unique(sim_data[self.filter_col]):
            in_filt = np.where(sim_data[self.filter_col] == filtername)[0]
            # Calculate the length of the on-sky time per EXPOSURE
            exptime = sim_data[self.exptime_col][in_filt] / sim_data[self.nexp_col][in_filt]
            # Calculate sky counts per pixel per second
            # from skybrightness + zeropoint (e/1s)
            sky_counts = (
                10.0 ** (0.4 * (self.zeropoints[filtername] - sim_data[self.skybrightness_col][in_filt]))
                * self.pixscale**2
            )
            # Total sky counts in each exposure
            sky_counts = sky_counts * exptime
            # The counts available to the source (at peak) in each exposure is
            # the difference between saturation and sky
            remaining_counts_peak = self.saturation_e - sky_counts
            # Now to figure out how many counts there would be total, if there
            # are that many in the peak
            sigma = sim_data[self.seeing_col][in_filt] / 2.354
            source_counts = remaining_counts_peak * 2.0 * np.pi * (sigma / self.pixscale) ** 2
            # source counts = counts per exposure (expTimeCol / nexp)
            # Translate to counts per second, to apply zeropoint
            count_rate = source_counts / exptime
            sim_data["saturation_mag"][in_filt] = -2.5 * np.log10(count_rate) + self.zeropoints[filtername]
            # Airmass correction
            sim_data["saturation_mag"][in_filt] -= self.km[filtername] * (
                sim_data[self.airmass_col][in_filt] - 1.0
            )
            # Explicitly make sure if sky has saturated we return NaN
            sim_data["saturation_mag"][np.where(remaining_counts_peak < 0)] = np.nan

        return sim_data


class NoteRootStacker(BaseStacker):
    """Strip off things after a comma in the scheduler_note"""

    cols_added = ["scheduler_note_root"]

    def __init__(self, note_col="scheduler_note"):
        self.units = ["mag"]
        self.cols_req = [note_col]
        self.note_col = note_col
        self.cols_added_dtypes = ["<U50"]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is fine.
            return sim_data
        new_note = [note.split(",")[0] for note in sim_data[self.note_col]]
        sim_data["scheduler_note_root"] = new_note
        return sim_data


class FiveSigmaStacker(BaseStacker):
    """Calculate the 5-sigma limiting depth for a point source in the given
    conditions.

    This is generally not needed, unless the m5 parameters have been updated
    or m5 was not previously calculated.

    Parameters
    ----------
    airmass_col : `str`, optional
        Name of the airmass column in the data.
    seeing_col : `str`, optional
        Name of the seeing column in the data.
        (FWHM of the single-Gaussian PSF)
    skybrightness_col : `str`, optional
        Name of the skybrightness column in the data.
    filter_col : `str`, optional
        Name of the filter bandpass column in the data.
    exptime_col : `str`, optional
        Name of the on-sky exposure time column in the data.
    """

    cols_added = ["m5_simsUtils"]

    def __init__(
        self,
        airmass_col="airmass",
        seeing_col="seeingFwhmEff",
        skybrightness_col="skyBrightness",
        filter_col="filter",
        exptime_col="visitExposureTime",
    ):
        self.units = ["mag"]
        self.cols_req = [
            airmass_col,
            seeing_col,
            skybrightness_col,
            filter_col,
            exptime_col,
        ]
        self.airmass_col = airmass_col
        self.seeing_col = seeing_col
        self.skybrightness_col = skybrightness_col
        self.filter_col = filter_col
        self.exptime_col = exptime_col

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is fine.
            return sim_data
        filts = np.unique(sim_data[self.filter_col])
        for filtername in filts:
            infilt = np.where(sim_data[self.filter_col] == filtername)
            sim_data["m5_simsUtils"][infilt] = m5_flat_sed(
                filtername,
                sim_data[infilt][self.skybrightness_col],
                sim_data[infilt][self.seeing_col],
                sim_data[infilt][self.exptime_col],
                sim_data[infilt][self.airmass_col],
            )
        return sim_data


class NormAirmassStacker(BaseStacker):
    """Adds a calculated normairmass for each pointing.

    The normalized airmass is the airmass divided by the minimum airmass
    achievable at each pointing (which is defined by the declination of
    the field).

    Parameters
    ----------
    airmass_col : `str`, optional
        The name of the airmass column in the data.
    dec_col : `str`, optional
        The name of the declination column in the data.
    degrees : `bool`, optional
        If True, angle columns are assumed to be in degrees and returned in
        degrees. If False, uses and calculates radians.
    telescope_lat : `float`, optional
        The latitude of the telescope, in degrees.
    """

    cols_added = ["normairmass"]

    def __init__(
        self,
        airmass_col="airmass",
        dec_col="fieldDec",
        degrees=True,
        telescope_lat=-30.2446388,
    ):
        self.units = ["X / Xmin"]
        self.cols_req = [airmass_col, dec_col]
        self.airmass_col = airmass_col
        self.dec_col = dec_col
        self.telescope_lat = telescope_lat
        self.degrees = degrees

    def _run(self, sim_data, cols_present=False):
        # Run method is required to calculate column.
        # Driver runs getColInfo to know what columns are needed from db &
        # which are calculated,  then gets data from db and then calculates
        # additional columns (via run methods here).
        if cols_present:
            # Column already present in data; assume it is correct
            # and does not need recalculating.
            return sim_data
        dec = sim_data[self.dec_col]
        if self.degrees:
            dec = np.radians(dec)
        min_z_possible = np.abs(dec - np.radians(self.telescope_lat))
        min_airmass_possible = 1.0 / np.cos(min_z_possible)
        sim_data["normairmass"] = sim_data[self.airmass_col] / min_airmass_possible
        return sim_data


class ZenithDistStacker(BaseStacker):
    """Adds a calculated zenithDistance value for each pointing.

    Parameters
    ----------
    alt_col : `str`, optional
        The name of the altitude column in the data.
    degrees : `bool`, optional
        If True, data in alt_col is in degrees, and values for zenithDistance
        will be in degrees. (Default).
        If False, data in alt_col is in radians and zenithDistance values
        will be in radians.
    """

    cols_added = ["zenithDistance"]

    def __init__(self, alt_col="altitude", degrees=True):
        self.alt_col = alt_col
        self.degrees = degrees
        if self.degrees:
            self.units = ["degrees"]
        else:
            self.unit = ["radians"]
        self.cols_req = [self.alt_col]

    def _run(self, sim_data, cols_present=False):
        """Calculate new column for zenith distance."""
        if cols_present:
            # Column already present in data; assume it is correct and does not
            # need recalculating.
            return sim_data
        if self.degrees:
            sim_data["zenithDistance"] = 90.0 - sim_data[self.alt_col]
        else:
            sim_data["zenithDistance"] = np.pi / 2.0 - sim_data[self.alt_col]
        return sim_data


class ParallaxFactorStacker(BaseStacker):
    """Add a parallax factor (in arcseconds) column for each visit.

    Parameters
    ----------
    ra_col : `str`, optional
        Name of the RA column in the data.
    dec_col : `str`, optional
        Name of the declination column in the data.
    date_col : `str`, optional
        Name of the exposure start time column in the data.
        Date should be in units of MJD.
    degrees : `bool`, optional
        If true, assumes angles are in degrees. If False, radians.
    """

    cols_added = ["ra_pi_amp", "dec_pi_amp"]

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        date_col="observationStartMJD",
        degrees=True,
    ):
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.date_col = date_col
        self.units = ["arcsec", "arcsec"]
        self.cols_req = [ra_col, dec_col, date_col]
        self.degrees = degrees

    def _gnomonic_project_toxy(self, ra1, dec1, r_acen, deccen):
        # ra, dec values in RADIANS
        # also used in Global Telescope Network website
        cosc = np.sin(deccen) * np.sin(dec1) + np.cos(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)
        x = np.cos(dec1) * np.sin(ra1 - r_acen) / cosc
        y = (np.cos(deccen) * np.sin(dec1) - np.sin(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)) / cosc
        return x, y

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct
            # and does not need recalculating.
            return sim_data
        ra_pi_amp = np.zeros(np.size(sim_data), dtype=[("ra_pi_amp", "float")])
        dec_pi_amp = np.zeros(np.size(sim_data), dtype=[("dec_pi_amp", "float")])
        ra = sim_data[self.ra_col]
        dec = sim_data[self.dec_col]
        if self.degrees:
            ra = np.radians(ra)
            dec = np.radians(dec)

        times = Time(sim_data[self.date_col], format="mjd")
        c = SkyCoord(ra * u.rad, dec * u.rad, obstime=times)
        geo_far = c.transform_to(GCRS)
        c_near = SkyCoord(ra * u.rad, dec * u.rad, distance=1 * u.pc, obstime=times)
        geo_near = c_near.transform_to(GCRS)

        x_geo1, y_geo1 = self._gnomonic_project_toxy(geo_near.ra.rad, geo_near.dec.rad, ra, dec)
        x_geo, y_geo = self._gnomonic_project_toxy(geo_far.ra.rad, geo_far.dec.rad, ra, dec)

        # Return ra_pi_amp and dec_pi_amp in arcseconds.
        ra_pi_amp[:] = np.degrees(x_geo1 - x_geo) * 3600.0
        dec_pi_amp[:] = np.degrees(y_geo1 - y_geo) * 3600.0
        sim_data["ra_pi_amp"] = ra_pi_amp
        sim_data["dec_pi_amp"] = dec_pi_amp
        return sim_data


class DcrStacker(BaseStacker):
    """Add columns representing the expected RA/Dec offsets expected for
    an object due to differential chromatic refraction, per visit.

    For DCR calculation, we also need zenithDistance, HA, and PA -- but these
    will be explicitly handled within this stacker so that setup is consistent
    and they run in order. If those values have already been calculated
    elsewhere, they will not be overwritten.

    Parameters
    ----------
    filter_col : `str`, optional
        The name of the column with filter names. Default 'filter'.
    altCol : `str`, optional
        Name of the column with altitude info. Default 'altitude'.
    ra_col : `str`, optional
        Name of the column with RA. Default 'fieldRA'.
    dec_col : `str`, optional
        Name of the column with Dec. Default 'fieldDec'.
    lstCol : `str`, optional
        Name of the column with local sidereal time. Default
        'observationStartLST'.
    site : `str` or `rubin_scheduler.utils.Site`, optional
        Name of the observory or a rubin_scheduler.utils.Site object.
        Default 'LSST'.
    mjdCol : `str`, optional
        Name of column with modified julian date.
        Default 'observationStartMJD'
    dcr_magnitudes : dict
        Magnitude of the DCR offset for each filter at an
        altitude/zenith distance of 45 degrees.
        Defaults u=0.07, g=0.07, r=0.50, i=0.045, z=0.042, y=0.04
        (all values should be in arcseconds).

    Returns
    -------
    data : `numpy.array`
        Returns array with additional columns 'ra_dcr_amp' and 'dec_dcr_amp'
        with the DCR offsets for each observation.  Also runs ZenithDistStacker
        and ParallacticAngleStacker.
    """

    cols_added = ["ra_dcr_amp", "dec_dcr_amp"]  # zenithDist, HA, PA

    def __init__(
        self,
        filter_col="filter",
        alt_col="altitude",
        degrees=True,
        ra_col="fieldRA",
        dec_col="fieldDec",
        lst_col="observationStartLST",
        site="LSST",
        mjd_col="observationStartMJD",
        dcr_magnitudes=None,
    ):
        self.units = ["arcsec", "arcsec"]
        if dcr_magnitudes is None:
            # DCR amplitudes are in arcseconds.
            self.dcr_magnitudes = {
                "u": 0.07,
                "g": 0.07,
                "r": 0.050,
                "i": 0.045,
                "z": 0.042,
                "y": 0.04,
            }
        else:
            self.dcr_magnitudes = dcr_magnitudes
        self.zd_col = "zenithDistance"
        self.pa_col = "PA"
        self.filter_col = filter_col
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.degrees = degrees
        self.cols_req = [filter_col, ra_col, dec_col, alt_col, lst_col]
        #  'zenithDist', 'PA', 'HA' are additional columns required, coming
        #  from other stackers which must also be configured -- so we handle
        #  this explicitly here.
        self.zstacker = ZenithDistStacker(alt_col=alt_col, degrees=self.degrees)
        self.pastacker = ParallacticAngleStacker(
            ra_col=ra_col,
            dec_col=dec_col,
            mjd_col=mjd_col,
            degrees=self.degrees,
            lst_col=lst_col,
            site=site,
        )
        # Note that RA/Dec could be coming from a dither stacker!
        # But we will assume that coord stackers will be handled separately.

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not
            # need recalculating.
            return sim_data
        # Need to make sure the Zenith stacker gets run first Call _run method
        # because already added these columns due to 'colsAdded' line.
        sim_data = self.zstacker.run(sim_data)
        sim_data = self.pastacker.run(sim_data)
        if self.degrees:
            zenith_tan = np.tan(np.radians(sim_data[self.zd_col]))
            parallactic_angle = np.radians(sim_data[self.pa_col])
        else:
            zenith_tan = np.tan(sim_data[self.zd_col])
            parallactic_angle = sim_data[self.pa_col]
        dcr_in_ra = zenith_tan * np.sin(parallactic_angle)
        dcr_in_dec = zenith_tan * np.cos(parallactic_angle)
        for filtername in np.unique(sim_data[self.filter_col]):
            fmatch = np.where(sim_data[self.filter_col] == filtername)
            dcr_in_ra[fmatch] = self.dcr_magnitudes[filtername] * dcr_in_ra[fmatch]
            dcr_in_dec[fmatch] = self.dcr_magnitudes[filtername] * dcr_in_dec[fmatch]
        sim_data["ra_dcr_amp"] = dcr_in_ra
        sim_data["dec_dcr_amp"] = dcr_in_dec
        return sim_data


class HourAngleStacker(BaseStacker):
    """Add the Hour Angle (in decimal hours) for each observation.

    Parameters
    ----------
    lst_col : `str`, optional
        Name of the LST column in the data.
    ra_col : `str`, optional
        Name of the RA column in the data.
    degrees : `bool`, optional
        If True, assumes angles (RA and LST) are in degrees.
        If False, assumes radians.
    """

    cols_added = ["HA"]

    def __init__(self, lst_col="observationStartLST", ra_col="fieldRA", degrees=True):
        self.units = ["Hours"]
        self.cols_req = [lst_col, ra_col]
        self.lst_col = lst_col
        self.ra_col = ra_col
        self.degrees = degrees

    def _run(self, sim_data, cols_present=False):
        """HA = LST - RA"""
        if cols_present:
            # Column already present in data; assume it is correct and does not
            # need recalculating.
            return sim_data
        if len(sim_data) == 0:
            return sim_data
        if self.degrees:
            ra = np.radians(sim_data[self.ra_col])
            lst = np.radians(sim_data[self.lst_col])
        else:
            ra = sim_data[self.ra_col]
            lst = sim_data[self.lst_col]
        # Check that LST is reasonable
        if (np.min(lst) < 0) | (np.max(lst) > 2.0 * np.pi):
            warnings.warn("LST values are not between 0 and 2 pi")
        # Check that RA is reasonable
        if (np.min(ra) < 0) | (np.max(ra) > 2.0 * np.pi):
            warnings.warn("RA values are not between 0 and 2 pi")
        ha = lst - ra
        # Wrap the results so HA between -pi and pi
        ha = np.where(ha < -np.pi, ha + 2.0 * np.pi, ha)
        ha = np.where(ha > np.pi, ha - 2.0 * np.pi, ha)
        # Convert radians to hours
        sim_data["HA"] = ha * 12 / np.pi
        return sim_data


class ParallacticAngleStacker(BaseStacker):
    """Add the calculated parallactic angle to each visit.

    Parameters
    ----------
    ra_col : `str`, optional
        Name of the RA column in the data.
    dec_col : `str`, optional
        Name of the declination column in the data.
    degrees : `bool`, optional
        If True, assumes ra and dec in degrees and returns Parallactic Angle
        in degrees. If False, assumes and returns radians.
    mjd_col : `str`, optional
        Name of the observation MJD column in the data.
    lst_col : `str`, optional
        Name of the LST column in the data.
    site : `str` or `rubin_scheduler.utils.Site`, optional
        Name of the observory or a rubin_scheduler.utils.Site object.
        Default 'LSST'.
    """

    cols_added = ["PA"]

    def __init__(
        self,
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
        mjd_col="observationStartMJD",
        lst_col="observationStartLST",
        site="LSST",
    ):
        self.lst_col = lst_col
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.degrees = degrees
        self.mjd_col = mjd_col
        self.site = Site(name=site)
        self.units = ["radians"]
        self.cols_req = [self.ra_col, self.dec_col, self.mjd_col, self.lst_col]
        self.ha_stacker = HourAngleStacker(lst_col=lst_col, ra_col=ra_col, degrees=self.degrees)

    def _run(self, sim_data, cols_present=False):
        # Equation from:
        # http://www.gb.nrao.edu/~rcreager/GBTMetrology/140ft/l0058/gbtmemo52/memo52.html
        # or
        # http://www.gb.nrao.edu/GBT/DA/gbtidl/release2pt9/contrib/contrib/parangle.pro
        if cols_present:
            # Column already present in data; assume it is correct and does not
            # need recalculating.
            return sim_data
        # Using the run method (not _run) means that if HA is present, it will
        # not be recalculated.
        sim_data = self.ha_stacker.run(sim_data)
        if self.degrees:
            dec = np.radians(sim_data[self.dec_col])
        else:
            dec = sim_data[self.dec_col]
        sim_data["PA"] = np.arctan2(
            np.sin(sim_data["HA"] * np.pi / 12.0),
            (
                np.cos(dec) * np.tan(self.site.latitude_rad)
                - np.sin(dec) * np.cos(sim_data["HA"] * np.pi / 12.0)
            ),
        )
        if self.degrees:
            sim_data["PA"] = np.degrees(sim_data["PA"])
        return sim_data


class OverheadStacker(BaseStacker):
    """Add time between visits in seconds.

    Parameters
    ----------
    max_gap : `float`, optional
        The maximum gap between observations, in minutes.
        Assume anything longer the dome has closed.
        Defaults to infinity.
    mjd_col : `str`, optional
        The name of the column with the observation start MJD.
        Defaults to "observationStartMJD".
    visit_time_col : `str`, optional
        The name of the column with the total visit time (on-sky plus
        shutter and other overheads).
        Defaults to "visitTime".
    exposure_time_col : `str`, optional
        The name of the column with the visit on-sky exposure time.
        Defaults to "visitExposureTime."
    """

    cols_added = ["overhead"]

    def __init__(
        self,
        max_gap=np.inf,
        mjd_col="observationStartMJD",
        visit_time_col="visitTime",
        exposure_time_col="visitExposureTime",
    ):
        # Set max_gap in minutes to match API of existing BruteOSFMetric
        self.max_gap = max_gap
        self.mjd_col = mjd_col
        self.visit_time_col = visit_time_col
        self.exposure_time_col = exposure_time_col
        self.units = ["seconds"]
        self.day_obs_mjd_stacker = DayObsMJDStacker(self.mjd_col)
        self.cols_req = [self.mjd_col, self.visit_time_col, self.exposure_time_col]
        self.cols_req.extend(self.day_obs_mjd_stacker.cols_req)
        self.cols_req = list(set(self.cols_req))

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not
            # need recalculating.
            return sim_data

        # Count overhead as any time from the shutter close of the previous
        # visit, to the shutter close of the current visit, minus the
        # current on-sky exposure time.
        # This is all non-exposure time required for this visit --
        # includes slew time to this target, any shutter overheads for this
        # visit, and any readouts not absorbed in slewtime for this visit.
        observation_end_mjd = sim_data[self.mjd_col] + sim_data[self.visit_time_col] / (24 * 60 * 60)
        overhead = (
            np.diff(observation_end_mjd, prepend=np.nan) * 24 * 60 * 60 - sim_data[self.exposure_time_col]
        )

        # Rough heuristic not to count downtime due to weather or instrument
        # problems as overhead.  A more reliable way to do this would be to
        # use other sources of weather data or problem reporting to identify
        # these gaps.
        # We might also explicitly want to look at the gaps that arise from
        # these problems in order to measure time lost to these causes.
        overhead[overhead > self.max_gap * 60] = np.nan

        # If the gap includes a change of night, it isn't really overhead. For
        # most reasonable finite values of self.max_gap, this is never
        # relevant, but it's sometimes useful to set max_gap to inifinity to
        # catch all delays in the night. (The comment above.) But, in
        # even in this case, we would still not want to include gaps between
        # nights.
        day_obs_mjd = self.day_obs_mjd_stacker.run(sim_data)["day_obs_mjd"]
        different_night = np.diff(day_obs_mjd, prepend=np.nan) != 0
        overhead[different_night] = np.nan

        sim_data[self.cols_added[0]] = overhead
        return sim_data
