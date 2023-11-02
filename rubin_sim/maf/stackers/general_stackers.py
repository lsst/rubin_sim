__all__ = (
    "NormAirmassStacker",
    "ParallaxFactorStacker",
    "HourAngleStacker",
    "ZenithDistStacker",
    "ParallacticAngleStacker",
    "DcrStacker",
    "FiveSigmaStacker",
    "SaturationStacker",
)

import warnings

import numpy as np
import palpy
from rubin_scheduler.utils import Site, m5_flat_sed

from rubin_sim.maf.utils import load_inst_zeropoints

from .base_stacker import BaseStacker


class SaturationStacker(BaseStacker):
    """Calculate the saturation limit of a point source. Assumes Guassian PSF.

    Parameters
    ----------
    pixscale : float, optional (0.2)
        Arcsec per pixel
    saturation_e : float, optional (150e3)
        The saturation level in electrons
    zeropoints : dict-like, optional (None)
        The zeropoints for the telescope. Keys should be str with filter names, values in mags.
        If None, will use Rubin-like zeropoints.
    km : dict-like, optional (None)
        Atmospheric extinction values.  Keys should be str with filter names.
        If None, will use Rubin-like zeropoints.
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
            # Calculate sky counts per pixel per second from skybrightness + zeropoint (e/1s)
            sky_counts = (
                10.0 ** (0.4 * (self.zeropoints[filtername] - sim_data[self.skybrightness_col][in_filt]))
                * self.pixscale**2
            )
            # Total sky counts in each exposure
            sky_counts = sky_counts * exptime
            # The counts available to the source (at peak) in each exposure is the
            # difference between saturation and sky
            remaining_counts_peak = self.saturation_e - sky_counts
            # Now to figure out how many counts there would be total, if there are that many in the peak
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


class FiveSigmaStacker(BaseStacker):
    """
    Calculate the 5-sigma limiting depth for a point source in the given conditions.

    This is generally not needed, unless the m5 parameters have been updated
    or m5 was not previously calculated.
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
    """Calculate the normalized airmass for each opsim pointing."""

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
        """Calculate new column for normalized airmass."""
        # Run method is required to calculate column.
        # Driver runs getColInfo to know what columns are needed from db & which are calculated,
        #  then gets data from db and then calculates additional columns (via run methods here).
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return sim_data
        dec = sim_data[self.dec_col]
        if self.degrees:
            dec = np.radians(dec)
        min_z_possible = np.abs(dec - np.radians(self.telescope_lat))
        min_airmass_possible = 1.0 / np.cos(min_z_possible)
        sim_data["normairmass"] = sim_data[self.airmass_col] / min_airmass_possible
        return sim_data


class ZenithDistStacker(BaseStacker):
    """Calculate the zenith distance for each pointing.
    If 'degrees' is True, then assumes alt_col is in degrees and returns degrees.
    If 'degrees' is False, assumes alt_col is in radians and returns radians.
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
            # Column already present in data; assume it is correct and does not need recalculating.
            return sim_data
        if self.degrees:
            sim_data["zenithDistance"] = 90.0 - sim_data[self.alt_col]
        else:
            sim_data["zenithDistance"] = np.pi / 2.0 - sim_data[self.alt_col]
        return sim_data


class ParallaxFactorStacker(BaseStacker):
    """Calculate the parallax factors for each opsim pointing.  Output parallax factor in arcseconds."""

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
        """Calculate x/y projection of ra1/dec1 in system with center at r_acen, Deccenp.
        Input radians.
        """
        # also used in Global Telescope Network website
        cosc = np.sin(deccen) * np.sin(dec1) + np.cos(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)
        x = np.cos(dec1) * np.sin(ra1 - r_acen) / cosc
        y = (np.cos(deccen) * np.sin(dec1) - np.sin(deccen) * np.cos(dec1) * np.cos(ra1 - r_acen)) / cosc
        return x, y

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return sim_data
        ra_pi_amp = np.zeros(np.size(sim_data), dtype=[("ra_pi_amp", "float")])
        dec_pi_amp = np.zeros(np.size(sim_data), dtype=[("dec_pi_amp", "float")])
        ra_geo1 = np.zeros(np.size(sim_data), dtype="float")
        dec_geo1 = np.zeros(np.size(sim_data), dtype="float")
        ra_geo = np.zeros(np.size(sim_data), dtype="float")
        dec_geo = np.zeros(np.size(sim_data), dtype="float")
        ra = sim_data[self.ra_col]
        dec = sim_data[self.dec_col]
        if self.degrees:
            ra = np.radians(ra)
            dec = np.radians(dec)

        for i, ack in enumerate(sim_data):
            mtoa_params = palpy.mappa(2000.0, sim_data[self.date_col][i])
            # Object with a 1 arcsec parallax
            ra_geo1[i], dec_geo1[i] = palpy.mapqk(ra[i], dec[i], 0.0, 0.0, 1.0, 0.0, mtoa_params)
            # Object with no parallax
            ra_geo[i], dec_geo[i] = palpy.mapqk(ra[i], dec[i], 0.0, 0.0, 0.0, 0.0, mtoa_params)
        x_geo1, y_geo1 = self._gnomonic_project_toxy(ra_geo1, dec_geo1, ra, dec)
        x_geo, y_geo = self._gnomonic_project_toxy(ra_geo, dec_geo, ra, dec)
        # Return ra_pi_amp and dec_pi_amp in arcseconds.
        ra_pi_amp[:] = np.degrees(x_geo1 - x_geo) * 3600.0
        dec_pi_amp[:] = np.degrees(y_geo1 - y_geo) * 3600.0
        sim_data["ra_pi_amp"] = ra_pi_amp
        sim_data["dec_pi_amp"] = dec_pi_amp
        return sim_data


class DcrStacker(BaseStacker):
    """Calculate the RA,Dec offset expected for an object due to differential chromatic refraction.

    For DCR calculation, we also need zenithDistance, HA, and PA -- but these will be explicitly
    handled within this stacker so that setup is consistent and they run in order. If those values
    have already been calculated elsewhere, they will not be overwritten.

    Parameters
    ----------
    filter_col : str
        The name of the column with filter names. Default 'fitler'.
    altCol : str
        Name of the column with altitude info. Default 'altitude'.
    ra_col : str
        Name of the column with RA. Default 'fieldRA'.
    dec_col : str
        Name of the column with Dec. Default 'fieldDec'.
    lstCol : str
        Name of the column with local sidereal time. Default 'observationStartLST'.
    site : str or rubin_sim.utils.Site
        Name of the observory or a rubin_sim.utils.Site object. Default 'LSST'.
    mjdCol : str
        Name of column with modified julian date. Default 'observationStartMJD'
    dcr_magnitudes : dict
        Magitude of the DCR offset for each filter at altitude/zenith distance of 45 degrees.
        Defaults u=0.07, g=0.07, r=0.50, i=0.045, z=0.042, y=0.04 (all arcseconds).

    Returns
    -------
    numpy.array
        Returns array with additional columns 'ra_dcr_amp' and 'dec_dcr_amp' with the DCR offsets
        for each observation.  Also runs ZenithDistStacker and ParallacticAngleStacker.
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
        #  'zenithDist', 'PA', 'HA' are additional columns required, coming from other stackers which must
        #  also be configured -- so we handle this explicitly here.
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
            # Column already present in data; assume it is correct and does not need recalculating.
            return sim_data
        # Need to make sure the Zenith stacker gets run first
        # Call _run method because already added these columns due to 'colsAdded' line.
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
    """Add the Hour Angle for each observation.
    Always in HOURS.
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
            # Column already present in data; assume it is correct and does not need recalculating.
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
    """Add the parallactic angle to each visit.
    If 'degrees' is True, this will be in degrees (as are all other angles). If False, then in radians.
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
            # Column already present in data; assume it is correct and does not need recalculating.
            return sim_data
        # Using the run method (not _run) means that if HA is present, it will not be recalculated.
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
