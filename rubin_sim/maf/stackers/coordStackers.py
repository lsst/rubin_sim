import numpy as np
import ephem
from rubin_sim.utils import _galacticFromEquatorial, calcLmstLast

from .baseStacker import BaseStacker
from .ditherStackers import wrapRA

__all__ = ['mjd2djd', 'raDec2AltAz', 'GalacticStacker', 'EclipticStacker']


def mjd2djd(mjd):
    """Convert MJD to the Dublin Julian date used by ephem.

    Parameters
    ----------
    mjd : float or numpy.ndarray
        The modified julian date.
    Returns
    -------
    float or numpy.ndarray
        The dublin julian date.
    """
    doff = ephem.Date(0)-ephem.Date('1858/11/17')
    djd = mjd-doff
    return djd


def raDec2AltAz(ra, dec, lat, lon, mjd, altonly=False):
    """Convert RA/Dec (and telescope site lat/lon) to alt/az.

    This uses simple equations and ignores aberation, precession, nutation, etc.

    Parameters
    ----------
    ra : array_like
        RA, in radians.
    dec : array_like
        Dec, in radians. Must be same length as `ra`.
    lat : float
        Latitude of the observatory in radians.
    lon : float
        Longitude of the observatory in radians.
    mjd : float
        Modified Julian Date.
    altonly : bool, opt
        Calculate altitude only.

    Returns
    -------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Radians.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Radians.
    """
    lmst, last = calcLmstLast(mjd, lon)
    lmst = lmst / 12. * np.pi  # convert to rad
    ha = lmst - ra
    sindec = np.sin(dec)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    sinalt = sindec * sinlat + np.cos(dec) * coslat * np.cos(ha)
    # make sure sinalt is in the expected range.
    sinalt = np.where(sinalt < -1, -1, sinalt)
    sinalt = np.where(sinalt > 1, 1, sinalt)
    alt = np.arcsin(sinalt)
    if altonly:
        az = None
    else:
        cosaz = (sindec-np.sin(alt)*sinlat)/(np.cos(alt)*coslat)
        cosaz = np.where(cosaz < -1, -1, cosaz)
        cosaz = np.where(cosaz > 1, 1, cosaz)
        az = np.arccos(cosaz)
        signflip = np.where(np.sin(ha) > 0)
        az[signflip] = 2.*np.pi-az[signflip]
    return alt, az


class GalacticStacker(BaseStacker):
    """Add the galactic coordinates of each RA/Dec pointing: gall, galb

    Parameters
    ----------
    raCol : str, opt
        Name of the RA column. Default fieldRA.
    decCol : str, opt
        Name of the Dec column. Default fieldDec.
    """
    colsAdded = ['gall', 'galb']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True):
        self.colsReq = [raCol, decCol]
        self.raCol = raCol
        self.decCol = decCol
        self.degrees = degrees
        if self.degrees:
            self.units = ['degrees', 'degrees']
        else:
            self.units = ['radians', 'radians']

    def _run(self, simData, cols_present=False):
        # raCol and DecCol in radians, gall/b in radians.
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        if self.degrees:
            simData['gall'], simData['galb'] = _galacticFromEquatorial(np.radians(simData[self.raCol]),
                                                                       np.radians(simData[self.decCol]))
        else:
            simData['gall'], simData['galb'] = _galacticFromEquatorial(simData[self.raCol],
                                                                       simData[self.decCol])
        return simData


class EclipticStacker(BaseStacker):
    """Add the ecliptic coordinates of each RA/Dec pointing: eclipLat, eclipLon
    Optionally subtract off the sun's ecliptic longitude and wrap.

    Parameters
    ----------
    mjdCol : str, opt
        Name of the MJD column. Default expMJD.
    raCol : str, opt
        Name of the RA column. Default fieldRA.
    decCol : str, opt
        Name of the Dec column. Default fieldDec.
    subtractSunLon : bool, opt
        Flag to subtract the sun's ecliptic longitude. Default False.
    """
    colsAdded = ['eclipLat', 'eclipLon']

    def __init__(self, mjdCol='observationStartMJD', raCol='fieldRA', decCol='fieldDec', degrees=True,
                 subtractSunLon=False):

        self.colsReq = [mjdCol, raCol, decCol]
        self.subtractSunLon = subtractSunLon
        self.degrees = degrees
        if self.degrees:
            self.units = ['degrees', 'degrees']
        else:
            self.units = ['radians', 'radians']
        self.mjdCol = mjdCol
        self.raCol = raCol
        self.decCol = decCol

    def _run(self, simData, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        for i in np.arange(simData.size):
            if self.degrees:
                coord = ephem.Equatorial(np.radians(simData[self.raCol][i]),
                                         np.radians(simData[self.decCol][i]), epoch=2000)
            else:
                coord = ephem.Equatorial(simData[self.raCol][i],
                                         simData[self.decCol][i], epoch=2000)
            ecl = ephem.Ecliptic(coord)
            simData['eclipLat'][i] = ecl.lat
            if self.subtractSunLon:
                djd = mjd2djd(simData[self.mjdCol][i])
                sun = ephem.Sun(djd)
                sunEcl = ephem.Ecliptic(sun)
                lon = wrapRA(ecl.lon - sunEcl.lon)
                simData['eclipLon'][i] = lon
            else:
                simData['eclipLon'][i] = ecl.lon
        if self.degrees:
            simData['eclipLon'] = np.degrees(simData['eclipLon'])
            simData['eclipLat'] = np.degrees(simData['eclipLat'])
        return simData
