import numpy as np
from rubin_sim.utils import _galacticFromEquatorial, calcLmstLast
from astropy import units as u
from astropy.coordinates import SkyCoord, get_sun
from .baseStacker import BaseStacker
from .ditherStackers import wrapRA
from astropy.time import Time


__all__ = ['raDec2AltAz', 'GalacticStacker', 'EclipticStacker']


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
    altonly : bool, optional
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
    raCol : str, optional
        Name of the RA column. Default fieldRA.
    decCol : str, optional
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
    mjdCol : str, optional
        Name of the MJD column. Default expMJD.
    raCol : str, optional
        Name of the RA column. Default fieldRA.
    decCol : str, optional
        Name of the Dec column. Default fieldDec.
    subtractSunLon : bool, optional
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
                coord = SkyCoord(ra=simData[self.raCol]*u.degree, dec=simData[self.decCol]*u.degree)
            else:
                coord = SkyCoord(ra=simData[self.raCol]*u.rad, dec=simData[self.decCol]*u.rad)
            coord_ecl = coord.geocentricmeanecliptic
            simData['eclipLat'] = coord_ecl.lat.rad

            if self.subtractSunLon:
                times = Time(simData[self.mjdCol])
                sun = get_sun(times)
                sunEcl = sun.geocentricmeanecliptic
                lon = wrapRA(coord_ecl.lon.rad - sunEcl.lon.rad)
                simData['eclipLon'] = lon
            else:
                simData['eclipLon'] = coord_ecl.lon.rad
        if self.degrees:
            simData['eclipLon'] = np.degrees(simData['eclipLon'])
            simData['eclipLat'] = np.degrees(simData['eclipLat'])
        return simData
