from builtins import zip
import numpy as np
from .baseStacker import BaseStacker
from .coordStackers import raDec2AltAz

__all__ = ['findTelescopes', 'NFollowStacker']


def findTelescopes(minSize=3.):
    """Finds telescopes larger than minSize, from list of large telescopes based on
    http://astro.nineplanets.org/bigeyes.html.

    Returns
    -------
    np.recarray
        Array of large telescopes with columns [aperture, name, lat, lon].
    """
    # Aperture  Name Location http://astro.nineplanets.org/bigeyes.html
    telescopes = [
        [10.4, 'Gran Canarias', 'La Palma'],
        [10.0, 'Keck', 'Mauna Kea'],
        [10.0, 'Keck II', 'Mauna Kea'],
        [9.2, 'SALT', 'South African Astronomical Observatory'],
        [9.2, 'Hobby-Eberly', 'Mt. Fowlkes'],
        [8.4, 'Large Binocular Telescope', 'Mt. Graham'],
        [8.3, 'Subaru', 'Mauna Kea'],
        [8.2, 'Antu', 'Cerro Paranal'],
        [8.2, 'Kueyen', 'Cerro Paranal'],
        [8.2, 'Melipal', 'Cerro Paranal'],
        [8.2, 'Yepun', 'Cerro Paranal'],
        [8.1, 'Gemini North', 'Mauna Kea'],
        [8.1, 'Gemini South', 'Cerro Pachon'],
        [6.5, 'MMT', 'Mt. Hopkins'],
        [6.5, 'Walter Baade', 'La Serena'],
        [6.5, 'Landon Clay', 'La Serena'],
        [6.0, 'Bolshoi Teleskop Azimutalnyi', 'Nizhny Arkhyz'],
        [6.0, 'LZT', 'British Columbia'],
        [5.0, 'Hale', 'Palomar Mountain'],
        [4.3, 'Dicovery Channel', 'Lowell Observatory'],
        [4.2, 'William Herschel', 'La Palma'],
        [4.2, 'SOAR', 'Cerro Pachon'],
        [4.2, 'LAMOST', 'Xinglong Station'],
        [4.0, 'Victor Blanco', 'Cerro Tololo'],
        [4.0, 'Vista', 'Cerro Paranal'],
        [3.9, 'Anglo-Australian', 'Coonabarabran'],
        [3.8, 'Mayall', 'Kitt Peak'],
        [3.8, 'UKIRT', 'Mauna Kea'],
        [3.6, '360', 'Cerro La Silla'],
        [3.6, 'Canada-France-Hawaii', 'Mauna Kea'],
        [3.6, 'Telescopio Nazionale Galileo', 'La Palma'],
        [3.5, 'MPI-CAHA', 'Calar Alto'],
        [3.5, 'New Technology', 'Cerro La Silla'],
        [3.5, 'ARC', 'Apache Point'],
        [3.5, 'WIYN', 'Kitt Peak'],
        [3.0, 'Shane', 'Mount Hamilton'],
        [3.0, 'NASA IRTF', 'Mauna Kea'],
    ]

    scopes = np.zeros(len(telescopes), dtype = list(zip(
        ['aperture', 'name', 'lat', 'lon'], [float, (np.str_, 38), float, float])))

    # name, lat (S negative), lon (W negative)
    observatories = [
        ['Cerro Paranal', -24, 38, -70, 24],
        ['Nizhny Arkhyz', 43, 39, 41, 26],
        ['Cerro La Silla', -29, 15, -70, 44],
        ['Lowell Observatory', 35, 12, -111, 40],
        ['Apache Point', 32, 47, -105, 49],
        ['Mount Hamilton', 37, 21, -121, 38],
        ['South African Astronomical Observatory', -32, 23, 20, 49],
        ['Cerro Pachon', -30, 20, -70, 59],
        ['Coonabarabran', -31, 17, 149, 0o4],
        ['Mt. Fowlkes', 30, 40, -104, 1],
        ['La Palma', 28, 46, -17, 53],
        ['Mt. Graham', 32, 42, -109, 53],
        ['Calar Alto', 37, 13, -2, 33],
        ['British Columbia', 49, 17, -122, 34],
        ['Kitt Peak', 31, 57, -111, 37],
        ['La Serena', -30, 10, -70, 48],
        ['Palomar Mountain', 33, 21, -116, 52],
        ['Xinglong Station', 40, 23, 105, 50],
        ['Mt. Hopkins', 31, 41, -110, 53],
        ['Cerro Tololo', -30, 10, -70, 49],
        ['Mauna Kea', 19, 50, -155, 28]
    ]

    # Make a nice little dict to look up the observatory positions
    obs = {}
    for i, ob in enumerate(observatories):
        obs[ob[0]] = [(np.abs(ob[1])+ob[2]/60.)*(ob[1]/np.abs(ob[1])),
                      (np.abs(ob[3])+ob[4]/60.)*(ob[3]/np.abs(ob[3]))]

    for i, telescope in enumerate(telescopes):
        scopes['aperture'][i] = telescope[0]
        scopes['name'][i] = telescope[1]
        scopes['lat'][i], scopes['lon'][i] = obs[telescope[2]]

    scopes = scopes[np.where(scopes['aperture'] >= minSize)]
    return scopes


class NFollowStacker(BaseStacker):
    """Add the number of telescopes ('nObservatories') that could follow up any visit
    at (any of the) times in timeStep, specifying the minimum telescope size (in meters) and airmass limit.

    Parameters
    ----------
    minSize: float, opt
        The minimum telescope aperture to use, in meters. Default 3.0.
    airmassLimit: float, opt
        The maximum airmass allowable at the follow-up observatory. Default 2.5.
    timeSteps: np.array or list of floats, opt
        The timesteps to check for followup opportunities, in hours. Default is np.arange(0.5, 12., 3.0).
    mjdCol: str, opt
        The exposure MJD column name. Default 'observationStartMJD'.
    raCol: str, opt
        The RA column name. Default 'fieldRA'.
    decCol: str, opt
        The Dec column name. Default 'fieldDec'.
    raDecDeg: bool, opt
        Flag whether RA/Dec are in degrees (True) or radians (False).
    """
    colsAdded = ['nObservatories']

    def __init__(self, minSize=3.0, airmassLimit=2.5, timeSteps=np.arange(0.5, 12., 3.0),
                 mjdCol='observationStartMJD', raCol='fieldRA', decCol='fieldDec', degrees=True):
        self.mjdCol = mjdCol
        self.raCol = raCol
        self.decCol = decCol
        self.degrees = degrees
        self.colsAddedDtypes = [int]
        self.colsReq = [self.mjdCol, self.raCol, self.decCol]
        self.units = ['#']
        self.airmassLimit = airmassLimit
        self.timeSteps = timeSteps
        self.telescopes = findTelescopes(minSize = minSize)

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        simData['nObservatories'] = 0
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        for obs in self.telescopes:
            obsGotIt = np.zeros(len(simData[self.raCol]), int)
            obsLon = np.radians(obs['lon'])
            obsLat = np.radians(obs['lat'])
            for step in self.timeSteps:
                alt, az = raDec2AltAz(ra, dec, obsLon, obsLat,
                                      simData[self.mjdCol] + step / 24.0,
                                      altonly=True)
                airmass = 1. / (np.cos(np.pi / 2. - alt))
                followed = np.where((airmass <= self.airmassLimit) & (airmass >= 1.))
                # If the observatory got an observation, save this into obsGotIt.
                # obsGotIt will be 1 if ANY of the times got an observation.
                obsGotIt[followed] = 1
            # If an observatory got an observation, count it in nObservatories.
            simData['nObservatories'] += obsGotIt
        return simData
