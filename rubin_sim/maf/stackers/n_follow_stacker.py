__all__ = ("find_telescopes", "NFollowStacker")


import numpy as np

from .base_stacker import BaseStacker
from .coord_stackers import ra_dec2_alt_az


def find_telescopes(min_size=3.0):
    """Finds telescopes larger than min_size,
    from list of large telescopes based on
    http://astro.nineplanets.org/bigeyes.html.

    Returns
    -------
    np.recarray
        Array of large telescopes with columns [aperture, name, lat, lon].
    """
    # Aperture  Name Location http://astro.nineplanets.org/bigeyes.html
    telescopes = [
        [10.4, "Gran Canarias", "La Palma"],
        [10.0, "Keck", "Mauna Kea"],
        [10.0, "Keck II", "Mauna Kea"],
        [9.2, "SALT", "South African Astronomical Observatory"],
        [9.2, "Hobby-Eberly", "Mt. Fowlkes"],
        [8.4, "Large Binocular Telescope", "Mt. Graham"],
        [8.3, "Subaru", "Mauna Kea"],
        [8.2, "Antu", "Cerro Paranal"],
        [8.2, "Kueyen", "Cerro Paranal"],
        [8.2, "Melipal", "Cerro Paranal"],
        [8.2, "Yepun", "Cerro Paranal"],
        [8.1, "Gemini North", "Mauna Kea"],
        [8.1, "Gemini South", "Cerro Pachon"],
        [6.5, "MMT", "Mt. Hopkins"],
        [6.5, "Walter Baade", "La Serena"],
        [6.5, "Landon Clay", "La Serena"],
        [6.0, "Bolshoi Teleskop Azimutalnyi", "Nizhny Arkhyz"],
        [6.0, "LZT", "British Columbia"],
        [5.0, "Hale", "Palomar Mountain"],
        [4.3, "Dicovery Channel", "Lowell Observatory"],
        [4.2, "William Herschel", "La Palma"],
        [4.2, "SOAR", "Cerro Pachon"],
        [4.2, "LAMOST", "Xinglong Station"],
        [4.0, "Victor Blanco", "Cerro Tololo"],
        [4.0, "Vista", "Cerro Paranal"],
        [3.9, "Anglo-Australian", "Coonabarabran"],
        [3.8, "Mayall", "Kitt Peak"],
        [3.8, "UKIRT", "Mauna Kea"],
        [3.6, "360", "Cerro La Silla"],
        [3.6, "Canada-France-Hawaii", "Mauna Kea"],
        [3.6, "Telescopio Nazionale Galileo", "La Palma"],
        [3.5, "MPI-CAHA", "Calar Alto"],
        [3.5, "New Technology", "Cerro La Silla"],
        [3.5, "ARC", "Apache Point"],
        [3.5, "WIYN", "Kitt Peak"],
        [3.0, "Shane", "Mount Hamilton"],
        [3.0, "NASA IRTF", "Mauna Kea"],
    ]

    scopes = np.zeros(
        len(telescopes),
        dtype=list(zip(["aperture", "name", "lat", "lon"], [float, (np.str_, 38), float, float])),
    )

    # name, lat (S negative), lon (W negative)
    observatories = [
        ["Cerro Paranal", -24, 38, -70, 24],
        ["Nizhny Arkhyz", 43, 39, 41, 26],
        ["Cerro La Silla", -29, 15, -70, 44],
        ["Lowell Observatory", 35, 12, -111, 40],
        ["Apache Point", 32, 47, -105, 49],
        ["Mount Hamilton", 37, 21, -121, 38],
        ["South African Astronomical Observatory", -32, 23, 20, 49],
        ["Cerro Pachon", -30, 20, -70, 59],
        ["Coonabarabran", -31, 17, 149, 0o4],
        ["Mt. Fowlkes", 30, 40, -104, 1],
        ["La Palma", 28, 46, -17, 53],
        ["Mt. Graham", 32, 42, -109, 53],
        ["Calar Alto", 37, 13, -2, 33],
        ["British Columbia", 49, 17, -122, 34],
        ["Kitt Peak", 31, 57, -111, 37],
        ["La Serena", -30, 10, -70, 48],
        ["Palomar Mountain", 33, 21, -116, 52],
        ["Xinglong Station", 40, 23, 105, 50],
        ["Mt. Hopkins", 31, 41, -110, 53],
        ["Cerro Tololo", -30, 10, -70, 49],
        ["Mauna Kea", 19, 50, -155, 28],
    ]

    # Make a nice little dict to look up the observatory positions
    obs = {}
    for i, ob in enumerate(observatories):
        obs[ob[0]] = [
            (np.abs(ob[1]) + ob[2] / 60.0) * (ob[1] / np.abs(ob[1])),
            (np.abs(ob[3]) + ob[4] / 60.0) * (ob[3] / np.abs(ob[3])),
        ]

    for i, telescope in enumerate(telescopes):
        scopes["aperture"][i] = telescope[0]
        scopes["name"][i] = telescope[1]
        scopes["lat"][i], scopes["lon"][i] = obs[telescope[2]]

    scopes = scopes[np.where(scopes["aperture"] >= min_size)]
    return scopes


class NFollowStacker(BaseStacker):
    """Add the number of telescopes ('nObservatories') that could
    follow up any visit at (any of the) times in timeStep,
    specifying the minimum telescope size (in meters) and airmass limit.

    Parameters
    ----------
    minSize: float, optional
        The minimum telescope aperture to use, in meters. Default 3.0.
    airmass_limit: float, optional
        The maximum airmass allowable at the follow-up observatory.
        Default 2.5.
    time_steps: np.array or list of floats, optional
        The timesteps to check for followup opportunities, in hours.
        Default is np.arange(0.5, 12., 3.0).
    mjd_col: str, optional
        The exposure MJD column name. Default 'observationStartMJD'.
    ra_col: str, optional
        The RA column name. Default 'fieldRA'.
    dec_col: str, optional
        The Dec column name. Default 'fieldDec'.
    raDecDeg: bool, optional
        Flag whether RA/Dec are in degrees (True) or radians (False).
    """

    cols_added = ["nObservatories"]

    def __init__(
        self,
        min_size=3.0,
        airmass_limit=2.5,
        time_steps=np.arange(0.5, 12.0, 3.0),
        mjd_col="observationStartMJD",
        ra_col="fieldRA",
        dec_col="fieldDec",
        degrees=True,
    ):
        self.mjd_col = mjd_col
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.degrees = degrees
        self.cols_added_dtypes = [int]
        self.cols_req = [self.mjd_col, self.ra_col, self.dec_col]
        self.units = ["#"]
        self.airmass_limit = airmass_limit
        self.time_steps = time_steps
        self.telescopes = find_telescopes(min_size=min_size)

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            return sim_data
        sim_data["nObservatories"] = 0
        if self.degrees:
            ra = np.radians(sim_data[self.ra_col])
            dec = np.radians(sim_data[self.dec_col])
        else:
            ra = sim_data[self.ra_col]
            dec = sim_data[self.dec_col]
        for obs in self.telescopes:
            obs_got_it = np.zeros(len(sim_data[self.ra_col]), int)
            obs_lon = np.radians(obs["lon"])
            obs_lat = np.radians(obs["lat"])
            for step in self.time_steps:
                alt, az = ra_dec2_alt_az(
                    ra,
                    dec,
                    obs_lon,
                    obs_lat,
                    sim_data[self.mjd_col] + step / 24.0,
                    altonly=True,
                )
                airmass = 1.0 / (np.cos(np.pi / 2.0 - alt))
                followed = np.where((airmass <= self.airmass_limit) & (airmass >= 1.0))
                # If the observatory got an observation, save this
                # into obs_got_it.
                # obs_got_it will be 1 if ANY of the times got an observation.
                obs_got_it[followed] = 1
            # If an observatory got an observation, count it in nObservatories.
            sim_data["nObservatories"] += obs_got_it
        return sim_data
