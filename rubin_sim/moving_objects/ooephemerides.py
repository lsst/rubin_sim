__all__ = ("get_oorb_data_dir", "PyOrbEphemerides")

import os
import time
import warnings
from itertools import repeat

import numpy as np
import pandas as pd

try:
    import pyoorb as oo

    PYOORB_PRESENT = True

except ModuleNotFoundError:
    PYOORB_PRESENT = False


def dtime(time_prev):
    return (time.time() - time_prev, time.time())


def get_oorb_data_dir():
    """Find where the oorb files should be installed"""
    data_path = os.getenv("OORB_DATA")
    if data_path is None:
        # See if we are in a conda enviroment and can find it
        conda_dir = os.getenv("CONDA_PREFIX")
        if conda_dir is not None:
            data_path = os.path.join(conda_dir, "share/openorb")
            if not os.path.isdir(data_path):
                data_path = None
        os.environ["OORB_DATA"] = data_path
    if data_path is None:
        warnings.warn(
            "Failed to find path for oorb data files. "
            "No $OORB_DATA environment variable set, and they are not the usual conda spot"
        )
    return data_path


class PyOrbEphemerides:
    """Generate ephemerides and propagate orbits,
    using the python interface to Oorb.

    PyOrbEphemerides handles the packing and unpacking of the fortran style
    arrays that pyoorb uses, to and from more user-friendly pandas arrays.

    Parameters
    ----------
    ephfile : `str`, optional
        Planetary ephemerides file for Oorb (i.e. de430 or de405).
        Default $OORB_DATA/de430.dat  ($OORB_DATA = $OORB_DIR/data).

    Examples
    --------
    Typical usage:

    >>> pyephs = PyOrbEphemerides()
    >>> pyephs.set_orbits(orbits)
    >>> ephs = pyephs.generateEphemerides(times, timeScale, obscode)
    """

    def __init__(self, ephfile=None):

        if not PYOORB_PRESENT:
            warnings.warn("No pyoorb available, use another ephemeris generator.")
            raise ModuleNotFoundError

        # Set translation from timescale to OpenOrb numerical representation.
        # Note all orbits are assumed to be in TT timescale.
        # Also, all dates are expected to be in MJD.
        self.time_scales = {"UTC": 1, "UT1": 2, "TT": 3, "TAI": 4}
        self.elem_type = {"CAR": 1, "COM": 2, "KEP": 3, "DEL": 4, "EQX": 5}

        # Set up oorb. Call this once.
        if ephfile is None:
            ephfile = os.path.join(get_oorb_data_dir(), "de430.dat")
        self.ephfile = ephfile
        self._init_oorb()
        self.oorb_elem = None
        self.orb_format = None

    def _init_oorb(self):
        oo.pyoorb.oorb_init(ephemeris_fname=self.ephfile)

    def set_orbits(self, orbit_obj):
        """Set the orbits, to be used to generate ephemerides.

        Immediately calls self._convertOorbElem to translate to the
        'packed' oorb format.

        Parameters
        ----------
        orbit_obj : `rubin_sim.moving_objects.Orbits`
           The orbits to use to generate ephemerides.
        """
        if len(orbit_obj) == 0:
            raise ValueError("There are no orbits in the Orbit object.")
        self._convert_to_oorb_elem(orbit_obj.orbits, orbit_obj.orb_format)

    def _convert_to_oorb_elem(self, orbit_dataframe, orb_format):
        """Convert orbital elements into the numpy fortran-format
        array OpenOrb requires.

        The OpenOrb element format is a single array with elements:
        0 : orbitId (cannot be a string)
        1-6 : orbital elements, using radians for angles
        7 : element 'type' code
        (1 = CAR, 2 = COM, 3 = KEP, 4 = DELauny, 5 = EQX (equinoctial))
        8 : epoch
        9 : timescale for epoch
        (1 = UTC, 2 = UT1, 3 = TT, 4 = TAI : always assumes TT)
        10 : magHv
        11 : g

        Sets self.oorb_elem, the orbit parameters in an array
        formatted for OpenOrb.
        """
        oorb_elem = np.zeros([len(orbit_dataframe), 12], dtype=np.double, order="F")
        # Put in simple values for objid, or add method to test if
        # any obj_id is a string.
        # NOTE THAT THIS MEANS WE'VE LOST THE OBJID
        oorb_elem[:, 0] = np.arange(0, len(orbit_dataframe), dtype=int) + 1
        # Add the appropriate element and epoch types:
        oorb_elem[:, 7] = np.zeros(len(orbit_dataframe), float) + self.elem_type[orb_format]
        oorb_elem[:, 9] = np.zeros(len(orbit_dataframe), float) + self.time_scales["TT"]
        # Convert other elements INCLUDING converting inclination,
        # node, argperi to RADIANS
        if orb_format == "KEP":
            oorb_elem[:, 1] = orbit_dataframe["a"]
            oorb_elem[:, 2] = orbit_dataframe["e"]
            oorb_elem[:, 3] = np.radians(orbit_dataframe["inc"])
            oorb_elem[:, 4] = np.radians(orbit_dataframe["Omega"])
            oorb_elem[:, 5] = np.radians(orbit_dataframe["argPeri"])
            oorb_elem[:, 6] = np.radians(orbit_dataframe["meanAnomaly"])
        elif orb_format == "COM":
            oorb_elem[:, 1] = orbit_dataframe["q"]
            oorb_elem[:, 2] = orbit_dataframe["e"]
            oorb_elem[:, 3] = np.radians(orbit_dataframe["inc"])
            oorb_elem[:, 4] = np.radians(orbit_dataframe["Omega"])
            oorb_elem[:, 5] = np.radians(orbit_dataframe["argPeri"])
            oorb_elem[:, 6] = orbit_dataframe["tPeri"]
        elif orb_format == "CAR":
            oorb_elem[:, 1] = orbit_dataframe["x"]
            oorb_elem[:, 2] = orbit_dataframe["y"]
            oorb_elem[:, 3] = orbit_dataframe["z"]
            oorb_elem[:, 4] = orbit_dataframe["xdot"]
            oorb_elem[:, 5] = orbit_dataframe["ydot"]
            oorb_elem[:, 6] = orbit_dataframe["zdot"]
        else:
            raise ValueError("Unknown orbit format %s: should be COM, KEP or CAR." % orb_format)
        oorb_elem[:, 8] = orbit_dataframe["epoch"]
        oorb_elem[:, 10] = orbit_dataframe["H"]
        oorb_elem[:, 11] = orbit_dataframe["g"]
        self.oorb_elem = oorb_elem
        self.orb_format = orb_format

    def convert_from_oorb_elem(self):
        """Translate pyoorb-style (fortran packed) orbital element array
        into a pandas dataframe. Operates on self.oorb_elem.

        Returns
        -------
        new_orbits : `pd.DataFrame`
            A DataFrame with the appropriate subset of columns
            relating to orbital elements.
        """
        if self.orb_format == "KEP":
            new_orbits = pd.DataFrame(
                self.oorb_elem.copy(),
                columns=[
                    "oorbId",
                    "a",
                    "e",
                    "inc",
                    "Omega",
                    "argPeri",
                    "meanAnomaly",
                    "elem_type",
                    "epoch",
                    "epoch_type",
                    "H",
                    "g",
                ],
            )
            new_orbits["meanAnomaly"] = np.degrees(new_orbits["meanAnomaly"])
        elif self.orb_format == "COM":
            new_orbits = pd.DataFrame(
                self.oorb_elem.copy(),
                columns=[
                    "oorbId",
                    "q",
                    "e",
                    "inc",
                    "Omega",
                    "argPeri",
                    "tPeri",
                    "elem_type",
                    "epoch",
                    "epoch_type",
                    "H",
                    "g",
                ],
            )
        elif self.orb_format == "CAR":
            new_orbits = pd.DataFrame(
                self.oorb_elem.copy(),
                columns=[
                    "oorbId",
                    "x",
                    "y",
                    "z",
                    "xdot",
                    "ydot",
                    "zdot",
                    "elem_type",
                    "epoch",
                    "epoch_type",
                    "H",
                    "g",
                ],
            )
        else:
            raise ValueError("Unknown orbit format %s: should be COM, KEP or CAR." % self.orb_format)
        # Convert from radians to degrees.
        if self.orb_format == "KEP" or self.orb_format == "COM":
            new_orbits["inc"] = np.degrees(new_orbits["inc"])
            new_orbits["Omega"] = np.degrees(new_orbits["Omega"])
            new_orbits["argPeri"] = np.degrees(new_orbits["argPeri"])
        # Drop columns we don't need and don't include in our standard columns.
        del new_orbits["elem_type"]
        del new_orbits["epoch_type"]
        del new_orbits["oorbId"]
        # To incorporate with original Orbits object, need to swap
        # back to original obj_ids as well as put back in original SEDs.
        return new_orbits

    def convert_orbit_format(self, orb_format="CAR"):
        """Convert orbital elements into `format`.

        Example: converts from self.oorb_elem[orb_format] (such as KEP)
        to oorb_format (such as CAR).

        Parameters
        ----------
        orb_format : `str`, optional
            Format to convert orbital elements into.
        """
        oorb_elem, err = oo.pyoorb.oorb_element_transformation(
            in_orbits=self.oorb_elem, in_element_type=self.elem_type[orb_format]
        )
        if err != 0:
            raise RuntimeError("Oorb returned error %s" % (err))
        del self.oorb_elem
        self.oorb_elem = oorb_elem
        self.orb_format = orb_format
        return

    def _convert_times(self, times, time_scale="UTC"):
        """Generate an oorb-format array of the times desired for the
        ephemeris generation.

        Parameters
        ----------
        times : `np.ndarray` or `float`
            The ephemeris times (MJD) desired
        time_scale : `str`, optional
            The timescale (UTC, UT1, TT, TAI) of the ephemeris MJD values.
            Default = UTC, MJD.

        Returns
        -------
        eph_times : `np.ndarray`
            The oorb-formatted 'eph_times' array.
        """
        if isinstance(times, float):
            times = np.array([times])
        if len(times) == 0:
            raise ValueError("Got zero times to convert for OpenOrb")
        eph_times = np.array(
            list(zip(times, repeat(self.time_scales[time_scale], len(times)))),
            dtype="double",
            order="F",
        )
        return eph_times

    def _generate_oorb_ephs_full(self, eph_times, obscode="I11", eph_mode="N"):
        """Generate full set of ephemeris output values using Oorb.

        Parameters
        ----------
        eph_times : `np.ndarray`
            Ephemeris times in oorb format (see self.convertTimes)
        obscode : `int` or `str`, optional
            The observatory code for ephemeris generation.
            Default=I11 (Cerro Pachon).
        eph_mode : `str`, optional
            What dynamical mode to use for generating ephemerides -
            "N" (n-body) or "2" (2-body).

        Returns
        -------
        ephemerides : `np.ndarray`
            The oorb-formatted ephemeris array.
        """
        oorb_ephems, err = oo.pyoorb.oorb_ephemeris_full(
            in_orbits=self.oorb_elem,
            in_obscode=obscode,
            in_date_ephems=eph_times,
            in_dynmodel=eph_mode,
        )
        if err != 0:
            raise RuntimeError("Oorb returned error %s" % (err))
        return oorb_ephems

    def _convert_oorb_ephs_full(self, oorb_ephs, by_object=True):
        """Converts oorb ephemeris array to np.ndarray.

        Here we convert to a numpy.ndarray, grouped either by object (default)
        or by time (if by_object=False).
        The resulting array is composed of columns (of each ephemeris element),
        where each column is 2-d array with first axes either 'object'
        or 'time'.
        - if by_object = True : [ephemeris elements][object][time]
        (i.e. the 'ra' column = 2-d array, where the [0] axis (length)
        equals the number of ephTimes)
        - if by_object = False : [ephemeris elements][time][object]
        (i.e. the 'ra' column = 2-d arrays, where the [0] axis (length)
        equals the number of objects)

        Parameters
        ----------
        oorb_ephs : `np.ndarray`
            The oorb-formatted ephemeris values
        by_object : `bool`, optional
            If True (default), resulting converted ephemerides are grouped
            by object.
            If False, resulting converted ephemerides are grouped by time.

        Returns
        -------
        ephemerides : `np.ndarray`
            The re-arranged ephemeris values, in a 3-d array.

        Notes
        -----
        The oorb ephemeris array is a 3-d array organized as:
        (object / times / eph@time)
        [objid][time][ephemeris information @ that time] with elements
        ! (1) modified julian date
        ! (2) right ascension (deg)
        ! (3) declination (deg)
        ! (4) dra/dt sky-motion (deg/day, including cos(dec) factor)
        ! (5) ddec/dt sky-motion (deg/day)
        ! (6) solar phase angle (deg)
        ! (7) solar elongation angle (deg)
        ! (8) heliocentric distance (au)
        ! (9) geocentric distance (au)
        ! (10) predicted apparent V-band magnitude
        ! (11) position angle for direction of motion (deg)
        ! (12) topocentric ecliptic longitude (deg)
        ! (13) topocentric ecliptic latitude (deg)
        ! (14) opposition-centered topocentric ecliptic longitude (deg)
        ! (15) opposition-centered topocentric ecliptic latitude (deg)
        ! (16) heliocentric ecliptic longitude (deg)
        ! (17) heliocentric ecliptic latitude (deg)
        ! (18) opposition-centered heliocentric ecliptic longitude (deg)
        ! (19) opposition-centered heliocentric ecliptic latitude (deg)
        ! (20) topocentric object altitude (deg)
        ! (21) topocentric solar altitude (deg)
        ! (22) topocentric lunar altitude (deg)
        ! (23) lunar phase [0...1]
        ! (24) lunar elongation (deg, distance between the target and the Moon)
        ! (25) heliocentric ecliptic cartesian x coordinate for the object (au)
        ! (26) helio ecliptic cartesian y coordinate for the object (au)
        ! (27) helio ecliptic cartesian z coordinate for the objects (au)
        ! (28) helio ecliptic cartesian x rate for the object (au/day))
        ! (29) helio ecliptic cartesian y rate for the object (au/day)
        ! (30) helio ecliptic cartesian z rate for the objects (au/day)
        ! (31) helio ecliptic cartesian coordinates for the observatory (au)
        ! (32) helio ecliptic cartesian coordinates for the observatory (au)
        ! (33) helio ecliptic cartesian coordinates for the observatory (au)
        ! (34) true anomaly (currently only a dummy value)
        """
        ephs = np.swapaxes(oorb_ephs, 2, 0)
        velocity = np.sqrt(ephs[3] ** 2 + ephs[4] ** 2)
        if by_object:
            ephs = np.swapaxes(ephs, 2, 1)
            velocity = np.swapaxes(velocity, 1, 0)
        # Create a numpy recarray.
        names = [
            "time",
            "ra",
            "dec",
            "dradt",
            "ddecdt",
            "phase",
            "solarelon",
            "helio_dist",
            "geo_dist",
            "magV",
            "pa",
            "topo_lon",
            "topo_lat",
            "opp_topo_lon",
            "opp_topo_lat",
            "helio_lon",
            "helio_lat",
            "opp_helio_lon",
            "opp_helio_lat",
            "topo_obj_alt",
            "topo_solar_alt",
            "topo_lunar_alt",
            "lunar_phase",
            "lunar_dist",
            "helio_x",
            "helio_y",
            "helio_z",
            "helio_dx",
            "helio_dy",
            "helio_dz",
            "obs_helio_x",
            "obs_helio_y",
            "obs_helio_z",
            "trueAnom",
        ]
        arraylist = []
        for i, n in enumerate(names):
            arraylist.append(ephs[i])
        arraylist.append(velocity)
        names.append("velocity")
        ephs = np.rec.fromarrays(arraylist, names=names)
        return ephs

    def _generate_oorb_ephs_basic(self, eph_times, obscode="I11", eph_mode="N"):
        """Generate ephemerides using OOrb with two body mode.

        Parameters
        ----------
        ephtimes : `np.ndarray`
            Ephemeris times in oorb format (see self.convertTimes).
        obscode : `int` or `str`, optional
            The observatory code for ephemeris generation.
            Default=I11 (Cerro Pachon).

        Returns
        -------
        oorb_ephems : `np.ndarray`
            The oorb-formatted ephemeris array.
        """
        oorb_ephems, err = oo.pyoorb.oorb_ephemeris_basic(
            in_orbits=self.oorb_elem,
            in_obscode=obscode,
            in_date_ephems=eph_times,
            in_dynmodel=eph_mode,
        )
        if err != 0:
            raise RuntimeError("Oorb returned error %s" % (err))
        return oorb_ephems

    def _convert_oorb_ephs_basic(self, oorb_ephs, by_object=True):
        """Converts oorb ephemeris array to numpy recarray,
        with labeled columns.

        Parameters
        ----------
        oorb_ephs : `np.ndarray`
            The oorb-formatted ephemeris values
        by_object : `bool`, optional
            If True (default), resulting converted ephemerides are grouped
            by object.
            If False, resulting converted ephemerides are grouped by time.

        Returns
        -------
        ephs : `np.ndarray`
            The re-arranged ephemeris values, in a 3-d array.

        Notes
        -----
        The oorb ephemeris array is a 3-d array organized as:
        (object / times / eph@time)
        [objid][time][ephemeris information @ that time] with ephemeris
        elements
        ! (1) modified julian date
        ! (2) right ascension (deg)
        ! (3) declination (deg)
        ! (4) dra/dt sky-motion (deg/day, including cos(dec) factor)
        ! (5) ddec/dt sky-motion (deg/day)
        ! (6) solar phase angle (deg)
        ! (7) solar elongation angle (deg)
        ! (8) heliocentric distance (au)
        ! (9) geocentric distance (au)
        ! (10) predicted apparent V-band magnitude
        ! (11) true anomaly (currently only a dummy value)

        Here we convert to a numpy array, grouped either by object (default)
        or by time (if by_object=False).
        The resulting array is composed of columns (of each ephemeris element),
        where each column is 2-d array with first axes either 'object'
        or 'time'.
        - if by_object = True : [ephemeris elements][object][time]
        (i.e. the 'ra' column = 2-d array, where the [0] axis (length)
        equals the number of ephTimes)
        - if by_object = False : [ephemeris elements][time][object]
        (i.e. the 'ra' column = 2-d arrays, where the [0] axis (length)
        equals the number of objects)
        """
        ephs = np.swapaxes(oorb_ephs, 2, 0)
        velocity = np.sqrt(ephs[3] ** 2 + ephs[4] ** 2)
        if by_object:
            ephs = np.swapaxes(ephs, 2, 1)
            velocity = np.swapaxes(velocity, 1, 0)
        # Create a numpy recarray.
        names = [
            "time",
            "ra",
            "dec",
            "dradt",
            "ddecdt",
            "phase",
            "solarelon",
            "helio_dist",
            "geo_dist",
            "magV",
            "trueAnomaly",
        ]
        arraylist = []
        for i, n in enumerate(names):
            arraylist.append(ephs[i])
        arraylist.append(velocity)
        names.append("velocity")
        ephs = np.rec.fromarrays(arraylist, names=names)
        return ephs

    def generate_ephemerides(
        self,
        times,
        time_scale="UTC",
        obscode="I11",
        by_object=True,
        eph_mode="nbody",
        eph_type="basic",
    ):
        """Calculate ephemerides for all orbits at times `times`.

        All returned positions and angles are in degrees, velocities
        are degrees/day and distances are in AU.

        Parameters
        ----------
        times : `np.ndarray`, (N,)
            Ephemeris times.
        time_scale : `str`, optional
            Time scale (UTC, TT, TAI) of times.
        obscode : `int` or `str`, optional
            The observatory code for ephemeris generation.
        by_object : `bool`, optional
            If True (default), resulting converted ephemerides are
            grouped by object.
            If False, resulting converted ephemerides are grouped by time.
        eph_mode : `str`, optional
            Dynamical model to use for ephemeris generation - nbody or 2body.
            Accepts 'nbody', '2body', 'N' or '2'. Default nbody.
        eph_type : `str`, optional
            Generate full (more data) ephemerides or basic (less data)
            ephemerides.
            Default basic.

        Returns
        -------
        ephemerides : `np.ndarray`
            The ephemeris values, organized as chosen by the user.


        Notes
        -----
        The returned ephemerides are a numpy array that can be grouped
        by object or by time.

        If they are grouped by object (by_object = True), the array
        is organized as `ephemeris_values[object][time]`.
        Here the "ra" column is a 2-d array where the [0] axis
        length equals the number of ephemeris times.

        If they are grouped by time (by_object=False), the array
        is organized as `ephemeris_values[time][object]`.
        Here the "ra" column is a 2-d array where the [0] axis length
        equals the number of objects.
        """
        if eph_mode.lower() in ("nbody", "n"):
            eph_mode = "N"
        elif eph_mode.lower() in ("2body", "2"):
            eph_mode = "2"
        else:
            raise ValueError("eph_mode should be 2body or nbody (or '2' or 'N').")

        # t = time.time()
        eph_times = self._convert_times(times, time_scale=time_scale)
        if eph_type.lower() == "basic":
            oorb_ephs, err = oo.pyoorb.oorb_ephemeris_basic(
                in_orbits=self.oorb_elem,
                in_obscode=obscode,
                in_date_ephems=eph_times,
                in_dynmodel=eph_mode,
            )
            ephs = self._convert_oorb_ephs_basic(oorb_ephs, by_object=by_object)
        elif eph_type.lower() == "full":
            oorb_ephs = self._generate_oorb_ephs_full(eph_times, obscode=obscode, eph_mode=eph_mode)
            ephs = self._convert_oorb_ephs_full(oorb_ephs, by_object=by_object)
        else:
            raise ValueError("eph_type must be full or basic")
        # dt, t = dtime(t)
        # logging.debug("# Calculating ephemerides for %d objects over %d times
        # required %f seconds"
        #              % (len(self.oorb_elem), len(times), dt))
        return ephs

    def propagate_orbits(self, new_epoch, eph_mode="nbody"):
        """Propagate orbits from self.orbits.epoch to new epoch (MJD TT).

        Parameters
        ----------
        new_epoch : `float`
            MJD TT time for new epoch.
        """
        new_epoch = self._convert_times(new_epoch, time_scale="TT")
        if eph_mode.lower() in ("nbody", "n"):
            eph_mode = "N"
        elif eph_mode.lower() in ("2body", "2"):
            eph_mode = "2"
        else:
            raise ValueError("eph_mode should be 2body or nbody (or '2' or 'N').")

        new_oorb_elem, err = oo.pyoorb.oorb_propagation(
            in_orbits=self.oorb_elem, in_dynmodel=eph_mode, in_epoch=new_epoch
        )
        if err != 0:
            raise RuntimeError("Orbit propagation returned error %d" % err)
        self.oorb_elem = new_oorb_elem
        return
