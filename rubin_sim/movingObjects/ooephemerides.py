import os
import logging
from itertools import repeat
import numpy as np
import pandas as pd
import pyoorb as oo
from .orbits import Orbits
from .utils import get_oorb_data_dir

import time

__all__ = ['PyOrbEphemerides']


def dtime(time_prev):
    return (time.time() - time_prev, time.time())


class PyOrbEphemerides(object):
    """Generate ephemerides and propagate orbits using the python interface to Oorb.

    Typical usage:
    pyephs = PyOrbEphemerides()
    # Set the orbital parameters, using an lsst.sims.movingObjects.Orbits object
    pyephs.setOrbits(orbits)
    # Generate ephemerides at times 'times'.
    ephs = pyephs.generateEphemerides(times, timeScale='UTC', obscode='I11')

    This class handles the packing and unpacking of the fortran style arrays that
    pyoorb uses, to and from more user-friendly pandas arrays.

    Parameters
    ----------
    ephfile : str, optional
        Planetary ephemerides file for Oorb (i.e. de430 or de405).
        Default $OORB_DATA/de430.dat  ($OORB_DATA = $OORB_DIR/data).
    """
    def __init__(self, ephfile=None):
        # Set translation from timescale to OpenOrb numerical representation.
        # Note all orbits are assumed to be in TT timescale.
        # Also, all dates are expected to be in MJD.
        self.timeScales = {'UTC': 1, 'UT1': 2, 'TT': 3, 'TAI': 4}
        self.elemType = {'CAR': 1, 'COM': 2, 'KEP': 3, 'DEL': 4, 'EQX': 5}

        # Set up oorb. Call this once.
        if ephfile is None:
            # just making a copy on our own so we don't have to chase down oorb install dir
            ephfile = os.path.join(get_oorb_data_dir(), 'de430.dat')
        self.ephfile = ephfile
        self._init_oorb()
        self.oorbElem = None
        self.orb_format = None

    def _init_oorb(self):
        oo.pyoorb.oorb_init(ephemeris_fname=self.ephfile)

    def setOrbits(self, orbitObj):
        """Set the orbits, to be used to generate ephemerides.

        Immediately calls self._convertOorbElem to translate to the 'packed' oorb format.

        Parameters
        ----------
        orbitObj : Orbits
           The orbits to use to generate ephemerides.
        """
        if len(orbitObj) == 0:
            raise ValueError('There are no orbits in the Orbit object.')
        self._convertToOorbElem(orbitObj.orbits, orbitObj.orb_format)

    def _convertToOorbElem(self, orbitDataframe, orb_format):
        """Convert orbital elements into the numpy fortran-format array OpenOrb requires.

        The OpenOrb element format is a single array with elemenets:
        0 : orbitId (cannot be a string)
        1-6 : orbital elements, using radians for angles
        7 : element 'type' code (1 = CAR, 2 = COM, 3 = KEP, 4 = DELauny, 5 = EQX (equinoctial))
        8 : epoch
        9 : timescale for epoch (1 = UTC, 2 = UT1, 3 = TT, 4 = TAI : always assumes TT)
        10 : magHv
        11 : g

        Sets self.oorbElem, the orbit parameters in an array formatted for OpenOrb.
        """
        oorbElem = np.zeros([len(orbitDataframe), 12], dtype=np.double, order='F')
        # Put in simple values for objid, or add method to test if any objId is a string.
        # NOTE THAT THIS MEANS WE'VE LOST THE OBJID
        oorbElem[:,0] = np.arange(0, len(orbitDataframe), dtype=int) + 1
        # Add the appropriate element and epoch types:
        oorbElem[:,7] = np.zeros(len(orbitDataframe), float) + self.elemType[orb_format]
        oorbElem[:,9] = np.zeros(len(orbitDataframe), float) + self.timeScales['TT']
        # Convert other elements INCLUDING converting inclination, node, argperi to RADIANS
        if orb_format == 'KEP':
            oorbElem[:, 1] = orbitDataframe['a']
            oorbElem[:, 2] = orbitDataframe['e']
            oorbElem[:, 3] = np.radians(orbitDataframe['inc'])
            oorbElem[:, 4] = np.radians(orbitDataframe['Omega'])
            oorbElem[:, 5] = np.radians(orbitDataframe['argPeri'])
            oorbElem[:, 6] = np.radians(orbitDataframe['meanAnomaly'])
        elif orb_format == 'COM':
            oorbElem[:, 1] = orbitDataframe['q']
            oorbElem[:, 2] = orbitDataframe['e']
            oorbElem[:, 3] = np.radians(orbitDataframe['inc'])
            oorbElem[:, 4] = np.radians(orbitDataframe['Omega'])
            oorbElem[:, 5] = np.radians(orbitDataframe['argPeri'])
            oorbElem[:, 6] = orbitDataframe['tPeri']
        elif orb_format == 'CAR':
            oorbElem[:, 1] = orbitDataframe['x']
            oorbElem[:, 2] = orbitDataframe['y']
            oorbElem[:, 3] = orbitDataframe['z']
            oorbElem[:, 4] = orbitDataframe['xdot']
            oorbElem[:, 5] = orbitDataframe['ydot']
            oorbElem[:, 6] = orbitDataframe['zdot']
        else:
            raise ValueError('Unknown orbit format %s: should be COM, KEP or CAR.' % orb_format)
        oorbElem[:,8] = orbitDataframe['epoch']
        oorbElem[:,10] = orbitDataframe['H']
        oorbElem[:,11] = orbitDataframe['g']
        self.oorbElem = oorbElem
        self.orb_format = orb_format

    def convertFromOorbElem(self):
        """Translate pyoorb-style orbital element array back into dataframe.

        Parameters
        ----------
        oorbElem : numpy.ndarray
            The orbital elements in OpenOrb format.

        Returns
        -------
        pd.DataFrame
            A DataFrame with the appropriate subset of columns relating to orbital elements.
        """
        if self.orb_format == 'KEP':
            newOrbits = pd.DataFrame(self.oorbElem.copy(), columns=['oorbId', 'a', 'e', 'inc', 'Omega', 'argPeri',
                                                             'meanAnomaly', 'elem_type', 'epoch',
                                                             'epoch_type',
                                                             'H', 'g'])
            newOrbits['meanAnomaly'] = np.degrees(newOrbits['meanAnomaly'])
        elif self.orb_format == 'COM':
            newOrbits = pd.DataFrame(self.oorbElem.copy(), columns=['oorbId', 'q', 'e', 'inc', 'Omega', 'argPeri',
                                                             'tPeri', 'elem_type', 'epoch', 'epoch_type',
                                                             'H', 'g'])
        elif self.orb_format == 'CAR':
            newOrbits = pd.DataFrame(self.oorbElem.copy(), columns = ['oorbId', 'x', 'y', 'z',
                                                               'xdot', 'ydot', 'zdot', 'elem_type', 'epoch',
                                                               'epoch_type', 'H', 'g'])
        else:
            raise ValueError('Unknown orbit format %s: should be COM, KEP or CAR.' % self.orb_format)
        # Convert from radians to degrees.
        if self.orb_format == 'KEP' or self.orb_format =='COM':
            newOrbits['inc'] = np.degrees(newOrbits['inc'])
            newOrbits['Omega'] = np.degrees(newOrbits['Omega'])
            newOrbits['argPeri'] = np.degrees(newOrbits['argPeri'])
        # Drop columns we don't need and don't include in our standard columns.
        del newOrbits['elem_type']
        del newOrbits['epoch_type']
        del newOrbits['oorbId']
        # To incorporate with original Orbits object, need to swap back to original objIds
        # as well as put back in original SEDs.
        return newOrbits

    def convertOrbitFormat(self, orb_format='CAR'):
        """Convert orbital elements from the format in orbitObj into 'format'.

        Parameters
        ----------
        format : str, optional
            Format to convert orbital elements into.

        Returns
        -------
        """
        oorbElem, err = oo.pyoorb.oorb_element_transformation(in_orbits=self.oorbElem,
                                                              in_element_type=self.elemType[orb_format])
        if err != 0:
            raise RuntimeError('Oorb returned error %s' % (err))
        del self.oorbElem
        self.oorbElem = oorbElem
        self.orb_format = orb_format
        return

    def _convertTimes(self, times, timeScale='UTC'):
        """Generate an oorb-format array of the times desired for the ephemeris generation.

        Parameters
        ----------
        times : numpy.ndarray or float
            The ephemeris times (MJD) desired
        timeScale : str, optional
            The timescale (UTC, UT1, TT, TAI) of the ephemeris MJD values. Default = UTC, MJD.

        Returns
        -------
        numpy.ndarray
            The oorb-formatted 'ephTimes' array.
        """
        if isinstance(times, float):
            times = np.array([times])
        if len(times) == 0:
            raise ValueError('Got zero times to convert for OpenOrb')
        ephTimes = np.array(list(zip(times, repeat(self.timeScales[timeScale], len(times)))),
                            dtype='double', order='F')
        return ephTimes

    def _generateOorbEphsFull(self, ephTimes, obscode='I11', ephMode='N'):
        """Generate full set of ephemeris output values using Oorb.

        Parameters
        ----------
        ephtimes : numpy.ndarray
            Ephemeris times in oorb format (see self.convertTimes)
        obscode : int or str, optional
            The observatory code for ephemeris generation. Default=I11 (Cerro Pachon).

        Returns
        -------
        numpy.ndarray
            The oorb-formatted ephemeris array.
        """
        oorbEphems, err = oo.pyoorb.oorb_ephemeris_full(in_orbits=self.oorbElem,
                                                        in_obscode=obscode,
                                                        in_date_ephems=ephTimes,
                                                        in_dynmodel=ephMode)
        if err != 0:
            raise RuntimeError('Oorb returned error %s' % (err))
        return oorbEphems

    def _convertOorbEphsFull(self, oorbEphs, byObject=True):
        """Converts oorb ephemeris array to numpy recarray, with labeled columns.

        The oorb ephemeris array is a 3-d array organized as: (object / times / eph@time)
        [objid][time][ephemeris information @ that time] with ephemeris elements
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
        ! (26) heliocentric ecliptic cartesian y coordinate for the object (au)
        ! (27) heliocentric ecliptic cartesian z coordinate for the objects (au)
        ! (28) heliocentric ecliptic cartesian x rate for the object (au/day))
        ! (29) heliocentric ecliptic cartesian y rate for the object (au/day)
        ! (30) heliocentric ecliptic cartesian z rate for the objects (au/day)
        ! (31) heliocentric ecliptic cartesian coordinates for the observatory (au)
        ! (32) heliocentric ecliptic cartesian coordinates for the observatory (au)
        ! (33) heliocentric ecliptic cartesian coordinates for the observatory (au)
        ! (34) true anomaly (currently only a dummy value)

        Here we convert to a numpy recarray, grouped either by object (default)
        or by time (if byObject=False).
        The resulting numpy recarray is composed of columns (of each ephemeris element),
        where each column is 2-d array with first axes either 'object' or 'time'.
        - if byObject = True : [ephemeris elements][object][time]
        (i.e. the 'ra' column = 2-d array, where the [0] axis (length) equals the number of ephTimes)
        - if byObject = False : [ephemeris elements][time][object]
        (i.e. the 'ra' column = 2-d arrays, where the [0] axis (length) equals the number of objects)

        Parameters
        ----------
        oorbEphs : numpy.ndarray
            The oorb-formatted ephemeris values
        byObject : `bool`, optional
            If True (default), resulting converted ephemerides are grouped by object.
            If False, resulting converted ephemerides are grouped by time.

        Returns
        -------
        numpy.recarray
            The re-arranged ephemeris values, in a 3-d array.
        """
        ephs = np.swapaxes(oorbEphs, 2, 0)
        velocity = np.sqrt(ephs[3]**2 + ephs[4]**2)
        if byObject:
            ephs = np.swapaxes(ephs, 2, 1)
            velocity = np.swapaxes(velocity, 1, 0)
        # Create a numpy recarray.
        names = ['time', 'ra', 'dec', 'dradt', 'ddecdt', 'phase', 'solarelon',
                 'helio_dist', 'geo_dist', 'magV', 'pa',
                 'topo_lon', 'topo_lat', 'opp_topo_lon', 'opp_topo_lat',
                 'helio_lon', 'helio_lat', 'opp_helio_lon', 'opp_helio_lat',
                 'topo_obj_alt', 'topo_solar_alt', 'topo_lunar_alt', 'lunar_phase', 'lunar_dist',
                 'helio_x', 'helio_y', 'helio_z', 'helio_dx', 'helio_dy', 'helio_dz',
                 'obs_helio_x', 'obs_helio_y', 'obs_helio_z', 'trueAnom']
        arraylist = []
        for i, n in enumerate(names):
            arraylist.append(ephs[i])
        arraylist.append(velocity)
        names.append('velocity')
        ephs = np.rec.fromarrays(arraylist, names=names)
        return ephs

    def _generateOorbEphsBasic(self, ephTimes, obscode='I11', ephMode='N'):
        """Generate ephemerides using OOrb with two body mode.

        Parameters
        ----------
        ephtimes : numpy.ndarray
            Ephemeris times in oorb format (see self.convertTimes).
        obscode : int or str, optional
            The observatory code for ephemeris generation. Default=I11 (Cerro Pachon).

        Returns
        -------
        numpy.ndarray
            The oorb-formatted ephemeris array.
        """
        oorbEphems, err = oo.pyoorb.oorb_ephemeris_basic(in_orbits=self.oorbElem,
                                                         in_obscode=obscode,
                                                         in_date_ephems=ephTimes,
                                                         in_dynmodel=ephMode)
        if err != 0:
            raise RuntimeError('Oorb returned error %s' % (err))
        return oorbEphems

    def _convertOorbEphsBasic(self, oorbEphs, byObject=True):
        """Converts oorb ephemeris array to numpy recarray, with labeled columns.

        The oorb ephemeris array is a 3-d array organized as: (object / times / eph@time)
        [objid][time][ephemeris information @ that time] with ephemeris elements
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

        Here we convert to a numpy recarray, grouped either by object (default)
        or by time (if byObject=False).
        The resulting numpy recarray is composed of columns (of each ephemeris element),
        where each column is 2-d array with first axes either 'object' or 'time'.
        - if byObject = True : [ephemeris elements][object][time]
        (i.e. the 'ra' column = 2-d array, where the [0] axis (length) equals the number of ephTimes)
        - if byObject = False : [ephemeris elements][time][object]
        (i.e. the 'ra' column = 2-d arrays, where the [0] axis (length) equals the number of objects)

        Parameters
        ----------
        oorbEphs : numpy.ndarray
            The oorb-formatted ephemeris values
        byObject : `bool`, optional
            If True (default), resulting converted ephemerides are grouped by object.
            If False, resulting converted ephemerides are grouped by time.

        Returns
        -------
        numpy.recarray
            The re-arranged ephemeris values, in a 3-d array.
        """
        ephs = np.swapaxes(oorbEphs, 2, 0)
        velocity = np.sqrt(ephs[3]**2 + ephs[4]**2)
        if byObject:
            ephs = np.swapaxes(ephs, 2, 1)
            velocity = np.swapaxes(velocity, 1, 0)
        # Create a numpy recarray.
        names = ['time', 'ra', 'dec', 'dradt', 'ddecdt', 'phase', 'solarelon',
                 'helio_dist', 'geo_dist', 'magV', 'trueAnomaly']
        arraylist = []
        for i, n in enumerate(names):
            arraylist.append(ephs[i])
        arraylist.append(velocity)
        names.append('velocity')
        ephs = np.rec.fromarrays(arraylist, names=names)
        return ephs

    def generateEphemerides(self, times, timeScale='UTC', obscode='I11', byObject=True,
                            ephMode='nbody', ephType='basic'):
        """Calculate ephemerides for all orbits at times `times`.

        This is a public method, wrapping self._convertTimes, self._generateOorbEphs
        and self._convertOorbEphs (which include dealing with oorb-formatting of arrays).

        The return ephemerides are in a numpy recarray, with axes
        - if byObject = True : [ephemeris values][object][@time]
        (i.e. the 'ra' column = 2-d array, where the [0] axis (length) equals the number of ephTimes)
        - if byObject = False : [ephemeris values][time][@object]
        (i.e. the 'ra' column = 2-d arrays, where the [0] axis (length) equals the number of objects)

        The ephemeris values returned to the user (== columns of the recarray) are:
        ['delta', 'ra', 'dec', 'magV', 'time', 'dradt', 'ddecdt', 'phase', 'solarelon', 'velocity']
        where positions/angles are all in degrees, velocities are deg/day, and delta is the
        distance between the Earth and the object in AU.

        Parameters
        ----------
        ephtimes : numpy.ndarray
            Ephemeris times in oorb format (see self.convertTimes)
        obscode : int or str, optional
            The observatory code for ephemeris generation. Default=807 (Cerro Tololo).
        byObject : `bool`, optional
            If True (default), resulting converted ephemerides are grouped by object.
            If False, resulting converted ephemerides are grouped by time.
        ephMode : str, optional
            Dynamical model to use for ephemeris generation - nbody or 2body.
            Accepts 'nbody', '2body', 'N' or '2'. Default nbody.
        ephType : str, optional
            Generate full (more data) ephemerides or basic (less data) ephemerides.
            Default basic.

        Returns
        -------
        numpy.ndarray
            The ephemeris values, organized as chosen by the user.
        """
        if ephMode.lower() in ('nbody', 'n'):
            ephMode = 'N'
        elif ephMode.lower() in ('2body', '2'):
            ephMode = '2'
        else:
            raise ValueError("ephMode should be 2body or nbody (or '2' or 'N').")

        #t = time.time()
        ephTimes = self._convertTimes(times, timeScale=timeScale)
        if ephType.lower() == 'basic':
            #oorbEphs = self._generateOorbEphsBasic(ephTimes, obscode=obscode, ephMode=ephMode)
            oorbEphs, err = oo.pyoorb.oorb_ephemeris_basic(in_orbits=self.oorbElem,
                                                             in_obscode=obscode,
                                                             in_date_ephems=ephTimes,
                                                             in_dynmodel=ephMode)
            ephs = self._convertOorbEphsBasic(oorbEphs, byObject=byObject)
        elif ephType.lower() == 'full':
            oorbEphs = self._generateOorbEphsFull(ephTimes, obscode=obscode, ephMode=ephMode)
            ephs = self._convertOorbEphsFull(oorbEphs, byObject=byObject)
        else:
            raise ValueError('ephType must be full or basic')
        #dt, t = dtime(t)
        #logging.debug("# Calculating ephemerides for %d objects over %d times required %f seconds"
        #              % (len(self.oorbElem), len(times), dt))
        return ephs

    def propagateOrbits(self, newEpoch, ephMode='nbody'):
        """Propagate orbits from self.orbits.epoch to new epoch (MJD TT).

        Parameters
        ----------
        new_epoch : float
            MJD TT time for new epoch.
        """
        newEpoch = self._convertTimes(newEpoch, timeScale='TT')
        if ephMode.lower() in ('nbody', 'n'):
            ephMode = 'N'
        elif ephMode.lower() in ('2body', '2'):
            ephMode = '2'
        else:
            raise ValueError("ephMode should be 2body or nbody (or '2' or 'N').")

        newOorbElem, err = oo.pyoorb.oorb_propagation(in_orbits=self.oorbElem, 
                                                      in_dynmodel=ephMode, 
                                                      in_epoch=newEpoch)
        if err != 0:
            raise RuntimeError('Orbit propagation returned error %d' % err)
        self.oorbElem = newOorbElem
        return
