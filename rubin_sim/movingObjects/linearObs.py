import logging
import datetime
import numpy as np
from scipy import interpolate

from .ooephemerides import PyOrbEphemerides
from .baseObs import BaseObs

__all__ = ['LinearObs']


class LinearObs(BaseObs):
    """Generate observations for a set of Orbits using linear interpolation.

    Uses linear interpolations between grid of true ephemerides.
    Ephemerides can be generated using 2-body or n-body integration.

    Parameters
    ----------
    footPrint: str, opt
        Specify the footprint for the FOV. Options include "camera", "circle", "rectangle".
        'Camera' means use the actual LSST camera footprint (following a rough cut with a circular FOV).
        Default is circular FOV.
    rFov : float, opt
        If footprint is "circle", this is the radius of the fov (in degrees).
        Default 1.75 degrees.
    xTol : float, opt
        If footprint is "rectangle", this is half of the width of the (on-sky) fov in the RA
        direction (in degrees).
        Default 5 degrees. (so size of footprint in degrees will be 10 degrees in the RA direction).
    yTol : float, opt
        If footprint is "rectangular", this is half of the width of the fov in Declination (in degrees).
        Default is 3 degrees (so size of footprint in degrees will be 6 degrees in the Dec direction).
    ephMode: str, opt
        Mode for ephemeris generation - nbody or 2body. Default is nbody.
    ephType: str, opt
        Type of ephemerides to generate - full or basic.
        Full includes all values calculated by openorb; Basic includes a more basic set.
        Default is Basic.  (this includes enough information for most standard MAF metrics).
    ephFile: str or None, opt
        The name of the planetary ephemerides file to use for ephemeris generation.
        Default (None) will use the default for PyOrbEphemerides.
    obsCode: str, opt
        Observatory code for ephemeris generation. Default is "I11" - Cerro Pachon.
    obsTimeCol: str, opt
        Name of the time column in the obsData. Default 'observationStartMJD'.
    obsTimeScale: str, opt
        Type of timescale for MJD (TAI or UTC currently). Default TAI.
    seeingCol: str, opt
        Name of the seeing column in the obsData. Default 'seeingFwhmGeom'.
        This should be the geometric/physical seeing as it is used for the trailing loss calculation.
    visitExpTimeCol: str, opt
        Name of the visit exposure time column in the obsData. Default 'visitExposureTime'.
    obsRA: str, opt
        Name of the RA column in the obsData. Default 'fieldRA'.
    obsDec: str, opt
        Name of the Dec column in the obsData. Default 'fieldDec'.
    obsRotSkyPos: str, opt
        Name of the Rotator column in the obsData. Default 'rotSkyPos'.
    obsDegrees: bool, opt
        Whether the observational data is in degrees or radians. Default True (degrees).
    outfileName : str, opt
        The output file name.
        Default is 'lsst_obs.dat'.
    obsMetadata : str, opt
        A string that captures provenance information about the observations.
        For example: 'kraken_2026, MJD 59853-61677' or 'baseline2018a minus NES'
        Default ''.
    tstep : float, opt
        The time between points in the ephemeris grid, in days.
        Default 2 hours.
    """
    def __init__(self, footprint='circular', rFov=1.75, xTol=5, yTol=3,
                 ephMode='nbody', ephType='basic', obsCode='I11',
                 ephFile=None,
                 obsTimeCol='observationStartMJD', obsTimeScale='TAI',
                 seeingCol='seeingFwhmGeom', visitExpTimeCol='visitExposureTime',
                 obsRA='fieldRA', obsDec='fieldDec', obsRotSkyPos='rotSkyPos', obsDegrees=True,
                 outfileName='lsst_obs.dat', obsMetadata='', tstep=2.0/24.0):
        super().__init__(footprint=footprint, rFov=rFov, xTol=xTol, yTol=yTol,
                         ephMode=ephMode, ephType=ephType, obsCode=obsCode,
                         ephFile=ephFile, obsTimeCol=obsTimeCol, obsTimeScale=obsTimeScale,
                         seeingCol=seeingCol, visitExpTimeCol=visitExpTimeCol,
                         obsRA=obsRA, obsDec=obsDec, obsRotSkyPos=obsRotSkyPos, obsDegrees=obsDegrees,
                         outfileName=outfileName, obsMetadata=obsMetadata)
        self.tstep = tstep

    def _headerMeta(self):
        # Linear obs header metadata.
        self.outfile.write('# linear obs header metadata\n')
        self.outfile.write('# observation generation via %s\n' % self.__class__.__name__)
        self.outfile.write('# ephMode %s\n' % (self.ephMode))
        self.outfile.write('# time step for ephemeris grid %f\n' % self.tstep)

    # Linear interpolation
    def _makeInterps(self, ephs):
        """Generate the interpolation functions for the linear interpolation.

        Parameters
        ----------
        ephs : np.recarray
            Grid of actual ephemerides, for a single object.

        Returns
        -------
        dictionary
            Dictionary of the interpolation functions.
        """
        interpfuncs = {}
        for n in ephs.dtype.names:
            if n == 'time':
                continue
            interpfuncs[n] = interpolate.interp1d(ephs['time'], ephs[n], kind='linear',
                                                  assume_sorted=True, copy=False)
        return interpfuncs

    def _interpEphs(self, interpfuncs, times, columns=None):
        """Calculate the linear interpolation approximations of the ephemeride columns.

        Parameters
        ----------
        interpfuncs : dict
            Dictionary of the linear interpolation functions.
        times : np.ndarray
            Times at which to generate ephemerides.
        columns : list of str, opt
            List of the values to generate ephemerides for.
            Default None = generate all values.

        Returns
        -------
        np.recarray
            Array of interpolated ephemerides.
        """
        if columns is None:
            columns = interpfuncs.keys()
        dtype = []
        for col in columns:
            dtype.append((col, '<f8'))
        dtype.append(('time', '<f8'))
        ephs = np.recarray([len(times)], dtype=dtype)
        for col in columns:
            ephs[col] = interpfuncs[col](times)
        ephs['time'] = times
        return ephs

    def run(self, orbits, obsData):
        """Find and write the observations of each object to disk.

        For each object, identify the observations where the object is
        within rFOV of the pointing boresight (potentially, also in the camera footprint),
        and write the ephemeris values and observation metadata to disk.
        Uses linear interpolation between ephemeris gridpoints.

        Parameters
        ----------
        orbits: lsst.sims.movingObjects.Orbits
            The orbits to generate ephemerides for.
        obsData : np.recarray
            The simulated pointing history data.
        """
        # Set the times for the ephemeris grid.
        timeStep = float(self.tstep)
        timeStart = obsData[self.obsTimeCol].min() - timeStep
        timeEnd = obsData[self.obsTimeCol].max() + timeStep
        times = np.arange(timeStart, timeEnd + timeStep / 2.0, timeStep)
        logging.info('Generating ephemerides on a grid of %f day timesteps, then will extrapolate '
                     'to opsim times.' % (timeStep))
        # For each object, identify observations where the object is within the FOV (or camera footprint).
        for i, sso in enumerate(orbits):
            objid = sso.orbits['objId'].iloc[0]
            sedname = sso.orbits['sed_filename'].iloc[0]
            # Generate ephemerides on a grid.
            logging.debug(("%d/%d   id=%s : " % (i, len(orbits), objid)) + 
                          datetime.datetime.now().strftime("Start: %Y-%m-%d %H:%M:%S") + 
                          " nTimes: %s" % len(times))
            ephs = self.generateEphemerides(sso, times, ephMode=self.ephMode, ephType=self.ephType)[0]
            interpfuncs = self._makeInterps(ephs)
            ephs = self._interpEphs(interpfuncs, times=obsData[self.obsTimeCol], columns=['ra', 'dec'])
            logging.debug(("%d/%d   id=%s : " % (i, len(orbits), objid)) + 
                          datetime.datetime.now().strftime("Interp end: %Y-%m-%d %H:%M:%S"))
            # Find objects in the chosen footprint (circular, rectangular or camera)
            idxObs = self.ssoInFov(ephs, obsData)
            logging.info(("Object %d/%d   id=%s : " % (i, len(orbits), objid)) + 
                         "Object in %d visits" % (len(idxObs)))
            if len(idxObs) > 0:
                obsdat = obsData[idxObs]
                ephs = self._interpEphs(interpfuncs, times=obsdat[self.obsTimeCol])
                # Write these observations to disk.
                self.writeObs(objid, ephs, obsdat, sedname=sedname)
