import logging
import numpy as np
import datetime

from .baseObs import BaseObs

__all__ = ['DirectObs']


class DirectObs(BaseObs):
    """
    Generate observations of a set of moving objects: exact ephemeris at the times of each observation.

    First generates observations on a rough grid and looks for observations within a specified tolerance
    of the actual observations; for the observations which pass this cut, generates a precise ephemeris
    and checks if the object is within the FOV.

    Parameters
    ----------
    footPrint: str, opt
        Specify the footprint for the FOV. Options include "camera", "circle", "rectangle".
        'Camera' means use the actual LSST camera footprint (following a rough cut with a circular FOV).
        Default is circular FOV.
    rFov : float, opt
        If footprint is "circular", this is the radius of the fov (in degrees).
        Default 1.75 degrees.
    xTol : float, opt
        If footprint is "rectangular", this is half of the width of the (on-sky) fov in the RA
        direction (in degrees).
        Default 5 degrees. (so size of footprint in degrees will be 10 degrees in the RA direction).
    yTol : float, opt
        If footprint is "rectangular", this is half of the width of the fov in Declination (in degrees).
        Default is 3 degrees (so size of footprint in degrees will be 6 degrees in the Dec direction).
    ephMode: str, opt
        Mode for ephemeris generation - nbody or 2body. Default is nbody.
    prelimEphMode: str, opt
        Mode for preliminary ephemeris generation, if any is done. Default is 2body.
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
    tstep: float, opt
        The time between initial (rough) ephemeris generation points, in days.
        Default 1 day.
    roughTol: float, opt
        The initial rough tolerance value for positions, used as a first cut to identify potential
        observations (in degrees).
        Default 10 degrees.
    """
    def __init__(self, footprint='circular', rFov=1.75, xTol=5, yTol=3,
                 ephMode='nbody', prelimEphMode='2body', ephType='basic', obsCode='I11',
                 ephFile=None,
                 obsTimeCol='observationStartMJD', obsTimeScale='TAI',
                 seeingCol='seeingFwhmGeom', visitExpTimeCol='visitExposureTime',
                 obsRA='fieldRA', obsDec='fieldDec', obsRotSkyPos='rotSkyPos', obsDegrees=True,
                 outfileName='lsst_obs.dat', obsMetadata='', tstep=1.0, roughTol=10.0):
        super().__init__(footprint=footprint, rFov=rFov, xTol=xTol, yTol=yTol,
                         ephMode=ephMode, ephType=ephType, obsCode=obsCode,
                         ephFile=ephFile, obsTimeCol=obsTimeCol, obsTimeScale=obsTimeScale,
                         seeingCol=seeingCol, visitExpTimeCol=visitExpTimeCol,
                         obsRA=obsRA, obsDec=obsDec, obsRotSkyPos=obsRotSkyPos, obsDegrees=obsDegrees,
                         outfileName=outfileName, obsMetadata=obsMetadata)
        self.tstep = tstep
        self.roughTol = roughTol
        if prelimEphMode not in ('2body', 'nbody'):
            raise ValueError('Ephemeris generation must be 2body or nbody.')
        self.prelimEphMode = prelimEphMode

    def _headerMeta(self):
        # Specific header information for direct obs.
        self.outfile.write('# direct obs header metadata\n')
        self.outfile.write('# observation generation via %s\n' % self.__class__.__name__)
        self.outfile.write('# ephMode %s prelimEphMode %s\n' % (self.ephMode, self.prelimEphMode))
        self.outfile.write('# rough tolerance for preliminary match %f\n' % self.roughTol)
        self.outfile.write('# time step for preliminary match %f\n' % self.tstep)

    def run(self, orbits, obsData):
        """Find and write the observations of each object to disk.

        For each object, generate a very rough grid of ephemeris points (typically using 2body integration).
        Then identify pointings in obsData which are within

        Parameters
        ----------
        orbits: lsst.sims.movingObjects.Orbits
            The orbits to generate ephemerides for.
        obsData : np.recarray
            The simulated pointing history data.
        """
        # Set the times for the rough ephemeris grid.
        timeStep = float(self.tstep)
        timeStart = np.floor(obsData[self.obsTimeCol].min() + 0.16 - 0.5) - timeStep
        timeEnd = np.ceil(obsData[self.obsTimeCol].max() + 0.16 + 0.5) + timeStep
        rough_times = np.arange(timeStart, timeEnd + timeStep / 2.0, timeStep)
        logging.info('Generating preliminary ephemerides on a grid of %f day timesteps.' % (timeStep))
        # For each object, identify observations where the object is within the FOV (or camera footprint).
        for i, sso in enumerate(orbits):
            objid = sso.orbits['objId'].iloc[0]
            sedname = sso.orbits['sed_filename'].iloc[0]
            # Generate ephemerides on the rough grid.
            logging.debug(("%d/%d   id=%s : " % (i, len(orbits), objid)) +
                          datetime.datetime.now().strftime("Prelim start: %Y-%m-%d %H:%M:%S")
                          + " nRoughTimes: %s" % len(rough_times))
            ephs = self.generateEphemerides(sso, rough_times,
                                            ephMode=self.prelimEphMode, ephType=self.ephType)[0]
            mu = ephs['velocity']
            logging.debug(("%d/%d   id=%s : " % (i, len(orbits), objid)) + 
                          datetime.datetime.now().strftime("Prelim end: %Y-%m-%d %H:%M:%S") + 
                          " π(median, max), min(geo_dist): %.2f, %.2f deg/day  %.2f AU" 
                          % (np.median(mu), np.max(mu), np.min(ephs['geo_dist'])))
            
            # Find observations which come within roughTol of the fov.
            ephsIdxs = np.searchsorted(ephs['time'], obsData[self.obsTimeCol])
            roughIdxObs = self._ssoInCircleFov(ephs[ephsIdxs], obsData, self.roughTol)
            if len(roughIdxObs) > 0:
                # Generate exact ephemerides for these times.
                times = obsData[self.obsTimeCol][roughIdxObs]
                logging.debug(("%d/%d   id=%s : " % (i, len(orbits), objid)) + 
                              datetime.datetime.now().strftime("Exact start: %Y-%m-%d %H:%M:%S") + 
                              " nExactTimes: %s" % len(times))
                ephs = self.generateEphemerides(sso, times, ephMode=self.ephMode, ephType=self.ephType)[0]
                logging.debug(("%d/%d   id=%s : " % (i, len(orbits), objid)) + 
                              datetime.datetime.now().strftime("Exact end: %Y-%m-%d %H:%M:%S"))
                # Identify the objects which fell within the specific footprint.
                idxObs = self.ssoInFov(ephs, obsData[roughIdxObs])
                logging.info(("%d/%d   id=%s : " % (i, len(orbits), objid)) + 
                             "Object in %d out of %d potential fields (%.2f%% success rate)" 
                             % (len(idxObs), len(times), 100.*float(len(idxObs))/len(times)))
                # Write these observations to disk.
                self.writeObs(objid, ephs[idxObs], obsData[roughIdxObs][idxObs], sedname=sedname)

