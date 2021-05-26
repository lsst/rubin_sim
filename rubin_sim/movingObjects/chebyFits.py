import os
import warnings
import numpy as np
from .chebyshevUtils import chebfit, makeChebMatrix, makeChebMatrixOnlyX
from .orbits import Orbits
from .ooephemerides import PyOrbEphemerides
from .utils import get_oorb_data_dir

__all__ = ['ChebyFits']


def three_sixty_to_neg(ra):
    """Wrap discontiguous RA values into more-contiguous results."""
    if (ra.min() < 100) and (ra.max() > 270):
        ra = np.where(ra > 270, ra - 360, ra)
    return ra


class ChebyFits(object):
    """Generates chebyshev coefficients for a provided set of orbits.

    Calculates true ephemerides using PyEphemerides, then fits these positions with a constrained
    Chebyshev Polynomial, using the routines in chebyshevUtils.py.
    Many chebyshev polynomials are used to fit one moving object over a given timeperiod;
    typically, the length of each segment is typically about 2 days for MBAs.
    The start and end of each segment must match exactly, and the entire segments must
    fit into the total timespan an integer number of times. This is accomplished by setting nDecimal to
    the number of decimal places desired in the 'time' value. For faster moving objects, this number needs
    be greater to allow for smaller subdivisions. It's tempting to allow flexibility to the point of not
    enforcing this for non-database use; however, then the resulting ephemeris may have multiple values
    depending on which polynomial segment was used to calculate the ephemeris.
    The length of each chebyshev polynomial is related to the number of ephemeris positions used to fit that
    polynomial by ngran:
    length = timestep * ngran
    The length of each polynomial is adjusted so that the residuals in RA/Dec position
    are less than skyTolerance - default = 2.5mas.
    The polynomial length (and the resulting residuals) is affected by ngran (i.e. timestep).

    Default values are based on Yusra AlSayaad's work.

    Parameters
    ----------
    orbitsObj : Orbits
        The orbits for which to fit chebyshev polynomial coefficients.
    tStart : float
        The starting point in time to fit coefficients. MJD.
    tSpan : float
        The time span (starting at tStart) over which to fit coefficients. Days.
    timeScale : {'TAI', 'UTC', 'TT'}
        The timescale of the MJD time, tStart, and the timeScale that should be
        used with the chebyshev coefficients.
    obsCode : int, optional
        The observatory code of the location for which to generate ephemerides. Default 807 (CTIO).
    skyTolerance : float, optional
        The desired tolerance in mas between ephemerides calculated by OpenOrb and fitted values.
        Default 2.5 mas.
    nCoeff_position : int, optional
        The number of Chebyshev coefficients to fit for the RA/Dec positions. Default 14.
    nCoeff_vmag : int, optional
        The number of Chebyshev coefficients to fit for the V magnitude values. Default 9.
    nCoeff_delta : int, optional
        The number of Chebyshev coefficients to fit for the distance between Earth/Object. Default 5.
    nCoeff_elongation : int, optional
        The number of Chebyshev coefficients to fit for the solar elongation. Default 5.
    ngran : int, optional
        The number of ephemeris points within each Chebyshev polynomial segment. Default 64.
    ephFile : str, optional
        The path to the JPL ephemeris file to use. Default is '$OORB_DATA/de405.dat'.
    nDecimal : int, optional
        The number of decimal places to allow in the segment length (and thus the times of the endpoints)
        can be limited to nDecimal places. Default 10.
        For LSST SIMS moving object database, this should be 13 decimal places for NEOs and 0 for all others.
    """
    def __init__(self, orbitsObj, tStart, tSpan, timeScale='TAI',
                 obscode=807, skyTolerance=2.5,
                 nCoeff_position=14, nCoeff_vmag=9, nCoeff_delta=5,
                 nCoeff_elongation=6, ngran=64, ephFile=None, nDecimal=10):
        # Set up PyOrbEphemerides.
        if ephFile is None:
            self.ephFile = os.path.join(get_oorb_data_dir(), 'de405.dat')
        else:
            self.ephFile = ephFile
        self.pyephems = PyOrbEphemerides(self.ephFile)
        # And then set orbits.
        self._setOrbits(orbitsObj)
        # Save input parameters.
        # We have to play some games with the start and end times, using Decimal,
        # in order to get the subdivision and times to match exactly, up to nDecimal places.
        self.nDecimal = int(nDecimal)
        self.tStart = round(tStart, self.nDecimal)
        self.tSpan = round(tSpan, self.nDecimal)
        self.tEnd = round(self.tStart + self.tSpan, self.nDecimal)
        # print('input times', self.tStart, self.tSpan, self.tEnd, orbitsObj.orbits.objId.as_matrix())
        if timeScale.upper() == 'TAI':
            self.timeScale = 'TAI'
        elif timeScale.upper() == 'UTC':
            self.timeScale = 'UTC'
        elif timeScale.upper() == 'TT':
            self.timeScale = 'TT'
        else:
            raise ValueError('Do not understand timeScale; use TAI, UTC or TT.')
        self.obscode = obscode
        self.skyTolerance = skyTolerance
        self.nCoeff = {}
        self.nCoeff['position'] = int(nCoeff_position)
        self.nCoeff['geo_dist'] = int(nCoeff_delta)
        self.nCoeff['vmag'] = int(nCoeff_vmag)
        self.nCoeff['elongation'] = int(nCoeff_elongation)
        self.ngran = int(ngran)
        # Precompute multipliers (we only do this once, instead of per segment).
        self._precomputeMultipliers()
        # Initialize attributes to save the coefficients and residuals.
        self.coeffs = {'objId': [], 'tStart': [], 'tEnd': [],
                       'ra': [], 'dec': [], 'geo_dist': [], 'vmag': [], 'elongation': []}
        self.resids = {'objId': [], 'tStart': [], 'tEnd': [],
                       'pos': [], 'geo_dist': [], 'vmag': [], 'elongation': []}
        self.failed = []

    def _setOrbits(self, orbitsObj):
        """Set the orbits, to be used to generate ephemerides.

        Parameters
        ----------
        orbitsObj : Orbits
           The orbits to use to generate ephemerides.
        """
        if not isinstance(orbitsObj, Orbits):
            raise ValueError('Need to provide an Orbits object.')
        self.orbitsObj = orbitsObj
        self.pyephems.setOrbits(self.orbitsObj)

    def _precomputeMultipliers(self):
        """Calculate multipliers for Chebyshev fitting.

        Calculate these once, rather than for each segment.
        """
        # The nPoints are predetermined here, based on Yusra's earlier work.
        # The weight is based on Newhall, X. X. 1989, Celestial Mechanics, 45, p. 305-310
        self.multipliers = {}
        self.multipliers['position'] = makeChebMatrix(self.ngran + 1,
                                                      self.nCoeff['position'], weight=0.16)
        self.multipliers['vmag'] = makeChebMatrixOnlyX(self.ngran + 1, self.nCoeff['vmag'])
        self.multipliers['geo_dist'] = makeChebMatrixOnlyX(self.ngran + 1, self.nCoeff['geo_dist'])
        self.multipliers['elongation'] = makeChebMatrixOnlyX(self.ngran + 1, self.nCoeff['elongation'])

    def _lengthToTimestep(self, length):
        """Convert chebyshev polynomial segment lengths to the corresponding timestep over the segment.

        Parameters
        ----------
        length : float
            The chebyshev polynomial segment length (nominally, days).

        Returns
        -------
        float
            The corresponding timestep, = length/ngran (nominally, days).
        """
        return length / self.ngran

    def makeAllTimes(self):
        """Using tStart and tEnd, generate a numpy array containing times spaced at
        timestep = self.length/self.ngran.
        The expected use for this time array would be to generate ephemerides at each timestep.

        Returns
        -------
        numpy.ndarray
            Numpy array of times.
        """
        try:
            self.length
        except AttributeError:
            raise AttributeError('Need to set self.timestep first, using calcSegmentLength.')
        timestep = self._lengthToTimestep(self.length)
        times = np.arange(self.tStart, self.tEnd + timestep / 2, timestep)
        return times

    def generateEphemerides(self, times, byObject=True):
        """Generate ephemerides using OpenOrb for all orbits.

        Parameters
        ----------
        times : numpy.ndarray
            The times to use for ephemeris generation.
        """
        return self.pyephems.generateEphemerides(times, obscode=self.obscode,
                                                 ephMode='N', ephType='basic',
                                                 timeScale=self.timeScale, byObject=byObject)

    def _roundLength(self, length):
        """Modify length, to fit in an 'integer multiple' within the tStart/tEnd,
        and to have the desired number of decimal values.

        Parameters
        ----------
        length : float
            The input length value to be rounded.

        Returns
        -------
        float
            The rounded length value.
        """
        length = round(length, self.nDecimal)
        length_in = length
        # Make length an integer value within the time interval, to last decimal place accuracy.
        counter = 0
        prev_int_factor = 0
        numTolerance = 10. ** (-1 * (self.nDecimal - 1))
        while ((self.tSpan % length) > numTolerance) and (length > 0) and (counter < 20):
            int_factor = int(self.tSpan / length) + 1  # round up / ceiling
            if int_factor == prev_int_factor:
                int_factor = prev_int_factor + 1
            prev_int_factor = int_factor
            length = round(self.tSpan / int_factor, self.nDecimal)
            counter += 1
        if (self.tSpan % length) > numTolerance or (length <= 0):
            # Add this entire segment into the failed list.
            for objId in self.orbitsObj.orbits['objId'].as_matrix():
                self.failed.append((objId, self.tStart, self.tEnd))
            raise ValueError('Could not find a suitable length for the timespan (start %f, span %f), '
                             'starting with length %s, ending with length value %f'
                             % (self.tStart, self.tSpan, str(length_in), length))
        return length

    def _testResiduals(self, length, cutoff=99):
        """Calculate the position residual, for a test case.
        Convenience function to make calcSegmentLength easier to read.
        """
        # The pos_resid used will be the 'cutoff' percentile of all max residuals per object.
        max_pos_resids = np.zeros(len(self.orbitsObj), float)
        timestep = self._lengthToTimestep(length)
        # Test for one segment near the start (would do at midpoint, but for long timespans
        # this is not efficient .. a point near the start should be fine).
        times = np.arange(self.tStart, self.tStart + length + timestep / 2, timestep)
        # We must regenerate ephemerides here, because the timestep is different each time.
        ephs = self.generateEphemerides(times, byObject=True)
        # Look for the coefficients and residuals.
        for i, e in enumerate(ephs):
            coeff_ra, coeff_dec, max_pos_resids[i] = self._getCoeffsPosition(e)
        # Find a representative value and return.
        pos_resid = np.percentile(max_pos_resids, cutoff)
        ratio = pos_resid / self.skyTolerance
        return pos_resid, ratio

    def calcSegmentLength(self, length=None):
        """Set the typical initial ephemeris timestep and segment length for all objects between tStart/tEnd.

        Sets self.length.

        The segment length will fit into the time period between tStart/tEnd an approximately integer
        multiple of times, and will only have a given number of decimal places.

        Parameters
        ----------
        length : float, optional
            If specified, this value for the length is used, instead of calculating it here.
        """
        # If length is specified, use it and do nothing else.
        if length is not None:
            length = self._roundLength(length)
            pos_resid, ratio = self._testResiduals(length)
            if pos_resid > self.skyTolerance:
                warnings.warn('Will set length and timestep, but this value of length '
                              'produces residuals (%f) > skyTolerance (%f).' % (pos_resid, self.skyTolerance))
            self.length = length
            return
        # Otherwise, calculate an appropriate length and timestep.
        # Give a guess at a very approximate segment length, given the skyTolerance,
        # purposefully trying to overestimate this value.
        # The actual behavior of the residuals is not linear with segment length.
        # There is a linear increase at low residuals < ~2 mas / segment length < 2 days
        # Then at around 2 days the residuals blow up, increasing rapidly to about 5000 mas
        #   (depending on orbit .. TNOs, for example, increase but only to about 300 mas,
        #    when the residuals resume ~linear growth out to 70 day segments if ngran=128)
        # Make an arbitrary cap on segment length at 60 days, (25000 mas) ~.5 arcminute accuracy.
        maxLength = 60
        maxIterations = 50
        if self.skyTolerance < 5:
            # This is the cap of the low-linearity regime, looping below will refine this value.
            length = 2.0
        elif self.skyTolerance >= 5000:
            # Make a very rough guess.
            length = np.round((5000.0 / 20.0) * (self.skyTolerance - 5000.)) + 5.0
            length = np.min([maxLength, int(length * 10) / 10.0])
        else:
            # Try to pick a length somewhere in the middle of the fast increase.
            length = 4.0
        # Tidy up some characteristics of "length":
        # make it fit an integer number of times into overall timespan.
        # and use a given number of decimal places (easier for database storage).
        length = self._roundLength(length)
        # Check the resulting residuals.
        pos_resid, ratio = self._testResiduals(length)
        counter = 0
        # Now should be relatively close. Start to zero in using slope around the value.ngran
        while pos_resid > self.skyTolerance and counter <= maxIterations and length > 0:
            length = length / 2
            length = self._roundLength(length)
            pos_resid, ratio = self._testResiduals(length)
            counter += 1
            # print(counter, length, pos_resid, ratio)
        if counter > maxIterations or length <= 0:
            # Add this entire segment into the failed list.
            for objId in self.orbitsObj.orbits['objId'].as_matrix():
                self.failed.append((objId, self.tStart, self.tEnd))
            errorMessage = 'Could not find good segment length to meet skyTolerance %f' % (self.skyTolerance)
            errorMessage += ' milliarcseconds within %d iterations. ' % (maxIterations)
            errorMessage += 'Final residual was %f milli-arcseconds.' % (pos_resid)
            raise ValueError(errorMessage)
        else:
            self.length = length

    def _getCoeffsPosition(self, ephs):
        """Calculate coefficients for the ra/dec values of a single objects ephemerides.

        Parameters
        ----------
        times : numpy.ndarray
            The times of the ephemerides.
        ephs : numpy.ndarray
            The structured array returned by PyOrbEphemerides holding ephemeris values, for one object.

        Returns
        -------
        numpy.ndarray
            The ra coefficients
        numpy.ndarray
            The dec coefficients
        float
            The positional error residuals between fit and ephemeris values, in mas.
        """
        dradt_coord = ephs['dradt'] / np.cos(np.radians(ephs['dec']))
        coeff_ra, resid_ra, rms_ra_resid, max_ra_resid = \
            chebfit(ephs['time'], three_sixty_to_neg(ephs['ra']), dxdt=dradt_coord,
                    xMultiplier=self.multipliers['position'][0],
                    dxMultiplier=self.multipliers['position'][1],
                    nPoly=self.nCoeff['position'])
        coeff_dec, resid_dec, rms_dec_resid, max_dec_resid = \
            chebfit(ephs['time'], ephs['dec'], dxdt=ephs['ddecdt'],
                    xMultiplier=self.multipliers['position'][0],
                    dxMultiplier=self.multipliers['position'][1],
                    nPoly=self.nCoeff['position'])
        max_pos_resid = np.max(np.sqrt(resid_dec**2 +
                                       (resid_ra * np.cos(np.radians(ephs['dec'])))**2))
        # Convert position residuals to mas.
        max_pos_resid *= 3600.0 * 1000.0
        return coeff_ra, coeff_dec, max_pos_resid

    def _getCoeffsOther(self, ephs):
        """Calculate coefficients for the ra/dec values of a single objects ephemerides.

        Parameters
        ----------
        ephs : numpy.ndarray
            The structured array returned by PyOrbEphemerides holding ephemeris values, for one object.

        Returns
        -------
        dict
            Dictionary containing the coefficients for each of 'geo_dist', 'vmag', 'elongation'
        dict
            Dictionary containing the max residual values for each of 'geo_dist', 'vmag', 'elongation'.
        """
        coeffs = {}
        max_resids = {}
        for key, ephValue in zip(('geo_dist', 'vmag', 'elongation'), ('geo_dist', 'magV', 'solarelon')):
            coeffs[key], resid, rms, max_resids[key] = chebfit(ephs['time'], ephs[ephValue], dxdt=None,
                                                               xMultiplier=self.multipliers[key],
                                                               dxMultiplier=None, nPoly=self.nCoeff[key])
        return coeffs, max_resids

    def calcSegments(self):
        """Run the calculation of all segments over the entire time span.
        """
        # First calculate ephemerides for all objects, over entire time span.
        # For some objects, we will end up recalculating the ephemeride values, but most should be fine.
        times = self.makeAllTimes()
        ephs = self.generateEphemerides(times)
        eps = self._lengthToTimestep(self.length)/4.0
        # Loop through each object to generate coefficients.
        for orbitObj, e in zip(self.orbitsObj, ephs):
            tSegmentStart = self.tStart
            # Cycle through all segments until we reach the end of the period we're fitting.
            while tSegmentStart < (self.tEnd - eps):
                # Identify the subset of times and ephemerides which are relevant for this segment
                # (at the default segment size).
                tSegmentEnd = round(tSegmentStart + self.length, self.nDecimal)
                subset = np.where((times >= tSegmentStart) & (times < tSegmentEnd + eps))
                self.calcOneSegment(orbitObj, e[subset])
                tSegmentStart = tSegmentEnd

    def calcOneSegment(self, orbitObj, ephs):
        """Calculate the coefficients for a single Chebyshev segment, for a single object.

        Calculates the coefficients and residuals, and saves this information to self.coeffs,
        self.resids, and (if there are problems), self.failed.

        Parameters
        ----------
        orbitObj : Orbits
            The single Orbits object we're fitting at the moment.
        ephs : numpy.ndarray
            The ephemerides we're fitting at the moment (for the single object / single segment).
        """
        objId = orbitObj.orbits.objId.iloc[0]
        tSegmentStart = ephs['time'][0]
        tSegmentEnd = ephs['time'][-1]
        coeff_ra, coeff_dec, max_pos_resid = self._getCoeffsPosition(ephs)
        if max_pos_resid > self.skyTolerance:
            # print('subdividing segments', orbitObj.orbits.objId.iloc[0])
            self._subdivideSegment(orbitObj, ephs)
        else:
            # print('working on ', orbitObj.orbits.objId.iloc[0], 'at times', tSegmentStart, tSegmentEnd)
            coeffs, max_resids = self._getCoeffsOther(ephs)
            fitFailed = False
            for k in max_resids:
                if np.isnan(max_resids[k]):
                    fitFailed = True
            if fitFailed:
                warnings.warn('Fit failed for orbitObj %d for times between %f and %f'
                              % (objId, tSegmentStart, tSegmentEnd))
                self.failed.append((orbitObj.orbits['objId'], tSegmentStart, tSegmentEnd))
            else:
                # Consolidate items into the tracked coefficient values.
                self.coeffs['objId'].append(objId)
                self.coeffs['tStart'].append(tSegmentStart)
                self.coeffs['tEnd'].append(tSegmentEnd)
                self.coeffs['ra'].append(coeff_ra)
                self.coeffs['dec'].append(coeff_dec)
                self.coeffs['geo_dist'].append(coeffs['geo_dist'])
                self.coeffs['vmag'].append(coeffs['vmag'])
                self.coeffs['elongation'].append(coeffs['elongation'])
                # Consolidate items into the tracked residual values.
                self.resids['objId'].append(objId)
                self.resids['tStart'].append(tSegmentStart)
                self.resids['tEnd'].append(tSegmentEnd)
                self.resids['pos'].append(max_pos_resid)
                self.resids['geo_dist'].append(max_resids['geo_dist'])
                self.resids['vmag'].append(max_resids['geo_dist'])
                self.resids['elongation'].append(max_resids['elongation'])

    def _subdivideSegment(self, orbitObj, ephs):
        """Subdivide a segment, then calculate the segment coefficients.

        Parameters
        ----------
        orbitObj : Orbits
            The single Orbits object we're fitting at the moment.
        ephs : numpy.ndarray
            The ephemerides we're fitting at the moment (for the single object / single segment).
        """
        newCheby = ChebyFits(orbitObj, ephs['time'][0], (ephs['time'][-1] - ephs['time'][0]),
                             timeScale=self.timeScale, obscode=self.obscode,
                             skyTolerance=self.skyTolerance,
                             nCoeff_position=self.nCoeff['position'],
                             nCoeff_vmag=self.nCoeff['vmag'],
                             nCoeff_delta=self.nCoeff['geo_dist'],
                             nCoeff_elongation=self.nCoeff['elongation'],
                             ngran=self.ngran, ephFile=self.ephFile,
                             nDecimal=self.nDecimal)
        try:
            newCheby.calcSegmentLength()
        except ValueError as ve:
            # Could not find a good segment length.
            warningmessage = 'Objid %s, segment %f to %f ' % (orbitObj.orbits.objId.iloc[0],
                                                              ephs['time'][0], ephs['time'][-1])
            warningmessage += ' - error: %s' % (ve)
            warnings.warn(warningmessage)
            self.failed += newCheby.failed
            return
        newCheby.calcSegments()
        # Add subdivided segment values into tracked values here.
        for k in self.coeffs:
            self.coeffs[k] += newCheby.coeffs[k]
        for k in self.resids:
            self.resids[k] += newCheby.resids[k]
        self.failed += newCheby.failed

    def write(self, coeffFile, residFile, failedFile, append=False):
        """Write coefficients, residuals and failed fits to disk.

        Parameters
        ----------
        coeffFile : str
            The filename for the coefficient values.
        residFile : str
            The filename for the residual values.
        failedFile : str
            The filename to write the failed fit information (if failed objects exist).
        append : bool, optional
            Flag to append (or overwrite) the output files.
        """
        if append:
            openMode = 'aw'
        else:
            openMode = 'w'
        # Write a header to the coefficients file, if writing to a new file:
        if (not append) or (not os.path.isfile(coeffFile)):
            header = 'objId tStart tEnd '
            header += ' '.join(['ra_%d' % x for x in range(self.nCoeff['position'])]) + ' '
            header += ' '.join(['dec_%d' % x for x in range(self.nCoeff['position'])]) + ' '
            header += ' '.join(['geo_dist_%d' % x for x in range(self.nCoeff['geo_dist'])]) + ' '
            header += ' '.join(['vmag_%d' % x for x in range(self.nCoeff['vmag'])]) + ' '
            header += ' '.join(['elongation_%d' % x for x in range(self.nCoeff['elongation'])])
        else:
            header = None
        if (not append) or (not os.path.isfile(residFile)):
            resid_header = 'objId segNum tStart tEnd length pos geo_dist vmag elong'
        else:
            resid_header = None
        timeformat = '%.' + '%s' % self.nDecimal + 'f'
        with open(coeffFile, openMode) as f:
            if header is not None:
                print(header, file=f)
            for i, (objId, tStart, tEnd, cRa, cDec, cDelta, cVmag, cE) in \
                    enumerate(zip(self.coeffs['objId'], self.coeffs['tStart'],
                                  self.coeffs['tEnd'], self.coeffs['ra'],
                                  self.coeffs['dec'], self.coeffs['geo_dist'],
                                  self.coeffs['vmag'], self.coeffs['elongation'])):
                print("%s %s %s %s %s %s %s %s" % (objId, timeformat % tStart, timeformat % tEnd,
                                                   " ".join('%.14e' % j for j in cRa),
                                                   " ".join('%.14e' % j for j in cDec),
                                                   " ".join('%.7e' % j for j in cDelta),
                                                   " ".join('%.7e' % j for j in cVmag),
                                                   " ".join('%.7e' % j for j in cE)), file=f)

        with open(residFile, openMode) as f:
            if resid_header is not None:
                print(resid_header, file=f)
            for i, (objId, tStart, tEnd, rPos, rDelta, rVmag, rE) in \
                    enumerate(zip(self.resids['objId'], self.resids['tStart'],
                                  self.resids['tEnd'], self.resids['pos'],
                                  self.resids['geo_dist'], self.resids['vmag'],
                                  self.resids['elongation'])):
                print("%s %i %.14f %.14f %.14f %.14e %.14e %.14e %.14e" % (objId, i + 1,
                                                                           tStart, tEnd,
                                                                           (tEnd - tStart),
                                                                           rPos, rDelta,
                                                                           rVmag, rE), file=f)

        if len(self.failed) > 0:
            with open(failedFile, openMode) as f:
                for i, failed in enumerate(self.failed):
                    print(' '.join([str(x) for x in failed]), file=f)
