from builtins import zip
from builtins import range
import numpy as np
from .baseStacker import BaseStacker
import warnings

__all__ = ['setupDitherStackers', 'wrapRADec', 'wrapRA', 'inHexagon', 'polygonCoords',
           'BaseDitherStacker',
           'RandomDitherFieldPerVisitStacker', 'RandomDitherFieldPerNightStacker',
           'RandomDitherPerNightStacker',
           'SpiralDitherFieldPerVisitStacker', 'SpiralDitherFieldPerNightStacker',
           'SpiralDitherPerNightStacker',
           'HexDitherFieldPerVisitStacker', 'HexDitherFieldPerNightStacker',
           'HexDitherPerNightStacker',
           'RandomRotDitherPerFilterChangeStacker']

# Stacker naming scheme:
# [Pattern]Dither[Field]Per[Timescale].
#  Timescale indicates how often the dither offset is changed.
#  The presence of 'Field' indicates that a new offset is chosen per field, on the indicated timescale.
#  The absence of 'Field' indicates that all visits within the indicated timescale use the same dither offset.


# Original dither stackers (Random, Spiral, Hex) written by Lynne Jones (lynnej@uw.edu)
# Additional dither stackers written by Humna Awan (humna.awan@rutgers.edu), with addition of
# constraining dither offsets to be within an inscribed hexagon (code modifications for use here by LJ).

def setupDitherStackers(raCol, decCol, degrees, **kwargs):
    b = BaseStacker()
    stackerList = []
    if raCol in b.sourceDict:
        stackerList.append(b.sourceDict[raCol](degrees=degrees, **kwargs))
    if decCol in b.sourceDict:
        if b.sourceDict[raCol] != b.sourceDict[decCol]:
            stackerList.append(b.sourceDict[decCol](degrees=degrees, **kwargs))
    return stackerList


def wrapRADec(ra, dec):
    """
    Wrap RA into 0-2pi and Dec into +/0 pi/2.

    Parameters
    ----------
    ra : numpy.ndarray
        RA in radians
    dec : numpy.ndarray
        Dec in radians

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        Wrapped RA/Dec values, in radians.
    """
    # Wrap dec.
    low = np.where(dec < -np.pi / 2.0)[0]
    dec[low] = -1 * (np.pi + dec[low])
    ra[low] = ra[low] - np.pi
    high = np.where(dec > np.pi / 2.0)[0]
    dec[high] = np.pi - dec[high]
    ra[high] = ra[high] - np.pi
    # Wrap RA.
    ra = ra % (2.0 * np.pi)
    return ra, dec


def wrapRA(ra):
    """
    Wrap only RA values into 0-2pi (using mod).

    Parameters
    ----------
    ra : numpy.ndarray
        RA in radians

    Returns
    -------
    numpy.ndarray
        Wrapped RA values, in radians.
    """
    ra = ra % (2.0 * np.pi)
    return ra


def inHexagon(xOff, yOff, maxDither):
    """
    Identify dither offsets which fall within the inscribed hexagon.

    Parameters
    ----------
    xOff : numpy.ndarray
        The x values of the dither offsets.
    yoff : numpy.ndarray
        The y values of the dither offsets.
    maxDither : float
        The maximum dither offset.

    Returns
    -------
    numpy.ndarray
        Indexes of the offsets which are within the hexagon inscribed inside the 'maxDither' radius circle.
    """
    # Set up the hexagon limits.
    #  y = mx + b, 2h is the height.
    m = np.sqrt(3.0)
    b = m * maxDither
    h = m / 2.0 * maxDither
    # Identify offsets inside hexagon.
    inside = np.where((yOff < m * xOff + b) &
                      (yOff > m * xOff - b) &
                      (yOff < -m * xOff + b) &
                      (yOff > -m * xOff - b) &
                      (yOff < h) & (yOff > -h))[0]
    return inside


def polygonCoords(nside, radius, rotationAngle):
    """
    Find the x,y coords of a polygon.

    This is useful for plotting dither points and showing they lie within
    a given shape.

    Parameters
    ----------
    nside : int
        The number of sides of the polygon
    radius : float
        The radius within which to plot the polygon
    rotationAngle : float
        The angle to rotate the polygon to.

    Returns
    -------
    [float, float]
        List of x/y coordinates of the points describing the polygon.
    """
    eachAngle = 2 * np.pi / float(nside)
    xCoords = np.zeros(nside, float)
    yCoords = np.zeros(nside, float)
    for i in range(0, nside):
        xCoords[i] = np.sin(eachAngle * i + rotationAngle) * radius
        yCoords[i] = np.cos(eachAngle * i + rotationAngle) * radius
    return list(zip(xCoords, yCoords))


class BaseDitherStacker(BaseStacker):
    """Base class for dither stackers.

    The base class just adds an easy way to define a stacker as one of the 'dither' types of stackers.
    These run first, before any other stackers.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    colsAdded = []

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True,
                 maxDither=1.75, inHex=True):
        # Instantiate the RandomDither object and set internal variables.
        self.raCol = raCol
        self.decCol = decCol
        self.degrees = degrees
        # Convert maxDither to radians for internal use.
        self.maxDither = np.radians(maxDither)
        self.inHex = inHex
        # self.units used for plot labels
        if self.degrees:
            self.units = ['deg', 'deg']
        else:
            self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol]


class RandomDitherFieldPerVisitStacker(BaseDitherStacker):
    """
    Randomly dither the RA and Dec pointings up to maxDither degrees from center,
    with a different offset for each field, for each visit.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    randomSeed : int or None, optional
        If set, then used as the random seed for the numpy random number generation for the dither offsets.
        Default None.
    """
    # Values required for framework operation: this specifies the name of the new columns.
    colsAdded = ['randomDitherFieldPerVisitRa', 'randomDitherFieldPerVisitDec']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, maxDither=1.75,
                 inHex=True, randomSeed=None):
        """
        @ MaxDither in degrees
        """
        super().__init__(raCol=raCol, decCol=decCol, degrees=degrees, maxDither=maxDither, inHex=inHex)
        self.randomSeed = randomSeed

    def _generateRandomOffsets(self, noffsets):
        xOut = np.array([], float)
        yOut = np.array([], float)
        maxTries = 100
        tries = 0
        while (len(xOut) < noffsets) and (tries < maxTries):
            dithersRad = np.sqrt(self._rng.rand(noffsets * 2)) * self.maxDither
            dithersTheta = self._rng.rand(noffsets * 2) * np.pi * 2.0
            xOff = dithersRad * np.cos(dithersTheta)
            yOff = dithersRad * np.sin(dithersTheta)
            if self.inHex:
                # Constrain dither offsets to be within hexagon.
                idx = inHexagon(xOff, yOff, self.maxDither)
                xOff = xOff[idx]
                yOff = yOff[idx]
            xOut = np.concatenate([xOut, xOff])
            yOut = np.concatenate([yOut, yOff])
            tries += 1
        if len(xOut) < noffsets:
            raise ValueError('Could not find enough random points within the hexagon in %d tries. '
                             'Try another random seed?' % (maxTries))
        self.xOff = xOut[0:noffsets]
        self.yOff = yOut[0:noffsets]

    def _run(self, simData, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData
        # Generate random numbers for dither, using defined seed value if desired.
        if not hasattr(self, '_rng'):
            if self.randomSeed is not None:
                self._rng = np.random.RandomState(self.randomSeed)
            else:
                self._rng = np.random.RandomState(2178813)

        # Generate the random dither values.
        noffsets = len(simData[self.raCol])
        self._generateRandomOffsets(noffsets)
        # Add to RA and dec values.
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        simData['randomDitherFieldPerVisitRa'] = (ra + self.xOff / np.cos(dec))
        simData['randomDitherFieldPerVisitDec'] = dec + self.yOff
        # Wrap back into expected range.
        simData['randomDitherFieldPerVisitRa'], simData['randomDitherFieldPerVisitDec'] = \
            wrapRADec(simData['randomDitherFieldPerVisitRa'], simData['randomDitherFieldPerVisitDec'])
        # Convert to degrees
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class RandomDitherFieldPerNightStacker(RandomDitherFieldPerVisitStacker):
    """
    Randomly dither the RA and Dec pointings up to maxDither degrees from center,
    one dither offset per new night of observation of a field.
    e.g. visits within the same night, to the same field, have the same offset.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    randomSeed : int or None, optional
        If set, then used as the random seed for the numpy random number generation for the dither offsets.
        Default None.
    """
    # Values required for framework operation: this specifies the names of the new columns.
    colsAdded = ['randomDitherFieldPerNightRa', 'randomDitherFieldPerNightDec']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, fieldIdCol='fieldId',
                 nightCol='night', maxDither=1.75, inHex=True, randomSeed=None):
        """
        @ MaxDither in degrees
        """
        # Instantiate the RandomDither object and set internal variables.
        super().__init__(raCol=raCol, decCol=decCol, degrees=degrees,
                         maxDither=maxDither, inHex=inHex, randomSeed=randomSeed)
        self.nightCol = nightCol
        self.fieldIdCol = fieldIdCol
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.nightCol, self.fieldIdCol]

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        # Generate random numbers for dither, using defined seed value if desired.
        if not hasattr(self, '_rng'):
            if self.randomSeed is not None:
                self._rng = np.random.RandomState(self.randomSeed)
            else:
                self._rng = np.random.RandomState(872453)

        # Generate the random dither values, one per night per field.
        fields = np.unique(simData[self.fieldIdCol])
        nights = np.unique(simData[self.nightCol])
        self._generateRandomOffsets(len(fields) * len(nights))
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        # counter to ensure new random numbers are chosen every time
        delta = 0
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply dithers, increasing each night.
            nights = simData[self.nightCol][match]
            vertexIdxs = np.searchsorted(np.unique(nights), nights)
            vertexIdxs = vertexIdxs % len(self.xOff)
            # ensure that the same xOff/yOff entries are not chosen
            delta = delta + len(vertexIdxs)
            simData['randomDitherFieldPerNightRa'][match] = (ra[match] +
                                                             self.xOff[vertexIdxs] /
                                                             np.cos(dec[match]))
            simData['randomDitherFieldPerNightDec'][match] = (dec[match] +
                                                              self.yOff[vertexIdxs])
        # Wrap into expected range.
        simData['randomDitherFieldPerNightRa'], simData['randomDitherFieldPerNightDec'] = \
            wrapRADec(simData['randomDitherFieldPerNightRa'], simData['randomDitherFieldPerNightDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class RandomDitherPerNightStacker(RandomDitherFieldPerVisitStacker):
    """
    Randomly dither the RA and Dec pointings up to maxDither degrees from center,
    one dither offset per night.
    All fields observed within the same night get the same offset.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    randomSeed : int or None, optional
        If set, then used as the random seed for the numpy random number generation for the dither offsets.
        Default None.
    """
    # Values required for framework operation: this specifies the names of the new columns.
    colsAdded = ['randomDitherPerNightRa', 'randomDitherPerNightDec']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, nightCol='night',
                 maxDither=1.75, inHex=True, randomSeed=None):
        """
        @ MaxDither in degrees
        """
        # Instantiate the RandomDither object and set internal variables.
        super().__init__(raCol=raCol, decCol=decCol, degrees=degrees,
                         maxDither=maxDither, inHex=inHex, randomSeed=randomSeed)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.nightCol]

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        # Generate random numbers for dither, using defined seed value if desired.
        if not hasattr(self, '_rng'):
            if self.randomSeed is not None:
                self._rng = np.random.RandomState(self.randomSeed)
            else:
                self._rng = np.random.RandomState(66334)

        # Generate the random dither values, one per night.
        nights = np.unique(simData[self.nightCol])
        self._generateRandomOffsets(len(nights))
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        # Add to RA and dec values.
        for n, x, y in zip(nights, self.xOff, self.yOff):
            match = np.where(simData[self.nightCol] == n)[0]
            simData['randomDitherPerNightRa'][match] = (ra[match] +
                                                        x / np.cos(dec[match]))
            simData['randomDitherPerNightDec'][match] = dec[match] + y
        # Wrap RA/Dec into expected range.
        simData['randomDitherPerNightRa'], simData['randomDitherPerNightDec'] = \
            wrapRADec(simData['randomDitherPerNightRa'], simData['randomDitherPerNightDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class SpiralDitherFieldPerVisitStacker(BaseDitherStacker):
    """
    Offset along an equidistant spiral with numPoints, out to a maximum radius of maxDither.
    Each visit to a field receives a new, sequential offset.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    numPoints : int, optional
        The number of points in the spiral.
        Default 60.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    nCoils : int, optional
        The number of coils the spiral should have.
        Default 5.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    # Values required for framework operation: this specifies the names of the new columns.
    colsAdded = ['spiralDitherFieldPerVisitRa', 'spiralDitherFieldPerVisitDec']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, fieldIdCol='fieldId',
                 numPoints=60, maxDither=1.75, nCoils=5, inHex=True):
        """
        @ MaxDither in degrees
        """
        super().__init__(raCol=raCol, decCol=decCol, degrees=degrees, maxDither=maxDither, inHex=inHex)
        self.fieldIdCol = fieldIdCol
        # Convert maxDither from degrees (internal units for ra/dec are radians)
        self.numPoints = numPoints
        self.nCoils = nCoils
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.fieldIdCol]

    def _generateSpiralOffsets(self):
        # First generate a full archimedean spiral ..
        theta = np.arange(0.0001, self.nCoils * np.pi * 2., 0.001)
        a = self.maxDither/theta.max()
        if self.inHex:
            a = 0.85 * a
        r = theta * a
        # Then pick out equidistant points along the spiral.
        arc = a / 2.0 * (theta * np.sqrt(1 + theta**2) + np.log(theta + np.sqrt(1 + theta**2)))
        stepsize = arc.max()/float(self.numPoints)
        arcpts = np.arange(0, arc.max(), stepsize)
        arcpts = arcpts[0:self.numPoints]
        rpts = np.zeros(self.numPoints, float)
        thetapts = np.zeros(self.numPoints, float)
        for i, ap in enumerate(arcpts):
            diff = np.abs(arc - ap)
            match = np.where(diff == diff.min())[0]
            rpts[i] = r[match]
            thetapts[i] = theta[match]
        # Translate these r/theta points into x/y (ra/dec) offsets.
        self.xOff = rpts * np.cos(thetapts)
        self.yOff = rpts * np.sin(thetapts)

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        # Generate the spiral offset vertices.
        self._generateSpiralOffsets()
        # Now apply to observations.
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        for fieldid in np.unique(simData[self.fieldIdCol]):
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply sequential dithers, increasing with each visit.
            vertexIdxs = np.arange(0, len(match), 1)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['spiralDitherFieldPerVisitRa'][match] = (ra[match] +
                                                             self.xOff[vertexIdxs] /
                                                             np.cos(dec[match]))
            simData['spiralDitherFieldPerVisitDec'][match] = (dec[match] +
                                                              self.yOff[vertexIdxs])
        # Wrap into expected range.
        simData['spiralDitherFieldPerVisitRa'], simData['spiralDitherFieldPerVisitDec'] = \
            wrapRADec(simData['spiralDitherFieldPerVisitRa'], simData['spiralDitherFieldPerVisitDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class SpiralDitherFieldPerNightStacker(SpiralDitherFieldPerVisitStacker):
    """
    Offset along an equidistant spiral with numPoints, out to a maximum radius of maxDither.
    Each field steps along a sequential series of offsets, each night it is observed.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    numPoints : int, optional
        The number of points in the spiral.
        Default 60.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    nCoils : int, optional
        The number of coils the spiral should have.
        Default 5.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    # Values required for framework operation: this specifies the names of the new columns.
    colsAdded = ['spiralDitherFieldPerNightRa', 'spiralDitherFieldPerNightDec']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, fieldIdCol='fieldId',
                 nightCol='night', numPoints=60, maxDither=1.75, nCoils=5, inHex=True):
        """
        @ MaxDither in degrees
        """
        super().__init__(raCol=raCol, decCol=decCol, degrees=degrees, fieldIdCol=fieldIdCol,
                         numPoints=numPoints, maxDither=maxDither, nCoils=nCoils, inHex=inHex)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        self._generateSpiralOffsets()
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply a sequential dither, increasing each night.
            nights = simData[self.nightCol][match]
            vertexIdxs = np.searchsorted(np.unique(nights), nights)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['spiralDitherFieldPerNightRa'][match] = (ra[match] +
                                                             self.xOff[vertexIdxs] /
                                                             np.cos(dec[match]))
            simData['spiralDitherFieldPerNightDec'][match] = (dec[match] +
                                                              self.yOff[vertexIdxs])
        # Wrap into expected range.
        simData['spiralDitherFieldPerNightRa'], simData['spiralDitherFieldPerNightDec'] = \
            wrapRADec(simData['spiralDitherFieldPerNightRa'], simData['spiralDitherFieldPerNightDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class SpiralDitherPerNightStacker(SpiralDitherFieldPerVisitStacker):
    """
    Offset along an equidistant spiral with numPoints, out to a maximum radius of maxDither.
    All fields observed in the same night receive the same sequential offset, changing per night.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    numPoints : int, optional
        The number of points in the spiral.
        Default 60.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    nCoils : int, optional
        The number of coils the spiral should have.
        Default 5.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    # Values required for framework operation: this specifies the names of the new columns.
    colsAdded = ['spiralDitherPerNightRa', 'spiralDitherPerNightDec']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, fieldIdCol='fieldId',
                 nightCol='night', numPoints=60, maxDither=1.75, nCoils=5, inHex=True):
        """
        @ MaxDither in degrees
        """
        super().__init__(raCol=raCol, decCol=decCol, degrees=degrees, fieldIdCol=fieldIdCol,
                         numPoints=numPoints, maxDither=maxDither, nCoils=nCoils, inHex=inHex)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        self._generateSpiralOffsets()
        nights = np.unique(simData[self.nightCol])
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        # Add to RA and dec values.
        vertexIdxs = np.searchsorted(nights, simData[self.nightCol])
        vertexIdxs = vertexIdxs % self.numPoints
        simData['spiralDitherPerNightRa'] = (ra +
                                             self.xOff[vertexIdxs] / np.cos(dec))
        simData['spiralDitherPerNightDec'] = dec + self.yOff[vertexIdxs]
        # Wrap RA/Dec into expected range.
        simData['spiralDitherPerNightRa'], simData['spiralDitherPerNightDec'] = \
            wrapRADec(simData['spiralDitherPerNightRa'], simData['spiralDitherPerNightDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class HexDitherFieldPerVisitStacker(BaseDitherStacker):
    """
    Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.
    Sequential offset for each visit.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    # Values required for framework operation: this specifies the names of the new columns.
    colsAdded = ['hexDitherFieldPerVisitRa', 'hexDitherFieldPerVisitDec']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True,
                 fieldIdCol='fieldId', maxDither=1.75, inHex=True):
        """
        @ MaxDither in degrees
        """
        super().__init__(raCol=raCol, decCol=decCol, degrees=degrees, maxDither=maxDither, inHex=inHex)
        self.fieldIdCol = fieldIdCol
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.fieldIdCol]

    def _generateHexOffsets(self):
        # Set up basics of dither pattern.
        dith_level = 4
        nrows = 2**dith_level
        halfrows = int(nrows / 2.)
        # Calculate size of each offset
        dith_size_x = self.maxDither * 2.0 / float(nrows)
        dith_size_y = np.sqrt(3) * self.maxDither / float(nrows)  # sqrt 3 comes from hexagon
        if self.inHex:
            dith_size_x = 0.95 * dith_size_x
            dith_size_y = 0.95 * dith_size_y
        # Calculate the row identification number, going from 0 at center
        nid_row = np.arange(-halfrows, halfrows + 1, 1)
        # and calculate the number of vertices in each row.
        vert_in_row = np.arange(-halfrows, halfrows + 1, 1)
        # First calculate how many vertices we will create in each row.
        total_vert = 0
        for i in range(-halfrows, halfrows + 1, 1):
            vert_in_row[i] = (nrows+1) - abs(nid_row[i])
            total_vert += vert_in_row[i]
        self.numPoints = total_vert
        self.xOff = []
        self.yOff = []
        # Calculate offsets over hexagonal grid.
        for i in range(0, nrows+1, 1):
            for j in range(0, vert_in_row[i], 1):
                self.xOff.append(dith_size_x * (j - (vert_in_row[i] - 1) / 2.0))
                self.yOff.append(dith_size_y * nid_row[i])
        self.xOff = np.array(self.xOff)
        self.yOff = np.array(self.yOff)

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        self._generateHexOffsets()
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply sequential dithers, increasing with each visit.
            vertexIdxs = np.arange(0, len(match), 1)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['hexDitherFieldPerVisitRa'][match] = (ra[match] +
                                                          self.xOff[vertexIdxs] /
                                                          np.cos(dec[match]))
            simData['hexDitherFieldPerVisitDec'][match] = dec[match] + self.yOff[vertexIdxs]
        # Wrap into expected range.
        simData['hexDitherFieldPerVisitRa'], simData['hexDitherFieldPerVisitDec'] = \
            wrapRADec(simData['hexDitherFieldPerVisitRa'], simData['hexDitherFieldPerVisitDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class HexDitherFieldPerNightStacker(HexDitherFieldPerVisitStacker):
    """
    Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.
    Sequential offset for each night of visits.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    # Values required for framework operation: this specifies the names of the new columns.
    colsAdded = ['hexDitherFieldPerNightRa', 'hexDitherFieldPerNightDec']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True,
                 fieldIdCol='fieldId', nightCol='night',
                 maxDither=1.75, inHex=True):
        """
        @ MaxDither in degrees
        """
        super().__init__(raCol=raCol, decCol=decCol, fieldIdCol=fieldIdCol,
                         degrees=degrees, maxDither=maxDither, inHex=inHex)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        self._generateHexOffsets()
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply a sequential dither, increasing each night.
            vertexIdxs = np.arange(0, len(match), 1)
            nights = simData[self.nightCol][match]
            vertexIdxs = np.searchsorted(np.unique(nights), nights)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['hexDitherFieldPerNightRa'][match] = (ra[match] +
                                                          self.xOff[vertexIdxs] /
                                                          np.cos(dec[match]))
            simData['hexDitherFieldPerNightDec'][match] = (dec[match] +
                                                           self.yOff[vertexIdxs])
        # Wrap into expected range.
        simData['hexDitherFieldPerNightRa'], simData['hexDitherFieldPerNightDec'] = \
            wrapRADec(simData['hexDitherFieldPerNightRa'], simData['hexDitherFieldPerNightDec'])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class HexDitherPerNightStacker(HexDitherFieldPerVisitStacker):
    """
    Use offsets from the hexagonal grid of 'hexdither', but visit each vertex sequentially.
    Sequential offset per night for all fields.

    Parameters
    ----------
    raCol : str, optional
        The name of the RA column in the data.
        Default 'fieldRA'.
    decCol : str, optional
        The name of the Dec column in the data.
        Default 'fieldDec'.
    degrees : bool, optional
        Flag whether RA/Dec should be treated as (and kept as) degrees.
    fieldIdCol : str, optional
        The name of the fieldId column in the data.
        Used to identify fields which should be identified as the 'same'.
        Default 'fieldId'.
    nightCol : str, optional
        The name of the night column in the data.
        Default 'night'.
    maxDither : float, optional
        The radius of the maximum dither offset, in degrees.
        Default 1.75 degrees.
    inHex : bool, optional
        If True, offsets are constrained to lie within a hexagon inscribed within the maxDither circle.
        If False, offsets can lie anywhere out to the edges of the maxDither circle.
        Default True.
    """
    # Values required for framework operation: this specifies the names of the new columns.
    colsAdded = ['hexDitherPerNightRa', 'hexDitherPerNightDec']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True, fieldIdCol='fieldId',
                 nightCol='night', maxDither=1.75, inHex=True):
        """
        @ MaxDither in degrees
        """
        super().__init__(raCol=raCol, decCol=decCol, degrees=degrees,
                         fieldIdCol=fieldIdCol, maxDither=maxDither, inHex=inHex)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)
        self.addedRA = self.colsAdded[0]
        self.addedDec = self.colsAdded[1]

    def _run(self, simData, cols_present=False):
        if cols_present:
            return simData
        # Generate the spiral dither values
        self._generateHexOffsets()
        nights = np.unique(simData[self.nightCol])
        if self.degrees:
            ra = np.radians(simData[self.raCol])
            dec = np.radians(simData[self.decCol])
        else:
            ra = simData[self.raCol]
            dec = simData[self.decCol]
        # Add to RA and dec values.
        vertexID = 0
        for n in nights:
            match = np.where(simData[self.nightCol] == n)[0]
            vertexID = vertexID % self.numPoints
            simData[self.addedRA][match] = (ra[match] + self.xOff[vertexID] / np.cos(dec[match]))
            simData[self.addedDec][match] = dec[match] + self.yOff[vertexID]
            vertexID += 1
        # Wrap RA/Dec into expected range.
        simData[self.addedRA], simData[self.addedDec] = \
            wrapRADec(simData[self.addedRA], simData[self.addedDec])
        if self.degrees:
            for col in self.colsAdded:
                simData[col] = np.degrees(simData[col])
        return simData


class RandomRotDitherPerFilterChangeStacker(BaseDitherStacker):
    """
    Randomly dither the physical angle of the telescope rotator wrt the mount,
    after every filter change. Visits (in between filter changes) that cannot
    all be assigned an offset without surpassing the rotator limit are not
    dithered.

    Parameters
    ----------
    rotTelCol : str, optional
        The name of the column in the data specifying the physical angle
        of the telescope rotator wrt. the mount.
        Default: 'rotTelPos'.
    filterCol : str, optional
        The name of the filter column in the data.
        Default: 'filter'.
    degrees : boolean, optional
        True if angles in the database are in degrees (default).
        If True, returned dithered values are in degrees also.
        If False, angles assumed to be in radians and returned in radians.
    maxDither : float, optional
        Abs(maximum) rotational dither, in degrees. The dithers then will be
        between -maxDither to maxDither.
        Default: 90 degrees.
    maxRotAngle : float, optional
        Maximum rotator angle possible for the camera (degrees). Default 90 degrees.
    minRotAngle : float, optional
        Minimum rotator angle possible for the camera (degrees). Default -90 degrees.
    randomSeed: int, optional
        If set, then used as the random seed for the numpy random number
        generation for the dither offsets.
        Default: None.
    debug: bool, optinal
        If True, will print intermediate steps and plots histograms of
        rotTelPos for cases when no dither is applied.
        Default: False
    """
    # Values required for framework operation: this specifies the names of the new columns.
    colsAdded = ['randomDitherPerFilterChangeRotTelPos']

    def __init__(self, rotTelCol= 'rotTelPos', filterCol= 'filter', degrees=True,
                 maxDither=90., maxRotAngle=90, minRotAngle=-90, randomSeed=None,
                 debug=False):
        # Instantiate the RandomDither object and set internal variables.
        self.rotTelCol = rotTelCol
        self.filterCol = filterCol
        self.degrees = degrees
        self.maxDither = maxDither
        self.maxRotAngle = maxRotAngle
        self.minRotAngle = minRotAngle
        self.randomSeed = randomSeed
        # self.units used for plot labels
        if self.degrees:
            self.units = ['deg']
        else:
            self.units = ['rad']
            # Convert user-specified values into radians as well.
            self.maxDither = np.radians(self.maxDither)
            self.maxRotAngle = np.radians(self.maxRotAngle)
            self.minRotAngle = np.radians(self.minRotAngle)
        self.debug = debug

        # Values required for framework operation: specify the data columns required from the database.
        self.colsReq = [self.rotTelCol, self.filterCol]

    def _run(self, simData, cols_present=False):
        if self.debug: import matplotlib.pyplot as plt

        # Just go ahead and return if the columns were already in place.
        if cols_present:
            return simData

        # Generate random numbers for dither, using defined seed value if desired.
        # Note that we must define the random state for np.random, to ensure consistency in the build system.
        if not hasattr(self, '_rng'):
            if self.randomSeed is not None:
                self._rng = np.random.RandomState(self.randomSeed)
            else:
                self._rng = np.random.RandomState(544320)

        if len(np.where(simData[self.rotTelCol]>self.maxRotAngle)[0]) > 0:
            warnings.warn('Input data does not respect the specified maxRotAngle constraint: '
                          '(Re)Setting maxRotAngle to max value in the input data: %s'
                          % max(simData[self.rotTelCol]))
            self.maxRotAngle = max(simData[self.rotTelCol])
        if len(np.where(simData[self.rotTelCol]<self.minRotAngle)[0]) > 0:
            warnings.warn('Input data does not respect the specified minRotAngle constraint: '
                          '(Re)Setting minRotAngle to min value in the input data: %s'
                          % min(simData[self.rotTelCol]))
            self.minRotAngle = min(simData[self.rotTelCol])

        # Identify points where the filter changes.
        changeIdxs = np.where(simData[self.filterCol][1:] != simData[self.filterCol][:-1])[0]

        # Add the random offsets to the RotTelPos values.
        rotDither = self.colsAdded[0]

        if len(changeIdxs) == 0:
            # There are no filter changes, so nothing to dither. Just use original values.
            simData[rotDither] = simData[self.rotTelCol]
        else:
            # For each filter change, generate a series of random values for the offsets,
            # between +/- self.maxDither. These are potential values for the rotational offset.
            # The offset actually used will be  confined to ensure that rotTelPos for all visits in
            # that set of observations (between filter changes) fall within
            # the specified min/maxRotAngle -- without truncating the rotTelPos values.

            # Generate more offsets than needed - either 2x filter changes or 2500, whichever is bigger.
            # 2500 is an arbitrary number.
            maxNum = max(len(changeIdxs) * 2, 2500)

            rotOffset = np.zeros(len(simData), float)
            # Some sets of visits will not be assigned dithers: it was too hard to find an offset.
            n_problematic_ones = 0

            # Loop over the filter change indexes (current filter change, next filter change) to identify
            # sets of visits that should have the same offset.
            for (c, cn) in zip(changeIdxs, changeIdxs[1:]):
                randomOffsets = self._rng.rand(maxNum + 1) * 2.0 * self.maxDither - self.maxDither
                i = 0
                potential_offset = randomOffsets[i]
                # Calculate new rotTelPos values, if we used this offset.
                new_rotTel = simData[self.rotTelCol][c+1:cn+1] + potential_offset
                # Does it work? Do all values fall within minRotAngle / maxRotAngle?
                goodToGo = (new_rotTel >= self.minRotAngle).all() and (new_rotTel <= self.maxRotAngle).all()
                while ((not goodToGo) and (i < maxNum)):
                    # break if find a good offset or hit maxNum tries.
                    i += 1
                    potential_offset = randomOffsets[i]
                    new_rotTel = simData[self.rotTelCol][c+1:cn+1] + potential_offset
                    goodToGo = (new_rotTel >= self.minRotAngle).all() and \
                               (new_rotTel <= self.maxRotAngle).all()

                if not goodToGo:  # i.e. no good offset was found after maxNum tries
                    n_problematic_ones += 1
                    rotOffset[c+1:cn+1] = 0. # no dither
                else:
                    rotOffset[c+1:cn+1] = randomOffsets[i]  # assign the chosen offset

            # Handle the last set of observations (after the last filter change to the end of the survey).
            randomOffsets = self._rng.rand(maxNum + 1) * 2.0 * self.maxDither - self.maxDither
            i = 0
            potential_offset = randomOffsets[i]
            new_rotTel = simData[self.rotTelCol][changeIdxs[-1]+1:] + potential_offset
            goodToGo = (new_rotTel >= self.minRotAngle).all() and (new_rotTel <= self.maxRotAngle).all()
            while ((not goodToGo) and (i < maxNum)):
                # break if find a good offset or cant (after maxNum tries)
                i += 1
                potential_offset = randomOffsets[i]
                new_rotTel = simData[self.rotTelCol][changeIdxs[-1]+1:] + potential_offset
                goodToGo = (new_rotTel >= self.minRotAngle).all() and \
                           (new_rotTel <= self.maxRotAngle).all()

            if not goodToGo:  # i.e. no good offset was found after maxNum tries
                n_problematic_ones += 1
                rotOffset[c+1:cn+1] = 0.
            else:
                rotOffset[changeIdxs[-1]+1:] = potential_offset

        # Assign the dithers
        simData[rotDither] = simData[self.rotTelCol] + rotOffset

        # Final check to make sure things are okay
        goodToGo = (simData[rotDither] >= self.minRotAngle).all() and \
                   (simData[rotDither] <= self.maxRotAngle).all()
        if not goodToGo:
            message = 'Rotational offsets are not working properly:\n'
            message += ' dithered rotTelPos: %s\n' % (simData[rotDither])
            message += ' minRotAngle: %s ; maxRotAngle: %s' % (self.minRotAngle, self.maxRotAngle)
            raise ValueError(message)
        else:
            return simData
