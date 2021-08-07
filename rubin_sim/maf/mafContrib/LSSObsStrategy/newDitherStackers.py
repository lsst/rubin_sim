##############################################################################################################
# Purpose: implement new dithering strategies.

# The stackers here follow the naming scheme:  [Pattern]Dither[Field]Per[Timescale]. The absence of the
# keyword 'Field' implies dither assignment to all fields.

# Dithers are restricted to the hexagon inscribed in the circle with radius maxDither, where maxDither is the
# required dither offset (generally taken to be radius of the FOV).

# Humna Awan: humna.awan@rutgers.edu
# Last updated: 06/11/16
###############################################################################################################
import numpy as np
from rubin_sim.maf.stackers import (wrapRADec, polygonCoords)
from rubin_sim.maf.stackers import (BaseStacker, SpiralDitherFieldPerVisitStacker)
from rubin_sim.utils import calcSeason

__all__ = ['RepulsiveRandomDitherFieldPerVisitStacker',
           'RepulsiveRandomDitherFieldPerNightStacker',
           'RepulsiveRandomDitherPerNightStacker',
           'FermatSpiralDitherFieldPerVisitStacker',
           'FermatSpiralDitherFieldPerNightStacker',
           'FermatSpiralDitherPerNightStacker',
           'PentagonDiamondDitherFieldPerSeasonStacker',
           'PentagonDitherPerSeasonStacker',
           'PentagonDiamondDitherPerSeasonStacker',
           'SpiralDitherPerSeasonStacker']


class RepulsiveRandomDitherFieldPerVisitStacker(BaseStacker):
    """
    Repulsive-randomly dither the RA and Dec pointings up to maxDither degrees from center, 
    different offset per visit for each field.

    Note: dithers are confined to the hexagon inscribed in the circle with radius maxDither.

    Parameters
    -------------------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    fieldIdCol : str
        name of the fieldID column in the data. Default: 'fieldIdCol'.
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    randomSeed: int
        random seed for the numpy random number generation for the dither offsets.
        Default: None.
    printInfo: `bool`
        set to True to print out information about the number of squares considered,
        number of points chosen, and the filling factor. Default: False
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec',
                 fieldIdCol='fieldID', maxDither=1.75, randomSeed=None, printInfo= False):
        # Instantiate the RandomDither object and set internal variables.
        self.raCol = raCol
        self.decCol = decCol
        self.fieldIdCol= fieldIdCol
        # Convert maxDither from degrees (internal units for ra/dec are radians)
        self.maxDither = np.radians(maxDither)
        self.randomSeed = randomSeed
        self.printInfo= printInfo
        # self.units used for plot labels
        self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['repulsiveRandomDitherFieldPerVisitRa', 'repulsiveRandomDitherFieldPerVisitDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.fieldIdCol]

    def _generateRepRandomOffsets(self, noffsets, numTiles):
        # Goal: Tile the circumscribing square with squares. Discard those that fall outside the hexagon.
        # Then choose a square repulsive-randomly (i.e. choose without replacement), and choose a random
        # point from the chosen square.
        noffsets= int(noffsets)
        numTiles= int(numTiles)
        
        squareSide= self.maxDither*2   # circumscribing square. center at (0,0)
        tileSide= squareSide/ np.sqrt(numTiles)

        xCenter= np.zeros(numTiles)   # x-coords of the tiles' center
        yCenter= np.zeros(numTiles)   # y-coords of the tiles' center

        # fill in x-coordinates
        k= 0
        xCenter[k]= -tileSide*((np.sqrt(numTiles)/2.0)-0.5)   # far left x-coord

        tempXarr= []
        tempXarr.append(xCenter[k])
        while (k < (np.sqrt(numTiles)-1)):
            # fill xCoords for squares right above the x-axis
            k+=1
            xCenter[k]= xCenter[k-1]+tileSide
            tempXarr.append(xCenter[k])

        # fill in the rest of the xCenter array
        indices= np.arange(k+1,len(xCenter))
        indices= indices % len(tempXarr)
        tempXarr=np.array(tempXarr)
        xCenter[k+1:numTiles]= tempXarr[indices]

        # fill in the y-coords
        i=0
        temp= np.empty(len(tempXarr))
        while (i<numTiles):
            # the highest y-center coord above the x-axis
            if (i==0):
                temp.fill(tileSide*((np.sqrt(numTiles)/2.0)-0.5))
            # y-centers below the top one
            else:
                temp.fill(yCenter[i-1]-tileSide)
 
            yCenter[i:i+len(temp)]=  temp
            i+=len(temp)

        # set up the hexagon
        b= np.sqrt(3.0)*self.maxDither
        m= np.sqrt(3.0)
        h= self.maxDither*np.sqrt(3.0)/2.0
        
        # find the points that are inside hexagon
        insideHex= np.where((yCenter < m*xCenter+b) &
                            (yCenter > m*xCenter-b) &
                            (yCenter < -m*xCenter+b) &
                            (yCenter > -m*xCenter-b) &
                            (yCenter < h) &
                            (yCenter > -h))[0]  

        numPointsInsideHex= len(insideHex)
        if self.printInfo:
            print('NumPointsInsideHexagon: ', numPointsInsideHex)
            print('Total squares chosen: ', len(xCenter))
            print('Filling factor for repRandom (Number of points needed/Number of points in hexagon): ', float(noffsets)/numPointsInsideHex)

        # keep only the points that are inside the hexagon
        tempX= xCenter.copy()
        tempY= yCenter.copy()
        xCenter= list(tempX[insideHex])
        yCenter= list(tempY[insideHex])
        xCenter_copy=list(np.array(xCenter).copy())   # in case need to reuse the squares
        yCenter_copy=list(np.array(yCenter).copy())    # in case need to reuse the squares
        
        # initiate the offsets' array
        xOff= np.zeros(noffsets)
        yOff= np.zeros(noffsets)
        # randomly select a point from the insideHex points. assign a random offset from within that square and 
        # then delete it from insideHex array
        for q in range(0,noffsets):
            randNum= np.random.rand()
            randIndexForSquares= int(np.floor(randNum*numPointsInsideHex))
            
            if (randIndexForSquares > len(xCenter)):
                while (randIndexForSquares > len(xCenter)):
                    randNum= np.random.rand()
                    randIndexForSquares= int(np.floor(randNum*numPointsInsideHex))
            randNums= np.random.rand(2)
            randXOffset= (randNums[0]-0.5)*(tileSide/2.0)   # subtract 0.5 to get +/- delta
            randYOffset= (randNums[1]-0.5)*(tileSide/2.0)

            newX= xCenter[randIndexForSquares]+randXOffset
            newY= yCenter[randIndexForSquares]+randYOffset
            
            # make sure the offset is within the hexagon
            goodCondition= ((newY <= m*newX+b) & (newY >= m*newX-b) & (newY <= -m*newX+b) & (newY >= -m*newX-b) & (newY <= h) & (newY >= -h))
            if not(goodCondition):
                while not(goodCondition):
                    randNums= np.random.rand(2)
                    randXOffset= (randNums[0]-0.5)*(tileSide/2.0)   # subtract 0.5 to get +/- delta
                    randYOffset= (randNums[1]-0.5)*(tileSide/2.0)

                    newX= xCenter[randIndexForSquares]+randXOffset
                    newY= yCenter[randIndexForSquares]+randYOffset
                    
                    goodCondition= ((newY <= m*newX+b) & (newY >= m*newX-b) & (newY <= -m*newX+b) & (newY >= -m*newX-b) & (newY <= h) & (newY >= -h))
                    
            xOff[q]= xCenter[randIndexForSquares]+randXOffset;
            yOff[q]= yCenter[randIndexForSquares]+randYOffset;

            if (len(xCenter)==0):
                # have used all the squares ones
                print('Starting reuse of the squares inside the hexagon')
                xCenter= xCenter_copy.copy()
                yCenter= yCenter_copy.copy()
            xCenter.pop(randIndexForSquares)
            yCenter.pop(randIndexForSquares)
            numPointsInsideHex-=1

        self.xOff = xOff
        self.yOff = yOff

    def _run(self, simData):
        # Generate random numbers for dither, using defined seed value if desired.
        if self.randomSeed is not None:
            np.random.seed(self.randomSeed)

        # analysis is simplified if deal with each field separately.
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            noffsets= len(match)
            numTiles= np.ceil(np.sqrt(noffsets)*1.5)**2     # number of tiles must be a perfect square.
                                                            # arbitarily chosen factor of 1.5 to have more than necessary tiles inside hexagon.
            self._generateRepRandomOffsets(noffsets, numTiles)
            # Add to RA and dec values.
            simData['repulsiveRandomDitherFieldPerVisitRa'][match] = simData[self.raCol][match] + self.xOff/np.cos(simData[self.decCol][match])
            simData['repulsiveRandomDitherFieldPerVisitDec'][match] = simData[self.decCol][match] + self.yOff
            
        # Wrap back into expected range.
        simData['repulsiveRandomDitherFieldPerVisitRa'], simData['repulsiveRandomDitherFieldPerVisitDec'] = \
                            wrapRADec(simData['repulsiveRandomDitherFieldPerVisitRa'], simData['repulsiveRandomDitherFieldPerVisitDec'])
        return simData


class RepulsiveRandomDitherFieldPerNightStacker(RepulsiveRandomDitherFieldPerVisitStacker):
    """
    Repulsive-randomly dither the RA and Dec pointings up to maxDither degrees from center, one dither offset 
    per new night of observation of a field.

    Note: dithers are confined to the hexagon inscribed in the circle with with radius maxDither

    Parameters
    -------------------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    fieldIdCol : str
        name of the fieldID column in the data. Default: 'fieldIdCol'.
    nightCol : str
        name of the night column in the data. Default: 'night'.
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    randomSeed: int
        random seed for the numpy random number generation for the dither offsets.
        Default: None.
    printInfo: `bool`
        set to True to print out information about the number of squares considered,
        number of points chosen, and the filling factor. Default: False
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', fieldIdCol='fieldID', nightCol='night',
                 maxDither=1.75, randomSeed=None, printInfo= False):
        # Instantiate the RandomDither object and set internal variables.
        super(RepulsiveRandomDitherFieldPerNightStacker, self).__init__(raCol=raCol, decCol=decCol,
                                                                        fieldIdCol= fieldIdCol,
                                                                        maxDither=maxDither,
                                                                        randomSeed=randomSeed,
                                                                        printInfo= printInfo)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['repulsiveRandomDitherFieldPerNightRa', 'repulsiveRandomDitherFieldPerNightDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)
    
    def _run(self, simData):
        # Generate random numbers for dither, using defined seed value if desired.
        if self.randomSeed is not None:
            np.random.seed(self.randomSeed)
            
        for fieldid in np.unique(simData[self.fieldIdCol]):
            # Identify observations of this field.
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]

            noffsets= len(match)
            numTiles= np.ceil(np.sqrt(noffsets)*1.5)**2
            self._generateRepRandomOffsets(noffsets,numTiles)
            
            # Apply dithers, increasing each night.
            vertexIdxs = np.arange(0, len(match), 1)
            nights = simData[self.nightCol][match]
            vertexIdxs = np.searchsorted(np.unique(nights), nights)
            vertexIdxs = vertexIdxs % len(self.xOff)            

            simData['repulsiveRandomDitherFieldPerNightRa'][match] = simData[self.raCol][match] \
                                                                     + self.xOff[vertexIdxs]\
                                                                     /np.cos(simData[self.decCol][match])
            simData['repulsiveRandomDitherFieldPerNightDec'][match] = simData[self.decCol][match] \
                                                                      + self.yOff[vertexIdxs]
        # Wrap into expected range.
        simData['repulsiveRandomDitherFieldPerNightRa'], simData['repulsiveRandomDitherFieldPerNightDec'] = \
                                wrapRADec(simData['repulsiveRandomDitherFieldPerNightRa'],
                                          simData['repulsiveRandomDitherFieldPerNightDec'])
        return simData


class RepulsiveRandomDitherPerNightStacker(RepulsiveRandomDitherFieldPerVisitStacker):
    """
    Repulsive-randomly dither the RA and Dec pointings up to maxDither degrees from center, one dither offset 
    per night for all the fields.

    Note: dithers are confined to the hexagon inscribed in the circle with with radius maxDither

    Parameters
    -------------------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    nightCol : str
        name of the night column in the data. Default: 'night'.
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    randomSeed: int
        random seed for the numpy random number generation for the dither offsets.
        Default: None.
    printInfo: `bool`
        set to True to print out information about the number of squares considered,
        number of points chosen, and the filling factor. Default: False
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', nightCol='night',
                 maxDither=1.75, randomSeed=None, printInfo= False):
        # Instantiate the RepulsiveRandomDitherFieldPerVisitStacker object and set internal variables.
        super(RepulsiveRandomDitherPerNightStacker, self).__init__(raCol=raCol, decCol=decCol,
                                                                   maxDither=maxDither,
                                                                   randomSeed=randomSeed,
                                                                   printInfo= printInfo)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['repulsiveRandomDitherPerNightRa', 'repulsiveRandomDitherPerNightDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)

    def _run(self, simData):
        # Generate random numbers for dither, using defined seed value if desired.
        if self.randomSeed is not None:
            np.random.seed(self.randomSeed)

        # Generate the random dither values, one per night.
        nights = np.unique(simData[self.nightCol])
        numNights= len(nights)
        numTiles= np.ceil(np.sqrt(numNights)*1.5)**2
        self._generateRepRandomOffsets(numNights, numTiles)
            
        # Add to RA and dec values.
        for n, x, y in zip(nights, self.xOff, self.yOff):
            match = np.where(simData[self.nightCol] == n)[0]
            simData['repulsiveRandomDitherPerNightRa'][match] = simData[self.raCol][match] \
                                                                + x/np.cos(simData[self.decCol][match])
            simData['repulsiveRandomDitherPerNightDec'][match] = simData[self.decCol][match] + y
        # Wrap RA/Dec into expected range.
        simData['repulsiveRandomDitherPerNightRa'], simData['repulsiveRandomDitherPerNightDec'] = \
                wrapRADec(simData['repulsiveRandomDitherPerNightRa'],
                          simData['repulsiveRandomDitherPerNightDec'])
        return simData


class FermatSpiralDitherFieldPerVisitStacker(BaseStacker):
    """
    Offset along a Fermat's spiral with numPoints, out to a maximum radius of maxDither.
    Sequential offset for each visit to a field.

    Note: dithers are confined to the hexagon inscribed in the circle with with radius maxDither
    Note: Fermat's spiral is defined by r= c*sqrt(n), theta= n*angle, n= integer

    Parameters
    -------------------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    fieldIdCol : str
        name of the fieldID column in the data. Default: 'fieldID'.
    numPoints: int
        number of points in the spiral. Default: 60
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    goldAngle: float
        angle in degrees defining the spiral: theta= multiple of goldAngle
        Default: 137.508

    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', fieldIdCol='fieldID',
                 numPoints=60, maxDither=1.75, goldAngle=137.508):
        self.raCol = raCol
        self.decCol = decCol
        self.fieldIdCol = fieldIdCol
        # Convert maxDither from degrees (internal units for ra/dec are radians)
        self.numPoints = numPoints
        self.goldAngle = goldAngle
        self.maxDither = np.radians(maxDither)
        # self.units used for plot labels
        self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['fermatSpiralDitherFieldPerVisitRa', 'fermatSpiralDitherFieldPerVisitDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.fieldIdCol]

    def _generateFermatSpiralOffsets(self):
        # Fermat's spiral: r= c*sqrt(n), theta= n*angle
        # Golden spiral: r= c*sqrt(n), theta= n*137.508degrees
        n= np.arange(0, self.numPoints)
        theta= np.radians(n*self.goldAngle)
        rmax= np.sqrt(theta.max()/np.radians(self.goldAngle))
        scalingFactor= 0.8*self.maxDither/rmax
        r = scalingFactor*np.sqrt(n)
        
        self.xOff = r * np.cos(theta)
        self.yOff = r * np.sin(theta)

    def _run(self, simData):            
        # Generate the spiral offset vertices.
        self._generateFermatSpiralOffsets()
        # Now apply to observations.
        for fieldid in np.unique(simData[self.fieldIdCol]):
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply sequential dithers, increasing with each visit.
            vertexIdxs = np.arange(0, len(match), 1)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['fermatSpiralDitherFieldPerVisitRa'][match] = simData[self.raCol][match] + \
                                                                self.xOff[vertexIdxs]\
                                                                  /np.cos(simData[self.decCol][match])
            simData['fermatSpiralDitherFieldPerVisitDec'][match] = simData[self.decCol][match] \
                                                                   + self.yOff[vertexIdxs]
        # Wrap into expected range.
        simData['fermatSpiralDitherFieldPerVisitRa'], simData['fermatSpiralDitherFieldPerVisitDec'] = \
                                        wrapRADec(simData['fermatSpiralDitherFieldPerVisitRa'],
                                                  simData['fermatSpiralDitherFieldPerVisitDec'])
        return simData


class FermatSpiralDitherFieldPerNightStacker(FermatSpiralDitherFieldPerVisitStacker):
    """
    Offset along a Fermat's spiral with numPoints, out to a maximum radius of maxDither.
    one dither offset  per new night of observation of a field.

    Note: dithers are confined to the hexagon inscribed in the circle with with radius maxDither
    Note: Fermat's spiral is defined by r= c*sqrt(n), theta= n*angle, n= integer

    Parameters
    -----------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    fieldIdCol : str
        name of the fieldID column in the data. Default: 'fieldID'.
    nightCol : str
        name of the night column in the data. Default: 'night'.
    numPoints: int
        number of points in the spiral. Default: 60
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    goldAngle: float
        angle in degrees defining the spiral: theta= multiple of goldAngle. Default: 137.508
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', fieldIdCol='fieldID',nightCol='night',
                 numPoints=60, maxDither=1.75, goldAngle=137.508):
        super(FermatSpiralDitherFieldPerNightStacker,self).__init__(raCol= raCol, decCol=decCol,
                                                                    fieldIdCol= fieldIdCol,
                                                                    numPoints= numPoints,
                                                                    maxDither=maxDither,
                                                                    goldAngle=goldAngle)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['fermatSpiralDitherFieldPerNightRa', 'fermatSpiralDitherFieldPerNightDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)

    def _run(self, simData):            
        # Generate the spiral offset vertices.
        self._generateFermatSpiralOffsets()
        # Now apply to observations.
        for fieldid in np.unique(simData[self.fieldIdCol]):
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            # Apply sequential dithers, increasing with each visit.
            vertexIdxs = np.arange(0, len(match), 1)
            nights = simData[self.nightCol][match]
            vertexIdxs = np.searchsorted(np.unique(nights), nights)
            vertexIdxs = vertexIdxs % self.numPoints
            simData['fermatSpiralDitherFieldPerNightRa'][match] = simData[self.raCol][match] + \
                                                                self.xOff[vertexIdxs]/np.cos(simData[self.decCol][match])
            simData['fermatSpiralDitherFieldPerNightDec'][match] = simData[self.decCol][match] + self.yOff[vertexIdxs]
        # Wrap into expected range.
        simData['fermatSpiralDitherFieldPerNightRa'], simData['fermatSpiralDitherFieldPerNightDec'] = \
                                        wrapRADec(simData['fermatSpiralDitherFieldPerNightRa'], simData['fermatSpiralDitherFieldPerNightDec'])
        return simData


class FermatSpiralDitherPerNightStacker(FermatSpiralDitherFieldPerVisitStacker):
    """
    Offset along a Fermat's spiral with numPoints, out to a maximum radius of maxDither.
    Sequential offset per night for all fields.

    Note: dithers are confined to the hexagon inscribed in the circle with with radius maxDither
    Note: Fermat's spiral is defined by r= c*sqrt(n), theta= n*angle, n= integer

    Parameters
    ----------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    fieldIdCol : str
        name of the fieldID column in the data. Default: 'fieldID'.
    nightCol : str
        name of the night column in the data. Default: 'night'.
    numPoints: int
        number of points in the spiral. Default: 60
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    goldAngle: float
        angle in degrees defining the spiral: theta= multiple of goldAngle
        Default: 137.508
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', fieldIdCol='fieldID',nightCol='night',
                 numPoints=60, maxDither=1.75, goldAngle=137.508):
        super(FermatSpiralDitherPerNightStacker, self).__init__(raCol= raCol, decCol=decCol,
                                                                fieldIdCol= fieldIdCol,
                                                                numPoints= numPoints, maxDither=maxDither,
                                                                goldAngle=goldAngle)
        self.nightCol = nightCol
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['fermatSpiralDitherPerNightRa', 'fermatSpiralDitherPerNightDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq.append(self.nightCol)

    def _run(self, simData):            
        # Generate the spiral offset vertices.
        self._generateFermatSpiralOffsets()

        vertexID= 0
        nights = np.unique(simData[self.nightCol])
        for n in nights:
            match = np.where(simData[self.nightCol] == n)[0]
            vertexID= vertexID % self.numPoints

            simData['fermatSpiralDitherPerNightRa'][match] = simData[self.raCol][match] + \
                                                                self.xOff[vertexID]\
                                                             /np.cos(simData[self.decCol][match])
            simData['fermatSpiralDitherPerNightDec'][match] = simData[self.decCol][match] \
                                                              + self.yOff[vertexID]
            vertexID += 1
            
        # Wrap into expected range.
        simData['fermatSpiralDitherPerNightRa'], simData['fermatSpiralDitherPerNightDec'] = \
                                        wrapRADec(simData['fermatSpiralDitherPerNightRa'],
                                                  simData['fermatSpiralDitherPerNightDec'])
        return simData


class PentagonDitherFieldPerSeasonStacker(BaseStacker):
    """
    Offset along two pentagons, one inverted and inside the other.
    Sequential offset for each field on a visit in new season.

    Parameters
    -----------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    fieldIdCol : str
        name of the fieldID column in the data. Default: 'fieldID'.
    expMJDCol : str
        name of the date/time stamp column in the data. Default: 'expMJD'.
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    wrapLastSeason: `bool`
        set to False to all consider 11 seasons independently.
        set to True to wrap 0th and 10th season, leading to a total of 10 seasons.
        Default: True
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec',fieldIdCol='fieldID',
                 expMJDCol= 'expMJD', maxDither= 1.75, wrapLastSeason= True):
        self.raCol = raCol
        self.decCol = decCol
        self.fieldIdCol = fieldIdCol
        self.expMJDCol= expMJDCol
        # Convert maxDither from degrees (internal units for ra/dec are radians)
        self.maxDither = np.radians(maxDither)
        self.wrapLastSeason= wrapLastSeason
        # self.units used for plot labels
        self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['pentagonDitherFieldPerSeasonRa', 'pentagonDitherFieldPerSeasonDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.fieldIdCol, self.expMJDCol]
            
    def _generatePentagonOffsets(self):
        # inner pentagon tuples
        nside= 5
        inner= polygonCoords(nside, self.maxDither/2.5, 0.0)
        # outer pentagon tuples
        outerTemp= polygonCoords(nside, self.maxDither/1.3, np.pi)
        # reorder outer tuples' order
        outer= []
        outer[0:3]= outerTemp[2:5]
        outer[4:6]= outerTemp[0:2]
        # join inner and outer coordiantes' array
        self.xOff= np.concatenate((zip(*inner)[0],zip(*outer)[0]), axis=0)
        self.yOff= np.concatenate((zip(*inner)[1],zip(*outer)[1]), axis=0)

    def _run(self, simData):
        # find the seasons associated with each visit.
        seasons = calcSeason(simData[self.raCol], simdata[self.expMJDCol])
        # check how many entries in the >10 season
        ind= np.where(seasons > 9)[0]
        # should be only 1 extra seasons ..
        if len(np.unique(seasons[ind])) > 1:
            raise ValueError('Too many seasons (more than 11). Check SeasonStacker.')

        if self.wrapLastSeason:
            print('Seasons to wrap ', np.unique(seasons[ind]))
            # wrap the season around: 10th == 0th
            seasons[ind]= seasons[ind]%10

        # Generate the spiral offset vertices.
        self._generatePentagonOffsets()
        
        # Now apply to observations.
        for fieldid in np.unique(simData[self.fieldIdCol]):
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            seasonsVisited = seasons[match]
            # Apply sequential dithers, increasing with each season.
            vertexIdxs = np.searchsorted(np.unique(seasonsVisited), seasonsVisited)
            vertexIdxs = vertexIdxs % len(self.xOff)
            simData['pentagonDitherFieldPerSeasonRa'][match] = simData[self.raCol][match] + \
              self.xOff[vertexIdxs]/np.cos(simData[self.decCol][match])
            simData['pentagonDitherFieldPerSeasonDec'][match] = simData[self.decCol][match] \
                                                                + self.yOff[vertexIdxs]
        # Wrap into expected range.
        simData['pentagonDitherFieldPerSeasonRa'], simData['pentagonDitherFieldPerSeasonDec'] = \
                                        wrapRADec(simData['pentagonDitherFieldPerSeasonRa'],
                                                  simData['pentagonDitherFieldPerSeasonDec'])
        return simData


class PentagonDiamondDitherFieldPerSeasonStacker(BaseStacker):
    """
    Offset along a diamond circumscribed by a pentagon.
    Sequential offset for each field on a visit in new season.

    Parameters
    -------------------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    fieldIdCol : str
        name of the fieldID column in the data. Default: 'fieldID'.
    expMJDCol : str
        name of the date/time stamp column in the data. Default: 'expMJD'.
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    wrapLastSeason: `bool`
        set to False to all consider 11 seasons independently.
        set to True to wrap 0th and 10th season, leading to a total of 10 seasons.
        Default: True
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec',fieldIdCol='fieldID',
                 expMJDCol= 'expMJD', maxDither= 1.75, wrapLastSeason= True):
        self.raCol = raCol
        self.decCol = decCol
        self.fieldIdCol = fieldIdCol
        self.expMJDCol= expMJDCol
        # Convert maxDither from degrees (internal units for ra/dec are radians)
        self.maxDither = np.radians(maxDither)
        self.wrapLastSeason= wrapLastSeason
        # self.units used for plot labels
        self.units = ['rad', 'rad']
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['pentagonDiamondDitherFieldPerSeasonRa', 'pentagonDiamondDitherFieldPerSeasonDec']
        # Values required for framework operation: this specifies the data columns required from the database.
        self.colsReq = [self.raCol, self.decCol, self.fieldIdCol, self.expMJDCol]
           
    def _generateOffsets(self):
        # outer pentagon tuples
        pentCoord= polygonCoords(5,self.maxDither/1.3, 0)
        # inner diamond tuples
        diamondCoord= polygonCoords(4, self.maxDither/2.5, np.pi/2)

        # join inner and outer coordiantes' array + a point in the middle (origin)
        self.xOff= np.concatenate(([0],zip(*diamondCoord)[0],zip(*pentCoord)[0]), axis=0)
        self.yOff= np.concatenate(([0],zip(*diamondCoord)[1],zip(*pentCoord)[1]), axis=0)

    def _run(self, simData):
        # find the seasons associated with each visit.
        seasons = calcSeason(simData[self.raCol], simData[self.expMJDCol])

        # check how many entries in the >10 season
        ind= np.where(seasons > 9)[0]
        # should be only 1 extra seasons ..
        if len(np.unique(seasons[ind])) > 1:
            raise ValueError('Too many seasons (more than 11). Check SeasonStacker.')

        if self.wrapLastSeason:
            print('Seasons to wrap ', np.unique(seasons[ind]))
            # wrap the season around: 10th == 0th
            seasons[ind]= seasons[ind]%10
            
        # Generate the spiral offset vertices.
        self._generateOffsets()
        
        # Now apply to observations.
        for fieldid in np.unique(simData[self.fieldIdCol]):
            match = np.where(simData[self.fieldIdCol] == fieldid)[0]
            seasonsVisited = seasons[match]    
            # Apply sequential dithers, increasing with each season.
            vertexIdxs = np.searchsorted(np.unique(seasonsVisited), seasonsVisited)
            vertexIdxs = vertexIdxs % len(self.xOff)
            simData['pentagonDiamondDitherFieldPerSeasonRa'][match] = simData[self.raCol][match] + \
              self.xOff[vertexIdxs]/np.cos(simData[self.decCol][match])
            simData['pentagonDiamondDitherFieldPerSeasonDec'][match] = simData[self.decCol][match] \
                                                                       + self.yOff[vertexIdxs]
        # Wrap into expected range.
        simData['pentagonDiamondDitherFieldPerSeasonRa'], simData['pentagonDiamondDitherFieldPerSeasonDec'] = \
                                                          wrapRADec(simData['pentagonDiamondDitherFieldPerSeasonRa'],
                                                                    simData['pentagonDiamondDitherFieldPerSeasonDec'])
        return simData

class PentagonDitherPerSeasonStacker(PentagonDitherFieldPerSeasonStacker):
    """
    Offset along two pentagons, one inverted and inside the other.
    Sequential offset for all fields every season.

    Parameters
    -------------------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    fieldIdCol : str
        name of the fieldID column in the data. Default: 'fieldID'.
    expMJDCol : str
        name of the date/time stamp column in the data. Default: 'expMJD'.
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    wrapLastSeason: `bool`
        set to False to all consider 11 seasons independently.
        set to True to wrap 0th and 10th season, leading to a total of 10 seasons.
        Default: True
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec',
                 fieldIdCol='fieldID', expMJDCol= 'expMJD', nightCol='night',
                 maxDither= 1.75, wrapLastSeason= True):
        super(PentagonDitherPerSeasonStacker, self).__init__(raCol=raCol, decCol=decCol,
                                                             fieldIdCol=fieldIdCol, expMJDCol=expMJDCol,
                                                             maxDither=maxDither,
                                                             wrapLastSeason= wrapLastSeason)
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['pentagonDitherPerSeasonRa', 'pentagonDitherPerSeasonDec']
   
    def _run(self, simData):      
        # find the seasons associated with each visit.
        seasons = calcSeason(simData[self.raCol], simData[self.expMJDCol])
        years = simData[self.nightCol] % 365.25

         # check how many entries in the >10 season
        ind= np.where(seasons > 9)[0]
        # should be only 1 extra seasons ..
        if len(np.unique(seasons[ind])) > 1:
            raise ValueError('Too many seasons (more than 11). Check SeasonStacker.')

        if self.wrapLastSeason:
            # check how many entries in the >10 season
            print('Seasons to wrap ', np.unique(seasons[ind]), 'with total entries: ', len(seasons[ind]))
            seasons[ind]= seasons[ind]%10

        # Generate the spiral offset vertices.
        self._generatePentagonOffsets()
        # print details
        print('Total visits for all fields:', len(seasons))
        print('')
       
        # Add to RA and dec values.
        vertexID= 0
        for s in np.unique(seasons):
            match = np.where(seasons == s)[0]
            # print details
            print('season', s)
            print('numEntries ', len(match), '; ', float(len(match))/len(seasons)*100, '% of total')
            matchYears= np.unique(years[match])
            print('Corresponding years: ', matchYears)
            for i in matchYears: print('     Entries in year', i, ': ', len(np.where(i == years[match])[0]))
            print('')
            vertexID= vertexID %  len(self.xOff)
            simData['pentagonDitherPerSeasonRa'][match] = simData[self.raCol][match] + self.xOff[vertexID]\
                                                          /np.cos(simData[self.decCol][match])
            simData['pentagonDitherPerSeasonDec'][match] = simData[self.decCol][match] + self.yOff[vertexID]
            vertexID += 1

        # Wrap into expected range.
        simData['pentagonDitherPerSeasonRa'], simData['pentagonDitherPerSeasonDec'] = \
                                        wrapRADec(simData['pentagonDitherPerSeasonRa'],
                                                  simData['pentagonDitherPerSeasonDec'])
        return simData

class PentagonDiamondDitherPerSeasonStacker(PentagonDiamondDitherFieldPerSeasonStacker):
    """
    Offset along a diamond circumscribed by a pentagon.
    Sequential offset for all fields every season.

    Parameters
    -------------------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    fieldIdCol : str
        name of the fieldID column in the data. Default: 'fieldID'.
    expMJDCol : str
        name of the date/time stamp column in the data. Default: 'expMJD'.
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    wrapLastSeason: `bool`
        set to False to all consider 11 seasons independently.
        set to True to wrap 0th and 10th season, leading to a total of 10 seasons.
        Default: True
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec',
                 fieldIdCol='fieldID', expMJDCol= 'expMJD', maxDither= 1.75, wrapLastSeason= True):
        super(PentagonDiamondDitherPerSeasonStacker, self).__init__(raCol=raCol, decCol=decCol,
                                                                    fieldIdCol=fieldIdCol,
                                                                    expMJDCol=expMJDCol,
                                                                    maxDither=maxDither,
                                                                    wrapLastSeason= wrapLastSeason)
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['pentagonDiamondDitherPerSeasonRa', 'pentagonDiamondDitherPerSeasonDec']
   
    def _run(self, simData):           
        # find the seasons associated with each visit.
        seasons = calcSeason(simData[self.raCol], simData[self.expMJDCol])

        # check how many entries in the >10 season
        ind= np.where(seasons > 9)[0]
        # should be only 1 extra seasons ..
        if len(np.unique(seasons[ind])) > 1:
            raise ValueError('Too many seasons (more than 11). Check SeasonStacker.')

        if self.wrapLastSeason:
            print('Seasons to wrap ', np.unique(seasons[ind]))
            # wrap the season around: 10th == 0th
            seasons[ind]= seasons[ind]%10
        
        # Generate the spiral offset vertices.
        self._generateOffsets()

        uniqSeasons = np.unique(seasons)
        # Add to RA and dec values.
        vertexID= 0
        for s in uniqSeasons:
            match = np.where(seasons == s)[0]
            vertexID= vertexID %  len(self.xOff)
            simData['pentagonDiamondDitherPerSeasonRa'][match] = simData[self.raCol][match] \
                                                                 + self.xOff[vertexID]/np.cos(simData[self.decCol][match])
            simData['pentagonDiamondDitherPerSeasonDec'][match] = simData[self.decCol][match] \
                                                                  + self.yOff[vertexID]
            vertexID += 1

        # Wrap into expected range.
        simData['pentagonDiamondDitherPerSeasonRa'], simData['pentagonDiamondDitherPerSeasonDec'] = \
                                        wrapRADec(simData['pentagonDiamondDitherPerSeasonRa'],
                                                  simData['pentagonDiamondDitherPerSeasonDec'])
        return simData

class SpiralDitherPerSeasonStacker(SpiralDitherFieldPerVisitStacker):
    """
    Offsets along a 10pt spiral. Sequential offset for all fields every seaso along a 10pt spiral.
    
    Parameters
    -------------------
    raCol: str
        name of the RA column in the data. Default: 'fieldRA'.
    decCol : str
        name of the Dec column in the data. Default: 'fieldDec'.
    fieldIdCol : str
        name of the fieldID column in the data. Default: 'fieldID'.
    expMJDCol : str
        name of the date/time stamp column in the data. Default: 'expMJD'.
    maxDither: float
        radius of the maximum dither offset, in degrees. Default: 1.75
    wrapLastSeason: `bool`
        set to False to all consider 11 seasons independently.
        set to True to wrap 0th and 10th season, leading to a total of 10 seasons.
        Default: True
    numPoints: int:  number of points in the spiral. Default: 10
    nCoils: int:  number of coils the spiral. Default: 3
    """
    def __init__(self, raCol='fieldRA', decCol='fieldDec', fieldIdCol='fieldID', expMJDCol= 'expMJD',
                 maxDither= 1.75, wrapLastSeason= True, numPoints=10, nCoils=3):
        super(SpiralDitherPerSeasonStacker, self).__init__(raCol=raCol, decCol=decCol,
                                                           fieldIdCol=fieldIdCol, nCoils=nCoils,
                                                           numPoints=numPoints, maxDither=maxDither)
        self.expMJDCol= expMJDCol
        self.wrapLastSeason= wrapLastSeason
        # Values required for framework operation: this specifies the names of the new columns.
        self.colsAdded = ['spiralDitherPerSeasonRa', 'spiralDitherPerSeasonDec']
        self.colsReq.append(self.expMJDCol)
        
    def _run(self, simData):            
        # find the seasons associated with each visit.
        seasons = calcSeason(simData[self.raCol], simData[self.expMJDCol])

        # check how many entries in the >10 season
        ind= np.where(seasons > 9)[0]
        # should be only 1 extra seasons ..
        if len(np.unique(seasons[ind])) > 1:
            raise ValueError('Too many seasons (more than 11). Check SeasonStacker.')

        if self.wrapLastSeason:
            print('Seasons to wrap ', np.unique(seasons[ind]))
            # wrap the season around: 10th == 0th
            seasons[ind]= seasons[ind]%10
        
        # Generate the spiral offset vertices.
        self._generateSpiralOffsets()

        # Add to RA and dec values.
        vertexID= 0
        for s in np.unique(seasons):
            match = np.where(seasons == s)[0]
            vertexID= vertexID % self.numPoints
            simData['spiralDitherPerSeasonRa'][match] = simData[self.raCol][match] \
                                                        + self.xOff[vertexID]/np.cos(simData[self.decCol][match])
            simData['spiralDitherPerSeasonDec'][match] = simData[self.decCol][match] + self.yOff[vertexID]
            vertexID += 1

        # Wrap into expected range.
        simData['spiralDitherPerSeasonRa'], simData['spiralDitherPerSeasonDec'] = \
                                        wrapRADec(simData['spiralDitherPerSeasonRa'],
                                                  simData['spiralDitherPerSeasonDec'])
        return simData

