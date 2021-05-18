from builtins import zip
from builtins import object
from .baseStacker import BaseStacker

__all__ = ['ColInfo']


class ColInfo(object):
    """Class to hold the unit and source locations for columns.

    The stacker classes which will generate stacker columns are tracked here, as well as
    some default units for common opsim columns.

    Inspect ColInfo.unitDict for more information."""
    def __init__(self):
        self.defaultDataSource = None
        self.defaultUnit = ''
        self.unitDict = {'fieldId': '#',
                         'filter': 'filter',
                         'seqnNum': '#',
                         'expMJD': 'MJD',
                         'observationStartMJD': 'MJD',
                         'observationStartLST': 'deg',
                         'visitExposureTime': 's',
                         'slewTime': 's',
                         'slewDist': 'rad',
                         'rotSkyPos': 'deg',
                         'rotTelPos': 'deg',
                         'rawSeeing': 'arcsec',
                         'finSeeing': 'arcsec',
                         'FWHMeff': 'arcsec',
                         'FWHMgeom': 'arcsec',
                         'seeingFwhmEff': 'arcsec',
                         'seeingFwhmGeom': 'arcsec',
                         'seeingFwhm500': 'arcsec',
                         'seeing': 'arcsec',
                         'airmass': 'X',
                         'night': 'days',
                         'moonRA': 'rad',
                         'moonDec': 'rad',
                         'moonAlt': 'rad',
                         'dist2Moon': 'rad',
                         'filtSkyBrightness': 'mag/sq arcsec',
                         'skyBrightness': 'mag/sq arcsec',
                         'fiveSigmaDepth': 'mag',
                         'solarElong': 'degrees'}
        # Go through the available stackers and add any units, and identify their
        #   source methods.
        self.sourceDict = {}
        for stackerClass in BaseStacker.registry.values():
            stacker = stackerClass()
            for col, units in zip(stacker.colsAdded, stacker.units):
                self.sourceDict[col] = stackerClass
                self.unitDict[col] = units
        # Note that a 'unique' list of methods should be built from the resulting returned
        #  methods, at whatever point the derived data columns will be calculated. (i.e. in the driver)

    def getUnits(self, colName):
        """Return the appropriate units for colName.

        If no units have been defined for a given column, return the default units ('').

        Parameters
        ----------
        colName : str
            The name of the column

        Returns
        -------
        str
        """
        if colName is None or colName not in self.unitDict:
            return self.defaultUnit
        else:
            return self.unitDict[colName]

    def getDataSource(self, colName):
        """Identify the appropriate source for a given column.

        For values which are calculated via a stacker, the returned value is the stacker class.
        For values which do not have a recorded source or are known to be coming from the database, the result
        is self.defaultDataSource (None), which will be assumed to be queryable from the database.

        Parameters
        ----------
        colName : str
            The name of the column.

        Returns
        -------
        rubin_sim.maf.stacker or None
        """
        if colName in self.sourceDict:
            return self.sourceDict[colName]
        else:
            return self.defaultDataSource
