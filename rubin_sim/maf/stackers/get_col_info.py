from builtins import zip
from builtins import object
from .base_stacker import BaseStacker

__all__ = ["ColInfo"]


class ColInfo(object):
    """Class to hold the unit and source locations for columns.

    The stacker classes which will generate stacker columns are tracked here, as well as
    some default units for common opsim columns.

    Inspect ColInfo.unit_dict for more information."""

    def __init__(self):
        self.default_data_source = None
        self.default_unit = ""
        self.unit_dict = {
            "fieldId": "#",
            "filter": "filter",
            "seqnNum": "#",
            "expMJD": "MJD",
            "observationStartMJD": "MJD",
            "observationStartLST": "deg",
            "visitExposureTime": "s",
            "slewTime": "s",
            "slewDist": "rad",
            "rotSkyPos": "deg",
            "rotTelPos": "deg",
            "rawSeeing": "arcsec",
            "finSeeing": "arcsec",
            "FWHMeff": "arcsec",
            "FWHMgeom": "arcsec",
            "seeingFwhmEff": "arcsec",
            "seeingFwhmGeom": "arcsec",
            "seeingFwhm500": "arcsec",
            "seeing": "arcsec",
            "airmass": "X",
            "night": "days",
            "moonRA": "rad",
            "moonDec": "rad",
            "moonAlt": "rad",
            "dist2Moon": "rad",
            "filtSkyBrightness": "mag/sq arcsec",
            "skyBrightness": "mag/sq arcsec",
            "fiveSigmaDepth": "mag",
            "solarElong": "degrees",
        }
        # Go through the available stackers and add any units, and identify their
        #   source methods.
        self.source_dict = {}
        for stacker_class in BaseStacker.registry.values():
            stacker = stacker_class()
            for col, units in zip(stacker.cols_added, stacker.units):
                self.source_dict[col] = stacker_class
                self.unit_dict[col] = units
        # Note that a 'unique' list of methods should be built from the resulting returned
        #  methods, at whatever point the derived data columns will be calculated. (i.e. in the driver)

    def get_units(self, col_name):
        """Return the appropriate units for col_name.

        If no units have been defined for a given column, return the default units ('').

        Parameters
        ----------
        col_name : str
            The name of the column

        Returns
        -------
        str
        """
        if col_name is None or col_name not in self.unit_dict:
            return self.default_unit
        else:
            return self.unit_dict[col_name]

    def get_data_source(self, col_name):
        """Identify the appropriate source for a given column.

        For values which are calculated via a stacker, the returned value is the stacker class.
        For values which do not have a recorded source or are known to be coming from the database, the result
        is self.default_data_source (None), which will be assumed to be queryable from the database.

        Parameters
        ----------
        col_name : str
            The name of the column.

        Returns
        -------
        rubin_sim.maf.stacker or None
        """
        if col_name in self.source_dict:
            return self.source_dict[col_name]
        else:
            return self.default_data_source
