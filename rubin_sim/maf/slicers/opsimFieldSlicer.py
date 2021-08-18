# Class for opsim field based slicer.

import numpy as np
from functools import wraps
import warnings
from rubin_sim.maf.plots.spatialPlotters import OpsimHistogram, BaseSkyMap

from .baseSpatialSlicer import BaseSpatialSlicer

__all__ = ['OpsimFieldSlicer']


class OpsimFieldSlicer(BaseSpatialSlicer):
    """A spatial slicer that evaluates pointings based on matched IDs between the simData and fieldData.

    Note that this slicer uses the fieldId of the simulated data fields to generate the spatial matches.
    Thus, it is not suitable for use in evaluating dithering or high resolution metrics
    (use the HealpixSlicer instead for those use-cases).

    When the slicer is set up, it takes two arrays: fieldData and simData. FieldData is a numpy.recarray
    containing the information about the fields - this is the basis for slicing.
    The simData is a numpy.recarray that holds the information about the pointings - this is the data that
    is matched against the fieldData.

    Parameters
    ----------
    simDataFieldIDColName : str, optional
        Name of the column in simData for the fieldId
        Default fieldId.
    simDataFieldRaColName : str, optional
        Name of the column in simData for the RA.
        Default fieldRA.
    simDataFieldDecColName : str, optional
        Name of the column in simData for the fieldDec.
        Default fieldDec.
    latLongDeg : bool, optional
        Whether the RA/Dec values in *fieldData* are in degrees.
        If using a standard metricBundleGroup to run the metric, FieldData is fetched
        by utils.getFieldData, which always returns radians (so the default here is False).
    fieldIdColName : str, optional
        Name of the column in the fieldData for the fieldId (to match with simData).
        Default fieldId.
    fieldRaColName : str, optional
        Name of the column in the fieldData for the RA (used for plotting).
        Default fieldRA.
    fieldDecColName : str, optional
        Name of the column in the fieldData for the Dec (used for plotting).
        Default fieldDec.
    verbose : `bool`, optional
        Flag to indicate whether or not to write additional information to stdout during runtime.
        Default True.
    badval : float, optional
        Bad value flag, relevant for plotting. Default -666.
    """
    def __init__(self, simDataFieldIdColName='fieldId',
                 simDataFieldRaColName='fieldRA', simDataFieldDecColName='fieldDec', latLonDeg=False,
                 fieldIdColName='fieldId', fieldRaColName='fieldRA', fieldDecColName='fieldDec',
                 verbose=True, badval=-666):
        super(OpsimFieldSlicer, self).__init__(verbose=verbose, badval=badval)
        self.fieldId = None
        self.simDataFieldIdColName = simDataFieldIdColName
        self.fieldIdColName = fieldIdColName
        self.fieldRaColName = fieldRaColName
        self.fieldDecColName = fieldDecColName
        self.latLonDeg = latLonDeg
        self.columnsNeeded = [simDataFieldIdColName, simDataFieldRaColName, simDataFieldDecColName]
        while '' in self.columnsNeeded:
            self.columnsNeeded.remove('')
        self.fieldColumnsNeeded = [fieldIdColName, fieldRaColName, fieldDecColName]
        self.slicer_init = {'simDataFieldIdColName': simDataFieldIdColName,
                            'simDataFieldRaColName': simDataFieldRaColName,
                            'simDataFieldDecColName': simDataFieldDecColName,
                            'fieldIdColName': fieldIdColName,
                            'fieldRaColName': fieldRaColName,
                            'fieldDecColName': fieldDecColName, 'badval': badval}
        self.plotFuncs = [BaseSkyMap, OpsimHistogram]
        self.needsFields = True

    def setupSlicer(self, simData, fieldData, maps=None):
        """Set up opsim field slicer object.

        Parameters
        -----------
        simData : numpy.recarray
            Contains the simulation pointing history.
        fieldData : numpy.recarray
            Contains the field information (ID, Ra, Dec) about how to slice the simData.
            For example, only fields in the fieldData table will be matched against the simData.
            RA and Dec should be in degrees.
        maps : list of rubin_sim.maf.maps objects, optional
            Maps to run and provide additional metadata at each slicePoint. Default None.
        """
        if 'ra' in self.slicePoints:
            warning_msg = 'Warning: this OpsimFieldSlicer was already set up once. '
            warning_msg += 'Re-setting up an OpsimFieldSlicer can change the field information. '
            warning_msg += 'Rerun metrics if this was intentional. '
            warnings.warn(warning_msg)
        # Set basic properties for tracking field information, in sorted order.
        idxs = np.argsort(fieldData[self.fieldIdColName])
        # Set needed values for slice metadata.
        self.slicePoints['sid'] = fieldData[self.fieldIdColName][idxs]
        if self.latLonDeg:
            self.slicePoints['ra'] = np.radians(fieldData[self.fieldRaColName][idxs])
            self.slicePoints['dec'] = np.radians(fieldData[self.fieldDecColName][idxs])
        else:
            self.slicePoints['ra'] = fieldData[self.fieldRaColName][idxs]
            self.slicePoints['dec'] = fieldData[self.fieldDecColName][idxs]
        self.nslice = len(self.slicePoints['sid'])
        self._runMaps(maps)
        # Set up data slicing.
        self.simIdxs = np.argsort(simData[self.simDataFieldIdColName])
        simFieldsSorted = np.sort(simData[self.simDataFieldIdColName])
        self.left = np.searchsorted(simFieldsSorted, self.slicePoints['sid'], 'left')
        self.right = np.searchsorted(simFieldsSorted, self.slicePoints['sid'], 'right')

        self.spatialExtent = [simData[self.simDataFieldIdColName].min(),
                              simData[self.simDataFieldIdColName].max()]
        self.shape = self.nslice

        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            idxs = self.simIdxs[self.left[islice]:self.right[islice]]
            # Build dict for slicePoint info
            slicePoint = {}
            for key in self.slicePoints:
                if (np.shape(self.slicePoints[key])[0] == self.nslice) & \
                        (key != 'bins') & (key != 'binCol'):
                    slicePoint[key] = self.slicePoints[key][islice]
                else:
                    slicePoint[key] = self.slicePoints[key]
            return {'idxs': idxs, 'slicePoint': slicePoint}
        setattr(self, '_sliceSimData', _sliceSimData)

    def __eq__(self, otherSlicer):
        """Evaluate if two grids are equivalent."""
        result = False
        if isinstance(otherSlicer, OpsimFieldSlicer):
            if np.all(otherSlicer.shape == self.shape):
                # Check if one or both slicers have been setup
                if (self.slicePoints['ra'] is not None) or (otherSlicer.slicePoints['ra'] is not None):
                    if (np.array_equal(self.slicePoints['ra'], otherSlicer.slicePoints['ra']) &
                            np.array_equal(self.slicePoints['dec'], otherSlicer.slicePoints['dec']) &
                            np.array_equal(self.slicePoints['sid'], otherSlicer.slicePoints['sid'])):
                        result = True
                # If they have not been setup, check that they have same fields
                elif ((otherSlicer.fieldIdColName == self.fieldIdColName) &
                      (otherSlicer.fieldRaColName == self.fieldRaColName) &
                      (otherSlicer.fieldDecColName == self.fieldDecColName)):
                    result = True
        return result
