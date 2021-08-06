import numpy as np
from scipy.interpolate import interp1d
from rubin_sim.maf.metrics import BaseMetric
import healpy as hp

# Modifying from Knut Olson's fork at:
# https://github.com/knutago/sims_maf_contrib/blob/master/tutorials/CrowdingMetric.ipynb

__all__ = ['CrowdingM5Metric', 'CrowdingMagUncertMetric', 'NstarsMetric']


def _compCrowdError(magVector, lumFunc, seeing, singleMag=None):
    """
    Compute the photometric crowding error given the luminosity function and best seeing.

    Parameters
    ----------
    magVector : np.array
        Stellar magnitudes.
    lumFunc : np.array
        Stellar luminosity function.
    seeing : float
        The best seeing conditions. Assuming forced-photometry can use the best seeing conditions
        to help with confusion errors.
    singleMag : float (None)
        If singleMag is None, the crowding error is calculated for each mag in magVector. If
        singleMag is a float, the crowding error is interpolated to that single value.

    Returns
    -------
    np.array
        Magnitude uncertainties.

    Equation from Olsen, Blum, & Rigaut 2003, AJ, 126, 452
    """
    lumAreaArcsec = 3600.0 ** 2
    lumVector = 10 ** (-0.4 * magVector)
    coeff = np.sqrt(np.pi / lumAreaArcsec) * seeing / 2.
    myInt = (np.add.accumulate((lumVector ** 2 * lumFunc)[::-1]))[::-1]
    temp = np.sqrt(myInt) / lumVector
    if singleMag is not None:
        interp = interp1d(magVector, temp)
        temp = interp(singleMag)
    crowdError = coeff * temp
    return crowdError


class CrowdingM5Metric(BaseMetric):
    """Return the magnitude at which the photometric error exceeds crowding_error threshold.
    """
    def __init__(self, crowding_error=0.1, filtername='r', seeingCol='seeingFwhmGeom',
                 metricName=None, maps=['StellarDensityMap'], **kwargs):
        """
        Parameters
        ----------
        crowding_error : float, optional
            The magnitude uncertainty from crowding in magnitudes. Default 0.1 mags.
        filtername: str, optional
            The bandpass in which to calculate the crowding limit. Default r.
        seeingCol : str, optional
            The name of the seeing column.
        m5Col : str, optional
            The name of the m5 depth column.
        maps : list of str, optional
            Names of maps required for the metric.

        Returns
        -------
        float
        The magnitude of a star which has a photometric error of `crowding_error`
        """

        cols = [seeingCol]
        units = 'mag'
        self.crowding_error = crowding_error
        self.filtername = filtername
        self.seeingCol = seeingCol
        if metricName is None:
            metricName = 'Crowding to Precision %.2f' % (crowding_error)
        super().__init__(col=cols, maps=maps, units=units, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        # Set magVector to the same length as starLumFunc (lower edge of mag bins)
        magVector = slicePoint[f'starMapBins_{self.filtername}'][1:]
        # Pull up density of stars at this point in the sky
        lumFunc = slicePoint[f'starLumFunc_{self.filtername}']
        # Calculate the crowding error using the best seeing value (in any filter?)
        crowdError = _compCrowdError(magVector, lumFunc,
                                     seeing=min(dataSlice[self.seeingCol]))
        # Locate at which point crowding error is greater than user-defined limit
        aboveCrowd = np.where(crowdError >= self.crowding_error)[0]

        if np.size(aboveCrowd) == 0:
            result = max(magVector)
        else:
            crowdMag = magVector[max(aboveCrowd[0]-1, 0)]
            result = crowdMag

        return result


class NstarsMetric(BaseMetric):
    """Return the number of stars visible above some uncertainty limit,
    taking image depth and crowding into account.
    """
    def __init__(self, crowding_error=0.1, filtername='r', seeingCol='seeingFwhmGeom',
                 m5Col='fiveSigmaDepth',
                 metricName=None, maps=['StellarDensityMap'], ignore_crowding=False, **kwargs):
        """
        Parameters
        ----------
        crowding_error : float, optional
            The magnitude uncertainty from crowding in magnitudes. Default 0.1 mags.
        filtername: str, optional
            The bandpass in which to calculate the crowding limit. Default r.
        seeingCol : str, optional
            The name of the seeing column.
        m5Col : str, optional
            The name of the m5 depth column.
        maps : list of str, optional
            Names of maps required for the metric.
        ignore_crowding : bool (False)
            Ignore the cowding limit.

        Returns
        -------
        float
        The number of stars above the error limit
        """

        cols = [seeingCol, m5Col]
        units = 'N stars'
        self.crowding_error = crowding_error
        self.m5Col = m5Col
        self.filtername = filtername
        self.seeingCol = seeingCol
        self.ignore_crowding = ignore_crowding
        if metricName is None:
            metricName = 'N stars to Precision %.2f' % (crowding_error)
        super().__init__(col=cols, maps=maps, units=units, metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):

        pix_area = hp.nside2pixarea(slicePoint['nside'], degrees=True)
        # Set magVector to the same length as starLumFunc (lower edge of mag bins)
        magVector = slicePoint[f'starMapBins_{self.filtername}'][1:]
        # Pull up density of stars at this point in the sky
        lumFunc = slicePoint[f'starLumFunc_{self.filtername}']
        # Calculate the crowding error using the best seeing value (in any filter?)
        crowdError = _compCrowdError(magVector, lumFunc,
                                     seeing=min(dataSlice[self.seeingCol]))
        # Locate at which point crowding error is greater than user-defined limit
        aboveCrowd = np.where(crowdError >= self.crowding_error)[0]

        if np.size(aboveCrowd) == 0:
            crowdMag = max(magVector)
        else:
            crowdMag = magVector[max(aboveCrowd[0]-1, 0)]

        # Compute the coadded depth, and the mag where that depth hits the error specified
        coadded_depth = 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.m5Col])))
        mag_limit = -2.5*np.log10(1./(self.crowding_error*(1.09*5)))+coadded_depth

        # Use the shallower depth, crowding or coadded
        if self.ignore_crowding:
            min_mag = mag_limit
        else:
            min_mag = np.min([crowdMag, mag_limit])

        # Interpolate to the number of stars
        result = np.interp(min_mag, slicePoint[f'starMapBins_{self.filtername}'][1:],
                           slicePoint[f'starLumFunc_{self.filtername}']) * pix_area

        return result


class CrowdingMagUncertMetric(BaseMetric):
    """
    Given a stellar magnitude, calculate the mean uncertainty on the magnitude from crowding.
    """
    def __init__(self, rmag=20., seeingCol='seeingFwhmGeom', units='mag',
                 metricName=None, filtername='r', maps=['StellarDensityMap'], **kwargs):
        """
        Parameters
        ----------
        rmag : float
            The magnitude of the star to consider.

        Returns
        -------
        float
            The uncertainty in magnitudes caused by crowding for a star of rmag.
        """

        self.filtername = filtername
        self.seeingCol = seeingCol
        self.rmag = rmag
        if metricName is None:
            metricName = 'CrowdingError at %.2f' % (rmag)
        super().__init__(col=[seeingCol], maps=maps, units=units,
                         metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        magVector = slicePoint[f'starMapBins_{self.filtername}'][1:]
        lumFunc = slicePoint[f'starLumFunc_{self.filtername}']
        # Magnitude uncertainty given crowding
        dmagCrowd = _compCrowdError(magVector, lumFunc,
                                    dataSlice[self.seeingCol], singleMag=self.rmag)
        result = np.mean(dmagCrowd)
        return result
