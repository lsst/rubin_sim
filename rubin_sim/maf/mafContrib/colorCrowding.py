import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import os
import healpy as hp
from scipy.interpolate import griddata, interp1d, interpn
from scipy.interpolate import RectBivariateSpline

import rubin_sim.maf as maf
# will need single-band crowding error as well
def _compCrowdError(magVector, lumFunc, seeing, singleMag=None):
        """
        Compute the crowding error for each observation

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

        lumAreaArcsec = 3600.**2
        lumVector = 10**(-0.4*magVector)
        coeff=np.sqrt(np.pi/lumAreaArcsec)*seeing/2.
        myIntegral = (np.add.accumulate((lumVector**2*lumFunc)[::-1]))[::-1]
        temp = np.sqrt(myIntegral)/lumVector
        if singleMag is not None:
            interp = interp1d(magVector, temp)
            temp = interp(singleMag)

        crowdError = coeff*temp

        return crowdError

# compute color error for a given magnitude vector, joint LF, and seeing in two bands
def _compColorCrowdError(magVector1, magVector2, lumFunc12, seeing1, seeing2, singleMag1=None, singleMag2=None):
        """
        Compute the crowding error for each observation

        Parameters
        ----------
        magVector1 : np.array
            Stellar magnitudes in band 1.
        magVector2 : np.array
            Stellar magnitudes in band 2.
        lumFunc12 : np.array (2D)
            Joint stellar luminosity function in bands 1 and 2.
        seeing1 : float
            Seeing in band 1.
        seeing2 : float
            Seeing in band 2.
        singleMag1 : float (None)
            If singleMag is None, the crowding error is calculated for each mag in magVector. If
            singleMag is a float, the crowding error is interpolated to that single value.

        Returns
        -------
        np.array
            Magnitude uncertainties.


        Equation from Olsen, Blum, & Rigaut 2003, AJ, 126, 452
        """

        lumAreaArcsec = 3600.**2
        lumVector1, lumVector2 = np.meshgrid(10**(-0.4*magVector1), 10**(-0.4*magVector2))

        coeff=np.sqrt(np.pi/lumAreaArcsec)*np.min([seeing1,seeing2])/2.
        innerInt = np.add.accumulate(np.flip(lumVector1*lumFunc12),axis=1)
        outerInt = np.add.accumulate(np.flip(lumVector2)*innerInt,axis=0)
        int2d = np.flip(outerInt)

        temp = np.sqrt(int2d)/np.sqrt(lumVector1*lumVector2)

        crowdError12 = coeff*temp
        crowdError1 = _compCrowdError(magVector1,np.sum(lumFunc12,axis=0),seeing1)
        crowdError2 = _compCrowdError(magVector2,np.sum(lumFunc12,axis=1),seeing2)
        ce1, ce2 = np.meshgrid(crowdError1,crowdError2)
        
        colCrowdError = np.sqrt(ce1**2 + ce2**2 - 2*crowdError12**2)
        if singleMag1 is not None and singleMag2 is not None:
            interp = interp1d(magVector1, crowdError1)
            crowdError1 = interp(singleMag1)
            interp = interp1d(magVector2, crowdError2)
            crowdError2 = interp(singleMag2)
            #xvec,yvec = np.meshgrid(magVector1,magVector2)
            interp = interpn((magVector1,magVector2),colCrowdError,(singleMag1,singleMag2))
            colCrowdError = interp

        return crowdError1, crowdError2, colCrowdError

def _calcNearestUncty(unctyTarg=0.1, xFine=np.array([]), yFine=np.array([]), \
                    ff=None, useNearest=False, constColor=None):
    
    """Copied from notebook by Will Clarkson: 
    https://github.com/LSSTScienceCollaborations/SMWLV-metrics/blob/main/notebooks/CrowdingColor_wVecs.ipynb

    Method to return the nearest mag1, mag2 to the target uncertainty. 
    
    Arguments:
    
    unctyTarg = target color uncertainty. Scalar.
    
    xFine, yFine = mag0, mag1 1D arrays describing the locus
    
    ff = bivariate spline object describing the interpolation of the surface

    useNearest: return nearest value to the crossing rather than interpolating?
    
    constColor: if useInterp=True, then this color is added to the nearest value 
    of mag1 (i.e. xClose) to construct the mag2 value (i.e. yClose) to return.
    
    Returns: xClose, yClose, zClose, lSor4
    
    """
    
    # do we have the surface spline? (If not we could reconstruct here if given the z values)
    try:
        zeval = ff.ev(xFine, yFine)
    except:
        return -99., -99., -99., np.array([])
    
    # If the target uncertainty is outside the range of values of zeval, return 
    # the supplied target uncertainty, and then magnitudes that are much fainter
    # or much brighter than the input range as appropriate. 
    if unctyTarg > np.max(zeval):
        return 99., 99., unctyTarg, np.array([])
    if unctyTarg < np.min(zeval):
        return -99., -99., unctyTarg, np.array([])
    
    # Algorithm: at a crossing, the samples go from below to above the target level (in that order).
    # So, find the pairs that define crossings in that direction and pick the brightest. Then 
    # pick the nearest (vertically) of the two points in the pair 
    lOne = np.arange(xFine.size) # this could be argsort on some other property.
    lTwo = np.roll(lOne, -1)
    
    # Identify the candidate crossings...
    bCross = (zeval[lTwo] > zeval[lOne]) & \
        (zeval[lTwo] > unctyTarg) & \
        (zeval[lOne] <= unctyTarg) & \
        (xFine[lTwo] > xFine[lOne])
   
    # If there are no crossings in the increasing direction but the previous off-range conditions
    # were not triggered, then our limit must be at least as bright as the brightest point. So 
    # return that condition.
    if np.sum(bCross) < 1:
        return xFine[lOne[0]], yFine[lOne[0]], zeval[lOne[0]], np.array([])
    
    # ... find the brightest...
    lLo = lOne[bCross]
    iBri = np.argmin(xFine[lLo])
    lCross = np.array([lOne[bCross][iBri], lTwo[bCross][iBri] ])
    
    # For all reasonable crowding uncertainty curves, all points brighter than the first
    # crossing will have LOWER crowding uncertainty. However, let's make sure by adding
    # a condition: if any points brighter than the brightest crossing have greater crowding
    # uncertainty than either of the ends of the crossing, trap that condition and return.
    # In this case, the returned zeval will NOT be the same as the target uncertainty.
    bBri = xFine < xFine[lCross[0]]
    if np.sum(bBri) > 0:
        if np.max(zeval[bBri]) >= np.max(zeval[lCross]):
            iBri = np.argmax(zeval[bBri])
            return xFine[bBri][iBri], yFine[bBri][iBri], zeval[bBri][iBri], lCross
    
    # If we only want the closest of the input grid, return that here.
    if useNearest:
        lMin = np.argmin(np.abs(zeval[lCross] - unctyTarg))
        iMin = lCross[lMin]
        return xFine[iMin], yFine[iMin], zFine[iMin], lCross    
        
    # Otherwise, use our linear interpolation method
    xInterp = np.interp(unctyTarg, zeval[lCross], xFine[lCross])
    if constColor != None:
        yInterp = xInterp - constColor
    else:
        yInterp = np.interp(unctyTarg, zeval[lCross], yFine[lCross])
    
    return xInterp, yInterp, unctyTarg, lCross

# returns single-mag error in each band and the error in color
class CrowdingMagColorUncertMetric(maf.BaseMetric):
    """
    Given a stellar magnitude in two bands, calculate the mean uncertainty on the magnitude and color from crowding.
    """
    def __init__(self, mag1=20., mag2=20., filter1name='g', filter2name='r', seeingCol='seeingFwhmGeom', units='mag',
                 metricName=None, lfpath='~/lsst/SMWLV-metrics.orig/', **kwargs):
        """
        Parameters
        ----------
        mag1 : float, optional
            The magnitude of the star in filter1 to consider. Default 20.0.
        mag2 : float, optional
            The magnitude of the star in filter2 to consider. Default 20.0.
        filter1name : str, optional
            The name of the first filter forming the color.  Default 'g'.
        filter2name : str, optional
            The name of the second filter forming the color.  Default 'r'.            
        seeingCol : str, optional
            The name of the seeing column.
        units : str, optional
            The units of the output.
        

        Returns
        -------
        float
            The uncertainty in magnitudes caused by crowding for a star of mag1 and mag2 and color mag1-mag2.
        """

        colors = ['ug','ur','ui','uz','gr','gi','gz','ri','rz','iz']     
        if filter1name + filter2name not in colors:
            print('Color '+ filter1name + filter2name + ' not in list.  Please pick one of ' + str(colors)) 
            return
        self.filter1name = filter1name
        self.filter2name = filter2name
        self.seeingCol = seeingCol
        self.mag1 = mag1
        self.mag2 = mag2
        self.colorname = filter1name + filter2name
        self.jointlf_file = lfpath + 'jointlf_' + self.colorname + '_rd32_wVecs.fits'
        jointlf = Table.read(os.path.expanduser(self.jointlf_file))
        self.jointlf = jointlf
        self.magvec1 = np.array(self.jointlf.meta['MAGS0'])
        self.magvec2 = np.array(self.jointlf.meta['MAGS1'])
        self.maggrid1, self.maggrid2 = np.meshgrid(self.magvec1,self.magvec2)
        
        if metricName is None:
            metricName = 'CrowdingMagColorUncertMetric at %s = %.2f %s = %.2f' % (filter1name,mag1,filter2name,mag2)
        super().__init__(col=[seeingCol,'filter'], units=units,
                         metricName=metricName, **kwargs)


        
    def run(self, dataSlice, slicePoint=None):

        lf12t = self.jointlf[self.jointlf['ring32'] == slicePoint['sid']]
        if len(lf12t)>0:
            try:
                lf12 = griddata((lf12t[self.filter1name],lf12t[self.filter2name]),lf12t['n'],(self.maggrid1, self.maggrid2),fill_value=0.)                                                        
                # Magnitude and color uncertainty given crowding
                sig1, sig2, sig12 = _compColorCrowdError(self.magvec1, self.magvec2, lf12, 
                                               dataSlice[self.seeingCol].min(), 
                                               dataSlice[self.seeingCol].min(),
                                               singleMag1=self.mag1, singleMag2=self.mag2)
            except:
                sig1, sig2, sig12 = 0., 0., 0.
        else:
            sig1, sig2, sig12 = 0., 0., 0.

        result = {'sig1': sig1, 'sig2': sig2, 'sig12': sig12}
        
        return result
    
    def reduce_sig1(self, metricValue):
        return metricValue['sig1']
    
    def reduce_sig2(self, metricValue):
        return metricValue['sig2']

    def reduce_sig12(self, metricValue):
        return metricValue['sig12']
    
class CrowdingColorM5Metric(maf.BaseMetric):
    """
    Return the magnitude in two bands at which the photometric error in color exceeds color_crowding_error threshold, 
    for a given color.
    """
    def __init__(self, color_crowding_error=0.1, color=0.0, filter1name='g', filter2name='r', seeingCol='seeingFwhmGeom', 
                 units='mag', metricName=None, lfpath='~/lsst/SMWLV-metrics.orig/', **kwargs):

        """
        Parameters
        ----------
        color_crowding_error : float, optional
            The largest desired color uncertainty from crowding in magnitudes. Default 0.1 mags.
        color: float, optional
            The color of the star for which the crowding error should be calculated.  Default 0.0 mags.
        filter1name : str, optional
            The name of the first filter forming the color.  Default 'g'.
        filter2name : str, optional
            The name of the second filter forming the color.  Default 'r'.            
        seeingCol : str, optional
            The name of the seeing column.  Default 'seeingFwhmGeom'.

        Returns
        -------
        float
        The magnitude of a star in each band which has a photometric color error of `color_crowding_error`
        """

        colors = ['ug','ur','ui','uz','gr','gi','gz','ri','rz','iz']     
        if filter1name + filter2name not in colors:
            print('Color '+ filter1name + filter2name + ' not in list.  Please pick one of ' + str(colors)) 
            return
        self.crowding_error = color_crowding_error
        self.filter1name = filter1name
        self.filter2name = filter2name
        self.seeingCol = seeingCol
        self.color = color
        self.colorname = filter1name + filter2name
        self.jointlf_file = lfpath + 'jointlf_' + self.colorname + '_rd32_wVecs.fits'
        jointlf = Table.read(os.path.expanduser(self.jointlf_file))
        self.jointlf = jointlf
        self.magvec1 = np.array(self.jointlf.meta['MAGS0'])
        self.magvec2 = np.array(self.jointlf.meta['MAGS1'])
        self.maggrid1, self.maggrid2 = np.meshgrid(self.magvec1,self.magvec2)
  
        # the locus is a pair of vectors forming a constant color, but could be made an isochrone
        self.locus1 = self.magvec1
        self.locus2 = self.magvec1 + self.color
        # Chop the arrays down so that we don't go out of the mag-mag space
        bOK = (self.locus2 >= np.min(self.magvec2)) & (self.locus2 <= np.max(self.magvec2))
        self.locus1 = self.locus1[bOK]
        self.locus2 = self.locus2[bOK]
        
        if metricName is None:
            metricName = 'CrowdingColorM5Metric at error < %.2f at %s - %s = %.2f' % (color_crowding_error, filter1name,filter2name,color)
        super().__init__(col=[seeingCol,'filter'], units=units,
                         metricName=metricName, **kwargs)

    def run(self, dataSlice, slicePoint=None):

        lf12t = self.jointlf[self.jointlf['ring32'] == slicePoint['sid']]
        t0=0.
        t1=0.
        t2=0.
        t3=0.
        t4=0.
        if len(lf12t)>0:
            try:
                lf12 = griddata((lf12t[self.filter1name],lf12t[self.filter2name]),lf12t['n'],(self.maggrid1, self.maggrid2),fill_value=0.)                                                        
                # Magnitude and color uncertainty given crowding
                sig1, sig2, sig12 = _compColorCrowdError(self.magvec1, self.magvec2, lf12, 
                                               dataSlice[self.seeingCol].min(), 
                                               dataSlice[self.seeingCol].min())
                # try interpolating the grid of sig_gi so that we can draw a locus and
                # interpolate over it
                ff = RectBivariateSpline(self.magvec1, self.magvec2, sig12, kx=5, ky=5)
                mag1lim, mag2lim, color_error_interp, icrossing = _calcNearestUncty(self.crowding_error, self.locus1, self.locus2, ff,
                                                             useNearest=False)
            except:
                mag1lim, mag2lim = 0., 0.
        else:
            mag1lim, mag2lim = 0., 0.

        #if mag1lim != 0:
        #    import pdb; pdb.set_trace()
        result = {'mag1lim': mag1lim, 'mag2lim': mag2lim}
        
        return result
    
    def reduce_mag1lim(self, metricValue):
        return metricValue['mag1lim']
    
    def reduce_mag2lim(self, metricValue):
        return metricValue['mag2lim']
