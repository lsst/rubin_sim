"""
Photometric precision metrics 
Authors: Sergey Koposov, Thomas Collett
"""

import numpy as np
from rubin_sim.maf.metrics import BaseMetric

__all__ = ['SNMetric', 'ThreshSEDSNMetric', 'SEDSNMetric']


twopi = 2.0*np.pi


class RelRmsMetric(BaseMetric):
    """Relative scatter metric (RMS over median).
    """
    def run(self, dataSlice, slicePoint=None):
        return np.std(dataSlice[self.colname])/np.median(dataSlice[self.colname])


class SNMetric(BaseMetric):
    """Calculate the signal to noise metric in a given filter for an object of a given magnitude.
    We assume point source aperture photometry and assume that we do 
    the measurement over the stack
    """
    def __init__(self, m5Col = 'fiveSigmaDepth', seeingCol='finSeeing',
                     skyBCol='filtSkyBrightness',
                     expTCol='visitExpTime',
                     filterCol='filter',
                     metricName='SNMetric',
                     filter=None,
                     mag=None,
                     **kwargs):
        super(SNMetric, self).__init__(col=[m5Col, seeingCol, skyBCol,expTCol,filterCol],
                                       metricName=metricName, **kwargs)
        self.filter = filter
        self.mag = mag

    def run(self, dataSlice, slicePoint=None):
        #print 'x'
        npoints = len(dataSlice[self.seeingCol])
        seeing= dataSlice[self.seeingCol]
        depth5 = dataSlice[self.m5Col]
        #mag = depth5 
        mag = self.mag

        zpt0 = 25.85
        curfilt = self.filter #'r'
        zpts = {'u': zpt0,
                'g': zpt0,
                'r': zpt0,
                'i': zpt0,
                'z': zpt0,
                'y': zpt0}

        gain = 4.5

        zptArr= np.zeros(npoints)
        for filt in 'ugrizy':
            zptArr[dataSlice[self.filterCol]==filt]=zpts[filt]
        sky_mag_arcsec=dataSlice[self.skyBCol]
        exptime = dataSlice[self.expTCol]
        sky_adu = 10**(-(sky_mag_arcsec-zptArr)/2.5) * exptime
        sky_adu = sky_adu * np.pi * seeing**2 # adu per seeing circle

        source_fluxes = 10**(-mag/2.5)
        source_adu = 10**(-(mag-zptArr)/2.5)*exptime
        err_adu = np.sqrt(source_adu+sky_adu)/np.sqrt(gain)
        err_fluxes = err_adu * (source_fluxes/source_adu)

        ind = dataSlice[self.filterCol]==curfilt
        flux0 = source_fluxes
        stack_flux_err=1./np.sqrt((1/err_fluxes[ind]**2).sum())
        errMag = 2.5/np.log(10)*stack_flux_err/flux0
        #return errMag
        return flux0/stack_flux_err
        #return (source_fluxes/err_fluxes).mean()
        #1/0
        #return errMag
        #return 1.25 * np.log10(np.sum(10.**(.8*dataSlice['fiveSigmaDepth'])))


class SEDSNMetric(BaseMetric):
    """Computes the S/Ns for a given SED.
    """
    def __init__(self, m5Col = 'fiveSigmaDepth', 
                     seeingCol='finSeeing',
                     skyBCol='filtSkyBrightness',
                     expTCol='visitExpTime',
                     filterCol='filter',
                     metricName='SEDSNMetric',
                     #filter=None,
                     mags=None,
                     **kwargs):
        super(SEDSNMetric, self).__init__(col=[m5Col, seeingCol, skyBCol,expTCol,filterCol],
                                          metricName=metricName, **kwargs)
        self.mags=mags
        self.metrics={}
        for curfilt, curmag in mags.items():
                        self.metrics[curfilt]=SNMetric(mag=curmag,filter=curfilt)
        #self.filter = filter
        #self.mag = mag

    def run(self, dataSlice, slicePoint=None):
        res={}
        for curf, curm in self.metrics.items():
            curr=curm.run(dataSlice, slicePoint=slicePoint)
            res['sn_'+curf]=curr
        return res

    def reduceSn_g(self, metricValue):
        #print 'x',metricValue['sn_g']
        return metricValue['sn_g']

    def reduceSn_r(self, metricValue):
        #print 'x',metricValue['sn_r']
        return metricValue['sn_r']

    def reduceSn_i(self, metricValue):
        return metricValue['sn_i']

class ThreshSEDSNMetric(BaseMetric):
    """Computes the metric whether the S/N is bigger than the threshold in all the bands for a given SED
    """
    def __init__(self, m5Col = 'fiveSigmaDepth', 
                     seeingCol='finSeeing',
                     skyBCol='filtSkyBrightness',
                     expTCol='visitExpTime',
                     filterCol='filter',
                     metricName='ThreshSEDSNMetric',
                     snlim=20,
                     #filter=None,
                     mags=None,
                     **kwargs):
        """Instantiate metric."""
        super(ThreshSEDSNMetric, self).__init__(col=[m5Col, seeingCol, skyBCol,
                                                             expTCol, filterCol], metricName=metricName, **kwargs)
        self.xmet = SEDSNMetric(mags=mags)
        self.snlim = snlim
        #self.filter = filter
        #self.mag = mag

    def run(self, dataSlice, slicePoint=None):
        res=self.xmet.run(dataSlice, slicePoint=slicePoint)
        cnt=0
        for k,v in res.items():
            if v>self.snlim:
                cnt+=1
        if cnt>0:
            cnt=1
        return cnt
