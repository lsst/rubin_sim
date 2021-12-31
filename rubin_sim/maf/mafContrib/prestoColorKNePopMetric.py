
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
from lsst.sims.maf.utils import m52snr
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.plots as plots

import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.utils import equatorialFromGalactic, hpid2RaDec, _healbin, healbin
from lsst.utils import getPackageDir
#from mafContrib import KN_lc, KNePopMetric, generateKNPopSlicer
import healpy as hp
import os

import glob    
from itertools import combinations
import pickle
#from lsst.sims.utils import uniformSphere
#from lsst.sims.photUtils import Dust_values


__all__ = ["PrestoColorKNePopMetric", "Dust_values", "KN_lc", "generateKNPopSlicer"]

class Dust_values(object):
    """Calculate extinction values
    Parameters
    ----------
    R_v : float (3.1)
        Extinction law parameter (3.1).
    bandpassDict : dict (None)
        A dict with keys of filtername and values of rubin_sim.photUtils.Bandpass objects. Default
        of None will load the standard ugrizy bandpasses.
    ref_ev : float (1.)
        The reference E(B-V) value to use. Things in MAF assume 1.
    """
    def __init__(self, R_v=3.1, bandpassDict=None, ref_ebv=1.):
        # Calculate dust extinction values
        self.Ax1 = {}
        if bandpassDict is None:
            bandpassDict = BandpassDict.loadTotalBandpassesFromFiles(['u', 'g', 'r', 'i', 'z', 'y'])

        for filtername in bandpassDict:
            wavelen_min = bandpassDict[filtername].wavelen.min()
            wavelen_max = bandpassDict[filtername].wavelen.max()
            testsed = Sed()
            testsed.setFlatSED(wavelen_min=wavelen_min, wavelen_max=wavelen_max, wavelen_step=1.0)
            self.ref_ebv = ref_ebv
            # Calculate non-dust-extincted magnitude
            flatmag = testsed.calcMag(bandpassDict[filtername])
            # Add dust
            a, b = testsed.setupCCM_ab()
            testsed.addDust(a, b, ebv=self.ref_ebv, R_v=R_v)
            # Calculate difference due to dust when EBV=1.0 (m_dust = m_nodust - Ax, Ax > 0)
            self.Ax1[filtername] = testsed.calcMag(bandpassDict[filtername]) - flatmag


def uniformSphere(npoints, seed=42):
    """
    Just make RA, dec points on a sphere
    """
    np.random.seed(seed)
    u = np.random.uniform(size=npoints)
    v = np.random.uniform(size=npoints)

    ra = 2.*np.pi * u
    dec = np.arccos(2.*v - 1.)
    # astro convention of -90 to 90
    dec -= np.pi/2.
    return np.degrees(ra), np.degrees(dec)


class KN_lc(object):
    """
    Read in some KNe lightcurves
    Parameters
    ----------
    file_list : list of str (None)
        List of file paths to load. If None, loads up all the files from data/bns/
    """
    def __init__(self, file_list=None):
        if file_list is None:
            sims_maf_contrib_dir = os.getenv("SIMS_MAF_CONTRIB_DIR")
            # Get files, model grid developed by M. Bulla
            file_list = glob.glob(os.path.join(sims_maf_contrib_dir, 'data/bns/*.dat'))

        filts = ["u", "g", "r", "i", "z", "y"]
        magidxs = [1, 2, 3, 4, 5, 6]

        # Let's organize the data in to a list of dicts for easy lookup
        self.data = []
        for filename in file_list:
            mag_ds = np.loadtxt(filename)
            t = mag_ds[:, 0]
            new_dict = {}
            for ii, (filt, magidx) in enumerate(zip(filts, magidxs)):
                new_dict[filt] = {'ph': t, 'mag': mag_ds[:, magidx]}
            self.data.append(new_dict)

    def interp(self, t, filtername, lc_indx=0):
        """
        t : array of floats
            The times to interpolate the light curve to.
        filtername : str
            The filter. one of ugrizy
        lc_index : int (0)
            Which file to use.
        """

        result = np.interp(t, self.data[lc_indx][filtername]['ph'],
                           self.data[lc_indx][filtername]['mag'],
                           left=99, right=99)
        return result


def radec2gal(ra, dec):
    '''convert from ra/dec to galactic l/b'''
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    c = SkyCoord(ra=ra, dec=dec, 
                 unit=(u.degree, u.degree))
    
    l = c.galactic.l.degree
    b = c.galactic.b.degree
    return l, b


def generateKNPopSlicer(skyregion='galactic',
                        t_start=1, t_end=3652, n_events=10000, 
                        seed=42, n_files=100, d_min=10, d_max=300):
    """ Generate a population of KNe events, and put the info about them
    into a UserPointSlicer object
    Parameters
    ----------
    skyregion : string
        The skyregion of interst. Only two options: 'galactic' and 'extragalaxtic'        
    t_start : float (1)
        The night to start kilonova events on (days)
    t_end : float (3652)
        The final night of kilonova events
    n_events : int (10000)
        The number of kilonova events to generate
    seed : float
        The seed passed to np.random
    n_files : int (7)
        The number of different kilonova lightcurves to use
    d_min : float or int (10)
        Minimum luminosity distance (Mpc)
    d_max : float or int (300)
        Maximum luminosity distance (Mpc)
    """

    def rndm(a, b, g, size=1):
        """Power-law gen for pdf(x)\propto x^{g-1} for a<=x<=b"""
        r = np.random.random(size=size)
        ag, bg = a**g, b**g
        return (ag + (bg - ag)*r)**(1./g)

    ra, dec = uniformSphere(n_events, seed=seed)
    
    ###Convert ra, dec to gl, gb
    gl, gb = radec2gal(ra, dec)
   
    ###Determine if the object is in the Galaxy plane
    
    if skyregion is 'galactic':#keep the glactic events
        ra = ra[np.abs(gb)<20]
        dec= dec[np.abs(gb)<20]
    elif skyregion is 'extragalaxtic': #keep the extragalactic events.
        ra = ra[np.abs(eb)>20]
        dec= dec[np.abs(eb)>20]
    
    n_events = len(ra) 
    #len(ra)>n_events
    #ra = ra[:n_events]
    #dec = dec[]
    
    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)
    file_indx = np.floor(np.random.uniform(low=0, high=n_files,
                                           size=n_events)).astype(int)

    # Define the distance
    distance = rndm(d_min, d_max, 4, size=n_events)

    # Set up the slicer to evaluate the catalog we just made
    slicer = slicers.UserPointsSlicer(ra, dec, latLonDeg=True, badval=0)
    # Add any additional information about each object to the slicer
    slicer.slicePoints['peak_time'] = peak_times
    slicer.slicePoints['file_indx'] = file_indx
    slicer.slicePoints['distance'] = distance
    
    return slicer

class PrestoColorKNePopMetric(metrics.BaseMetric):
    def __init__(self, metricName='KNePopMetric', mjdCol='observationStartMJD',
                 m5Col='fiveSigmaDepth', filterCol='filter', nightCol='night',
                 ptsNeeded=2, file_list=None, mjd0=59853.5, outputLc=False,
                 skyregion='galactic',
                 filePathGalactic='/home/idies/workspace/Storage/lianmign/persistent/DataCube/TotalCubeNorm_1000Obj.pkl',
                 filePathExtragalactic='/home/idies/workspace/Storage/lianmign/persistent/DataCube/TotalCubeNorm_1000Obj.pkl',
                 **kwargs):        
        """ 
        Parameters
        ----------
        skyregion : string
            The skyregion of interst. Only two options: 'galactic' and 'extragalaxtic'        
        filePathGalactic : string
            The path to the file contains galactic Prest-Color phase space information
        filePathExtragalactic : string
            The path to the file contains galactic Prest-Color phase space information
        """
        maps = ['DustMap']
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.ptsNeeded = ptsNeeded # detection points threshold
        # Boolean variable, if True the light curve will be exported
        self.outputLc = outputLc
        
        if skyregion is 'galactic':
            self.filePath = filePathGalactic
        elif skyregion is 'extragalactic':
            self.filePath = filePathExtragalactic            
        
        with open(self.filePath, 'rb') as f:
            self.InfoDict = pickle.load( f )
            self.HashTable = pickle.load( f )            
        
        # read in file as light curve object;
        self.lightcurves = KN_lc(file_list=file_list)
        self.mjd0 = mjd0

        dust_properties = Dust_values()
        self.Ax1 = dust_properties.Ax1

        cols = [self.mjdCol, self.m5Col, self.filterCol, self.nightCol]
        super(KNePopMetric, self).__init__(col=cols, units='Detected, 0 or 1',
                                           metricName=metricName, maps=maps,
                                           **kwargs)

    def _multi_detect(self, around_peak):
        """
        Simple detection criteria: detect at least a certain number of times
        """
        result = 1
        # Detected data points
        if np.size(around_peak) < self.ptsNeeded:
            return 0

        return result
    
    def _presto_color_detect(self, around_peak, filters):
        """
        detection criteria of presto cadence: at least three detections at two filters;
        
        Parameters
        ----------
        around_peak : array
            indexes corresponding to 5sigma detections
        filters : array
            filters in which detections happened
        """
        result = 1
        
        if np.size(around_peak)<3:
            result = 0
        
        flts, flts_count = np.unique(filters, return_counts=True,)
        if np.size(flts) < 2:
            result = 0
        elif np.max(flts_count) < 2:
            # if no filters have visits larger than 2, set detection false
            result = 0

        return result
    
    def _enquiry(self, HashTable, InfoDict, Band1, Band2, dT1, dT2, dMag, Color):
        """
        Return the value in the probability cube provided the coordinates 
        in the Presto-Color phase space of an observation triplet.
        
        Parameters
        ----------
        HashTable : array
            Contains the values of the 6-D Presto-Color phase space
        InfoDict : dictionary
            Contains the essential information of the HashTable abobe.
        
        HashTable and InfoDict have to be loaded from premade data Presto-Color data file.
        
        Band1, Band2 : string
            The two filters that comprise the Presto-Color observation triplet. The filters are 
            the 6 bands of LSST: u, g, r, i, z, y. Band1 and Band2 should be different.
            
        dT1, dT2 : float
            The time gaps of the Presto-Color observation triplet.
            
        dMag : float
            The magnitude change calculated from the observations of the same band
            
        Color : float
            The difference in magnitude of observations in different bands.
            
        """
    
#         if abs(dT1) > abs(dT1-dT2):
#             dT1, dT2 = dT1-dT2, -dT2

        if not ( InfoDict['BinMag'][0]<=dMag<InfoDict['BinMag'][-1] and InfoDict['BinColor'][0]<=Color<InfoDict['BinColor'][-1] ):
            return 0

        Ind1 = InfoDict['BandPairs'].index(Band1+Band2)

        dT1grid = InfoDict['dT1s'][ abs( dT1 - InfoDict['dT1s'] ).argmin() ]
        dT2grid = InfoDict['dT2s'][ abs( dT2 - InfoDict['dT2s'] ).argmin() ]
        TimePairGrid = [ InfoDict['dT1s'][ abs( dT1 - InfoDict['dT1s'] ).argmin() ], InfoDict['dT2s'][ abs( dT2 - InfoDict['dT2s'] ).argmin() ] ]

        Ind2 = np.where( (InfoDict['TimePairs'] == TimePairGrid ).all(axis=1) )[0][0]        
        Ind3 = np.where( dMag >= InfoDict['BinMag'] )[0][-1]        
        Ind4 = np.where( Color >= InfoDict['BinColor'] )[0][-1]
        
        return HashTable[Ind1, Ind2, Ind3, Ind4]
    
    def _getScore(self, result, HashTable, InfoDict, scoreType='S', thr=0.003):
        """
        Get the score of a strategy from the Presto-Color perspective.
        
        Parameters
        ----------
        result : dataframe
            Dataframe that contains the results of the observations. The comlums include
            t: the time of the observation
            mag: the detected magnitude
            maglim: the limit of magnitude that can be detected by LSST, fiveSigmaDepth
            filter: the filter used for the observation
            
        HashTable : array
            Contains the values of the 6-D Presto-Color phase space
        InfoDict : dictionary
            Contains the essential information of the HashTable abobe.
        
        HashTable and InfoDict have to be loaded from premade data Presto-Color data file.
        
        scoreType : string
            Two types of scores were designed:
            'S' type involves a threshold, 
            'P' type work without a threshold.
            
        thr : float
            The threashold need for type 'S' score. The default value is 0.003 (3-sigma)
        
        """
        
        TimeLim1 = 8.125/24 # 8 h 7.5 min
        TimeLim2 = 32.25/24 # 32 h 15 min
        
        Detects = result[result.mag<result.maglim]        
        
        # reset index 
        Detects = Detects.reset_index(drop=True)
        
        #global df_detects
        #df_detects = Detects
        
        Ts = Detects.t.values  #Times for valid detections
        dTs = Ts.reshape(1,len(Ts)) - Ts.reshape(len(Ts),1) #Find out the differences between each pair

        dTindex0, dTindex1 = np.where( abs(dTs)<TimeLim2 ) #The time differences should be within 32 hours (2 nights)
        
        phaseSpaceCoords = []
        
        #loop through the rows of the matrix of valid time differences
        for ii in range(dTs.shape[0]):
            
            groupsOfThree = np.array( [ [ii] + list(jj) for jj in list(combinations( dTindex1[ (dTindex0==ii) * (dTindex1>ii) ], 2)) ] )

            for indices in groupsOfThree:

                Bands = Detects["filter"][indices].values
                
                #print('Bands: ', Bands)
                if len(np.unique(Bands)) != 2:
                    continue

                #The band appears once will be Band2  
                occurence = np.array([ np.count_nonzero(ii==Bands) for ii in Bands ]) 

                index2 = indices[occurence==1][0] #The index of observation in Band2
                index11 = indices[occurence==2][0] #The index of the first observation in Band1
                index12 = indices[occurence==2][1] #The index of the second observation in Band1

                if abs(dTs[ index12, index2 ])<abs(dTs[ index11, index2 ]) and abs(dTs[ index12, index2 ])<TimeLim1:
                    index11, index12 = index12, index11
                elif abs(dTs[ index11, index2 ])>TimeLim1:
                    continue

                dT1 = dTs[index11, index2]        
                dT2 = dTs[index11, index12]

                Band1 = Bands[ occurence==2 ][0]
                Band2 = Bands[ occurence==1 ][0]

                dMag = (Detects.mag[index11] - Detects.mag[index12]) * np.sign(dT2)
                Color = Detects.mag[index11] - Detects.mag[index2]

                phaseSpaceCoords.append( [Band1, Band2, dT1, dT2, dMag, Color] )        
                    
        if scoreType == 'S':
            
            score = 0
            for phaseSpaceCoord in phaseSpaceCoords:
                rate = self._enquiry(HashTable, InfoDict, *phaseSpaceCoord)
                
                if rate < thr:
                    score = 1
                    break
                    
                return score
            
        elif scoreType == 'P':
            
            scores = []
            
            for phaseSpaceCoord in phaseSpaceCoords:
                rate = self._enquiry(HashTable, InfoDict, *phaseSpaceCoord)
                
                scores.append( ( 1-rate ) )
                
                return max(scores)
            

    def _ztfrest_simple(self, around_peak, mags, t, filters, min_dt=0.125,
                        min_fade=0.3, max_rise=-1., selectRed=False):
        """
        Selection criteria based on rise or decay rate; simplified version of
        the methods employed by the ZTFReST project
        (Andreoni & Coughlin et al., 2021)
        Parameters
        ----------
        around_peak : array
            indexes corresponding to 5sigma detections
        mags : array
            magnitudes obtained interpolating models on the dataSlice
        t : array
            relative times
        filters : array
            filters in which detections happened
        min_dt : float
            minimum time gap between first and last detection in a given band
        min_fade : float
            fade rate threshold (positive, mag/day)
        max_rise : float
            rise rate threshold (negative, mag/day)
        selectRed : bool
            if True, only red 'izy' filters will be considered
        Examples
        ----------
        A transient:
            rising by 0.74 mag/day will pass a threshold max_rise=-0.5
            rising by 0.74 mag/day will not pass a threshold max_rise=-1.0
            fading by 0.6 mag/day will pass a threshold min_fade=0.3
            fading by 0.2 mag/day will not pass a threshold min_fade=0.3
        """
        result = 1

        # Quick check on the number of detected points
        if np.size(around_peak) < self.ptsNeeded:
            return 0
        # Quick check on the time gap between first and last detection
        elif np.max(t[around_peak]) - np.min(t[around_peak]) < min_dt:
            return 0
        else:
            evol_rate = []
            fil = []
            # Check time gaps and rise or fade rate for each band
            for f in set(filters):
                if selectRed is True and not (f in 'izy'):
                    continue
                times_f = t[around_peak][np.where(filters == f)[0]]
                mags_f = mags[around_peak][np.where(filters == f)[0]]
                dt_f = np.max(times_f) - np.min(times_f)
                # Calculate the evolution rate, if the time gap condition is met
                if dt_f > min_dt:
                    evol_rate_f = (np.max(mags_f) - np.min(mags_f)) / (times_f[np.where(mags_f == np.max(mags_f))[0]][0] - times_f[np.where(mags_f == np.min(mags_f))[0]][0])
                    evol_rate.append(evol_rate_f)
                else:
                    evol_rate.append(0)
                fil.append(f)
            if len(evol_rate) == 0:
                return 0
            # Check if the conditions on the evolution rate are met
            if np.max(evol_rate) < min_fade and np.min(evol_rate) > max_rise:
                return 0

        return result

    def _multi_color_detect(self, filters):
        """
        Color-based simple detection criteria: detect at least twice,
        with at least two filters
        """
        result = 1
        # detected in at least two filters
        if np.size(np.unique(filters)) < 2:
            return 0

        return result

    def _red_color_detect(self, filters, min_det=4):
        """
        Detected at least min_det times in either izy colors
        Parameters
        ----------
        filters : array
            filters in which detections happened
        min_det : float or int
            minimum number of detections required in izy bands
        """
        result = 1
        # Number of detected points in izy bands
        n_red_det = np.size(np.where(filters == 'i')[0]) + np.size(np.where(filters == 'z')[0]) + np.size(np.where(filters == 'y')[0])
        # Condition
        if n_red_det < min_det:
            return 0

        return result

    def _blue_color_detect(self, filters, min_det=4):
        """
        Detected at least min_det times in either ugr colors
        Parameters
        ----------
        filters : array
            filters in which detections happened
        min_det : float or int
            minimum number of detections required in ugr bands
        """
        result = 1
        # Number of detected points in ugr bands
        n_blue_det = np.size(np.where(filters == 'u')[0]) + np.size(np.where(filters == 'g')[0]) + np.size(np.where(filters == 'r')[0])
        # Condition
        if n_blue_det < min_det:
            return 0

        return result

    def run(self, dataSlice, slicePoint=None):
        dataSlice.sort(order=self.mjdCol)
        result = {}
        t = dataSlice[self.mjdCol] - self.mjd0 - slicePoint['peak_time']
        mags = np.zeros(t.size, dtype=float)
        
        for filtername in np.unique(dataSlice[self.filterCol]):
            infilt = np.where(dataSlice[self.filterCol] == filtername)
            mags[infilt] = self.lightcurves.interp(t[infilt], filtername,
                                                   lc_indx=slicePoint['file_indx'])
            # Apply dust extinction on the light curve
            A_x = self.Ax1[filtername] * slicePoint['ebv']
            mags[infilt] += A_x

            distmod = 5*np.log10(slicePoint['distance']*1e6) - 5.0
            mags[infilt] += distmod

        # Find the detected points
        around_peak = np.where((t > 0) & (t < 30) & (mags < dataSlice[self.m5Col]))[0]        
        # Filters in which the detections happened
        filters = dataSlice[self.filterCol][around_peak]
        
        #result['multi_detect'] = self._multi_detect(around_peak)
        #result['ztfrest_simple'] = self._ztfrest_simple(around_peak, mags, t,
        #                                                filters,
        #                                                selectRed=False)
        #result['ztfrest_simple_red'] = self._ztfrest_simple(around_peak, mags,
        #                                                    t, filters,
        #                                                    selectRed=True)
        #result['multi_color_detect'] = self._multi_color_detect(filters)
        #result['red_color_detect'] = self._red_color_detect(filters)
        #result['blue_color_detect'] = self._blue_color_detect(filters)
        
        # presto color 
        result['presto_color_detect'] = self._presto_color_detect(around_peak, filters)

        # Export the light curve
        if self.outputLc is True:
            # mags[np.where(mags > 50)[0]] = 99.
            #result['lc'] = [dataSlice[self.mjdCol], mags,
            #                dataSlice[self.m5Col], dataSlice[self.filterCol]]
            
            idx = np.where(mags < 100)[0]
            
            result['lc'] = {'t': dataSlice[self.mjdCol][idx], 
                            'mag': mags[idx],
                            'maglim': dataSlice[self.m5Col][idx], 
                            'filter': dataSlice[self.filterCol][idx] }
            
            result['slicePoint'] = slicePoint
            
            #### 
            if result['presto_color_detect'] == 1:
                result['score'] = self._getScore( pd.DataFrame(result['lc']), HashTable=self.HashTable, InfoDict=self.InfoDict, scoreType='S', thr=0.003)
            else:
                result['score'] = None
            # mag: interpolated mag at filter
            # maglim: observtion fiveSigmaDepth
            #print(slicePoint, )
            #print('presto detect', result['presto_color_detect'], 'score', result['score'])
        return result

    #def reduce_multi_detect(self, metric):
    #    return metric['multi_detect']

    #def reduce_ztfrest_simple(self, metric):
    #    return metric['ztfrest_simple']

    #def reduce_ztfrest_simple_red(self, metric):
    #    return metric['ztfrest_simple_red']

    #def reduce_multi_color_detect(self, metric):
    #    return metric['multi_color_detect']

    #def reduce_red_color_detect(self, metric):
    #    return metric['red_color_detect']

    #def reduce_blue_color_detect(self, metric):
    #    return metric['blue_color_detect']
    
    def reduce_presto_color_detect(self, metric):
        return metric['presto_color_detect']
    
    def reduce_getScore(self, metric):
        return metric['score']

