import numpy as np
import astropy.units as u

from opsimUtils import *

def find_completed_runs(n_metrics, resultDbPath, metricDataPath, bundleDicts=None):
    
    # get a dictionary of resultDb from the results directory.
    resultDbs = getResultsDbs(resultDbPath)
    
    #Load the metrics that have been completed if that dictionary has not been provided as input.
    if bundleDicts is None:
        bundleDicts = dict()
        for runName in resultDbs:
            bundleDicts[runName] = bundleDictFromDisk(resultDbs[runName], runName, metricDataPath)
    
    #Find the ones that have been completed so we do not run them again.
    completed_runs = list()
    for key in list(bundleDicts.keys()):
        if len(list(bundleDicts[key].keys()))==n_metrics:
            completed_runs.append(key)
    return completed_runs

def get_opsim_areas(key, bds, wfd_mask=None):
    runs = list(bds.keys())
    area = np.zeros(len(runs))
    for k, run in enumerate(runs):
        mb = bds[run][key]
        pix_area = (mb.slicer.pixArea * u.sr).to(u.deg**2).value
        mask = mb.metricValues.mask
        if wfd_mask is not None:
            mask = mask | wfd_mask[run]
        area[k] = len(mask[~mask])*pix_area
    return area


def get_wfd_mask(folder_mafoutput, resultDbPath, metricDataPath):
    
    # get a dictionary of resultDb from given directory
    resultDbs = getResultsDbs(resultDbPath)
    
    # retrieve metricBundles for each opsim run and store them in a dictionary
    bundleDicts = dict()
    for runName in resultDbs:
        bundleDicts[runName] = bundleDictFromDisk(resultDbs[runName], runName, metricDataPath)
        
    
    #Load the WFD footprint for each OpSim run as a mask. 
    Key = (1, 'nvisitsLong')
    wfd_mask = dict()
    for run in resultDbs:
        wfd_footprint = bundleDicts[run][Key].metricValues.filled(0)
        wfd_mask[run] = np.where(wfd_footprint > 750, False, True)
        
    return wfd_mask
    
    