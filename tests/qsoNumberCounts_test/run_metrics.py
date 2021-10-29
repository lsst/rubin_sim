import os
CWD = os.getcwd()

import sys
sys.path = [CWD+"/../.."]+sys.path

import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metricBundles as metricBundles
from rubin_sim.maf.metrics import QSONumberCountsMetric

from opsimUtils import *
from ExgalM5_with_cuts_AGN import ExgalM5_with_cuts_AGN
from script_utils import find_completed_runs

#We will use the same slicer and constraint for each metric. 
NSIDE=64
slicer = slicers.HealpixSlicer(nside=NSIDE)
constraint = 'note not like "DD%"' #remove DDFs

#Set the metrics. Use the i-band for both. 
filter = 'i'
metric_em5  = ExgalM5_with_cuts_AGN(lsstFilter=filter)
metric_nqso = QSONumberCountsMetric(lsstFilter=filter)

#Setup the metric bundles.
constraint_use = 'filter = "{}" and '.format(filter) + constraint
EM5 = metricBundles.MetricBundle(metric_em5 , slicer, constraint_use)
NQSO = metricBundles.MetricBundle(metric_nqso, slicer, constraint_use)

#Setup the bundle dictionary
bundleDict = dict()
bundleDict['EM5'] = EM5
bundleDict['NQSO'] = NQSO

#Setup the output folders.
your_username = "rjassef"
folder_mafoutput = "NQSO_test_{0:d}_v3".format(NSIDE)
outDir = '/home/idies/workspace/Temporary/{0}/scratch/MAFOutput/{1}'.format(your_username, folder_mafoutput)
if not os.path.exists(os.path.abspath(outDir)):
    os.mkdir(os.path.abspath(outDir))

resultDbPath  = '/home/idies/workspace/Temporary/{0}/scratch/MAFOutput/{1}'.format(
    your_username, folder_mafoutput)
metricDataPath = '/home/idies/workspace/Temporary/{0}/scratch/MAFOutput/{1}/MetricData/'.format(
    your_username, folder_mafoutput)

#Find the list of completed runs.
n_metrics = 2
completed_runs = find_completed_runs(n_metrics, resultDbPath, metricDataPath)

#Run all OpSims in FBS 1.5, 1.6 and 1.7
FBS_versions = ["1.5", "1.6", "1.7"]
for FBS_version in FBS_versions:
    dbDir = '/home/idies/workspace/lsst_cadence/FBS_{}/'.format(FBS_version)
    opSimDbs, resultDbs = connect_dbs(dbDir, outDir)
    dbRuns = show_opsims(dbDir)
    for run in dbRuns:
        if run in completed_runs:
            continue
        EM5.setRunName(run)
        NQSO.setRunName(run)
        metricGroup = metricBundles.MetricBundleGroup(bundleDict,\
                            opSimDbs[run], metricDataPath, resultDbs[run])
        metricGroup.runAll()