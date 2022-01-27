# Metrics File Name: YoungStellarObjectsMetric.py

# Script description 
This metrics computes the number of young stellar objects (YSO) with age t<10 Myrs and mass M>0.3 Msun.
Four Opsim surveys are considered. The total counts of YSOs are estimated for each OpSim, assuming to observe only with gri filters. 


## How to use

Within DataLab  LSST-2021-10-13 Py3 Kernel the code can be virified with a Notebook using the following commands

import YoungStellarObjectsMetric
YoungStellarObjectsMetric.run_examples_datalab()


## Notes 

Update version of the Young Stellar Objects metrics. No additional software dependencies are required since the code automatically downloads the repository and the 3D extinction map.  The Dust Map name has been changed to avoid collisions with the default LSST MAF Dust map.  The metrics can be tested with a Notebook using the following commands:

import YoungStellarObjectsMetric
YoungStellarObjectsMetric.run_examples_datalab()

The notebook demonstrating the use of the metrics is  added to rubin_sim_notebooks.
Please, contact  loredana.prisinzano@inaf.it for  updates  about the process of integration of this metrics into MAF and
if you have any questions or need further info.
