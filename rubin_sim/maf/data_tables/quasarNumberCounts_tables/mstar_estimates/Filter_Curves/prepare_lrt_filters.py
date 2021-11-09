#!/usr/bin/env python

import numpy as np
import subprocess

filters = ['u', 'g', 'r', 'i', 'z', 'y']
bands_file = open("bandmag.dat","w")
for filter in filters:

    in_fname = "LSST_LSST.{}.dat".format(filter)
    out_fname = "LSST{}.filter".format(filter)

    filt = np.loadtxt(in_fname)
    out_file = open(out_fname,"w")
    out_file.write("{}\n".format(filt.shape[0]))
    np.savetxt(out_file, filt, fmt='%10.1f %15.7f')
    out_file.close()

    bands_file.write("{0:15s} 3   3631.0\n".format("LSST"+filter))

bands_file.close()

subprocess.call("mv LSST?.filter ~/.lrt/Filters/",shell=True)
