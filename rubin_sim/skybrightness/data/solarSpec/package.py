import os

import lsst.sims.photUtils.Sed as Sed
import numpy as np

dataDir = os.getenv("SIMS_SKYBRIGHTNESS_DATA_DIR")

data = np.genfromtxt(
    os.path.join(dataDir, "solarSpec/solarSpec.dat"),
    dtype=list(zip(["microns", "Irr"], [float] * 2)),
)
# #convert W/m2/micron to erg/s/cm2/nm (HA, it's the same!)
# data['Irr'] = data['Irr']*1

sun = Sed()
sun.setSED(data["microns"] * 1e3, flambda=data["Irr"])

# Match the wavelength spacing and range to the ESO spectra
airglowSpec = np.load(os.path.join(dataDir, "ESO_Spectra/Airglow/airglowSpectra.npz"))
sun.resampleSED(wavelen_match=airglowSpec["wave"])

np.savez(os.path.join(dataDir, "solarSpec/solarSpec.npz"), wave=sun.wavelen, spec=sun.flambda)
