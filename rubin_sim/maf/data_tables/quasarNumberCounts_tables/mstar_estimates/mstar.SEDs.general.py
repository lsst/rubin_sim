#!/usr/bin/env python

import numpy as np
import astropy.units as u
from astropy.constants import c
from astropy.cosmology import Planck13 as cosmo
from astropy.table import Table

#This convoluted way of importing pysynphot is so that it does not generate a warning and take a long time to load. This happens because some tasks of pysynphot expect a number of files installed in the path 'PYSYN_CDBS' to work correctly. None of those tasks are being used in this script.
import os
os.environ['PYSYN_CDBS'] = "."
import warnings
warnings.simplefilter("ignore")
import pysynphot as S

#Module with the implementation of the QLF models.
import sys
sys.path.append("../QLFs/")


def load_SED(sed):

    if sed=='Richards06':
        #Load the blue quasar spectrum from Richards et al. (2006).
        qso_spec = Table.read("SEDs/Richards_06.dat",format='ascii.cds')
        nu    = 10.**qso_spec['LogF'].data * u.Hz
        nuLnu = 10.**qso_spec['Blue'].data * 1e-7 * u.W
        fnu   = (nuLnu/nu)/(4.*np.pi*(10*u.pc)**2)
        lam   = (c/nu).to(u.AA)
        flam  = (fnu * c/lam**2).to(u.erg/u.s/u.cm**2/u.AA).value
        wave  = (lam).to(u.AA).value

    elif sed=='vandenberk':
        #Load the vanden Berk et al. (2001).
        qso_spec = Table.read("SEDs/vandenberk_composite.txt",format='ascii.cds')
        wave=qso_spec['Wave']
        flam=qso_spec['FluxD']*1e-17

    #Load the pysynphot ArraySpectrum.
    qso = S.ArraySpectrum(wave=wave, waveunits='angstrom', flux=flam,   fluxunits='flam')

    return qso

"""
This script creates the table mstar_z.SED_model.qlf_module.qlf_model.dat, which holds the apparent magnitude of an L* quasar as a function of redshift for all the LSST bands, as well as the absolute magnitude at 1450A, M_1450, and at i-band.

All output magnitudes are in the AB system.

Command line arguments:
-----------------------
SED_model: str
    Can be Richards06 or vandenberk

qlf_module: str
    Can be Hopkins07, Shen20, or any mother module in the QLFs folder.

model: str
    QLF model to use (mostly Full for Hopkins07, and A or B for Shen20).

"""

if len(sys.argv)!=4:
    print("Correct use: python",sys.argv[0],"SED_model qlf_module model")
    sys.exit()
SED_model  = sys.argv[1]
qlf_module = sys.argv[2]
qlf_model  = sys.argv[3]

exec("import {}".format(qlf_module))

#Create the QLF object.
exec("qlf = {0}.QLF(model=\"{1}\")".format(qlf_module, qlf_model))

#Now, load the mean quasar SED.
qso = load_SED(SED_model)

# If Shen20, use the 1450A UV band as the calibration band. If Hopkins, use B-band instead, as they do not give the bolometric correction for 1450A.
if qlf_module == "Shen20":
    #Create the UV band. Following the description in Table 1 of Shen et al. (2020), the filter is a box-shaped filter with a width of 100A centered at 1450A. Note that since we want the filter to cover the rest-frame 1450 angstrom flux density, we will need to redshift it with the spectrum.

    #The next 3 lines make a synthetic filter curve covering 1350A to 1550A, set to 1 between 1400 and 1500A and 0 otherwise.
    lam_rest_cal = np.arange(1350., 1550. , 1.)*u.AA
    R_cal = np.zeros(len(lam_rest_cal))
    R_cal[(lam_rest_cal>=1400.*u.AA) & (lam_rest_cal<=1500.*u.AA)] = 1
    cal_band_name = "1450"
    nu_cal = c/(1450.*u.AA)
    qlf_Lcal = qlf.L1450

elif qlf_module == "Hopkins07":
    #Read the B-band filter curve.
    B_curve    = np.loadtxt("Filter_Curves/B_bessell.filter", skiprows=1)
    lam_rest_cal = B_curve[:,0] * u.AA
    R_cal        = B_curve[:,1]
    cal_band_name = "B"
    nu_cal = c/(4380.*u.AA)
    qlf_Lcal = qlf.L_B

#Load the LSST filter curves as pysynphot ArrayBandpass objects. These filtercurves were downloaded from SVO: http://svo2.cab.inta-csic.es/svo/theory//fps3/index.php?&mode=browse&gname=LSST&gname2=LSST
filters = ['u', 'g', 'r', 'i', 'z', 'y']
filtercurve = dict()
for filter in filters:
    data = np.loadtxt("Filter_Curves/LSST_LSST.{}.dat".format(filter), skiprows=1)
    filtercurve[filter] = S.ArrayBandpass(data[:,0], data[:,1], name="{}band".format(filter))

#Calculate the m_cal - m_i color at z=0 so that we can easily transform M_cal to M_i later in the code. Note that M_i for an L* quasar is a needed quantity to replicate Table 10.2 of https://www.lsst.org/sites/default/files/docs/sciencebook/SB_10.pdf. See code Table10_2.py for further details.
Cal_band = S.ArrayBandpass(lam_rest_cal.to(u.AA).value, R_cal, name=cal_band_name)
obs_cal  = S.Observation(qso, Cal_band, binset=qso.wave)
obs_i    = obs = S.Observation(qso, filtercurve['i'], binset=qso.wave)
#We define color_cal_i = M_cal - M_i
color_cal_i = obs_cal.effstim('abmag') - obs_i.effstim('abmag')

#Redshift grid.
zmin = 0.01
zmax = 7.0
dz = 0.01
zs = np.arange(zmin, zmax+0.1*dz, dz)

#For each redshift bin, calculate the apparent magnitude of an L* quasar in a each LSST band.
mstar = np.zeros((len(zs),len(filters)))
M_cal = np.zeros(len(zs))
M_i = np.zeros(len(zs))
for k,z in enumerate(zs):

    #Redshift the qso template to redshift z. Note that qso is an ArraySpectrum pysynphot object, so we use the pysynphot.ArraySpectrum.redshift() method to create a redshifted version of the spectrum.
    qso_z = qso.redshift(z)

    #Setup the redshifted UV band as a pysynphot.ArrayBandpass object.
    Cal_band = S.ArrayBandpass(lam_rest_cal.to(u.AA).value*(1.+z), R_cal, name=cal_band_name)

    #Get the bolometric luminosity of an L* quasar.
    Lbol = 10.**(qlf.log_Lstar(z))*qlf.Lstar_units

    #Get the 1450A monochromatic luminosity associated to the L* quasar given its bolometric luminosity. See Shen20.L1450() method documentation for further details.
    Lcal = qlf_Lcal(Lbol)

    #Tranform the monochromatic luminosity into a luminosity density at 1450A, which is defined as nu * Lnu (nu=1450A) in Shen et al. (2020).

    Lnu_cal = Lcal/nu_cal

    #Get the AB absolute magnitude in the UV band.
    #The flux density, fnu, observed at a luminosity distance DL for a source of luminosity density Lnu is given by:
    #
    # Lnu = (4pi DL^2)/(1+z) * fnu
    #
    # This corresponds to eqn. (6) of Hogg et al. (2002, arXiv:astro-ph/0210394). See that reference for further discussion.
    #
    # So at 10pc, we can assume that the redshift is 0, and we have that the observed flux density would be:
    #
    # Fnu = Lnu / [4pi (10pc)^2]
    Fnu_at_10pc = Lnu_cal / (4.*np.pi*(10.*u.pc)**2)
    #Now, the AB magnitude is just m = -2.5 log10 ( fnu / 3631 Jy) by definition. So the AB absolute magnitude at 1450A is just:
    M_cal[k] = -2.5*np.log10( Fnu_at_10pc / (3631*u.Jy) )

    #And we can easily get the absolute i band magnitude of the L* quasar by using the color term color_1450_i = M_1450 - M_i estimated earlier.
    M_i[k] = M_cal[k] - color_cal_i

    #The task now is to renormalize our redshifted spectrum so that it has a 1450A monochromatic luminosity equal to that of an L* quasar.
    #Start by transforming the 1450A luminosity density of the L* quasar to the observed flux density, using the equation listed above.
    DL = cosmo.luminosity_distance(z)
    fnu_cal = (Lnu_cal * (1.+z) / (4.*np.pi*DL**2) )

    #Transform fnu_1450 to be in units of erg / s /cm^2 / Hz and then strip the unit markers. This is important because the renorm method of the redshifted spectrum object expects this units for the flux density provided, but it will not accept astropy units, only floats.
    fnu_cal = fnu_cal.to(u.erg/u.s/u.cm**2/u.Hz)
    #Strip the units.
    fnu_cal = fnu_cal.value

    #Renormalize the redshifted qso spectrum to have a flux_density of fnu_1450 in the UV band. For this, we used the renorm method of the pysynphot.ArraySpectrum class.
    qso_z_renorm = qso_z.renorm(fnu_cal, 'fnu', Cal_band)
    #obs_1450 = S.Observation(qso_z_renorm, UV_band, binset=qso_z_renorm.wave)

    #Finally, convolve the redshifted quasar template with the LSST filter curves to obtain the observed magnitudes of the L* quasar.
    for j, filter in enumerate(filters):
        try:
            obs = S.Observation(qso_z_renorm, filtercurve[filter], binset=qso_z_renorm.wave)
            mstar[k,j] = obs.effstim('abmag')
        except (S.exceptions.PartialOverlap,S.exceptions.DisjointError):
            #If the spectrum does not overlap or only partially overlaps with the filter curve, set the magnitude to 99. to indicate we could not estimate it.
            mstar[k,j] = np.nan

#Print the apparent magnitudes of an L* quasar as a function of redshift.
output = np.zeros((len(zs),len(filters)+3))
output[:,0] = zs
output[:,1:len(filters)+1]  = mstar
output[:,-2] = M_cal
output[:,-1] = M_i
table_file = open("mstar_z.{0}.{1}.{2}.dat".format(SED_model, qlf_module, qlf_model),"w")
table_file.write("#Redshift")
for filter in filters:
    table_file.write("\tLSST{}".format(filter))
table_file.write("\tM_{}".format(cal_band_name))
table_file.write("\tM_i\n")
np.savetxt(table_file,output,fmt='%15.5f')
table_file.close()

#Make a plot of the magnitude of an L* quasar as a function of redshift for each of the LSST bands.
import matplotlib.pyplot as plt
for j,filter in enumerate(filters):
    cond = (~np.isnan(mstar[:,j]))
    plt.plot(zs[cond],mstar[cond,j],label='lsst'+filter)
plt.legend()
plt.ylim([13.,25.])
plt.xlabel('Redshift')
plt.ylabel('Observed magnitude of L* quasar (AB)')
plt.title('{0} SED, {1} QLF, {2} model'.format(SED_model, qlf_module, qlf_model))
plt.savefig("mstar_z.{0}.{1}.{2}.png".format(SED_model, qlf_module, qlf_model))
