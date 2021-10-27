# -*- coding: utf-8 -*-"""""

from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
 
import lsst.sims.maf.utils.astrometryUtils 
import lsst.sims.maf.db.opsimDatabase
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.metricBundles as metricBundles
import lsst.sims.maf.batches
import lsst.sims.maf.db as db
from lsst.sims.maf.utils import m52snr
import pandas as pd
import mafContrib
import gatspy
from scipy.interpolate import interp1d
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
from pandas import DataFrame
import os
from os import path
from astropy.table import Table
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from SaturationStacker import SaturationStacker

    # input:
    # ra: right ascension of the target. It is a double in degrees
    # dec: declination of the target. It is a double in degrees
    # Ebv: reddening. It is a double
    # Runname: this is the RunName for the Bundle. It is a string. For example: 'footprint_big_wfdv1.5_10yrs.db'
    # Dbfile: this is the file of the simulation. It is a string. For example: '/sims_maf/fbs_1.5/footprints/footprint_big_wfdv1.5_10yrs.db'
    # yearsStar:  an integer describing the start year for simulation. 
    # yearsFinish:  an integer describing the final year for simulation.
    # filenameModel: name of the file with  the theoretical model (before the point)
    # numberOfSigmaToNoisingLc: the number of sigma used to generate the noise for the simulated light curve. It is an integer:if 0 the lc will be not noised
    # outputDir: the directory where to store the generated figure with the light curves. It is a string.
    # optionPlot: boolean (True, False) - true if you want plots of phased light curves.
    # outputDir2: the directory where to store the generated file ascii with the temporal series. It is a string.
    # optionFile: boolean (True, False) - true if you want plots of phased light curves.
    # modelName: name of the file with  the theoretical model (after the point)
    # do_remove_saturated: boolean (True, False) - true if you want to remove from the plots the saturated visits (computed with saturation_stacker)
    # perc_blend: percentage of mean flux that will be added to flux in each band and visit. 0 = no blend
    
    # 
    # 
    # Output: 
    # Temporal series (ascii file in outputDir2 (MJD, theorical mag, simulated mag, flag_sat (1 if saturated), flag_det (1 if s/n <5) ), Light curve (multipanel plot in outputDir) and a  dictionary:
    #
    # LC={'timeu':epoch in u band,'timeg':epoch in g,
    #            'timer':epoch in r,'timei':epoch in i,
    #            'timez':epoch in z,
    #            'timey':epoch in y,'magu': magnitude in u band,'magg': magnitude in g band,
    #            'magr':magnitude in r band,'magi':magnitude in i band,
    #            'magz':magnitude in z band,'magy':magnitude in y band,
    #            'phaseu': phase in u band,'phaseg':phase in g band,'phaser':phase in r band,
    #            'phasei':phase in i band,'phasez':phase in i band,'phasey':phase in y band,
    #            'mag_all':list of the simulated magnitude for all bands sorted chronologically, 'time_all':list of the mean epochs for all the observations sorted chronologically,
    #            'dmag_all':list of the noise simulated at one sigma for all the observations sorted chronologically',
    #             noise_all':list of the computed noise (dmag * random double between -1 and 1) for all the observations sorted chronologically}
    # 
    # mv SQL output from OpSim query.
    
    

def main(ra,dec,distanceMod,Ebv,RunName,Dbfile,yearsStart,
         yearsFinish,filenameModel,numberOfSigmaToNoisingLc,outputDir,optionPlot,
         outputDir2,optionFile,modelName,do_remove_saturated,label,perc_blend):
    nSigma=numberOfSigmaToNoisingLc
    distMod=distanceMod
    ebv=Ebv
    ra_Target = [ra]
    dec_Target=[dec]
    filenameForModel=filenameModel
    #lcTheoric=ReadTeoSim(filenameForModel,distMod,ebv)

    opsdb = db.OpsimDatabase(Dbfile)
    resultsDb = db.ResultsDb(outDir=outputDir)
    version=lsst.sims.maf.db.opsimDatabase.testOpsimVersion(Dbfile, driver='sqlite', host=None, port=None)
    print('Version of db :' +str(version))
    
    #SQL
    sql = 'night between %d and %d' % (365.25 * yearsStart, 365.25 * yearsFinish)
    
    slicer=slicers.UserPointsSlicer(ra=ra_Target,dec=dec_Target, radius=1.75)
    metric=metrics.PassMetric(cols=['observationStartMJD', 'fiveSigmaDepth', 'filter', 
                                'fieldRA', 'night','visitExposureTime','numExposures',
                               'skyBrightness','seeingFwhmEff','airmass'])
    bundle = metricBundles.MetricBundle(metric, slicer, sql, runName=RunName)
    bgroup = metricBundles.MetricBundleGroup(
    {0: bundle}, opsdb, outDir=outputDir, resultsDb=resultsDb)
    bgroup.runAll()
    mv = bundle.metricValues[0]
    filters = np.unique(mv['filter'])
    colors = {'u': 'b','g': 'g','r': 'r',
          'i': 'purple',"z": 'y',"y": 'magenta'}
    

    print('%i Observations total at this sky position (All SNR levels)' % (
        bundle.metricValues.data[0].size))

#read the model of variable star and introduce  the effect of the blending  if perc_blend is not equal to zero
    lcTheoric=ReadTeoSim(filenameForModel,distMod,ebv)
    lcTheoric_blend=ReadTeoSim_blend(lcTheoric,perc_blend)  
    
#SNR retrieving from mv.

    snr=retrieveSnR(mv,lcTheoric)       
    
#definition of time and filters from simulation and add the noise

    time_lsst=np.asarray(mv['observationStartMJD']+mv['visitExposureTime']/2.)
    filters_lsst=np.asarray(mv['filter'])

    LcTeoLSST=generateLC(time_lsst,filters_lsst,lcTheoric_blend)
    LcTeoLSST_noised=noising(LcTeoLSST,snr,nSigma,perc_blend)
    
    index_notsaturated=count_saturation(mv,LcTeoLSST_noised,do_remove_saturated)
    saturation_index_u=[1]*len(LcTeoLSST['timeu'])
    saturation_index_g=[1]*len(LcTeoLSST['timeg'])
    saturation_index_r=[1]*len(LcTeoLSST['timer'])
    saturation_index_i=[1]*len(LcTeoLSST['timei'])
    saturation_index_z=[1]*len(LcTeoLSST['timez'])
    saturation_index_y=[1]*len(LcTeoLSST['timey'])
        
    for i in index_notsaturated['ind_notsaturated_u']:
        saturation_index_u[i]=0
    for i in index_notsaturated['ind_notsaturated_g']:
        saturation_index_g[i]=0
    for i in index_notsaturated['ind_notsaturated_r']:
        saturation_index_r[i]=0
    for i in index_notsaturated['ind_notsaturated_i']:
        saturation_index_i[i]=0
    for i in index_notsaturated['ind_notsaturated_z']:
        saturation_index_z[i]=0
    for i in index_notsaturated['ind_notsaturated_y']:
        saturation_index_y[i]=0

    detection_index_u=[0]*len(LcTeoLSST['timeu'])
    detection_index_g=[0]*len(LcTeoLSST['timeg'])
    detection_index_r=[0]*len(LcTeoLSST['timer'])
    detection_index_i=[0]*len(LcTeoLSST['timei'])
    detection_index_z=[0]*len(LcTeoLSST['timez'])
    detection_index_y=[0]*len(LcTeoLSST['timey'])
        
    ind_detection_u=(np.where(snr['u'] < 5.))[0]
    ind_detection_g=(np.where(snr['g'] < 5.))[0]
    ind_detection_r=(np.where(snr['r'] < 5.))[0]
    ind_detection_i=(np.where(snr['i'] < 5.))[0]
    ind_detection_z=(np.where(snr['z'] < 5.))[0]
    ind_detection_y=(np.where(snr['y'] < 5.))[0]

    for i in ind_detection_u:
        detection_index_u[i]=1
    for i in ind_detection_g:
        detection_index_g[i]=1
    for i in ind_detection_r:
        detection_index_r[i]=1
    for i in ind_detection_i:
        detection_index_i[i]=1
    for i in ind_detection_z:
        detection_index_z[i]=1
    for i in ind_detection_y:
        detection_index_y[i]=1  

#dizionario simulazioni sim1, sim2...
    
    if optionFile:
        
        if not os.path.exists(outputDir2):
            os.makedirs(outputDir2)
        
        wfile= open(outputDir2+label+'.csv','w')
        wfile.write('HJD,mag_theo,mag_sim,filter,saturation,detection\n')
        for (a,b,c,d,e) in zip(LcTeoLSST['timeu'],LcTeoLSST['magu'],LcTeoLSST_noised['magu'],saturation_index_u,detection_index_u):
            wfile.write("{0:13.5f},{1:8.3f},{2:8.3f},u,{3:1d},{4:1d}\n".format(a,b,c,d,e))
        for (a,b,c,d,e) in zip(LcTeoLSST['timeg'],LcTeoLSST['magg'],LcTeoLSST_noised['magg'],saturation_index_g,detection_index_g):
            wfile.write("{0:13.5f},{1:8.3f},{2:8.3f},g,{3:1d},{4:1d}\n".format(a,b,c,d,e))
        for (a,b,c,d,e) in zip(LcTeoLSST['timer'],LcTeoLSST['magr'],LcTeoLSST_noised['magr'],saturation_index_r,detection_index_r):
            wfile.write("{0:13.5f},{1:8.3f},{2:8.3f},r,{3:1d},{4:1d}\n".format(a,b,c,d,e))
        for (a,b,c,d,e) in zip(LcTeoLSST['timei'],LcTeoLSST['magi'],LcTeoLSST_noised['magi'],saturation_index_i,detection_index_i):
            wfile.write("{0:13.5f},{1:8.3f},{2:8.3f},i,{3:1d},{4:1d}\n".format(a,b,c,d,e))
        for (a,b,c,d,e) in zip(LcTeoLSST['timez'],LcTeoLSST['magz'],LcTeoLSST_noised['magz'],saturation_index_z,detection_index_z):
            wfile.write("{0:13.5f},{1:8.3f},{2:8.3f},z,{3:1d},{4:1d}\n".format(a,b,c,d,e))
        for (a,b,c,d,e) in zip(LcTeoLSST['timey'],LcTeoLSST['magy'],LcTeoLSST_noised['magy'],saturation_index_y,detection_index_y):
            wfile.write("{0:13.5f},{1:8.3f},{2:8.3f},y,{3:1d},{4:1d}\n".format(a,b,c,d,e))
        wfile.close()

        
#Make the plot    

    if optionPlot:
        if not os.path.exists(outputDir):
            os.makedirs(outputDir)

        plotting_LSST('LC multipanel',lcTheoric_blend['phase'],lcTheoric_blend['u'],lcTheoric_blend['phase'],lcTheoric_blend['g'],
                         lcTheoric_blend['phase'],lcTheoric_blend['r'],lcTheoric_blend['phase'],lcTheoric_blend['i'],lcTheoric_blend['phase'],
                         lcTheoric_blend['z'],lcTheoric_blend['phase'],lcTheoric_blend['y'],
                         LcTeoLSST_noised['phaseu'],LcTeoLSST_noised['magu'],
                         LcTeoLSST_noised['phaseg'],LcTeoLSST_noised['magg'],
                         LcTeoLSST_noised['phaser'],LcTeoLSST_noised['magr'],
                         LcTeoLSST_noised['phasei'],LcTeoLSST_noised['magi'],
                         LcTeoLSST_noised['phasez'],LcTeoLSST_noised['magz'],
                         LcTeoLSST_noised['phasey'],LcTeoLSST_noised['magy'],
                        snr['u'],snr['g'],snr['r'],snr['i'],snr['z'],snr['y'],yearsStart,yearsFinish,outputDir,label)
    return LcTeoLSST,LcTeoLSST_noised,lcTheoric_blend,mv,index_notsaturated

#mv viene sovrascritto dal SaturationStacker


# !!! count of saturation
def count_saturation(mv,LcTeoLSST_noised,do_remove_saturated):
    satStacker = SaturationStacker()
    mv = satStacker.run(mv)
    satlevel_lsst=np.asarray(mv['saturation_mag'])

    ind_mv_u = np.where(mv['filter'] == 'u')
    ind_mv_g = np.where(mv['filter'] == 'g')
    ind_mv_r = np.where(mv['filter'] == 'r')
    ind_mv_i = np.where(mv['filter'] == 'i')
    ind_mv_z = np.where(mv['filter'] == 'z')
    ind_mv_y = np.where(mv['filter'] == 'y')


    if do_remove_saturated:
        ind_notsaturated=(np.where(LcTeoLSST_noised['mag_all'] > satlevel_lsst))[0]
        ind_notsaturated_u=(np.where(LcTeoLSST_noised['magu'] > satlevel_lsst[ind_mv_u]))[0]
        ind_notsaturated_g=(np.where(LcTeoLSST_noised['magg'] > satlevel_lsst[ind_mv_g]))[0]
        ind_notsaturated_r=(np.where(LcTeoLSST_noised['magr'] > satlevel_lsst[ind_mv_r]))[0]
        ind_notsaturated_i=(np.where(LcTeoLSST_noised['magi'] > satlevel_lsst[ind_mv_i]))[0]
        ind_notsaturated_z=(np.where(LcTeoLSST_noised['magz'] > satlevel_lsst[ind_mv_z]))[0]
        ind_notsaturated_y=(np.where(LcTeoLSST_noised['magy'] > satlevel_lsst[ind_mv_y]))[0]

        index_notsaturated={'ind_notsaturated_u':ind_notsaturated_u,'ind_notsaturated_g':ind_notsaturated_g,
                    'ind_notsaturated_r':ind_notsaturated_r,'ind_notsaturated_i':ind_notsaturated_i,
                    'ind_notsaturated_z':ind_notsaturated_z,'ind_notsaturated_y':ind_notsaturated_y,
                    'ind_notsaturated_all':ind_notsaturated}
    else:
        index_notsaturated={'ind_notsaturated_u':np.arange(len(ind_mv_u[0])),
                           'ind_notsaturated_g':np.arange(len(ind_mv_g[0])),
                    'ind_notsaturated_r':np.arange(len(ind_mv_r[0])),'ind_notsaturated_i':np.arange(len(ind_mv_i[0])),
                    'ind_notsaturated_z':np.arange(len(ind_mv_z[0])),'ind_notsaturated_y':np.arange(len(ind_mv_y[0])),
                    'ind_notsaturated_all':np.arange(len(mv['filter']))}
        
    print('Useful (at all S/N and NOT saturated)  Nvisits in ugrizy bands')
    print(len(index_notsaturated['ind_notsaturated_u']),len(index_notsaturated['ind_notsaturated_g']),
          len(index_notsaturated['ind_notsaturated_r']),len(index_notsaturated['ind_notsaturated_i']),
          len(index_notsaturated['ind_notsaturated_z']),len(index_notsaturated['ind_notsaturated_y']))

    return index_notsaturated

#Alternatively if you want all points use this index
#index={'ind_notsaturated_u':range(len(LcTeoLSST_noised['magu'])),'ind_notsaturated_g':range(len(LcTeoLSST_noised['magg'])),'ind_notsaturated_r':range(len(LcTeoLSST_noised['magr'])),'ind_notsaturated_i':range(len(LcTeoLSST_noised['magi'])),'ind_notsaturated_z':range(len(LcTeoLSST_noised['magz'])),'ind_notsaturated_y':range(len(LcTeoLSST_noised['magy']))}



def meanmag_antilog(mag):
    mag=np.asarray(mag)
    flux=10.**(-mag/2.5)
    return (-2.5)*np.log10(sum(flux)/len(flux))
def mag_antilog(mag):
    mag=np.asarray(mag)
    flux=10.**(-mag/2.5)
    return flux
def ReadTeoSim_blend(lcTheoric=[],perc_blend=[0.,0.,0.,0.,0.,0.]):
    flux_u=mag_antilog(lcTheoric['u'])
    flux_g=mag_antilog(lcTheoric['g'])
    flux_r=mag_antilog(lcTheoric['r'])
    flux_i=mag_antilog(lcTheoric['i'])
    flux_z=mag_antilog(lcTheoric['z'])
    flux_y=mag_antilog(lcTheoric['y'])
    fluxblend_u=flux_u+perc_blend[0]*lcTheoric['mean_flux_u']
    fluxblend_g=flux_g+perc_blend[1]*lcTheoric['mean_flux_g']
    fluxblend_r=flux_r+perc_blend[2]*lcTheoric['mean_flux_r']
    fluxblend_i=flux_i+perc_blend[3]*lcTheoric['mean_flux_i']
    fluxblend_z=flux_z+perc_blend[4]*lcTheoric['mean_flux_z']
    fluxblend_y=flux_y+perc_blend[5]*lcTheoric['mean_flux_y']
    u_blend=(-2.5)*np.log10(fluxblend_u)
    g_blend=(-2.5)*np.log10(fluxblend_g)
    r_blend=(-2.5)*np.log10(fluxblend_r)
    i_blend=(-2.5)*np.log10(fluxblend_i)
    z_blend=(-2.5)*np.log10(fluxblend_z)
    y_blend=(-2.5)*np.log10(fluxblend_y)
    meanu_blend=meanmag_antilog(u_blend)
    meang_blend=meanmag_antilog(g_blend)
    meanr_blend=meanmag_antilog(r_blend)
    meani_blend=meanmag_antilog(i_blend)
    meanz_blend=meanmag_antilog(z_blend)
    meany_blend=meanmag_antilog(y_blend)
    amplu=max(u_blend)-min(u_blend)
    amplg=max(g_blend)-min(g_blend)
    amplr=max(r_blend)-min(r_blend)
    ampli=max(i_blend)-min(i_blend)
    amplz=max(z_blend)-min(z_blend)
    amply=max(y_blend)-min(y_blend)
    print('Theoretical amplitudes ugrizy')
    print(amplu,amplg,amplr,ampli,amplz,amply)
    
   
    #    return time_model,phase_model,u_model,g_model,r_model,i_model,z_model,y_model
    output_blend={'time':lcTheoric['time'], 'phase':lcTheoric['phase'],'period':lcTheoric['period'],'u':u_blend, 'g': g_blend, 'r': r_blend, 'i': i_blend, 'z': z_blend, 'y': y_blend,'meanu':meanu_blend,'meang':meang_blend,'meanr':meanr_blend,'meani':meani_blend,'meanz':meanz_blend,'meany':meany_blend,'amplu':amplu,'amplg':amplg,'amplr':amplr,'ampli':ampli,'amplz':amplz,'amply':amply}
    return output_blend
    

def ReadTeoSim(filename,dmod=0.,ebv=0.,t0_input=0.):
    
    time_model=[]
    u_model=[]
    g_model=[]
    r_model=[]
    i_model=[]
    z_model=[]
    y_model=[]
    phase_model=[]

    f=open(filename,'r')
    c=-1
    for line in f:
        c=c+1
        if c==0:
            header=line
            continue
        ll=line.split(',')
        if c==1:
            period_model=float(ll[8])*86400.
            if abs(t0_input)<1e-6:
                time_0=float(ll[0])
            else:
                time_0=t0_input
        time_model.append(float(ll[0]))
        phase_model.append((float(ll[0])-time_0)/period_model % 1)
        u_model.append(float(ll[2])+dmod+1.55607*3.1*ebv)
        g_model.append(float(ll[3])+dmod+1.18379*3.1*ebv)
        r_model.append(float(ll[4])+dmod+1.87075*3.1*ebv)
        i_model.append(float(ll[5])+dmod+0.67897*3.1*ebv)
        z_model.append(float(ll[6])+dmod+0.51683*3.1*ebv)
        y_model.append(float(ll[7])+dmod+0.42839*3.1*ebv)
    f.close()
    
    meanu=meanmag_antilog(u_model)
    meang=meanmag_antilog(g_model)
    meanr=meanmag_antilog(r_model)
    meani=meanmag_antilog(i_model)
    meanz=meanmag_antilog(z_model)
    meany=meanmag_antilog(y_model)
    u_model_flux=mag_antilog(u_model)
    g_model_flux=mag_antilog(g_model)
    r_model_flux=mag_antilog(r_model)
    i_model_flux=mag_antilog(i_model)
    z_model_flux=mag_antilog(z_model)
    y_model_flux=mag_antilog(y_model)
    mean_flux_u=np.mean(u_model_flux)
    mean_flux_g=np.mean(g_model_flux)
    mean_flux_r=np.mean(r_model_flux)
    mean_flux_i=np.mean(i_model_flux)
    mean_flux_z=np.mean(z_model_flux)
    mean_flux_y=np.mean(y_model_flux)
    amplu=max(u_model)-min(u_model)
    amplg=max(g_model)-min(g_model)
    amplr=max(r_model)-min(r_model)
    ampli=max(i_model)-min(i_model)
    amplz=max(z_model)-min(z_model)
    amply=max(y_model)-min(y_model)

    phase_model.append(1.)
    ind_0=phase_model.index(0.)
    u_model.append(u_model[ind_0])
    g_model.append(g_model[ind_0])
    r_model.append(r_model[ind_0])
    i_model.append(i_model[ind_0])
    z_model.append(z_model[ind_0])
    y_model.append(y_model[ind_0])
    
#    return time_model,phase_model,u_model,g_model,r_model,i_model,z_model,y_model
    output={'time':time_model, 'phase':phase_model,'period':period_model,
            'u':u_model, 'g': g_model, 'r': r_model, 'i': i_model, 'z': z_model, 'y': y_model,
            'meanu':meanu,'meang':meang,'meanr':meanr,'meani':meani,'meanz':meanz,'meany':meany,
            'amplu':amplu,'amplg':amplg,'amplr':amplr,'ampli':ampli,'amplz':amplz,'amply':amply,
            'mean_flux_u':mean_flux_u,'mean_flux_g':mean_flux_g,'mean_flux_r':mean_flux_r,'mean_flux_i':mean_flux_i,'mean_flux_z':mean_flux_z,'mean_flux_y':mean_flux_y,}
    return output

def generateLC(time_lsst,filters_lsst,output_ReadLCTeo,period_true=-99,
               ampl_true=1.,phase_true=0.,do_normalize=False):
    
    u_model=output_ReadLCTeo['u']
    g_model=output_ReadLCTeo['g']
    r_model=output_ReadLCTeo['r']
    i_model=output_ReadLCTeo['i']
    z_model=output_ReadLCTeo['z']
    y_model=output_ReadLCTeo['y']
    phase_model=output_ReadLCTeo['phase']
    
    #Se diamo un periodo arbitrario, usare quello (Caso generico), altrimenti
    #si usa il periodo del modello di Marcella.
    if period_true < -90:
        period_final=(output_ReadLCTeo['period'])/86400.
    else:
        period_final=period_true

    if do_normalize:
        meanu=output_ReadLCTeo['meanu']
        meang=output_ReadLCTeo['meang']
        meanr=output_ReadLCTeo['meanr']
        meani=output_ReadLCTeo['meani']
        meanz=output_ReadLCTeo['meanz']
        meany=output_ReadLCTeo['meany']
        amplg=output_ReadLCTeo['amplg']

        #normalizzo ad ampiezza g
        meanmags=[meanu,meang,meanr,meani,meanz,meany]
        mags_model_norm=[ [(u_model-meanu)/amplg],
                     [(g_model-meang)/amplg],
                     [(r_model-meanr)/amplg],
                     [(i_model-meani)/amplg],
                     [(z_model-meanz)/amplg],
                     [(y_model-meany)/amplg] ]
        model_u=interp1d(phase_model,mags_model_norm[0])
        model_g=interp1d(phase_model,mags_model_norm[1])
        model_r=interp1d(phase_model,mags_model_norm[2])
        model_i=interp1d(phase_model,mags_model_norm[3])
        model_z=interp1d(phase_model,mags_model_norm[4])
        model_y=interp1d(phase_model,mags_model_norm[5])
    else:
        model_u=interp1d(phase_model,u_model)
        model_g=interp1d(phase_model,g_model)
        model_r=interp1d(phase_model,r_model)
        model_i=interp1d(phase_model,i_model)
        model_z=interp1d(phase_model,z_model)
        model_y=interp1d(phase_model,y_model)
        meanmags=[0.,0.,0.,0.,0.,0.]

    allmodels=[ model_u,model_g,model_r,model_i,model_z,model_y ]
    
    t_time_0=min(time_lsst)
    
    ind_u=(np.where(filters_lsst == 'u'))[0]
    ind_g=(np.where(filters_lsst == 'g'))[0]
    ind_r=(np.where(filters_lsst == 'r'))[0]
    ind_i=(np.where(filters_lsst == 'i'))[0]
    ind_z=(np.where(filters_lsst == 'z'))[0]
    ind_y=(np.where(filters_lsst == 'y'))[0]

    timeLSSTu=time_lsst[ind_u]
    timeLSSTg=time_lsst[ind_g]
    timeLSSTr=time_lsst[ind_r]
    timeLSSTi=time_lsst[ind_i]
    timeLSSTz=time_lsst[ind_z]
    timeLSSTy=time_lsst[ind_y]
    
    magLSSTu=np.empty(len(ind_u))
    magLSSTg=np.empty(len(ind_g))
    magLSSTr=np.empty(len(ind_r))
    magLSSTi=np.empty(len(ind_i))
    magLSSTz=np.empty(len(ind_z))
    magLSSTy=np.empty(len(ind_y))
    
    def interpola(timeLSST,meanmags,model):
        magLSST=np.empty(len(timeLSST))
        phaselsst=np.empty(len(timeLSST))
        for i in np.arange(len(timeLSST)):
            phase_lsst_temp=((timeLSST[i]-t_time_0)/period_final) % 1.
            phaselsst[i]=phase_lsst_temp
            magLSST[i]=meanmags+ampl_true*model(phase_lsst_temp)
        return phaselsst,magLSST

    phaselsst_u,magLSSTu=interpola(timeLSSTu,meanmags[0],model_u)
    phaselsst_g,magLSSTg=interpola(timeLSSTg,meanmags[1],model_g)
    phaselsst_r,magLSSTr=interpola(timeLSSTr,meanmags[2],model_r)
    phaselsst_i,magLSSTi=interpola(timeLSSTi,meanmags[3],model_i)
    phaselsst_z,magLSSTz=interpola(timeLSSTz,meanmags[4],model_z)
    phaselsst_y,magLSSTy=interpola(timeLSSTy,meanmags[5],model_y)
    
    mag_all=np.empty(len(time_lsst))
    phase_all=np.empty(len(time_lsst))
    time_all=np.empty(len(time_lsst))

    #mag_all è ordinato come time_lsst    
    mag_all[ind_u]=magLSSTu
    mag_all[ind_g]=magLSSTg
    mag_all[ind_r]=magLSSTr
    mag_all[ind_i]=magLSSTi
    mag_all[ind_z]=magLSSTz
    mag_all[ind_y]=magLSSTy

    phase_all[ind_u]=phaselsst_u
    phase_all[ind_g]=phaselsst_g
    phase_all[ind_r]=phaselsst_r
    phase_all[ind_i]=phaselsst_i
    phase_all[ind_z]=phaselsst_z
    phase_all[ind_y]=phaselsst_y

    time_all[ind_u]=timeLSSTu
    time_all[ind_g]=timeLSSTg
    time_all[ind_r]=timeLSSTr
    time_all[ind_i]=timeLSSTi
    time_all[ind_z]=timeLSSTz
    time_all[ind_y]=timeLSSTy
        
    return {'timeu':timeLSSTu,'timeg':timeLSSTg,
                'timer':timeLSSTr,'timei':timeLSSTi,'timez':timeLSSTz,
                'timey':timeLSSTy,'magu':magLSSTu,'magg':magLSSTg,
                'magr':magLSSTr,'magi':magLSSTi,
                'magz':magLSSTz,'magy':magLSSTy,
           'phaseu':phaselsst_u,'phaseg':phaselsst_g,'phaser':phaselsst_r,
           'phasei':phaselsst_i,'phasez':phaselsst_z,'phasey':phaselsst_y,
           'mag_all':mag_all,'phase_all':phase_all,'time_all':time_all,
           'ind_u':ind_u,'ind_g':ind_g,'ind_r':ind_r,
           'ind_i':ind_i,'ind_z':ind_z,'ind_y':ind_y,'period':period_final}

    
def noising(LcTeoLSST,snr,sigma,perc_blend):
    
#noising 
    def noisingBand(timeLSSTteo,magLSSTteo,snr,sigma,blend):
        magNoised=[]
        for j in range(len(timeLSSTteo)):            
            dmag = 2.5*np.log10(1.+1./snr[j])
            if blend >0:
                dmag=np.sqrt(2)*dmag  #non sono sicuro. Forse non bisogna peggiorare il rumore così... perchè quello che si vede è il flusso totale
            noise = np.random.uniform(-sigma,sigma)*dmag
            magNoisedComp=magLSSTteo[j]+noise
            magNoised.append(magNoisedComp)
            
        return magNoised, noise ,dmag

    magNoisedu,noiseu,dmagu=noisingBand(LcTeoLSST['timeu'],LcTeoLSST['magu'],snr['u'],sigma,perc_blend[0])
    magNoisedg,noiseg,dmagg=noisingBand(LcTeoLSST['timeg'],LcTeoLSST['magg'],snr['g'],sigma,perc_blend[1])
    magNoisedr,noiser,dmagr=noisingBand(LcTeoLSST['timer'],LcTeoLSST['magr'],snr['r'],sigma,perc_blend[2])
    magNoisedi,noisei,dmagi=noisingBand(LcTeoLSST['timei'],LcTeoLSST['magi'],snr['i'],sigma,perc_blend[3])
    magNoisedz,noisez,dmagz=noisingBand(LcTeoLSST['timez'],LcTeoLSST['magz'],snr['z'],sigma,perc_blend[4])
    magNoisedy,noisey,dmagy=noisingBand(LcTeoLSST['timey'],LcTeoLSST['magy'],snr['y'],sigma,perc_blend[5])
    
    #mag_all è ordinato come time_lsst
    mag_all=np.empty(len(LcTeoLSST['mag_all']))
    mag_all[LcTeoLSST['ind_u']]=magNoisedu
    mag_all[LcTeoLSST['ind_g']]=magNoisedg
    mag_all[LcTeoLSST['ind_r']]=magNoisedr
    mag_all[LcTeoLSST['ind_i']]=magNoisedi
    mag_all[LcTeoLSST['ind_z']]=magNoisedz
    mag_all[LcTeoLSST['ind_y']]=magNoisedy
    
    
    #noise_all è ordinato come time_lsst
    noise_all=np.empty(len(LcTeoLSST['mag_all']))
    noise_all[LcTeoLSST['ind_u']]=noiseu
    noise_all[LcTeoLSST['ind_g']]=noiseg
    noise_all[LcTeoLSST['ind_r']]=noiser
    noise_all[LcTeoLSST['ind_i']]=noisei
    noise_all[LcTeoLSST['ind_z']]=noisez
    noise_all[LcTeoLSST['ind_y']]=noisey
    
   #mag_all è ordinato come time_lsst
    dmag_all=np.empty(len(LcTeoLSST['mag_all']))
    dmag_all[LcTeoLSST['ind_u']]=dmagu
    dmag_all[LcTeoLSST['ind_g']]=dmagg
    dmag_all[LcTeoLSST['ind_r']]=dmagr
    dmag_all[LcTeoLSST['ind_i']]=dmagi
    dmag_all[LcTeoLSST['ind_z']]=dmagz
    dmag_all[LcTeoLSST['ind_y']]=dmagy
    
    output2={'timeu':LcTeoLSST['timeu'],'timeg':LcTeoLSST['timeg'],
                'timer':LcTeoLSST['timer'],'timei':LcTeoLSST['timei'],
                'timez':LcTeoLSST['timez'],'timey':LcTeoLSST['timey'], 
                'magu':np.asarray(magNoisedu), 'magg':np.asarray(magNoisedg),
                'magr':np.asarray(magNoisedr),'magi':np.asarray(magNoisedi),
                'magz':np.asarray(magNoisedz),'magy':np.asarray(magNoisedy),
                'phaseu':LcTeoLSST['phaseu'],'phaseg':LcTeoLSST['phaseg'],'phaser':LcTeoLSST['phaser'],
                'phasei':LcTeoLSST['phasei'],'phasez':LcTeoLSST['phasez'],'phasey':LcTeoLSST['phasey'],
                'mag_all':mag_all, 'time_all':LcTeoLSST['time_all'],'noise_all':noise_all,'dmag_all':dmag_all}
    return output2


def retrieveSnR(mv,theoreticModel):
    good = np.where(mv['filter'] == 'u')
    sn_u=m52snr(theoreticModel['meanu'],mv['fiveSigmaDepth'][good])
    good = np.where(mv['filter'] == 'g')
    sn_g=m52snr(theoreticModel['meang'],mv['fiveSigmaDepth'][good])
    good = np.where(mv['filter'] == 'r')
    sn_r=m52snr(theoreticModel['meanr'],mv['fiveSigmaDepth'][good])
    good = np.where(mv['filter'] == 'i')
    sn_i=m52snr(theoreticModel['meani'],mv['fiveSigmaDepth'][good])
    good = np.where(mv['filter'] == 'z')
    sn_z=m52snr(theoreticModel['meanz'],mv['fiveSigmaDepth'][good])
    good = np.where(mv['filter'] == 'y')
    sn_y=m52snr(theoreticModel['meany'],mv['fiveSigmaDepth'][good])
        
    snr={'u':sn_u,'g':sn_g,'r':sn_r,'i':sn_i,'z':sn_z,'y':sn_y}
    return snr




def magAndPhaseMod2(zp, frequency,tref,amplitudes, phases):
    
    
    timet=range(int(tref),int(tref*10))
    #finalPhases=np.empty(len(modelPhases))
    #for i in range(len(modelPhases)):
    #    finalPhases[i]=modelPhases[i]/1000
    
    phase=[]
    for i in range(len(timet)):
        phase.append((timet[i]-tref)*frequency-int((timet[i]-tref)*frequency))
    magModel=[]
    for j in range(len(timet)):
        magModelComp=zp
        for i in range(len(amplitudes)):
            magModelComp=magModelComp+amplitudes[i]*np.cos(2*np.pi*(i+1)*frequency*(timet[j]-tref)+phases[i])
           
        #print(magModelComp, j)    
        magModel.append(magModelComp)
   
    
    
    return phase,magModel 

def magAndPhaseMod3(time, zp, frequency,tref,amplitudes, phases):
    
    
    timet=time
    #finalPhases=np.empty(len(modelPhases))
    #for i in range(len(modelPhases)):
    #    finalPhases[i]=modelPhases[i]/1000
    
    phase=[]
    for i in range(len(timet)):
        phase.append((timet[i]-tref)*frequency-int((timet[i]-tref)*frequency))
        
    magModel=[]
    for j in range(len(timet)):
        magModelComp=zp
        for i in range(len(amplitudes)):
            magModelComp=magModelComp+amplitudes[i]*np.cos(2*np.pi*(i+1)*frequency*(timet[j]-tref)+phases[i])
           
        #print(magModelComp, j)    
        magModel.append(magModelComp)
   
    
    
    return phase,magModel 

def plotting_LSST(title,phaseModelu,magModelu,phaseModelg,magModelg,
              phaseModelr,magModelr,phaseModeli,magModeli,phaseModelz,magModelz,phaseModely,magModely,
              phaseLSSTu,magLSSTu,phaseLSSTg,magLSSTg,phaseLSSTr,magLSSTr,
              phaseLSSTi,magLSSTi,phaseLSSTz,magLSSTz,phaseLSSTy,magLSSTy,
                  snru,snrg,snrr,snri,snrz,snry,yearsStart,yearsFinish,outputDir,label):
    
#    phaseLSSTAll=np.concatenate((phaseLSSTu,phaseLSSTg,phaseLSSTr,phaseLSSTi,phaseLSSTz,phaseLSSTy))
#    magLSSTAll=np.concatenate((magLSSTu,magLSSTg,magLSSTr,magLSSTi,magLSSTz,magLSSTy))
    
    if len(magLSSTu)!=0: 
        deltau=max(magLSSTu)-min(magLSSTu)
        meanu=np.mean(magModelu)
    else:
        deltau=.1
        meanu=10
        
    if len(magLSSTg)!=0: 
        deltag=max(magLSSTg)-min(magLSSTg)
        meang=np.mean(magModelg)
    else:
        deltag=.1
        meang=10
        
    if len(magLSSTr)!=0: 
        deltar=max(magLSSTr)-min(magLSSTr)
        meanr=np.mean(magModelr)
    else:
        deltar=.1
        meanr=10
        
    if len(magLSSTi)!=0: 
        deltai=max(magLSSTi)-min(magLSSTi)
        meani=np.mean(magModeli)
    else:
        deltai=.1
        meani=10
        
    if len(magLSSTz)!=0: 
        deltaz=max(magLSSTz)-min(magLSSTz)
        meanz=np.mean(magModelz)
    else:
        deltaz=.1
        meanz=10
        
    if len(magLSSTy)!=0: 
        deltay=max(magLSSTy)-min(magLSSTy)
        meany=np.mean(magModely)
    else:
        deltay=.1
        meany=10
    
    deltamax=max([deltau,deltag,deltar,deltai,deltaz,deltay])


    
    fig=plt.figure(figsize=(10,16), dpi=80)
    plt.subplots_adjust(top = 0.95, bottom = 0.1, right = 0.95
                        , left = 0.1, hspace = 0.08, wspace = 0.2)
    fig.suptitle(title)
   

    ax2 = plt.subplot2grid((4,2), (0,0),  colspan=2, rowspan=1) # topleft    
    ax2.set_ylabel('ALLbands (mag)')
    ax2.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax2.invert_yaxis()
    ax2.set_xlim([0,1])
    colors = {'u': 'b','g': 'g','r': 'r',
          'i': 'purple',"z": 'y',"y": 'magenta'}
    ax2.plot(phaseLSSTu,magLSSTu,'o',color='b')
    ax2.plot(phaseLSSTg,magLSSTg,'o',color='g')
    ax2.plot(phaseLSSTr,magLSSTr,'o',color='r')
    ax2.plot(phaseLSSTi,magLSSTi,'o',color='purple')
    ax2.plot(phaseLSSTz,magLSSTz,'o',color='y')
    ax2.plot(phaseLSSTy,magLSSTy,'o',color='magenta')
    ax2.set_xlabel('phase')

    
    ax3 = plt.subplot2grid((4,2), (1,0),  colspan=1, rowspan=1) # topleft    
    ax3.set_ylabel('u (mag)')
    ax3.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax3.invert_yaxis()
    ax3.set_xlim([0,1])
    ax3.set_ylim([meanu+.7*deltamax,meanu-.7*deltamax])
    ax3.set_xticklabels([])
    ax3.plot(phaseModelu, magModelu,'.',color='gray')
    ax3.plot(phaseLSSTu,magLSSTu,'o',color='b')

    ax4 = plt.subplot2grid((4,2), (2,0),  colspan=1, rowspan=1) # topleft    
    ax4.set_ylabel('g (mag)')
    ax4.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax4.invert_yaxis()
    ax4.set_xlim([0,1])
    ax4.set_ylim([meang+.7*deltamax,meang-.7*deltamax])
    ax4.set_xticklabels([])
    ax4.plot(phaseModelg, magModelg,'.',color='gray')
    ax4.plot(phaseLSSTg,magLSSTg,'o',color='g')
    
    ax5 = plt.subplot2grid((4,2), (3,0),  colspan=1, rowspan=1) # topleft    
    ax5.set_ylabel('r (mag)')
    ax5.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax5.invert_yaxis()
    ax5.set_xlim([0,1])
    ax5.set_ylim([meanr+.7*deltamax,meanr-.7*deltamax])
    ax5.set_xlabel('phase')
    ax5.plot(phaseModelr, magModelr,'.',color='gray')
    ax5.plot(phaseLSSTr,magLSSTr,'o',color='r')
    
    ax6 = plt.subplot2grid((4,2), (1,1),  colspan=1, rowspan=1) # topleft    
    ax6.set_ylabel('i (mag)')
    ax6.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax6.invert_yaxis()
    ax6.set_xticklabels([])
    ax6.set_xlim([0,1])
    ax6.set_ylim([meani+.7*deltamax,meani-.7*deltamax])
    ax6.plot(phaseModeli, magModeli,'.',color='gray')
    ax6.plot(phaseLSSTi,magLSSTi,'o',color='purple')
    
    ax7 = plt.subplot2grid((4,2), (2,1),  colspan=1, rowspan=1) # topleft    
    ax7.set_ylabel('z (mag)')
    ax7.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax7.invert_yaxis()
    ax7.set_xlim([0,1])
    ax7.set_ylim([meanz+.7*deltamax,meanz-.7*deltamax])
    ax7.plot(phaseModelz, magModelz,'.',color='gray')
    ax7.plot(phaseLSSTz,magLSSTz,'o',color='y')
    ax7.set_xticklabels([])
    
    ax8 = plt.subplot2grid((4,2), (3,1),  colspan=1, rowspan=1) # topleft    
    ax8.set_ylabel('y (mag)')
    ax8.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax8.invert_yaxis()
    ax8.set_xlim([0,1])
    ax8.set_ylim([meany+.7*deltamax,meany-.7*deltamax])
    ax8.plot(phaseModely, magModely,'.',color='gray')
    ax8.plot(phaseLSSTy,magLSSTy,'o',color='magenta')
    ax8.set_xlabel('phase')

    plt.savefig(outputDir+'LCnoised_'+label+'.pdf')   

def plotting_LSST_saturation(title,phaseModelu,magModelu,phaseModelg,magModelg,
              phaseModelr,magModelr,phaseModeli,magModeli,phaseModelz,magModelz,phaseModely,magModely,
              phaseLSSTu,magLSSTu,phaseLSSTg,magLSSTg,phaseLSSTr,magLSSTr,
              phaseLSSTi,magLSSTi,phaseLSSTz,magLSSTz,phaseLSSTy,magLSSTy,
              phaseLSSTu_satlevel,magLSSTu_satlevel,phaseLSSTg_satlevel,magLSSTg_satlevel,phaseLSSTr_satlevel,magLSSTr_satlevel,
              phaseLSSTi_satlevel,magLSSTi_satlevel,phaseLSSTz_satlevel,magLSSTz_satlevel,phaseLSSTy_satlevel,magLSSTy_satlevel,
                  snru,snrg,snrr,snri,snrz,snry,yearsStart,yearsFinish,outputDir):
    
    
    if len(magLSSTu)!=0: 
        deltau=max(magLSSTu)-min(magLSSTu)
        meanu=np.mean(magModelu)
    else:
        deltau=.1
        meanu=10
        
    if len(magLSSTg)!=0: 
        deltag=max(magLSSTg)-min(magLSSTg)
        meang=np.mean(magModelg)
    else:
        deltag=.1
        meang=10
        
    if len(magLSSTr)!=0: 
        deltar=max(magLSSTr)-min(magLSSTr)
        meanr=np.mean(magModelr)
    else:
        deltar=.1
        meanr=10
        
    if len(magLSSTi)!=0: 
        deltai=max(magLSSTi)-min(magLSSTi)
        meani=np.mean(magModeli)
    else:
        deltai=.1
        meani=10
        
    if len(magLSSTz)!=0: 
        deltaz=max(magLSSTz)-min(magLSSTz)
        meanz=np.mean(magModelz)
    else:
        deltaz=.1
        meanz=10
        
    if len(magLSSTy)!=0: 
        deltay=max(magLSSTy)-min(magLSSTy)
        meany=np.mean(magModely)
    else:
        deltay=.1
        meany=10
    
    deltamax=max([deltau,deltag,deltar,deltai,deltaz,deltay])


    
    fig=plt.figure(figsize=(10,16), dpi=80)
    plt.subplots_adjust(top = 0.95, bottom = 0.1, right = 0.95
                        , left = 0.1, hspace = 0.08, wspace = 0.2)
    fig.suptitle(title)
   

    ax2 = plt.subplot2grid((4,2), (0,0),  colspan=2, rowspan=1) # topleft    
    ax2.set_ylabel('ALLbands (mag)')
    ax2.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax2.invert_yaxis()
    ax2.set_xlim([0,1])
    ax2.plot(phaseLSSTu,magLSSTu,'o',color='purple')
    ax2.plot(phaseLSSTg,magLSSTg,'o',color='g')
    ax2.plot(phaseLSSTr,magLSSTr,'o',color='r')
    ax2.plot(phaseLSSTi,magLSSTi,'o',color='k')
    ax2.plot(phaseLSSTz,magLSSTz,'o',color='magenta')
    ax2.plot(phaseLSSTy,magLSSTy,'o',color='y')
    ax2.set_xlabel('phase')

    
    ax3 = plt.subplot2grid((4,2), (1,0),  colspan=1, rowspan=1) # topleft    
    ax3.set_ylabel('u (mag)')
    ax3.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax3.invert_yaxis()
    ax3.set_xlim([0,1])
    ax3.set_ylim([meanu+.7*deltamax,meanu-.7*deltamax])
    ax3.set_xticklabels([])
    ax3.plot(phaseModelu, magModelu,'.')
    ax3.plot(phaseLSSTu,magLSSTu,'o',color='r')

    ax4 = plt.subplot2grid((4,2), (2,0),  colspan=1, rowspan=1) # topleft    
    ax4.set_ylabel('g (mag)')
    ax4.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax4.invert_yaxis()
    ax4.set_xlim([0,1])
    ax4.set_ylim([meang+.7*deltamax,meang-.7*deltamax])
    ax4.set_xticklabels([])
    ax4.plot(phaseModelg, magModelg,'.')
    ax4.plot(phaseLSSTg,magLSSTg,'o',color='r')
    
    ax5 = plt.subplot2grid((4,2), (3,0),  colspan=1, rowspan=1) # topleft    
    ax5.set_ylabel('r (mag)')
    ax5.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax5.invert_yaxis()
    ax5.set_xlim([0,1])
    ax5.set_ylim([meanr+.7*deltamax,meanr-.7*deltamax])
    ax5.set_xlabel('phase')
    ax5.plot(phaseModelr, magModelr,'.')
    ax5.plot(phaseLSSTr,magLSSTr,'o',color='r')
    
    ax6 = plt.subplot2grid((4,2), (1,1),  colspan=1, rowspan=1) # topleft    
    ax6.set_ylabel('i (mag)')
    ax6.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax6.invert_yaxis()
    ax6.set_xticklabels([])
    ax6.set_xlim([0,1])
    ax6.set_ylim([meani+.7*deltamax,meani-.7*deltamax])
    ax6.plot(phaseModeli, magModeli,'.')
    ax6.plot(phaseLSSTi,magLSSTi,'o',color='r')
    
    ax7 = plt.subplot2grid((4,2), (2,1),  colspan=1, rowspan=1) # topleft    
    ax7.set_ylabel('z (mag)')
    ax7.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax7.invert_yaxis()
    ax7.set_xlim([0,1])
    ax7.set_ylim([meanz+.7*deltamax,meanz-.7*deltamax])
    ax7.plot(phaseModelz, magModelz,'.')
    ax7.plot(phaseLSSTz,magLSSTz,'o',color='r')
    ax7.set_xticklabels([])
    
    ax8 = plt.subplot2grid((4,2), (3,1),  colspan=1, rowspan=1) # topleft    
    ax8.set_ylabel('y (mag)')
    ax8.tick_params(axis='both',bottom=True, top=True, left=True, right=True, direction='in',which='major')
    ax8.invert_yaxis()
    ax8.set_xlim([0,1])
    ax8.set_ylim([meany+.7*deltamax,meany-.7*deltamax])
    ax8.plot(phaseModely, magModely,'.')
    ax8.plot(phaseLSSTy,magLSSTy,'o',color='r')
    ax8.set_xlabel('phase')


    plt.savefig(outputDir+'LcConstructor_simulated_noised_allbands_notSaturated'+'_'+str(yearsStart)+'_'+str(yearsFinish)+'.pdf')
