#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#This metric  perform the fit of phased light curve  with a given numberOfHarmonics and compute  the dimension of the max distance from two consecutive phases of the light curve in each band and the  number gaps larger than factorForDimensionGap 
# Input:
# data is a dictionary were the time and mag of all bands are stored.
# data={'timeu':time in u Band , 'timeg':time in g Band ,'timer':time in r Band ,
#       'timei':time in i Band ,'timez':time in z Band ,'timez':time in z Band ,
#       'magu': mag in u Band,'magg': mag in g Band,'magr': mag in r Band,
#       'magi': mag in i Band,'magz': mag in z Band,'magy': mag in y Band}
#
# index is a dictionary where are stored the index of general mag array where the saturation is not present.
# index{'ind_notsaturated_u': index of not saturated measurements for u band,
#      'ind_notsaturated_u': index of not saturated measurements for u band,
#      'ind_notsaturated_g': index of not saturated measurements for g band,
#      'ind_notsaturated_r': index of not saturated measurements for r band,
#      'ind_notsaturated_i': index of not saturated measurements for i band,
#      'ind_notsaturated_z': index of not saturated measurements for z band,}   
#
# period is the period of the lc. It is a double
# numberOfHarmonics is an integer and defines the number of harmonics of the model fit    
# factorForDimensionGap is a double that multiplyes the max distance from two consecutive phases of the light curve. Used to count the gap in 
# the light curve, should be < 1. 
# label is a label to identify the plots of the results.
# outDir is a String identifying the directory where to store the results. 
#
# Output:
# - multi panel plot with the modelled lc, saved in a file in the outputdir
# - a dictinary with the  finalResult:
# finalResult={'mean_u':mean magnitude in u band,'mean_g':mean magnitude in g band,'mean_r':mean magnitude in r band,
#              'mean_i':'mean magnitude in i band,'mean_z':mean magnitude in z band,'mean_y':mean magnitude in y band,
#              'ampl_u':amplitude in u,'ampl_g':amplitude in g band,'ampl_r':amplitude in r band,
#                 'ampl_i':amplitude in i band,'ampl_z':amplitude in z band,'ampl_y':amplitude in y band,
#                 'chi_u': chi squared of the model fitting in u band,'chi_g':chi squared of the model fitting in g band,
#                 'chi_r':chi squared of the model fitting in r band,
#                 'chi_i':chi squared of the model fitting in i band,'chi_z':chi squared of the model fitting in z band,
#                 'chi_y':chi squared of the model fitting in y band,
#                  'fittingParametersAllband': model fitting of all bands dictinary*, 
#                  'maxGapDimension_u': max dimension of the gap in u band,
#                  'maxGapDimension_g': max dimension of the gap in g band, 
#                  'maxGapDimension_r': max dimension of the gap in r band,,
#                  'maxGapDimension_i': max dimension of the gap in i band,,
#                  'maxGapDimension_z': max dimension of the gap in z band,,
#                  'maxGapDimension_y': max dimension of the gap in z band,
#                  'numberOfGaps_u': number of gaps of the light curve in the u band,
#                  'numberOfGaps_g':number of gaps of the light curve in the g band,
#                  'numberOfGaps_r':number of gaps of the light curve in the r band,
#                  'numberOfGaps_i':number of gaps of the light curve in the i band,
#                  'numberOfGaps_z':number of gaps of the light curve in the z band,
#                  'numberOfGaps_y':number of gaps of the light curve in the y band}   
    
# * model fit dictionary:
# results={'u':fit parameters** for u band,'g':fit parameters for g band,'r':fit parameters for r band,
#           'i':fit parameters for i band,'z':fit parameters for z,'y':fit parameters for y band,
#           'chi_u': chi squared of the model fitting in u band,'chi_g':chi squared of the model fitting in g band,
#                 'chi_r':chi squared of the model fitting in r band,
#                 'chi_i':chi squared of the model fitting in i band,'chi_z':chi squared of the model fitting in z band,
#                 'chi_y':chi squared of the model fitting in y band,}  

#** fit parameters: it is an array with dimension= 2*harmonics+2, with this structure= [period,zp,amplitude1,amplitude2,..., phase1,phase2,....]




import numpy as np
from scipy import stats as stat
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

#from astropy.table import Table
def modelToFit(time,param):
    #time in days
    magModel=[]
    amplitudes=[]
    phases=[]
    numberOfHarmonics=int((len(param)-2)/2)
    
    zp=param[1]
    period=param[0]
    for i in range(numberOfHarmonics):
        amplitudes.append(param[2+i])
        phases.append(param[numberOfHarmonics+2+i])
    for i in range(len(time)):
        y=zp
        for j in range(0,int(numberOfHarmonics)):
            y=y+amplitudes[j]*np.cos((2*np.pi/period)*(j+1)*(time[i])+phases[j])
        magModel.append(y)
    return magModel

def chisqr(residual,Ndat,Nvariable):
    chi=sum(pow(residual,2))/(Ndat-Nvariable)
    return chi

def chisqr2(datax,datay,fitparameters,Ndat,Nvariable):
    residuals=modelToFit(datax,fitparameters)-datay
    chi2=sum(pow(residuals,2)/modelToFit(datax,fitparameters))*1/(Ndat-Nvariable)
    return chi2

def residuals(datax,datay,fitparameters):
    residuals=modelToFit(datax,fitparameters)-datay
    return residuals

def meanmag_antilog(mag):
    mag=np.asarray(mag)
    flux=10.**(-mag/2.5)
    if len(flux)>0:
        result=(-2.5)*np.log10(sum(flux)/len(flux))
    else:
        result=9999.
    return result

def computingLcModel(data,period,numberOfHarmonics,index,outDir):
    

    def modelToFit2_fit(coeff):
        fit = modelToFit(x,coeff)
        return (fit - y_proc)
    
    

    time_u=data['timeu'][index['ind_notsaturated_u']]#time must be in days
    time_g=data['timeg'][index['ind_notsaturated_g']]
    time_r=data['timer'][index['ind_notsaturated_r']]
    time_i=data['timei'][index['ind_notsaturated_i']]
    time_z=data['timez'][index['ind_notsaturated_z']]
    time_y=data['timey'][index['ind_notsaturated_y']]
    mag_u=data['magu'][index['ind_notsaturated_u']]
    mag_g=data['magg'][index['ind_notsaturated_g']]
    mag_r=data['magr'][index['ind_notsaturated_r']]
    mag_i=data['magi'][index['ind_notsaturated_i']]
    mag_z=data['magz'][index['ind_notsaturated_z']]
    mag_y=data['magy'][index['ind_notsaturated_y']] 

    
    parametersForLcFit=[period,1] # period,zp
    for i in range(numberOfHarmonics):
        parametersForLcFit.append(1)#added ampl
        parametersForLcFit.append(1)#added phase
    x=time_u
    
    
    
    y_proc = np.copy(mag_u)
    if len(y_proc)>(numberOfHarmonics*2)+2:
        print('fitting u band')
        fit_u,a=leastsq( modelToFit2_fit,parametersForLcFit)
        residual=residuals(x,y_proc,fit_u)
        chi_u=chisqr2(x,y_proc,fit_u,len(x),len(fit_u))
    else:
        fit_u=[9999.]
        chi_u=9999.
    x=time_g
    y_proc = np.copy(mag_g)
    if len(y_proc)>(numberOfHarmonics*2)+2:
        print('fitting g band')
        fit_g,a=leastsq( modelToFit2_fit,parametersForLcFit)
        residual=residuals(x,y_proc,fit_g)
        chi_g=chisqr2(x,y_proc,fit_g,len(x),len(fit_g))
    else:
        fit_g=[9999.]
        chi_g=9999.
    y_proc = np.copy(mag_r)
    x=time_r
    if len(y_proc)>(numberOfHarmonics*2)+2:
        print('fitting r band')
        fit_r,a=leastsq( modelToFit2_fit,parametersForLcFit)
        residual=residuals(x,y_proc,fit_r)
        chi_r=chisqr2(x,y_proc,fit_r,len(x),len(fit_r))
    else:
        fit_r=[9999.]
        chi_r=9999.
    x=time_i
    y_proc = np.copy(mag_i)
    if len(y_proc)>(numberOfHarmonics*2)+2:
        print('fitting i band')
        fit_i,a=leastsq( modelToFit2_fit,parametersForLcFit)
        residual=residuals(x,y_proc,fit_i)
        chi_i=chisqr2(x,y_proc,fit_i,len(x),len(fit_i))
    else:
        fit_i=[9999.]
        chi_i=9999.
    x=time_z
    y_proc = np.copy(mag_z)
    if len(y_proc)>(numberOfHarmonics*2)+2:
        print('fitting z band')
        fit_z,a=leastsq( modelToFit2_fit,parametersForLcFit)
        residual=residuals(x,y_proc,fit_z)
        chi_z=chisqr2(x,y_proc,fit_z,len(x),len(fit_z))
    else:
        fit_z=[9999.]
        chi_z=9999.
    x=time_y
    y_proc = np.copy(mag_y)
    if len(y_proc)>(numberOfHarmonics*2)+2:
        print('fitting y band')
        fit_y,a=leastsq( modelToFit2_fit,parametersForLcFit)
        residual=residuals(x,y_proc,fit_y)
        chi_y=chisqr2(x,y_proc,fit_y,len(x),len(fit_y))
    else:
        fit_y=[9999.]
        chi_y=9999.
        
    results={'u':fit_u,'g':fit_g,'r':fit_r,'i':fit_i,'z':fit_z,'y':fit_y,'chi_u':chi_u,
             'chi_g':chi_g,'chi_r':chi_r,'chi_i':chi_i,'chi_z':chi_z,'chi_y':chi_y}
    
    return results
def plotting(data,fittingParameters,period,label,zeroTimeRef,outDir):
       
    fig=plt.figure(figsize=(10,16), dpi=80)
    plt.subplots_adjust(top = 0.95, bottom = 0.1, right = 0.95
                        , left = 0.1, hspace = 0.08, wspace = 0.2)
    
    ax1 = plt.subplot2grid((3,2), (0,0)) # topleft    
    ax1.set_ylabel('u (mag)')
    ax1.tick_params(axis='both',bottom=True, top=True, left=True, right=True,
                    direction='in',which='major')
    ax1.invert_yaxis()
    ax1.set_xlim([0,1])
    ax1.plot(((data['timeu']-zeroTimeRef)/period)%1,data['magu'],'o',color='b')
    if len(fittingParameters['u'])>1:
        timeForModel=np.arange(data['timeu'][0],data['timeu'][0]+2*period,0.01)
        magModelForPlot=modelToFit(timeForModel,fittingParameters['u'])
        ax1.plot(((timeForModel-             zeroTimeRef)/period)%1,magModelForPlot,'.',color='black')
    ax1.set_xlabel('phase')
    ax2 = plt.subplot2grid((3,2), (1,0)) # topleft    
    ax2.set_ylabel('g (mag)')
    ax2.tick_params(axis='both',bottom=True, top=True, left=True, right=True,
                    direction='in',which='major')
    ax2.invert_yaxis()
    ax2.set_xlim([0,1])
    ax2.plot(((data['timeg']-zeroTimeRef)/period)%1,data['magg'],'o',color='g')
    if len(fittingParameters['g'])>1:
        timeForModel=np.arange(data['timeg'][0],data['timeg'][0]+2*period,0.01)
        magModelForPlot=modelToFit(timeForModel,fittingParameters['g'])
        ax2.plot(((timeForModel-zeroTimeRef)/period)%1,magModelForPlot,'.',color='black')
    ax2.set_xlabel('phase')
    ax3 = plt.subplot2grid((3,2), (2,0)) # topleft    
    ax3.set_ylabel('r (mag)')
    ax3.tick_params(axis='both',bottom=True, top=True, left=True, right=True,
                    direction='in',which='major')
    ax3.invert_yaxis()
    ax3.set_xlim([0,1])
    ax3.plot(((data['timer']-zeroTimeRef)/period)%1,data['magr'],'o',color='r')
    if len(fittingParameters['r'])>1:
        timeForModel=np.arange(data['timer'][0],data['timer'][0]+2*period,0.01)
        magModelForPlot=modelToFit(timeForModel,fittingParameters['r'])
        ax3.plot(((timeForModel-zeroTimeRef)/period)%1,magModelForPlot,'.',color='black')
    ax3.set_xlabel('phase')
    ax4 = plt.subplot2grid((3,2), (0,1)) # topleft    
    ax4.set_ylabel('i (mag)')
    ax4.tick_params(axis='both',bottom=True, top=True, left=True, right=True,
                    direction='in',which='major')
    ax4.invert_yaxis()
    ax4.set_xlim([0,1])
    ax4.plot(((data['timei']-zeroTimeRef)/period) %1,data['magi'],'o',color='purple')
    if len(fittingParameters['i'])>1:
        timeForModel=np.arange(data['timei'][0],data['timei'][0]+2*period,0.01)
        magModelForPlot=modelToFit(timeForModel,fittingParameters['i'])
        ax4.plot(((timeForModel-zeroTimeRef)/period)%1,magModelForPlot,'.',color='black')
    ax4.set_xlabel('phase')
    ax5 = plt.subplot2grid((3,2), (1,1)) # topleft    
    ax5.set_ylabel('z (mag)')
    ax5.tick_params(axis='both',bottom=True, top=True, left=True, right=True,
                    direction='in',which='major')
    ax5.invert_yaxis()
    ax5.set_xlim([0,1])
    ax5.plot(((data['timez']-zeroTimeRef)/period) %1,data['magz'],'o',color='y')
    if len(fittingParameters['z'])>1:
        timeForModel=np.arange(data['timez'][0],data['timez'][0]+2*period,0.01)
        magModelForPlot=modelToFit(timeForModel,fittingParameters['z'])
        ax5.plot(((timeForModel-zeroTimeRef)/period)%1,magModelForPlot,'.',color='black')
    ax5.set_xlabel('phase')
    ax6 = plt.subplot2grid((3,2), (2,1)) # topleft    
    ax6.set_ylabel('y (mag)')
    ax6.tick_params(axis='both',bottom=True, top=True, left=True, right=True,
                    direction='in',which='major')
    ax6.invert_yaxis()
    ax6.set_xlim([0,1])
    ax6.plot(((data['timey']-zeroTimeRef)/period) %1,data['magy'],'o',color='magenta')
    if len(fittingParameters['y'])>1:
        timeForModel=np.arange(data['timey'][0],data['timey'][0]+2*period,0.01)
        magModelForPlot=modelToFit(timeForModel,fittingParameters['y'])
        ax6.plot(((timeForModel-zeroTimeRef)/period)%1,magModelForPlot,'.',color='black')
    ax6.set_xlabel('phase')
    
    plt.savefig(str(outDir)+'/LcFitting_'+str(label)+'.pdf')




    
def computation(data,index,period,numberOfHarmonics,factorForDimensionGap,label,outDir):
    
    zeroTimeRef=min(data['time_all'])

    print('fitting...')
    fitting=computingLcModel(data,period,numberOfHarmonics,index,outDir)
    timeForModel=np.arange(data['timeu'][0],data['timeu'][0]+2*period,0.01)
    #computing the magModelFromFit
    if len(fitting['u'])>1:
        magModelFromFit_u=modelToFit(timeForModel,fitting['u']) 
        ampl_u=max(magModelFromFit_u)-min(magModelFromFit_u)
    else:
        magModelFromFit_u=[9999.]
        ampl_u=9999.
    timeForModel=np.arange(data['timeg'][0],data['timeg'][0]+2*period,0.01)
    if len(fitting['g'])>1:
        #magModelFromFit_g=modelToFit(data['timeg'],fitting['g']) 
        magModelFromFit_g=modelToFit(timeForModel,fitting['g'])
        ampl_g=max(magModelFromFit_g)-min(magModelFromFit_g)
    else:
        magModelFromFit_g=[9999.]
        ampl_g=9999.
    timeForModel=np.arange(data['timer'][0],data['timer'][0]+2*period,0.01)
    if len(fitting['r'])>1:
        #magModelFromFit_r=modelToFit(data['timer'],fitting['r']) 
        magModelFromFit_r=modelToFit(timeForModel,fitting['r'])
        ampl_r=max(magModelFromFit_r)-min(magModelFromFit_r)
    else:
        magModelFromFit_r=[9999.]
        ampl_r=9999.
    timeForModel=np.arange(data['timei'][0],data['timei'][0]+2*period,0.01)
    
    if len(fitting['i'])>1:
        
        magModelFromFit_i=modelToFit(timeForModel,fitting['i'])
        
        if len(magModelFromFit_i)>0:
            ampl_i=max(magModelFromFit_i)-min(magModelFromFit_i)
        else:
            ampl_i=9999.
    else:
        magModelFromFit_i=[9999.]
        ampl_i=9999.
    timeForModel=np.arange(data['timez'][0],data['timez'][0]+2*period,0.01)    
    if len(fitting['z'])>1:
        
        magModelFromFit_z=modelToFit(timeForModel,fitting['z'])
        ampl_z=max(magModelFromFit_z)-min(magModelFromFit_z)
    else:
        magModelFromFit_z=[9999.]
        ampl_z=9999.
    timeForModel=np.arange(data['timey'][0],data['timey'][0]+2*period,0.01)    
    if len(fitting['y'])>1:
        
        magModelFromFit_y=modelToFit(timeForModel,fitting['y']) 
        ampl_y=max(magModelFromFit_y)-min(magModelFromFit_y)
    else:
        magModelFromFit_y=[9999.]
        ampl_y=9999.
    
   
    
    meanMag_u=meanmag_antilog(magModelFromFit_u)
    meanMag_g=meanmag_antilog(magModelFromFit_g)
    meanMag_r=meanmag_antilog(magModelFromFit_r)
    meanMag_i=meanmag_antilog(magModelFromFit_i)
    meanMag_z=meanmag_antilog(magModelFromFit_z)
    meanMag_y=meanmag_antilog(magModelFromFit_y)
    
   
    
    ampl_u=max(magModelFromFit_u)-min(magModelFromFit_u)
    ampl_g=max(magModelFromFit_g)-min(magModelFromFit_g)
    ampl_r=max(magModelFromFit_r)-min(magModelFromFit_r)
    ampl_i=max(magModelFromFit_i)-min(magModelFromFit_i)
    ampl_z=max(magModelFromFit_z)-min(magModelFromFit_z)
    ampl_y=max(magModelFromFit_y)-min(magModelFromFit_y)
    
    
    plotting(data,fitting,period,label,zeroTimeRef,outDir)

    
    finalResult={'mean_u':meanMag_u,'mean_g':meanMag_g,'mean_r':meanMag_r,
                 'mean_i':meanMag_i,'mean_z':meanMag_z,'mean_y':meanMag_y,
                 'ampl_u':ampl_u,'ampl_g':ampl_g,'ampl_r':ampl_r,
                 'ampl_i':ampl_i,'ampl_z':ampl_z,'ampl_y':ampl_y,
            'chi_u':fitting['chi_u'],'chi_g':fitting['chi_g'],'chi_r':fitting['chi_r'],
            'chi_i':fitting['chi_i'],'chi_z':fitting['chi_z'],'chi_y':fitting['chi_y'],
            'fittingParametersAllband':fitting}
    

#    finalResult2={'mean_u':meanMag_u,'mean_g':meanMag_g,'mean_r':meanMag_r,
#                 'mean_i':meanMag_i,'mean_z':meanMag_z,'mean_y':meanMag_y,
#                 'ampl_u':ampl_u,'ampl_g':ampl_g,'ampl_r':ampl_r,
#                 'ampl_i':ampl_i,'ampl_z':ampl_z,'ampl_y':ampl_y,
#            'chi_u':fitting['chi_u'],'chi_g':fitting['chi_g'],'chi_r':fitting['chi_r'],
#            'chi_i':fitting['chi_i'],'chi_z':fitting['chi_z'],'chi_y':fitting['chi_y'],
#            'fittingParametersAllband':fitting,
#            'uniformity_u':quality2['uniformity_u'],'uniformity_g':
##             quality2['uniformity_g'],'uniformity_r':
#             quality2['uniformity_r'],'uniformity_i':
#            quality2['uniformity_i'],'uniformity_z':
#             quality2['uniformity_z'],'uniformity_y': quality2['uniformity_y']}
    return finalResult