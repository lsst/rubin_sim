# factorForDimensionGap is a double that multiplyes the max distance from two consecutive phases of the light curve. Used to count the gap in 
# the light curve, should be < 1. 
import numpy as np
def main(data,period,index,factor1):
    time_u=data['timeu'][index['ind_notsaturated_u']]#time must be in days
    time_g=data['timeg'][index['ind_notsaturated_g']]
    time_r=data['timer'][index['ind_notsaturated_r']]
    time_i=data['timei'][index['ind_notsaturated_i']]
    time_z=data['timez'][index['ind_notsaturated_z']]
    time_y=data['timey'][index['ind_notsaturated_y']]
    n_u=len(time_u)
    n_g=len(time_g)
    n_r=len(time_r)
    n_i=len(time_i)
    n_z=len(time_z)
    n_y=len(time_y)                  
    maxGap_u,numberOfGaps_u=qualityCheck(time_u,period,factor1)
    maxGap_g,numberOfGaps_g=qualityCheck(time_g,period,factor1)
    maxGap_r,numberOfGaps_r=qualityCheck(time_r,period,factor1)
    maxGap_i,numberOfGaps_i=qualityCheck(time_i,period,factor1)
    maxGap_z,numberOfGaps_z=qualityCheck(time_z,period,factor1)
    maxGap_y,numberOfGaps_y= qualityCheck(time_y,period,factor1)
    uniformity_u=qualityCheck2(time_u,period)
    uniformity_g=qualityCheck2(time_g,period)
    uniformity_r=qualityCheck2(time_r,period)
    uniformity_i=qualityCheck2(time_i,period)
    uniformity_z=qualityCheck2(time_z,period)
    uniformity_y= qualityCheck2(time_y,period)
    uniformityKS_u=qualityCheck3(time_u,period)
    uniformityKS_g=qualityCheck3(time_g,period)
    uniformityKS_r=qualityCheck3(time_r,period)
    uniformityKS_i=qualityCheck3(time_i,period)
    uniformityKS_z=qualityCheck3(time_z,period)
    uniformityKS_y= qualityCheck3(time_y,period)
    
    finalResult={'n_u':n_u,'n_g':n_g,'n_r':n_r,'n_u':n_u,'n_i':n_i,'n_z':n_z,'n_y':n_y,
             'maxGap_u':maxGap_u,'maxGap_g':maxGap_g,'maxGap_r':maxGap_r,
             'maxGap_i':maxGap_i,'maxGap_z':maxGap_z,'maxGap_y':maxGap_y,
             'numberGaps_u':numberOfGaps_u,'numberGaps_g':numberOfGaps_g,'numberGaps_r':numberOfGaps_r,
             'numberGaps_i':numberOfGaps_i,'numberGaps_z':numberOfGaps_z,'numberGaps_y':numberOfGaps_y,
             'uniformity_u':uniformity_u,'uniformity_g':uniformity_g,'uniformity_r':uniformity_r,
             'uniformity_i':uniformity_i,'uniformity_z':uniformity_z,'uniformity_y':uniformity_y,
            'uniformityKS_u':uniformityKS_u,'uniformityKS_g':uniformityKS_g,'uniformityKS_r':uniformityKS_r,
             'uniformityKS_i':uniformityKS_i,'uniformityKS_z':uniformityKS_z,'uniformityKS_y':uniformityKS_y
             }
    
    return finalResult
def qualityCheck(time,period,factor1):
    if(len(time))>0:
    #period=param[0]
        phase= ((time-time[0])/period)%1
        indexSorted=np.argsort(phase)
   
        distances=[]
        indexStart=[]
        indexStop=[]
        leftDistance=phase[indexSorted[0]]
        rightDistance=1-phase[indexSorted[len(indexSorted)-1]]
        for i in range(len(phase)-1):
            dist=phase[indexSorted[i+1]]-phase[indexSorted[i]]
            distances.append(dist)
        
        
        #factor=sum(distances)/len(distances)*factor1
        distancesTotal=distances
        distancesTotal.append(leftDistance)
        distancesTotal.append(rightDistance)
        #factor=sum(distancesTotal)/len(distancesTotal)*factor1
        maxDistance=max(distancesTotal)
        factor = maxDistance*factor1
        for i in range(len(phase)-1):
            dist=phase[indexSorted[i+1]]-phase[indexSorted[i]]
            distances.append(dist)
            if (dist > factor):
                indexStart.append(indexSorted[i])
                indexStop.append(indexSorted[i+1])
        a=len(indexStart)
    else:
        maxDistance=999.
        a=999.   
    return maxDistance,a#,indexStart,indexStop

def qualityCheck2(time,period):
#This is based on Madore and Freedman (Apj 2005), uniformity definition   
    if(len(time))<=20 and (len(time))>1:
        phase= ((time-time[0])/period)%1
        indexSorted=np.argsort(phase)
   
        distances=[]
        indexStart=[]
        indexStop=[]
        leftDistance=phase[indexSorted[0]]
        rightDistance=1-phase[indexSorted[len(indexSorted)-1]]
        sumDistances=0
        for i in range(len(phase)-1):
            dist=phase[indexSorted[i+1]]-phase[indexSorted[i]]
            sumDistances=sumDistances+pow(dist,2)
            distances.append(dist)
        
        
        distancesTotal=distances
        distancesTotal.append(leftDistance)
        distancesTotal.append(rightDistance)
    
    #uniformity parameter
        u=len(time)/(len(time)-1)*(1-sumDistances)
    else:
        u=999.    
    return u 

def qualityCheck3(time,period):
#This is based on how a KS-test works: look at the cumulative distribution of observation dates,
#    and compare to a perfectly uniform cumulative distribution.
#    Perfectly uniform observations = 0, perfectly non-uniform = 1.
    if(len(time))>1:
        phase= ((time-time[0])/period)%1
        phase_sort=np.sort(phase)
        n_cum = np.arange(1, len(phase) + 1) / float(len(phase))
        D_max = np.max(np.abs(n_cum - phase_sort - phase_sort[0]))# ma in origine era phase_u_sort[1] ma non capisco il perché
    else:
        D_max=999.
    return D_max


