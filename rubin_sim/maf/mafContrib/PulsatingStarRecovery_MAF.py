import numpy as np
from rubin_sim.maf.utils import m52snr
from scipy.interpolate import interp1d
from gatspy import periodic
from rubin_sim.maf.stackers.generalStackers import SaturationStacker
from scipy.optimize import leastsq
from rubin_sim.maf.maps import dustMap
from rubin_sim.maf.maps import StellarDensityMap
from rubin_sim.photUtils import Dust_values
import rubin_sim.maf as maf
import matplotlib.pyplot as plt



__all__ = ['PulsatingStarRecovery']

class PulsatingStarRecovery(maf.BaseMetric):

    """
    
    This metric  studies how well a given cadence stategy 
    is able to recover the period and the shape of a light curve (from a template given in .csv file)
    in a  given point  of the sky. Returns a dictionary with the results of Lcsampling,Lcperiod, Lcfitting 
    
    Parameters
    ----------
     filename : str
        Ascii file containing the light curve of pulsating star. The file must
        contain nine columns - ['time', 'Mbol', 'u_sloan','g_sloan','r_sloan','i_sloan','z_sloan','y_sloan', period] -
     dmod : float
         Distance modulus     
     
     sigmaFORnoise: int
         the number of sigma used to generate the noise for the simulated light curve. It is an integer:if 0 the lc will be not noised
     
     do_remove_saturated :boolean (True, False)
          true if you want to remove from the plots the saturated visits (computed with saturation_stacker)
     
     numberOfHarmonics: int
          defines the number of harmonics used in LcFitting 
     
     factorForDimensionGap: float
          fraction of the size of the largest gap in the phase distribution  that is used to calculate numberGaps_X (see LcSampling)
          
        df came from lsst_sim.simdr2 and contains  magnitudes of the nearest stars (see query inn the notebook .If is empty no effect of blend is considered, see Notebook)
        
        and use/rename these columns of Opsim:
        mjdCol = observationStartMJD
        fiveSigmaDepth = fiveSigmaDepth
        filters = filter
        night = night
        visitExposureTime = visitExposureTime
        skyBrightness=skyBrightness
        numExposures=numExposures
        seeing=seeingFwhmEff
        airmass=airmass 

        
        
       
    """
    def __init__(self,filename,sigmaFORnoise,do_remove_saturated,numberOfHarmonics,factorForDimensionGap,df,mjdCol='observationStartMJD',fiveSigmaDepth='fiveSigmaDepth',filters= 'filter', night='night',visitExposureTime='visitExposureTime',skyBrightness='skyBrightness', numExposures='numExposures', seeing='seeingFwhmEff',airmass='airmass',**kwargs):



       
        self.mjdCol = mjdCol
        self.fiveSigmaDepth = fiveSigmaDepth
        self.filters = filters
        self.night = night
        self.visitExposureTime = visitExposureTime
        self.skyBrightness=skyBrightness
        self.numExposures=numExposures
        self.seeing=seeing
        self.airmass=airmass
 
      
        self.sigmaFORnoise=sigmaFORnoise
        self.do_remove_saturated=do_remove_saturated
        self.numberOfHarmonics=numberOfHarmonics
        self.factorForDimensionGap=factorForDimensionGap
        self.df=df

        
        
        
        cols = [self.mjdCol,self.fiveSigmaDepth,self.filters,self.night,self.visitExposureTime,self.skyBrightness,self.numExposures,self.seeing,self.airmass] 
        
        maps = ['DustMap']
        
        #the function 'ReadAsciifile' read filename with the theorical  light curve of a pulsating star
        self.lcModel_ascii=self.ReadAsciifile(filename)
        print(filename)
 
        super().__init__(col=cols, maps=maps, units='#', **kwargs, metricName="PulsatingStarRecovery_XLynne_blend", metricDtype="object")
        

        
    def run(self, dataSlice,slicePoint=None):
        

 
        #the function 'ReadTeoSim' puts the pulsating star at distance=dmod+Ax/Av*3.1*slicePoint('ebv').Gives a dictionary
        ebv1=slicePoint['ebv']
        dmod=5*np.log10(slicePoint['distance']*10**6) - 5
        print(slicePoint['distance'])
        print('sto analizzandola galassia con' )
        print(ebv1)
        print(dmod)
        ra1=slicePoint['ra']
        print(ra1)
        
        

#        if df.empty:
#            lcModel_noblend=self.ReadTeoSim(self.lcModel_ascii,self.dmod,ebv1)
#            lcModel_blend={'amplu':lcModel_noblend['amplu'],'amplg':lcModel_noblend['amplg'],'amplr':lcModel_noblend['amplr'],'ampli':lcModel_noblend['ampli'],'amplz':lcModel_noblend['amplz'],'amply':lcModel_noblend['amply'],'meanu':lcModel_noblend['meanu'],'meang':lcModel_noblend['meang'],'meanr':lcModel_noblend['meanr'],'meani':lcModel_noblend['meani'],'meanz':lcModel_noblend['meanz'],'meany':lcModel_noblend['meany']}
#        else:
#            lcModel_blend=self.ReadTeoSim_blend(df,self.lcModel_ascii,self.dmod,ebv1)
#            lcModel_noblend=self.ReadTeoSim(self.lcModel_ascii,self.dmod,ebv1)
#            
            
            
  
        lcModel_noblend=self.ReadTeoSim(self.lcModel_ascii,dmod,ebv1)
        
       

        
       
        
        
        
        #the class 'SaturationStacker' adds to dataSlice a column with the saturation
        satStacker = SaturationStacker()
        dataSlice = satStacker.run(dataSlice)

    
        
        
        #here we build -mv- that will be used in our metrics to build the simulated light curve:
        mv={'observationStartMJD':dataSlice[self.mjdCol],'fiveSigmaDepth':dataSlice[self.fiveSigmaDepth],
            'filter':dataSlice[self.filters], 'visitExposureTime':dataSlice[self.visitExposureTime],'night':dataSlice[self.night],
            'numExposures':dataSlice[self.numExposures],'skyBrightness':dataSlice[self.skyBrightness], 'seeingFwhmEff':dataSlice[self.seeing], 'airmass':dataSlice[self.airmass],'saturation_mag':dataSlice['saturation_mag']}


        #define time_lsst as array and filter
        time_lsst=np.asarray(mv['observationStartMJD']+mv['visitExposureTime']/2.)    
        filters_lsst=np.asarray(mv['filter'])
  

        #################################################
        # The following not consider the blend. 
        #################################################
        
         #the function self.generateLC generate the temporal series and simulate light curve    
        LcTeoLSST=self.generateLC(time_lsst,filters_lsst,lcModel_noblend)
        
        #the function 'noising' compute and add errors
        snr=self.retrieveSnR(mv,lcModel_noblend) 
        LcTeoLSST_noised=self.noising(LcTeoLSST,snr,self.sigmaFORnoise,[0.,0.,0.,0.,0.,0.])  
        
        
        #the function 'count_saturation' build an index to exclude saturated stars and those under detection limit.                                       
        index_notsaturated,saturation_index,detection_index=self.count_saturation(mv,snr,LcTeoLSST,LcTeoLSST_noised,self.do_remove_saturated)
        
        
        #The function 'Lcsampling' analize the sampling of the simulated light curve. Give a dictionary with UniformityPrameters obtained with three different methods
        #1) for each filter X calculates the number of points (n_X), the size in phase of the largest gap (maxGap_X) and the number of gaps largest than factorForDimensionGap*maxGap_X (numberGaps_X)
        #2) the uniformity parameters from Barry F. Madore and Wendy L. Freedman 2005 ApJ 630 1054 (uniformity_X)  useful for n_X<20
        #3) a modified version of UniformityMetric by Peter Yoachim (https://sims-maf.lsst.io/_modules/lsst/sims/maf/metrics/cadenceMetrics.html#UniformityMetric.run). Calculate how uniformly the observations are spaced in phase (not time)using KS test.Returns a value between 0 (uniform sampling) and 1 . uniformityKS_X

        period_model=LcTeoLSST['p_model']
        uni_meas=self.Lcsampling(LcTeoLSST_noised,period_model,index_notsaturated,self.factorForDimensionGap)
        
        #the function 'LcPeriod' analyse the periodogram with Gatspy and gives:
        #1)the best period (best_per_temp)
        #2)the difference between the recovered period and the  model's period(P) and
        #3)diffper_abs=(DeltaP/P)*100
        #4)diffcicli= DeltaP/P*1/number of cycle
        best_per_temp,diffper,diffper_abs,diffcicli=self.LcPeriodLight(mv,LcTeoLSST,LcTeoLSST_noised,index_notsaturated)
        period=best_per_temp #or period_model or fitLS_multi.best_period
        #period=LcTeoLSST['p_model']
        
        #The function 'LcFitting' fit the simulated light curve with number of harmonics=numberOfHarmonics.Return a dictionary with mean magnitudes, amplitudes and chi of the fits
        
        finalResult=self.LcFitting(LcTeoLSST_noised,index_notsaturated,period,self.numberOfHarmonics)
        
        #Some useful figure of merit on the recovery of the:
        #and shape.Difference between observed and derived mean magnitude (after fitting the light curve)
        deltamag_u=lcModel_noblend['meanu']-finalResult['mean_u']
        deltamag_g=lcModel_noblend['meang']-finalResult['mean_g']
        deltamag_r=lcModel_noblend['meanr']-finalResult['mean_r']
        deltamag_i=lcModel_noblend['meani']-finalResult['mean_i']
        deltamag_z=lcModel_noblend['meanz']-finalResult['mean_z']
        deltamag_y=lcModel_noblend['meany']-finalResult['mean_y']
        #the same can be done for the amplitudes (without the effect of blending for the momment.)
        deltaamp_u=lcModel_noblend['amplu']-finalResult['ampl_u']
        deltaamp_g=lcModel_noblend['amplg']-finalResult['ampl_g']
        deltaamp_r=lcModel_noblend['amplr']-finalResult['ampl_r']
        deltaamp_i=lcModel_noblend['ampli']-finalResult['ampl_i']
        deltaamp_z=lcModel_noblend['amplz']-finalResult['ampl_z']
        deltaamp_y=lcModel_noblend['amply']-finalResult['ampl_y']
        # Chi of the fit-->finalResult['chi_u']....
        
        if self.df.empty:
            output_metric={'n_u':uni_meas['n_u'],'n_g':uni_meas['n_g'],'n_r':uni_meas['n_r'],'n_i':uni_meas['n_i'],'n_z':uni_meas['n_z'],'n_y':uni_meas['n_y'],
                'maxGap_u':uni_meas['maxGap_u'],'maxGap_g':uni_meas['maxGap_g'],'maxGap_r':uni_meas['maxGap_r'],'maxGap_i':uni_meas['maxGap_i'],'maxGap_z':uni_meas['maxGap_z'],'maxGap_y':uni_meas['maxGap_y'],
                'numberGaps_u':uni_meas['numberGaps_u'],'numberGaps_g':uni_meas['numberGaps_g'],'numberGaps_r':uni_meas['numberGaps_r'],
                'numberGaps_i':uni_meas['numberGaps_i'],'numberGaps_z':uni_meas['numberGaps_z'],'numberGaps_y':uni_meas['numberGaps_y'],
                'uniformity_u':uni_meas['uniformity_u'],'uniformity_g':uni_meas['uniformity_g'],'uniformity_r':uni_meas['uniformity_r'],
                'uniformity_i':uni_meas['uniformity_i'],'uniformity_z':uni_meas['uniformity_z'],'uniformity_y':uni_meas['uniformity_y'],
                'uniformityKS_u':uni_meas['uniformityKS_u'],'uniformityKS_g':uni_meas['uniformityKS_g'],'uniformityKS_r':uni_meas['uniformityKS_r'],
                'uniformityKS_i':uni_meas['uniformityKS_i'],'uniformityKS_z':uni_meas['uniformityKS_z'],'uniformityKS_y':uni_meas['uniformityKS_y'],
                'P_gatpsy':best_per_temp,'Delta_Period':diffper,'Delta_Period_abs':diffper_abs,'Delta_Period_abs_cicli':diffcicli,
                'deltamag_u':deltamag_u,'deltamag_g':deltamag_g,'deltamag_r':deltamag_r,'deltamag_i':deltamag_i,'deltamag_z':deltamag_z,'deltamag_y':deltamag_y,
                'deltaamp_u':deltaamp_u, 'deltaamp_g':deltaamp_g,'deltaamp_r':deltaamp_r,'deltaamp_i':deltaamp_i,'deltaamp_z':deltaamp_z,'deltaamp_y':deltaamp_y,
                'chi_u':finalResult['chi_u'],'chi_g':finalResult['chi_g'],'chi_r':finalResult['chi_r'],'chi_i':finalResult['chi_i'],'chi_z':finalResult['chi_z'],'chi_y':finalResult['chi_y']}  
        else:
            lcModel_blend=self.ReadTeoSim_blend(self.df,self.lcModel_ascii,dmod,ebv1)
            
            
            LcTeoLSST_blend=self.generateLC(time_lsst,filters_lsst,lcModel_blend)
        
        #the function 'noising' compute and add errors
            snr_blend=self.retrieveSnR(mv,lcModel_blend) 
            LcTeoLSST_noised_blend=self.noising(LcTeoLSST_blend,snr_blend,self.sigmaFORnoise,[0.,0.,0.,0.,0.,0.])  
        
        
        #the function 'count_saturation' build an index to exclude saturated stars and those under detection limit.                                       
            index_notsaturated_blend,saturation_index_blend,detection_index_blend=self.count_saturation(mv,snr_blend,LcTeoLSST_blend,LcTeoLSST_noised_blend,self.do_remove_saturated)
        
        
        #The function 'Lcsampling' analize the sampling of the simulated light curve. Give a dictionary with UniformityPrameters obtained with three different methods
        #1) for each filter X calculates the number of points (n_X), the size in phase of the largest gap (maxGap_X) and the number of gaps largest than factorForDimensionGap*maxGap_X (numberGaps_X)
        #2) the uniformity parameters from Barry F. Madore and Wendy L. Freedman 2005 ApJ 630 1054 (uniformity_X)  useful for n_X<20
        #3) a modified version of UniformityMetric by Peter Yoachim (https://sims-maf.lsst.io/_modules/lsst/sims/maf/metrics/cadenceMetrics.html#UniformityMetric.run). Calculate how uniformly the observations are spaced in phase (not time)using KS test.Returns a value between 0 (uniform sampling) and 1 . uniformityKS_X

            period_model_blend=LcTeoLSST_blend['p_model']
            uni_meas_blend=self.Lcsampling(LcTeoLSST_noised_blend,period_model_blend,index_notsaturated_blend,self.factorForDimensionGap)
            
        #the function 'LcPeriod' analyse the periodogram with Gatspy and gives:
        #1)the best period (best_per_temp)
        #2)the difference between the recovered period and the  model's period(P) and
        #3)diffper_abs=(DeltaP/P)*100
        #4)diffcicli= DeltaP/P*1/number of cycle
            best_per_temp_blend,diffper_blend,diffper_abs_blend,diffcicli_blend=self.LcPeriodLight(mv,LcTeoLSST_blend,LcTeoLSST_noised_blend,index_notsaturated_blend)
            period_blend=best_per_temp_blend #or period_model or fitLS_multi.best_period
        #period=LcTeoLSST['p_model']
        
        #The function 'LcFitting' fit the simulated light curve with number of harmonics=numberOfHarmonics.Return a dictionary with mean magnitudes, amplitudes and chi of the fits
        
            finalResult_blend=self.LcFitting(LcTeoLSST_noised_blend,index_notsaturated_blend,period_blend,self.numberOfHarmonics)
        
        #Some useful figure of merit on the recovery of the:
        #and shape.Difference between observed and derived mean magnitude (after fitting the light curve)
            deltamag_u_blend=lcModel_blend['meanu']-finalResult_blend['mean_u']
            deltamag_g_blend=lcModel_blend['meang']-finalResult_blend['mean_g']
            deltamag_r_blend=lcModel_blend['meanr']-finalResult_blend['mean_r']
            deltamag_i_blend=lcModel_blend['meani']-finalResult_blend['mean_i']
            deltamag_z_blend=lcModel_blend['meanz']-finalResult_blend['mean_z']
            deltamag_y_blend=lcModel_blend['meany']-finalResult_blend['mean_y']
        #the same can be done for the amplitudes (without the effect of blending for the momment.)
            deltaamp_u_blend=lcModel_blend['amplu']-finalResult_blend['ampl_u']
            deltaamp_g_blend=lcModel_blend['amplg']-finalResult_blend['ampl_g']
            deltaamp_r_blend=lcModel_blend['amplr']-finalResult_blend['ampl_r']
            deltaamp_i_blend=lcModel_blend['ampli']-finalResult_blend['ampl_i']
            deltaamp_z_blend=lcModel_blend['amplz']-finalResult_blend['ampl_z']
            deltaamp_y_blend=lcModel_blend['amply']-finalResult_blend['ampl_y']
            
            
            
            output_metric={'n_u':uni_meas['n_u'],'n_g':uni_meas['n_g'],'n_r':uni_meas['n_r'],'n_i':uni_meas['n_i'],'n_z':uni_meas['n_z'],'n_y':uni_meas['n_y'],
                'maxGap_u':uni_meas['maxGap_u'],'maxGap_g':uni_meas['maxGap_g'],'maxGap_r':uni_meas['maxGap_r'],'maxGap_i':uni_meas['maxGap_i'],'maxGap_z':uni_meas['maxGap_z'],'maxGap_y':uni_meas['maxGap_y'],
                'numberGaps_u':uni_meas['numberGaps_u'],'numberGaps_g':uni_meas['numberGaps_g'],'numberGaps_r':uni_meas['numberGaps_r'],
                'numberGaps_i':uni_meas['numberGaps_i'],'numberGaps_z':uni_meas['numberGaps_z'],'numberGaps_y':uni_meas['numberGaps_y'],
                'uniformity_u':uni_meas['uniformity_u'],'uniformity_g':uni_meas['uniformity_g'],'uniformity_r':uni_meas['uniformity_r'],
                'uniformity_i':uni_meas['uniformity_i'],'uniformity_z':uni_meas['uniformity_z'],'uniformity_y':uni_meas['uniformity_y'],
                'uniformityKS_u':uni_meas['uniformityKS_u'],'uniformityKS_g':uni_meas['uniformityKS_g'],'uniformityKS_r':uni_meas['uniformityKS_r'],
                'uniformityKS_i':uni_meas['uniformityKS_i'],'uniformityKS_z':uni_meas['uniformityKS_z'],'uniformityKS_y':uni_meas['uniformityKS_y'],
                'P_gatpsy':best_per_temp,'Delta_Period':diffper,'Delta_Period_abs':diffper_abs,'Delta_Period_abs_cicli':diffcicli,
                'deltamag_u':deltamag_u,'deltamag_g':deltamag_g,'deltamag_r':deltamag_r,'deltamag_i':deltamag_i,'deltamag_z':deltamag_z,'deltamag_y':deltamag_y,
                'deltaamp_u':deltaamp_u, 'deltaamp_g':deltaamp_g,'deltaamp_r':deltaamp_r,'deltaamp_i':deltaamp_i,'deltaamp_z':deltaamp_z,'deltaamp_y':deltaamp_y,
                'chi_u':finalResult['chi_u'],'chi_g':finalResult['chi_g'],'chi_r':finalResult['chi_r'],'chi_i':finalResult['chi_i'],'chi_z':finalResult['chi_z'],'chi_y':finalResult['chi_y'],
                'P_gatpsy_blend':best_per_temp_blend,'Delta_Period_blend':diffper_blend,'Delta_Period_abs_blend':diffper_abs_blend,'Delta_Period_abs_cicli_blend':diffcicli_blend,
                'deltamag_u_blend':deltamag_u_blend,'deltamag_g_blend':deltamag_g_blend,'deltamag_r_blend':deltamag_r_blend,'deltamag_i_blend':deltamag_i_blend,'deltamag_z_blend':deltamag_z_blend, 'deltamag_y_blend':deltamag_y_blend,
                'deltaamp_u_blend':deltaamp_u_blend,'deltaamp_g_blend':deltaamp_g_blend,'deltaamp_r_blend':deltaamp_r_blend,'deltaamp_i_blend':deltaamp_i_blend,'deltaamp_z_blend':deltaamp_z_blend,'deltaamp_y_blend':deltaamp_y_blend,
                'chi_u_blend':finalResult_blend['chi_u'],'chi_g_blend':finalResult_blend['chi_g'],'chi_r_blend':finalResult_blend['chi_r'],'chi_i_blend':finalResult_blend['chi_i'],'chi_z_blend':finalResult_blend['chi_z'],'chi_y_blend':finalResult_blend['chi_y'],
                          
                          }    
        return output_metric
    

    def reduceP_gatpsy(self, metricValue):
        return metricValue['P_gatpsy']  

    def get_deltamag_u(self,metricValue):
        return metricValue['deltamag_u']
    def get_deltaamp_u(self,metricValue):
        return metricValue['deltaamp_u']
    def get_chi_u(self,metricValue):
        return metricValue['chi_u']
    def get_deltamag_g(self,metricValue):
        return metricValue['deltamag_g']
    def get_deltaamp_g(self,metricValue):
        return metricValue['deltaamp_g']
    def get_chi_g(self,metricValue):
        return metricValue['chi_g']
    
    def get_deltamag_r(self,metricValue):
        return metricValue['deltamag_r']
    def get_deltaamp_r(self,metricValue):
        return metricValue['deltaamp_r']
    def get_chi_r(self,metricValue):
        return metricValue['chi_r']
    def get_deltamag_i(self,metricValue):
        return metricValue['deltamag_i']
    def get_deltaamp_i(self,metricValue):
        return metricValue['deltaamp_i']
    def get_chi_i(self,metricValue):
        return metricValue['chi_i']
    def get_deltamag_z(self,metricValue):
        return metricValue['deltamag_z']
    def get_deltaamp_z(self,metricValue):
        return metricValue['deltaamp_z']
    def get_chi_z(self,metricValue):
        return metricValue['chi_z']
    def get_deltamag_y(self,metricValue):
        return metricValue['deltamag_y']
    def get_deltaamp_y(self,metricValue):
        return metricValue['deltaamp_y']
    def get_chi_y(self,metricValue):
        return metricValue['chi_y']

    def get_P_gatpsy_blend(self, metricValue):
        return metricValue['P_gatpsy_blend']

    def get_deltamag_u_blend(self,metricValue):
        return metricValue['deltamag_u_blend']
    def get_deltaamp_u_blend(self,metricValue):
        return metricValue['deltaamp_u_blend']
    def get_chi_u_blend(self,metricValue):
        return metricValue['chi_u_blend']
    
    def get_deltamag_g_blend(self,metricValue):
        return metricValue['deltamag_g_blend']
    def get_deltaamp_g_blend(self,metricValue):
        return metricValue['deltaamp_g_blend']
    def get_chi_g_blend(self,metricValue):
        return metricValue['chi_g_blend']
    
    def get_deltamag_r_blend(self,metricValue):
        return metricValue['deltamag_r_blend']
    def get_deltaamp_r_blend(self,metricValue):
        return metricValue['deltaamp_r_blend']
    def get_chi_r_blend(self,metricValue):
        return metricValue['chi_r_blend']
    
    def get_deltamag_i_blend(self,metricValue):
        return metricValue['deltamag_i_blend']
    def get_deltaamp_i_blend(self,metricValue):
        return metricValue['deltaamp_i_blend']
    def get_chi_i_blend(self,metricValue):
        return metricValue['chi_i_blend']
    
    def get_deltamag_z_blend(self,metricValue):
        return metricValue['deltamag_z_blend']
    def get_deltaamp_z_blend(self,metricValue):
        return metricValue['deltaamp_z_blend']
    def get_chi_z_blend(self,metricValue):
        return metricValue['chi_z_blend']
    
    def get_deltamag_y_blend(self,metricValue):
        return metricValue['deltamag_y_blend']
    def get_deltaamp_y_blend(self,metricValue):
        return metricValue['deltaamp_y_blend']
    def get_chi_y_blend(self,metricValue):
        return metricValue['chi_y_blend']
    
    



    def meanmag_antilog(self,mag):
        mag=np.asarray(mag)
        flux=10.**(-mag/2.5)
        if len(flux)>0:
            result=(-2.5)*np.log10(sum(flux)/len(flux))
        else:
            result=9999.
        return result
    
    def mag_antilog(self,mag):
        mag=np.asarray(mag)
        flux=10.**(-mag/2.5)
        return flux
    def ReadAsciifile(self,filename):
        """
        Reads in an ascii file the light curve of the pulsating stars that we want simulate. Must be
        in the following format, 9 columns: time, bolometric_mag, filterLSST_mag,period.
        Parameters
        -----------
        filename: str
            string containing the path to the ascii file containing the
            light curve.
        """
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
                period_sec=float(ll[8])*86400.
                period_day=float(ll[8])
                time_0=float(ll[0])
            time_model.append(float(ll[0]))
            phase_model.append((float(ll[0])-time_0)/period_sec % 1)
            u_model.append(float(ll[2]))
            g_model.append(float(ll[3]))
            r_model.append(float(ll[4]))
            i_model.append(float(ll[5]))
            z_model.append(float(ll[6]))
            y_model.append(float(ll[7]))
        f.close()
        output_ascii={'time':time_model, 'phase':phase_model,'period':period_sec,
                'u':u_model, 'g': g_model, 'r': r_model, 'i': i_model, 'z': z_model, 'y': y_model}
        return output_ascii
       
    
    def ReadTeoSim(self,model,dmod,ebv):
        """
        Put the  model at a given distance moduli using the E(b-v) given by dustMap.
        Parameters
        -----------
        model:dictionary
             output of ReadAsciifile
        dmod: float
             distance modulus 
        ebv: float
             E(B-V)=slicePoint('ebv')
            
        
        """

        time_model=model['time'].copy()
        u_mod=model['u'].copy()
        g_mod=model['g'].copy()
        r_mod=model['r'].copy()
        i_mod=model['i'].copy()
        z_mod=model['z'].copy()
        y_mod=model['y'].copy()
    
        u_model=[]
        g_model=[]
        r_model=[]
        i_model=[]
        z_model=[]
        y_model=[]
        
        
        phase_model=model['phase'].copy()
        
        
        period_model=model['period']

        time_0=time_model[0]

        for i in range(len(u_mod)):
            u_model.append(u_mod[i]+dmod+1.55607*3.1*ebv)
        for i in range(len(g_mod)):
            g_model.append(g_mod[i]+dmod+1.18379*3.1*ebv)
        for i in range(len(r_mod)):
            r_model.append(r_mod[i]+dmod+1.87075*3.1*ebv)
        for i in range(len(i_mod)):
            i_model.append(i_mod[i]+dmod+0.67897*3.1*ebv)
        for i in range(len(z_mod)):
            z_model.append(z_mod[i]+dmod+0.51683*3.1*ebv)
        for i in range(len(y_mod)):
            y_model.append(y_mod[i]+dmod+0.42839*3.1*ebv)
 
        
#compute the intensity means
        meanu=self.meanmag_antilog(u_model)
        meang=self.meanmag_antilog(g_model)
        meanr=self.meanmag_antilog(r_model)
        meani=self.meanmag_antilog(i_model)
        meanz=self.meanmag_antilog(z_model)
        meany=self.meanmag_antilog(y_model)
        u_model_flux=self.mag_antilog(u_model)
        g_model_flux=self.mag_antilog(g_model)
        r_model_flux=self.mag_antilog(r_model)
        i_model_flux=self.mag_antilog(i_model)
        z_model_flux=self.mag_antilog(z_model)
        y_model_flux=self.mag_antilog(y_model)
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
    
    def ReadTeoSim_blend(self,df,model,dmod,ebv):
        """
        Put the  model at a given distance moduli using the E(b-v) given by dustMap and take into account the blend using df
        Parameters
        -----------
        df:list of stars (max 100) from TRILEGAL('lsst_sim.simdr2) in a cone search centered on test_ra and test_dec and a given radii (in degree). 
        model:dictionary
             output of ReadAsciifile
        dmod: float
             distance modulus 
        ebv: float
             E(B-V)=slicePoint('ebv')
            
        
        """
       # df1= df['umag'][df['umag']<27]
        flux_blend_u=self.mag_antilog(df['umag'])
        flux_blend_g=self.mag_antilog(df['gmag'])
        flux_blend_r=self.mag_antilog(df['rmag'])
        flux_blend_i=self.mag_antilog(df['imag'])
        flux_blend_z=self.mag_antilog(df['zmag'])
        flux_blend_y=self.mag_antilog(df['ymag'])
        
        

       
    
    
    
        time_model_blend=model['time'].copy()
        u_mod_blend=model['u'].copy()
        g_mod_blend=model['g'].copy()  
        r_mod_blend=model['r'].copy()
        i_mod_blend=model['i'].copy()
        z_mod_blend=model['z'].copy()
        y_mod_blend=model['y'].copy()
    
        u_model_blend=[]
        g_model_blend=[]
        r_model_blend=[]
        i_model_blend=[]
        z_model_blend=[]
        y_model_blend=[]
        phase_model_blend=[]
        
        phase_model_blend=model['phase'].copy()
        period_model_blend=model['period']

        time_0=time_model_blend[0]

        for i in range(len(u_mod_blend)):
            u_model_blend.append(u_mod_blend[i]+dmod+1.55607*3.1*ebv)
        for i in range(len(g_mod_blend)):
            g_model_blend.append(g_mod_blend[i]+dmod+1.18379*3.1*ebv)
        for i in range(len(r_mod_blend)):
            r_model_blend.append(r_mod_blend[i]+dmod+1.87075*3.1*ebv)
        for i in range(len(i_mod_blend)):
            i_model_blend.append(i_mod_blend[i]+dmod+0.67897*3.1*ebv)
        for i in range(len(z_mod_blend)):
            z_model_blend.append(z_mod_blend[i]+dmod+0.51683*3.1*ebv)
        for i in range(len(y_mod_blend)):
            y_model_blend.append(y_mod_blend[i]+dmod+0.42839*3.1*ebv)
 
        
#compute the intensity means
        

        u_model_flux_blend=self.mag_antilog(u_model_blend)+sum(flux_blend_u)
        g_model_flux_blend=self.mag_antilog(g_model_blend)+sum(flux_blend_g)
        r_model_flux_blend=self.mag_antilog(r_model_blend)+sum(flux_blend_r)
        i_model_flux_blend=self.mag_antilog(i_model_blend)+sum(flux_blend_i)
        z_model_flux_blend=self.mag_antilog(z_model_blend)+sum(flux_blend_z)
        y_model_flux_blend=self.mag_antilog(y_model_blend)+sum(flux_blend_y)
        mean_flux_u_blend=np.mean(u_model_flux_blend)
        mean_flux_g_blend=np.mean(g_model_flux_blend)
        mean_flux_r_blend=np.mean(r_model_flux_blend)
        mean_flux_i_blend=np.mean(i_model_flux_blend)
        mean_flux_z_blend=np.mean(z_model_flux_blend)
        mean_flux_y_blend=np.mean(y_model_flux_blend)
        
        u_blend=(-2.5)*np.log10(u_model_flux_blend)
        g_blend=(-2.5)*np.log10(g_model_flux_blend)
        r_blend=(-2.5)*np.log10(r_model_flux_blend)
        i_blend=(-2.5)*np.log10(i_model_flux_blend)
        z_blend=(-2.5)*np.log10(z_model_flux_blend)
        y_blend=(-2.5)*np.log10(y_model_flux_blend)
        meanu_blend=self.meanmag_antilog(u_blend)
        meang_blend=self.meanmag_antilog(g_blend)
        meanr_blend=self.meanmag_antilog(r_blend)
        meani_blend=self.meanmag_antilog(i_blend)
        meanz_blend=self.meanmag_antilog(z_blend)
        meany_blend=self.meanmag_antilog(y_blend)
        amplu_blend=max(u_blend)-min(u_blend)
        amplg_blend=max(g_blend)-min(g_blend)
        amplr_blend=max(r_blend)-min(r_blend)
        ampli_blend=max(i_blend)-min(i_blend)
        amplz_blend=max(z_blend)-min(z_blend)
        amply_blend=max(y_blend)-min(y_blend)

        phase_model_blend.append(1.)
        ind_0_blend=phase_model_blend.index(0.)
        u_model_blend.append(u_model_blend[ind_0_blend])
        g_model_blend.append(g_model_blend[ind_0_blend])
        r_model_blend.append(r_model_blend[ind_0_blend])
        i_model_blend.append(i_model_blend[ind_0_blend])
        z_model_blend.append(z_model_blend[ind_0_blend])
        y_model_blend.append(y_model_blend[ind_0_blend])

#    return time_model,phase_model,u_model,g_model,r_model,i_model,z_model,y_model
        output={'time':time_model_blend, 'phase':phase_model_blend,'period':period_model_blend,
                'u':u_model_blend, 'g': g_model_blend, 'r': r_model_blend, 'i': i_model_blend, 'z': z_model_blend, 'y': y_model_blend,
                'meanu':meanu_blend,'meang':meang_blend,'meanr':meanr_blend,'meani':meani_blend,'meanz':meanz_blend,'meany':meany_blend,
                'amplu':amplu_blend,'amplg':amplg_blend,'amplr':amplr_blend,'ampli':ampli_blend,'amplz':amplz_blend,'amply':amply_blend,
                'mean_flux_u':mean_flux_u_blend,'mean_flux_g':mean_flux_g_blend,'mean_flux_r':mean_flux_r_blend,'mean_flux_i':mean_flux_i_blend,'mean_flux_z':mean_flux_z_blend,'mean_flux_y':mean_flux_y_blend,}
        return output
    
  
    def generateLC(self,time_lsst,filters_lsst,output_ReadLCTeo,period_true=-99,
                   ampl_true=1.,phase_true=0.,do_normalize=False):
        """
        Generate the observed teporal series and light curve from teplate  and opsim
        Parameters
        -----------
        Parameters:
        time_lsst: float
              
        filters_lsst: float
             E(B-V)=slicePoint('ebv')
        
        output_ReadLCTeo: dictionary
        
        period_true:
        
        ampl_true:
        
        phase_true:
        
        do_normalize:
            
        
        """

        u_model=output_ReadLCTeo['u']
        g_model=output_ReadLCTeo['g']
        r_model=output_ReadLCTeo['r']
        i_model=output_ReadLCTeo['i']
        z_model=output_ReadLCTeo['z']
        y_model=output_ReadLCTeo['y']
        phase_model=output_ReadLCTeo['phase']

        #you can use  different period
        if period_true < -90:
            period_final=(output_ReadLCTeo['period'])/86400.
        else:
            period_final=period_true


        #you can normalize
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
            print(len(phase_model), len(u_model))
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
               'ind_i':ind_i,'ind_z':ind_z,'ind_y':ind_y,'p_model':period_final}

 
    def retrieveSnR(self,mv,theoreticModel):
        ''''
        Generates s/n  based in the 5-sigma limiting depth
        of each observation
        '''''
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

    def noising(self,LcTeoLSST,snr,sigma,perc_blend):
                        
        """
        generate noise and add to the simulated light curve
        
        Parameters:
        -----------
        
        """

    #noising 
        def noisingBand(timeLSSTteo,magLSSTteo,snr,sigma,blend=0):
            magNoised=[]
            noise=[]
            dmag=[]
            magNoisedComp=[]
            for j in range(len(timeLSSTteo)):            
                dmag = 2.5*np.log10(1.+1./snr[j])
                if blend >0:
                    dmag=np.sqrt(2)*dmag  
                noise = np.random.uniform(-sigma,sigma)*dmag
                magNoisedComp=magLSSTteo[j]+noise
                magNoised.append(magNoisedComp)

            return magNoised, noise ,dmag
        magNoisedu,noiseu,dmagu=[],[],[]
        magNoisedg,noiseg,dmagg=[],[],[]
        magNoisedr,noiser,dmagr=[],[],[]
        magNoisedi,noisei,dmagi=[],[],[]
        magNoisedz,noisez,dmagz=[],[],[]
        magNoisedy,noisey,dmagy=[],[],[]

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

     
        
    def count_saturation(self,mv,snr,LcTeoLSST,LcTeoLSST_noised,do_remove_saturated):
        """
        Build an index that will be used to esclude saturated stars from the following analysis.
        Build an index  to flag those stars that are under detection limit
        
        Parameters:
        -----------
        
        """

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
        saturation_index={'u':saturation_index_u,'g':saturation_index_g,'r':saturation_index_r,'i':saturation_index_i,'z':saturation_index_z,'y':saturation_index_y}

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
        detection_index={'u':detection_index_u,'g':detection_index_g,'r':detection_index_r,'i':detection_index_i,'z':detection_index_z,'y':detection_index_y}
        return index_notsaturated,saturation_index,detection_index
    
  

    def Lcsampling(self,data,period,index,factor1):
        """
        Analyse the sampling of the simulated light curve (with the period=period_model) Give a dictionary with UniformityPrameters obtained with three different methods
        1) for each filter X calculates the number of points (n_X), the size in phase of the largest gap (maxGap_X) and the number of gaps largest than factorForDimensionGap*maxGap_X (numberGaps_X)
        2) the uniformity parameters from Barry F. Madore and Wendy L. Freedman 2005 ApJ 630 1054 (uniformity_X)  useful for n_X<20
        3) a modified version of UniformityMetric by Peter Yoachim (https://sims-maf.lsst.io/_modules/lsst/sims/maf/metrics/cadenceMetrics.html#UniformityMetric.run). Calculate how uniformly the observations are spaced in phase (not time)using KS test.Returns a value between 0 (uniform sampling) and 1 . uniformityKS_X
        Parameters:
        -----------
        data:
        period:
        index:
        factor1:
            factorForDimensionGap
        """
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
        maxGap_u,numberOfGaps_u=self.qualityCheck(time_u,period,factor1)
        maxGap_g,numberOfGaps_g=self.qualityCheck(time_g,period,factor1)
        maxGap_r,numberOfGaps_r=self.qualityCheck(time_r,period,factor1)
        maxGap_i,numberOfGaps_i=self.qualityCheck(time_i,period,factor1)
        maxGap_z,numberOfGaps_z=self.qualityCheck(time_z,period,factor1)
        maxGap_y,numberOfGaps_y=self.qualityCheck(time_y,period,factor1)
        uniformity_u=self.qualityCheck2(time_u,period)
        uniformity_g=self.qualityCheck2(time_g,period)
        uniformity_r=self.qualityCheck2(time_r,period)
        uniformity_i=self.qualityCheck2(time_i,period)
        uniformity_z=self.qualityCheck2(time_z,period)
        uniformity_y=self.qualityCheck2(time_y,period)
        uniformityKS_u=self.qualityCheck3(time_u,period)
        uniformityKS_g=self.qualityCheck3(time_g,period)
        uniformityKS_r=self.qualityCheck3(time_r,period)
        uniformityKS_i=self.qualityCheck3(time_i,period)
        uniformityKS_z=self.qualityCheck3(time_z,period)
        uniformityKS_y=self.qualityCheck3(time_y,period)

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
    def qualityCheck(self,time,period,factor1):
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

    def qualityCheck2(self,time,period):
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

    def qualityCheck3(self,time,period):
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




    
    def LcPeriodLight(self,mv,LcTeoLSST,LcTeoLSST_noised,index_notsaturated):
        """
        compute the period using Gatpsy and return differences with the period of the model.
        Does not make a figure in its light version
        
        Parameters:
        -----------
        
        """
        #########################################################################
        #period range in the periodogram for the plot of the periodogram
        period_model=LcTeoLSST['p_model'] 
 
        minper_opt=period_model-0.9*period_model
        maxper_opt=period_model+ 0.9*period_model
        periods = np.linspace(minper_opt, maxper_opt,1000)


 

        
        #########This is to measure the noise of the periodogramm but is not used yet

        LS_multi = periodic.LombScargleMultiband(Nterms_base=1, Nterms_band=0)
        LS_multi.fit(LcTeoLSST_noised['time_all'][index_notsaturated['ind_notsaturated_all']],LcTeoLSST_noised['mag_all'][index_notsaturated['ind_notsaturated_all']],LcTeoLSST_noised['dmag_all'][index_notsaturated['ind_notsaturated_all']], mv['filter'][index_notsaturated['ind_notsaturated_all']])
        P_multi = LS_multi.periodogram(periods)
        periodogram_noise=np.median(P_multi)
        periodogram_noise_mean=np.mean(P_multi)

        print('Noise level (median vs mean)')
        print(periodogram_noise,periodogram_noise_mean)
 


       ########This is to measure the best period 
        fitLS_multi= periodic.LombScargleMultiband(fit_period=True)
        fitLS_multi.optimizer.period_range=(minper_opt, maxper_opt)
        fitLS_multi.fit(LcTeoLSST_noised['time_all'][index_notsaturated['ind_notsaturated_all']],LcTeoLSST_noised['mag_all'][index_notsaturated['ind_notsaturated_all']],LcTeoLSST_noised['dmag_all'][index_notsaturated['ind_notsaturated_all']], mv['filter'][index_notsaturated['ind_notsaturated_all']])
        best_per_temp=fitLS_multi.best_period


        tmin=min(LcTeoLSST_noised['time_all'])
        tmax=max(LcTeoLSST_noised['time_all']) 
        cicli=(tmax-tmin)/period_model

        diffper=best_per_temp-period_model
        diffper_abs=abs(best_per_temp-period_model)/period_model*100
        diffcicli=abs(best_per_temp-period_model)/period_model*1/cicli
#        print(' Period of the model:')
#        print(period_model)
#        print(' Period found by Gatpy:')
#        print(best_per_temp)
#        print(' DeltaP/P (in perc):')
#        print(diffper)
#        print(' DeltaP/P*1/number of cycle:')
#        print(diffcicli)
#
 

        return best_per_temp,diffper,diffper_abs,diffcicli 
    
 
 

    def LcFitting(self,data,index,period,numberOfHarmonics):
        """
        Fit of the light curve and gives a dictionary with the mean amplitudes and magnitudes of the 
        fitting curve
        
        Parameters:
        -----------
        
        """

        zeroTimeRef=min(data['time_all'])

        print('fitting...')
        fitting=self.computingLcModel(data,period,numberOfHarmonics,index)
        #timeForModel=np.arange(data['timeu'][0],data['timeu'][0]+2*period,0.01)
        #computing the magModelFromFit
        if len(fitting['u'])>1:
            timeForModel=np.arange(data['timeu'][0],data['timeu'][0]+2*period,0.01)
            magModelFromFit_u=self.modelToFit(timeForModel,fitting['u']) 
            ampl_u=max(magModelFromFit_u)-min(magModelFromFit_u)
        else:
            magModelFromFit_u=[9999.]
            ampl_u=9999.
        #timeForModel=np.arange(data['timeg'][0],data['timeg'][0]+2*period,0.01)
        if len(fitting['g'])>1:
            timeForModel=np.arange(data['timeg'][0],data['timeg'][0]+2*period,0.01)
            #magModelFromFit_g=self.modelToFit(data['timeg'],fitting['g']) 
            magModelFromFit_g=self.modelToFit(timeForModel,fitting['g'])
            ampl_g=max(magModelFromFit_g)-min(magModelFromFit_g)
        else:
            magModelFromFit_g=[9999.]
            ampl_g=9999.
        #timeForModel=np.arange(data['timer'][0],data['timer'][0]+2*period,0.01)
        if len(fitting['r'])>1:
            timeForModel=np.arange(data['timer'][0],data['timer'][0]+2*period,0.01)
            #magModelFromFit_r=modelToFit(data['timer'],fitting['r']) 
            magModelFromFit_r=self.modelToFit(timeForModel,fitting['r'])
            ampl_r=max(magModelFromFit_r)-min(magModelFromFit_r)
        else:
            magModelFromFit_r=[9999.]
            ampl_r=9999.
       # timeForModel=np.arange(data['timei'][0],data['timei'][0]+2*period,0.01)

        if len(fitting['i'])>1:
            timeForModel=np.arange(data['timei'][0],data['timei'][0]+2*period,0.01)
            magModelFromFit_i=self.modelToFit(timeForModel,fitting['i'])

            #if len(magModelFromFit_i)>0:
            ampl_i=max(magModelFromFit_i)-min(magModelFromFit_i)
            #else:
                #ampl_i=9999.
        else:
            magModelFromFit_i=[9999.]
            ampl_i=9999.
        #timeForModel=np.arange(data['timez'][0],data['timez'][0]+2*period,0.01)    
        if len(fitting['z'])>1:
            timeForModel=np.arange(data['timez'][0],data['timez'][0]+2*period,0.01)
            magModelFromFit_z=self.modelToFit(timeForModel,fitting['z'])
            ampl_z=max(magModelFromFit_z)-min(magModelFromFit_z)
        else:
            magModelFromFit_z=[9999.]
            ampl_z=9999.
        #timeForModel=np.arange(data['timey'][0],data['timey'][0]+2*period,0.01)    
        if len(fitting['y'])>1:
            timeForModel=np.arange(data['timey'][0],data['timey'][0]+2*period,0.01)
            magModelFromFit_y=self.modelToFit(timeForModel,fitting['y']) 
            ampl_y=max(magModelFromFit_y)-min(magModelFromFit_y)
        else:
            magModelFromFit_y=[9999.]
            ampl_y=9999.



        meanMag_u=self.meanmag_antilog(magModelFromFit_u)
        meanMag_g=self.meanmag_antilog(magModelFromFit_g)
        meanMag_r=self.meanmag_antilog(magModelFromFit_r)
        meanMag_i=self.meanmag_antilog(magModelFromFit_i)
        meanMag_z=self.meanmag_antilog(magModelFromFit_z)
        meanMag_y=self.meanmag_antilog(magModelFromFit_y)







        finalResult={'mean_u':meanMag_u,'mean_g':meanMag_g,'mean_r':meanMag_r,
                     'mean_i':meanMag_i,'mean_z':meanMag_z,'mean_y':meanMag_y,
                     'ampl_u':ampl_u,'ampl_g':ampl_g,'ampl_r':ampl_r,
                     'ampl_i':ampl_i,'ampl_z':ampl_z,'ampl_y':ampl_y,
                'chi_u':fitting['chi_u'],'chi_g':fitting['chi_g'],'chi_r':fitting['chi_r'],
                'chi_i':fitting['chi_i'],'chi_z':fitting['chi_z'],'chi_y':fitting['chi_y'],
                'fittingParametersAllband':fitting}

        return finalResult   
    
    def modelToFit(self,time,param):
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

    def chisqr(self,residual,Ndat,Nvariable):
            chi=sum(pow(residual,2))/(Ndat-Nvariable)
            return chi

    def chisqr2(self,datax,datay,fitparameters,Ndat,Nvariable):
            residuals=self.modelToFit(datax,fitparameters)-datay
            chi2=sum(pow(residuals,2)/self.modelToFit(datax,fitparameters))*1/(Ndat-Nvariable)
            return chi2

    def residuals(self,datax,datay,fitparameters):
            residuals=self.modelToFit(datax,fitparameters)-datay
            return residuals


    def computingLcModel(self,data,period,numberOfHarmonics,index):
        def modelToFit2_fit(coeff):
            fit = self.modelToFit(x,coeff)
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
            fit_u,a=leastsq(modelToFit2_fit,parametersForLcFit)
            residual=self.residuals(x,y_proc,fit_u)
            chi_u=self.chisqr2(x,y_proc,fit_u,len(x),len(fit_u))
        else:
            fit_u=[9999.]
            chi_u=9999.
        x=time_g
        y_proc = np.copy(mag_g)
        if len(y_proc)>(numberOfHarmonics*2)+2:
            print('fitting g band')
            fit_g,a=leastsq(modelToFit2_fit,parametersForLcFit)
            residual=self.residuals(x,y_proc,fit_g)
            chi_g=self.chisqr2(x,y_proc,fit_g,len(x),len(fit_g))
        else:
            fit_g=[9999.]
            chi_g=9999.
        y_proc = np.copy(mag_r)
        x=time_r
        if len(y_proc)>(numberOfHarmonics*2)+2:
            print('fitting r band')
            fit_r,a=leastsq(modelToFit2_fit,parametersForLcFit)
            residual=self.residuals(x,y_proc,fit_r)
            chi_r=self.chisqr2(x,y_proc,fit_r,len(x),len(fit_r))
        else:
            fit_r=[9999.]
            chi_r=9999.
        x=time_i
        y_proc = np.copy(mag_i)
        if len(y_proc)>(numberOfHarmonics*2)+2:
            print('fitting i band')
            fit_i,a=leastsq(modelToFit2_fit,parametersForLcFit)
            residual=self.residuals(x,y_proc,fit_i)
            chi_i=self.chisqr2(x,y_proc,fit_i,len(x),len(fit_i))
        else:
            fit_i=[9999.]
            chi_i=9999.
        x=time_z
        y_proc = np.copy(mag_z)
        if len(y_proc)>(numberOfHarmonics*2)+2:
            print('fitting z band')
            fit_z,a=leastsq(modelToFit2_fit,parametersForLcFit)
            residual=self.residuals(x,y_proc,fit_z)
            chi_z=self.chisqr2(x,y_proc,fit_z,len(x),len(fit_z))
        else:
            fit_z=[9999.]
            chi_z=9999.
        x=time_y
        y_proc = np.copy(mag_y)
        if len(y_proc)>(numberOfHarmonics*2)+2:
            print('fitting y band')
            fit_y,a=leastsq(modelToFit2_fit,parametersForLcFit)
            residual=self.residuals(x,y_proc,fit_y)
            chi_y=self.chisqr2(x,y_proc,fit_y,len(x),len(fit_y))
        else:
            fit_y=[9999.]
            chi_y=9999.

        results={'u':fit_u,'g':fit_g,'r':fit_r,'i':fit_i,'z':fit_z,'y':fit_y,'chi_u':chi_u,
                 'chi_g':chi_g,'chi_r':chi_r,'chi_i':chi_i,'chi_z':chi_z,'chi_y':chi_y}

        return results
