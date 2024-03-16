__all__ = ("UniformMeanzBiasMetric",)

import numpy as np
"""
Still TODO:

"""
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import scipy
import sklearn
import rubin_sim
import rubin_sim.maf as maf
from rubin_sim.scheduler.utils import SkyAreaGenerator
from rubin_sim.data import get_baseline
import scipy.special as sc

from rubin_sim.maf.metrics.base_metric import BaseMetric

def compute_dzfromdm(zbins, band_ind, year, dzname):

    if dzname=='JQ':

        deriv = pd.read_pickle('/pscratch/sd/r/rhlozek/ObsStrat/code/meanz_uniformity/uniformity_pkl/meanzderiv.pkl')
        zvals = pd.read_pickle('/pscratch/sd/r/rhlozek/ObsStrat/code/meanz_uniformity/uniformity_pkl/meanzsy%i.pkl'%(year+1))
        meanzinterp = zvals[0:zbins,band_ind,5]
        dzdminterp = np.abs(deriv[year,band_ind,0:zbins])
    return dzdminterp, meanzinterp

def use_zbins(meanz_vals, figure_9_mean_z=np.array([0.2, 0.4, 0.7, 1.0]),  figure_9_width=0.2):
    max_z_use = np.max(figure_9_mean_z)+2*figure_9_width
    use_bins = meanz_vals < max_z_use
    return use_bins

def compute_Clbias(meanz_vals,scatter_mean_z_values,figure_9_mean_z=np.array([0.2, 0.4, 0.7, 1.0]), figure_9_Clbias =np.array([1e-3, 2e-3, 5e-3, 1.1e-2]),figure_9_width=0.2,figure_9_mean_z_scatter = 0.02):
    import numpy as np
    mzvals= np.array([float(mz) for mz in meanz_vals])
    sctz = np.array([float(sz)for sz in scatter_mean_z_values])
    
    fit_res = np.polyfit(figure_9_mean_z, figure_9_Clbias, 2)
    poly_fit = np.poly1d(fit_res)
    use_bins = use_zbins(meanz_vals,figure_9_mean_z, figure_9_width)

    mean_z_values_use = mzvals[use_bins]
    sctz_use = sctz[use_bins]

    Clbias = poly_fit(mean_z_values_use)
    rescale_fac =  sctz_use / figure_9_mean_z_scatter
    Clbias *= rescale_fac
    fit_res_bias = np.polyfit(mean_z_values_use, Clbias, 1)
    poly_fit_bias = np.poly1d(fit_res_bias)

    return poly_fit_bias(mean_z_values_use), mean_z_values_use

class UniformMeanzBiasMetricc(BaseMetric):
    """This calculates the bias in the weak lensing power given the scatter in the redshift of the tomographic sample
    induced by survey non-uniformity. It then computes the ratio of this bias to the desired y1 upper bound and the y10 
    DESC SRD requirement. Desire values are less than 1 by Y10. 

    This summary metric should be run on ExgalM5().


    """
    def __init__(
            self, nside=64, year=10, col=None, **kwargs
    ):
        self.nside = nside
        super().__init__(col=col, year=year, **kwargs)
        if col is None:
            self.col = "metricdata"
        self.year = year
        self.nside = nside

    def run(self, data_slice, slice_point=None):
        # Get a map that has zeros for the pixels we do not want to use.
        map = metric_values.data.copy()
        map[metric_values.mask] = 0

        # Start to set output.
        result = np.empty(1, dtype=[("name", np.str_, 20), ("value", float)])
        result["name"][0] = "UniformMeanzBiasMetric"

            # Get compute the quantities across all bands FoM for the overall map.  

        surveyAreas = SkyAreaGenerator(nside=nside)
        map_footprints, map_labels = surveyAreas.return_maps()
        days = self.year*365.25
        filter_list=["u","g","r","i"]
        zbins=5
        filts = dict(zip(filter_list, [None]*len(filter_list)))
        filtsz = dict(zip(filter_list, [[None]*zbins]*len(filter_list)))
        constraint_str='filter="YY" and note not like "DD%" and night <= XX and note not like "twilight_near_sun" '
        constraint_str = constraint_str.replace('XX','%d'%days)
        totdz=0
        avmeanz=0
        for filter_ind, filter in enumerate(filter_list):
            constraint_str = constraint_str.replace('YY','%s'%filter)

            # Now for some specifics - this part is just when using the overall map.
            use_slicer = maf.HealpixSubsetSlicer(
                nside=self.nside, use_cache=False, hpid=np.where(map_labels == "lowdust")[0])
            my_summary_stats = [maf.MedianMetric(), maf.MeanMetric(), maf.RmsMetric(), maf.PercentileMetric(percentile=25), maf.PercentileMetric(percentile=75)]

            depth_map_bundle = maf.MetricBundle(
                metric=maf.ExgalM5(), slicer=use_slicer, constraint=constraint_str, summary_metrics=my_summary_stats)
            bd = maf.metricBundles.make_bundles_dict_from_list([depth_map_bundle])
            dzdminterp, meanzinterp=compute_dzfromdm(zbins, filter_ind,self.year, 'JQ')
            stdz = [float(np.abs(dz))*float(bd[list(bd.keys())[0]].summary_values['Rms']) for dz in dzdminterp]
    
            clbias, meanz_use = compute_Clbias(meanzinterp,stdz)

            totdz+=[float(st**2) for st in stdz]
            avmeanz+=meanzinterp
            
        combined_tdz = [np.sqrt(tdz) for tdz in totdz]
        tmpclbias,tmpmeanz_use=compute_Clbias(avmeanz/len(filter_list),combined_tdz)
        

        y10_req = 0.003
        y1_goal = 0.013

        clbiastot = np.max(clbias)
        y10ratio = clbiastot/y10_req
        y1ratio = clbiastot/y1_goal
        result=[y1ratio,y10ratio]

        return result