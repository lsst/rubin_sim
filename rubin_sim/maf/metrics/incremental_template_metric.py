import numpy as np
from .base_metric import BaseMetric

__all__ = ["TemplateTime"]

class TemplateTime(BaseMetric):
    """Find the time at which we expect to hit incremental template availability.
    
    Note that there are some complications to real template generation that make this an 
    approximation and not an exact answer -- one aspect is that templates are generated in 
    `patches` and not per pixel. However, it may be possible to generate parts of these patches
    at about 5arcsecond scales, which implies running with a healpix slicer at nside=512 or 1024. 
    
    Parameters
    ----------
    n_images_in_template : `int`, opt
        Number of qualified visits required for incremental template generation. 
        Default 3. 
    seeing_percentile : `float`, opt
        Maximum percentile seeing to allow in the qualified images (0 - 100). 
    m5_percentile : `float`, opt
        Maximum percentile m5 to allow in the qualified images (0 - 100). Stand in for `weight` in template.
    seeingCol : `str`, opt
        Name of the seeing column to use.
    m5Col : `str`, opt
        Name of the five sigma depth columns.
    nightCol : `str`, opt
        Name of the column describing the night of the visit.
    mjd_col : `str`, opt
        Name of column describing time of the visit
    filter_col : `str`, opt
        Name of column describing filter
    """
    
    def __init__(self, n_images_in_template=3, seeing_percentile=25, m5_percentile=25, 
                 seeingCol='seeingFwhmEff', m5Col='fiveSigmaDepth',
                 nightCol = 'night', mjd_col = 'observationStartMJD', filter_col = 'filter', **kwargs):
        self.n_images_in_template = n_images_in_template
        self.seeing_percentile = seeing_percentile
        self.m5_percentile = m5_percentile
        self.seeingCol = seeingCol
        self.m5Col = m5Col
        self.nightCol = nightCol
        self.mjd_col = mjd_col
        self.filter_col = filter_col
        if 'metric_name' in kwargs:
            self.metric_name = kwargs['metric_name']
            del kwargs['metric_name']
        else:
            self.metric_name = 'TemplateTime'
        super().__init__(col=[self.seeingCol, self.m5Col, self.nightCol, self.mjd_col, self.filter_col],
                         metric_name=self.metric_name, units="days", **kwargs)
        # Looking at names of seeing columns, five sigma depth, and nights associated with template visits

    def run(self, dataSlice, slice_point=None):
        result = {}
        
        # Bail if not enough visits at all
        if len(dataSlice) < self.n_images_in_template:
            return self.badval
        
        # Check that the visits are sorted in time
        dataSlice.sort(order=self.mjd_col)
        
        # Find the threshold seeing and m5
       
        bench_seeing = np.percentile(dataSlice[self.seeingCol], self.seeing_percentile)
        bench_m5 = np.percentile(dataSlice[self.m5Col], 100 - self.m5_percentile)
        
        #Checks whether data was collected on night where seeing conditions were preferable
        seeing_ok = np.where(dataSlice[self.seeingCol] < bench_seeing, 
                            True, False)
        
        #Checks whether data was collected where seeing range was preferable
        m5_ok = np.where(dataSlice[self.m5Col] > bench_m5,
                        True, False)

        ok_template_input = np.where(seeing_ok & m5_ok)[0]
        if len(ok_template_input) < self.n_images_in_template: # If seeing_ok and/or m5_ok are "false", returned as bad value
            return self.badval
            
        idx_template_created = ok_template_input[self.n_images_in_template - 1] # Last image needed for template
        idx_template_inputs = ok_template_input[:self.n_images_in_template] # Images included in template
        
        Night_template_created = dataSlice[self.nightCol][idx_template_created]
        N_nights_without_template = Night_template_created - dataSlice[self.nightCol][0]
        
        where_template = dataSlice[self.nightCol] > Night_template_created # of later images where we have a template
        N_images_with_template = np.sum(where_template)
        
        template_m5 = 1.25 * np.log10(np.sum(10.0 ** (0.8 * dataSlice[self.m5Col][idx_template_inputs])))
        
        delta_m5 = -2.5 * np.log10(np.sqrt(1.0 + 10 ** (-0.8 * (template_m5 - dataSlice[self.m5Col]))))
        diff_m5s = dataSlice[self.m5Col] + delta_m5
        
        n_alerts_per_diffim = 1e4 * 10**(0.6 * (diff_m5s - 24.7))
        
        result["Night_template_created"] = Night_template_created
        result["N_nights_without_template"] = N_nights_without_template
        result["N_images_until_template"] = idx_template_created + 1
        result["N_images_with_template"] = N_images_with_template
        result["Template_m5"] = template_m5
        result["Total_alerts"] = np.sum(n_alerts_per_diffim[where_template])
        result["Template_input_m5s"] = dataSlice[self.m5Col][idx_template_inputs]
        result["Diffim_lc"] = {
            "mjd":dataSlice[self.mjd_col][where_template],
            "night":dataSlice[self.nightCol][where_template],
            "diff_m5":diff_m5s[where_template],
            "band":dataSlice[self.filter_col][where_template],
            "science_m5":dataSlice[self.m5Col][where_template],
            "n_alerts":n_alerts_per_diffim[where_template]
        }
        
        return result
    
        
    def reduceNight_template_created(self, metricVal): # returns night of template creation
        return metricVal["Night_template_created"]
    
    def reduceN_nights_without_template(self, metricVal): # returns number of nights needed to complete template
        return metricVal["N_nights_without_template"]
    
    def reduceN_images_until_template(self, metricVal): # returns number of images needed to complete template
        return metricVal["N_images_until_template"]
    
    def reduceN_images_with_template(self, metricVal): # returns number of images with a template
        return metricVal["N_images_with_template"]
    
    def reduceTemplate_m5(self, metricVal): # calculated coadded m5 of resulting template
        return metricVal["Template_m5"]
    
    def reduceTotal_alerts(self, metricVal):
        return metricVal["Total_alerts"]