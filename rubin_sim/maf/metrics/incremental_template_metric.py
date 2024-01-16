import numpy as np
from .base_metric import BaseMetric

__all__ = ["TemplateTime"]

class TemplateTime(maf.BaseMetric):
    """Find the time at which we expect to hit incremental template availability.
    
    Note that there are some complications to real template generation that make this an 
    approximation and not an exact answer -- one aspect is that templates are generated in 
    `patches` and not per pixel. However, it may be possible to generate parts of these patches
    at about 5arcsecond scales, which implies running with a healpix slicer at nside=512 or 1024. 
    
    Parameters
    ----------
    n_visits : `int`, opt
        Number of qualified visits required for incremental template generation. 
        Default 3. 
    seeing_range : `float`, opt
        Range of seeing to allow in the qualified images. 
    m5_range : `float`, opt
        Range of m5 values to allow in the qualified images. Stand in for `weight` in template.
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
    
    def __init__(self, n_visits=3, seeing_ratio=2.0, m5_range=0.5, 
                 seeingCol='seeingFwhmEff', m5Col='fiveSigmaDepth',
                 nightCol = 'night', mjd_col = 'observationStartMJD', filter_col = 'filter', **kwargs):
        self.n_visits = n_visits
        self.seeing_ratio = seeing_ratio
        self.m5_range = m5_range
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
        if len(dataSlice) < self.n_visits:
            return self.badval
        
        # Check that the visits are sorted in night
        dataSlice.sort(order=self.nightCol)
        
        # Find the best seeing in the first few images
        bench_seeing = np.percentile(dataSlice[self.seeingCol],25)
        bench_m5 = np.percentile(dataSlice[self.m5Col],25)
        
        #Checks whether data was collected on night where seeing conditions were preferable
        seeing_ok = np.where(dataSlice[self.seeingCol]/bench_seeing < self.seeing_ratio, 
                            True, False)
        
        #Checks whether data was collected where seeing range was preferable
        m5_ok = np.where(bench_m5 - dataSlice[self.m5Col] < self.m5_range,
                        True, False)

        both = np.where(seeing_ok & m5_ok)[0]
        if len(both) < self.n_visits: # If seeing_ok and/or m5_ok are "false", returned as bad value
            return self.badval
            
        idx_template = both[self.n_visits - 1] # Nights visited
        
        n_template = dataSlice[self.nightCol][idx_template] # Night of template creation
        d_n = n_template - dataSlice[self.nightCol][0] # Number of nights for a template
        
        where_template = dataSlice[self.nightCol] > n_template # of later images where we have a template
        images_with_template = np.sum(where_template)
        
        template_m5 = 1.25 * np.log10(np.sum(10.0 ** (0.8 * dataSlice[self.m5Col][idx_template])))
        
        delta_m5 = -2.5 * np.log10(np.sqrt(1.0 + 10 ** (-0.8 * (template_m5 - dataSlice[self.m5Col]))))
        diff_m5s = dataSlice[self.m5Col] + delta_m5
        
        n_alerts = 1e4 * 10**(0.6 * (diff_m5s - 24.7))
        
        result["Night_template_created"] = n_template
        result["N_nights_before_template"] = d_n
        result["N_images_for_template"] = idx_template + 1
        result["N_images_with_template"] = images_with_template
        result["Template_m5"] = template_m5
        result["Total_alerts"] = np.sum(n_alerts[where_template])
        result["Diffim_lc"] = {
            "mjd":dataSlice[self.mjd_col][where_template],
            "diff_m5":diff_m5s[where_template],
            "band":dataSlice[self.filter_col][where_template],
            "science_m5":dataSlice[self.m5Col][where_template],
            "n_alerts":n_alerts[where_template]
        }
        
        return result
    
        
    def reduceNight(self, metricVal): # returns night of template creation
        return metricVal["Night_template_created"]
    
    def reduceDeltaNight(self, metricVal): # returns number of nights needed to complete template
        return metricVal["N_nights_before_template"]
    
    def reduceNVis(self, metricVal): # returns number of images needed to complete template
        return metricVal["N_images_for_template"]
    
    def reduceImage(self, metricVal): # returns number of images with a template
        return metricVal["N_images_with_template"]
    
    def reduceTemplate_m5(self, metricVal): # calculated coadded m5 of resulting template
        return metricVal["Template_m5"]
    
    def reduceTotal_alerts(self, metricVal):
        return metricVal["Total_alerts"]