__all__ = ("MDwarfCountMetric", "MDwarfMeasureMetric")

from uncertainties import unumpy
from base_lcl_metric import BaseLCLMetric

class MDwarfCountMetric(BaseLCLMetric):
    """
    Metric based on lightcurvelynx parquet file.
    Counts the number of flares that have been observed at least once
    in u, g, or r by checks if the flare flux is significantly different 
    than the quiescent state.
    Averages all the quiescent flux that we simulated pre-flare in order compare
    the fluxes and flux errors to the flare observation.
    
    Goal: checks observations of time within 1 FWHM on the left and ?20? FWHM on the right
    FOR NOW: checks observations within 1 day on left and 2 days on right
    (since FWHM is not read in in the parquet file)
    """

    def __init__(self, *args, **kwargs):
        #add keywork for saving a new file
        super().__init__(*args, **kwargs)

    def run(self, data_slice, slice_point):
        """Calculate metric values.

        Parameters
        ----------
        data_slice : `numpy.ndarray`, (N,)
           Values passed to metric by the slicer, which the metric will
           use to calculate metric values at each slice_point.
        slice_point : `dict` or None
           Dictionary of slice_point metadata passed to each metric.
           E.g. the ra/dec of the healpix pixel.

        Returns
        -------
        metricValue : `int` `float` or `object`
            The metric value at each slice_point."""

        data_slice = data_slice[0][0] #lynne or max - does anyone know why?
        detected_mask = data_slice['snr']>10 #only using detected data points
        observed_points = data_slice[detected_mask]
        bands = ["u","g","r"]
        #for each band, see if any data points are observed in flare
        for band in bands:
            observations_in_filter = observed_points[observed_points['filter']==band]
            pre_flare_mask = observations_in_filter['time_rel']<-1 #more than one day pre-peak
            potential_flare_mask = (observations_in_filter['time_rel']>-1)*\
                (observations_in_filter['time_rel']<2)

            pre_flare_average = np.average(unumpy.uarray(
                observations_in_filter[potential_flare_mask]['flux'], 
                observations_in_filter[potential_flare_mask]['fluxerr']))
            
            for i, obs in observations_in_filter[potential_flare_mask.values].iterrows():

                flux_diff = obs['flux'] - pre_flare_average.nominal_value
                error_combo = np.sqrt(obs['fluxerr']**2 + pre_flare_average.std_dev**2)
                if np.abs(flux_diff / error_combo)>3:
                    return True
            
        
        return False 