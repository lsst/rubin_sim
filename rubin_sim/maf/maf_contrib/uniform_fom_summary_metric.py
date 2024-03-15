__all__ = ("UniformFoMMetric",)

import numpy as np
"""
Still TODO:
  - Set year as in Lynne's comment
  - Exclude DDFs, twilight exposures, and only use low dust regions.
"""
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd
import scipy
import sklearn

from rubin_sim.maf.metrics.base_metric import BaseMetric

def make_clustering_dataset(map, maskval=0, priority_fac=0.9, nside=64):
    """This utility takes a map of values across the sky, and puts it in a form that is convenient
    for running a clustering algorithm on it.

    It assumes masked regions are set to `maskval`, and masks them.  It also rescales the dimensions
    to be comparable in magnitude, modulo some priority factor that has a reasonable default.
    """
    if priority_fac<0 or priority_fac>=1:
        raise ValueError("priority_fac must lie between 0 and 1")

    # Get RA and Dec
    npix = hp.nside2npix(nside)
    ra, dec = hp.pix2ang(hp.npix2nside(npix), range(npix), lonlat=True)
    
    # Make a 3D numpy array containing the unmasked regions, including a rescaling factor as 
    # needed.
    n_unmasked = len(map[map>maskval])
    my_data = np.zeros((n_unmasked, 3))
    my_data[:,0] = ra[map>maskval]*(1-priority_fac)*np.std(map[map>maskval])/np.std(ra[map>maskval])
    my_data[:,1] = dec[map>maskval]*(1-priority_fac)*np.std(map[map>maskval])/np.std(dec[map>maskval])
    my_data[:,2] = map[map>maskval]
    return my_data

def apply_clustering(clustering_data):
    """This is just a thin wrapper around sklearn clustering routines.

    We are looking for patterns induced by rolling, which generally result in two clusters
    (deeper vs. shallower).  So we fix the number of clusters to two.
    """
    from sklearn.cluster import KMeans
    clustering = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(clustering_data)
    # We would like the labels to be 1 and 2.
    labels = clustering.labels_ + 1
    return labels

def expand_labels(map, labels, maskval=0):
    """A utility to apply the labels from a masked version of a depth map back to the entire depth map."""
    expanded_labels = np.zeros(hp.nside2npix(nside))
    expanded_labels[map>maskval] = labels
    return expanded_labels

def is_ngp(ra, dec):
    """Returns True if the location is in the northern galactic region, False if in the South."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    lat = c.galactic.b.deg
    return lat >= 0

def get_stats_by_region(map, nside, maskval=0, region='all'):
    """Get statistics of the input map -- either the entire thing, the northern galactic region, or
    the southern galactic region depending on the value of the region parameter.  Options: 'all',
    'north', 'south'."""
    if region not in ['all','north','south']:
        raise ValueError('Invalid region %s'%region)
    to_use = (map > maskval)
    
    if region != 'all':
        # Find the north/south part of the map as requested
        npix = hp.nside2npix(nside)
        ra, dec = hp.pix2ang(hp.npix2nside(npix), range(npix), lonlat=True)
        ngp_mask = is_ngp(ra, dec)
        if region=='north':
            reg_mask = ngp_mask
        else: 
            reg_mask = ~ngp_mask
        to_use = to_use & reg_mask
    
    # Calculate the desired stats
    reg_mad = scipy.stats.median_abs_deviation(map[to_use])
    reg_median = np.median(map[to_use])
    reg_std = np.std(map[to_use])
    
    # Return the values
    return(reg_mad, reg_median, reg_std)

def has_stripes(map, nside, threshold=0.1):
    """
    A utility to ascertain whether a particular routine has stripe-y features in a map.  It is tuned
    to identify the striping in the RIZ coadded exposure time based on the implementation of rolling
    in v3.3 and v3.4 simulations, though should work well unless major changes are made to the
    rolling patterns.
    """
    # Analyze the map to get MAD, median, std in north/south.
    mad = {}
    med = {}
    frac_scatter = {}
    regions = ['north', 'south']
    for region in regions:
        mad[region], med[region], _ = get_stats_by_region(map, nside, region=region)
        frac_scatter[region] = mad[region]/med[region]
    test_statistic = np.abs(frac_scatter['north']/frac_scatter['south']-1)
    if test_statistic < threshold:
        return False
    else:
        return True

class UniformFoMMetric(BaseMetric):
    """This calculates any impact on the dark energy Figure of Merit (FoM) for combined DESC static
    3x2pt measurements due to having to eliminate areas with coherent large-scale depth fluctuations
    due to residual rolling stripes in the coadd.  It is relevant in the limit that the striping is
    significant, as this is unlikely to be something we can model.

    This summary metric should be run on RIZDetectionCoaddExposureTime().

    A return value of 1 is the best option, as it means no area has to be removed (indicative of a
    realistically achievable level of uniformity). Return values below 1 indicate a loss of
    statistical constraining power. Non-rolling strategies generally return 1 for all years, while
    rolling strategies return 1 for year 1 and some years after that depending on the rolling
    implementation.

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

        # Check whether the map has stripe features
        stripes = has_stripes(map, self.nside)

        # Start to set output.
        result = np.empty(1, dtype=[("name", np.str_, 20), ("value", float)])
        result["name"][0] = "UniformFoMFraction"

        # If there are no stripes, we are done here.
        if not stripes:
            result["value"][0] = 1.0
            return result
        else:
            # But if there are stripes, we have to do the clustering to identify the
            # deeper/shallower regions.
            clustering_data = make_clustering_dataset(map)
            labels = apply_clustering(clustering_data)

            # Then we figure out which pixels are in each region.
            expanded_labels = expand_labels(map, labels)
            hpid_region = {}
            hpid_region[1] = np.where(labels == 1)[0]
            hpid_region[2] = np.where(labels == 2)[0]

            # Get 3x2pt FoM for the overall map.  Start with some basic definitions we will always use.
            surveyAreas = SkyAreaGenerator(nside=nside)
            map_footprints, map_labels = surveyAreas.return_maps()
            days = self.year*365.25
            use_filter = "i"
            constraint_str='filter="YY" and note not like "DD%" and night <= XX and note not like "twilight_near_sun" '
            constraint_str = constraint_str.replace('XX','%d'%days)
            constraint_str = constraint_str.replace('YY','%s'%use_filter)
            ThreebyTwoSummary = maf.StaticProbesFoMEmulatorMetric(
                nside=self.nside, metric_name="3x2ptFoM")

            # Now for some specifics - this part is just when using the overall map.
            use_slicer = maf.HealpixSubsetSlicer(
                nside=self.nside, use_cache=False, hpid=np.where(map_labels == "lowdust")[0])
            depth_map_bundle = maf.MetricBundle(
                metric=maf.ExgalM5(), slicer=use_slicer, constraint=constraint_str, summary_metrics=[ThreebyTwoSummary])
            bd = maf.metricBundles.make_bundles_dict_from_list([depth_map_bundle])
            ### Any other magic needed here to actually get the summary?
            overall_fom = bd[list(bd.keys())[0]].summary_values['3x2ptFoM']

            fom_region = {}
            for region in [1, 2]:
                # Need a slicer that selects for that region.
                use_slicer = maf.HealpixSubsetSlicer(
                    nside=self.nside, use_cache=False,
                    hpid=np.intersect1d(np.where(map_labels == "lowdust")[0], hpid_region[region]))
                depth_map_bundle = maf.MetricBundle(
                    metric=maf.ExgalM5(), slicer=use_slicer, constraint=constraint_str,
                    summary_metrics=[ThreebyTwoSummary])
                bd = maf.metricBundles.make_bundles_dict_from_list([depth_map_bundle])
                ### Any other magic needed here to actually get the summary?
                fom_region[region] = bd[list(bd.keys())[0]].summary_values['3x2ptFoM']

            # Now choose the larger FoM, and return the ratio between that and the one without any
            # area cuts
            fom = np.max((fom_region[1], fom_region[2]))
            result["value"][0] = fom / overall_fom
            return result
