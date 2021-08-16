import numpy as np
from rubin_sim.maf.utils import m52snr
import rubin_sim.maf.metrics as metrics
import os
from rubin_sim.utils import hpid2RaDec, equatorialFromGalactic
import rubin_sim.maf.slicers as slicers
from rubin_sim.data import get_data_dir

# Via Natasha Abrams nsabrams@college.harvard.edu
def microlensing_amplification(t, impact_parameter=1, crossing_time=1825., peak_time=100,
                               blending_factor = 1):
    """The microlensing amplification

    Parameters
    ----------
    t : float
        The time of observation (days)
    impact_parameter : float (1)
        The impact Parameter (0 means big amplification)
    crossing_time : float (1825)
        Einstein crossing time (days)
    peak_time : float (100)
        The peak time (days)
    blending_factor: float (1)
        The blending factor where 1 is unblended
    """

    lightcurve_u = np.sqrt(impact_parameter**2 + ((t-peak_time)**2/crossing_time**2))
    amplification = ((lightcurve_u**2 + 2)/(lightcurve_u*np.sqrt(lightcurve_u**2 + 4))
                     *blending_factor + (1 - blending_factor))

    return amplification

def info_peak_before_t0(impact_parameter=1, crossing_time=100.):
    """Time of Maximum Information before peak
    via Markus Hundertmark markus.hundertmark@uni-heidelberg.de
    
    Parameters
    ----------
    impact_parameter : float (1)
        The impact Parameter (0 means big amplification)
    crossing_time : float (1825)
        Einstein crossing time (days)
    """
    
    optimal_time = crossing_time*np.sqrt(-impact_parameter**2 +
                                         np.sqrt(9*impact_parameter**4 + 36*impact_parameter**2 + 4) - 2)/2
    return np.array(optimal_time)


class MicrolensingMetric(metrics.BaseMetric):
    """
    Quantifies detectability of Microlensing events.
    Can also return the number of datapoints within two crossing times of the peak of event.

    Parameters
    ----------
    metricCalc: str
        Type of metric. If metricCalc == 'detect', returns the number of microlensing events
        detected within certain parameters. If metricCalc == 'Npts', returns the number of points
        within two crossing times of the peak of the vent.
        Default is 'detect'
        
    ptsNeeded : int
        Number of an object's lightcurve points required to be above the 5-sigma limiting depth
        before it is considered detected.
        
    time_before_peak: int or str
        Number of days before lightcurve peak to qualify event as triggered.
        If time_before_peak == 'optimal', the number of days before the lightcurve peak
        is the time of maximal information.
        Default is 0.
    
    detect: bool
        By default we trigger which only looks at points before the peak of the lightcurve.
        When detect = True, observations on either side of the lightcurve are considered.
        Default is False.

    Notes
    -----
    Expects slicePoint to have keys of:
        peak_time : float (days)
        crossing_time : float (days)
        impact_parameter : float (positive)
        blending_factors : float (between 0 and 1 - optional)

    """
    def __init__(self, metricName='MicrolensingMetric', metricCalc = 'detect',
                 mjdCol='observationStartMJD', m5Col='fiveSigmaDepth',
                 filterCol='filter', nightCol='night',
                 ptsNeeded=3, rmag=20, detect_sigma=3., time_before_peak=0, detect=False, **kwargs):
        self.metricCalc = metricCalc
        if metricCalc == 'detect':
            self.units = 'Detected, 0 or 1'
        elif metricCalc == 'Npts':
            self.units = 'Npts within 2tE'
        else:
            raise Exception('metricCalc must be "detect" or "Npts"')
        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.nightCol = nightCol
        self.ptsNeeded = ptsNeeded
        self.detect_sigma = detect_sigma
        self.time_before_peak = time_before_peak
        self.detect = detect
        # For now, let's just have a flat SED
        # XXX--update to a stellar type
        filters = 'ugrizy'
        self.mags = {}
        for filtername in filters:
            self.mags[filtername] = rmag

        cols = [self.mjdCol, self.m5Col, self.filterCol, self.nightCol]
        super(MicrolensingMetric, self).__init__(col=cols,
                                                 units=self.units,
                                                 metricName=metricName,
                                                 **kwargs)

    def run(self, dataSlice, slicePoint=None):
        if self.detect == True and self.time_before_peak > 0:
            raise Exception("When detect = True, time_before_peak must be zero")
        # Generate the lightcurve for this object
        # make t a kind of simple way
        t = dataSlice[self.mjdCol] - np.min(dataSlice[self.nightCol])
        t = t - t.min()

        # Try for if a blending factor slice was added if not default to no blending factor
        try:
            amplitudes = microlensing_amplification(t, impact_parameter=slicePoint['impact_parameter'],
                                                    crossing_time=slicePoint['crossing_time'],
                                                    peak_time=slicePoint['peak_time'],
                                                    blending_factor=slicePoint['blending_factor'])
        
        except:
            amplitudes = microlensing_amplification(t, impact_parameter=slicePoint['impact_parameter'],
                                                    crossing_time=slicePoint['crossing_time'],
                                                    peak_time=slicePoint['peak_time'])

        filters = np.unique(dataSlice[self.filterCol])
        amplified_mags = amplitudes * 0

        for filtername in filters:
            infilt = np.where(dataSlice[self.filterCol] == filtername)[0]
            amplified_mags[infilt] = self.mags[filtername] - 2.5*np.log10(amplitudes[infilt])

        # The SNR of each point in the light curve
        snr = m52snr(amplified_mags, dataSlice[self.m5Col])
        # The magnitude uncertainties that go with amplified mags
        mag_uncert = 2.5*np.log10(1+1./snr)

        n_pre = []
        n_post = []
        for filtername in filters:
            if self.metricCalc == 'detect':
                if self.time_before_peak == 'optimal':
                    time_before_peak_optimal = info_peak_before_t0(slicePoint['impact_parameter'],
                                                                   slicePoint['crossing_time'])
                    # observations pre-peak and in the given filter
                    infilt = np.where((dataSlice[self.filterCol] == filtername) &
                                      (t < (slicePoint['peak_time'] - time_before_peak_optimal)))[0]
                
                else:
                    # observations pre-peak and in the given filter
                    infilt = np.where((dataSlice[self.filterCol] == filtername) &
                                      (t < (slicePoint['peak_time'] - self.time_before_peak)))[0]
                            
                # observations post-peak and in the given filter
                outfilt = np.where((dataSlice[self.filterCol] == filtername) &
                                   (t > slicePoint['peak_time']))[0]
                # Broadcast to calc the mag_i - mag_j
                diffs = amplified_mags[infilt] - amplified_mags[infilt][:, np.newaxis]
                diffs_uncert = np.sqrt(mag_uncert[infilt]**2 + mag_uncert[infilt][:, np.newaxis]**2)
                diffs_post = amplified_mags[outfilt] - amplified_mags[outfilt][:, np.newaxis]
                diffs_post_uncert = np.sqrt(mag_uncert[outfilt]**2 + mag_uncert[outfilt][:, np.newaxis]**2)

                # Calculating this as a catalog-level detection. In theory,
                # we could have a high SNR template image, so there would be
                # little to no additional uncertianty from the subtraction.

                sigma_above = np.abs(diffs)/diffs_uncert
                sigma_above_post = np.abs(diffs_post)/diffs_post_uncert
                # divide by 2 because array has i,j and j,i
                n_above = np.size(np.where(sigma_above > self.detect_sigma)[0])/2
                n_pre.append(n_above)
                n_above_post = np.size(np.where(sigma_above_post > self.detect_sigma)[0])/2
                n_post.append(n_above_post)
                            
            elif self.metricCalc == 'Npts':
                # observations pre-peak and in the given filter within 2tE
                infilt = np.where((dataSlice[self.filterCol] == filtername) &
                                  (t < (slicePoint['peak_time'])) &
                                  (t > (slicePoint['peak_time'] - slicePoint['crossing_time'])))[0]
                # observations post-peak and in the given filter within 2tE
                outfilt = np.where((dataSlice[self.filterCol] == filtername) &
                                   (t > (slicePoint['peak_time'])) &
                                   (t < (slicePoint['peak_time'] + slicePoint['crossing_time'])))[0]
                            
                n_pre.append(len(infilt))
                n_post.append(len(outfilt))

        npts = np.sum(n_pre)
        npts_post = np.sum(n_post)
        if self.metricCalc == 'detect':
            if self.detect == True:
                if npts >= self.ptsNeeded and npts_post >=self.ptsNeeded:
                    return 1
                else:
                    return 0
            else:
                if npts >= self.ptsNeeded:
                    return 1
                else:
                    return 0
        elif self.metricCalc == 'Npts':
            return npts + npts_post


def generateMicrolensingSlicer(min_crossing_time=1, max_crossing_time=10, t_start=1,
                               t_end=3652, n_events=10000, seed=42, nside=128, filtername='r'):
    """
    Generate a UserPointSlicer with a population of microlensing events. To be used with
    MicrolensingMetric

    Parameters
    ----------
    min_crossing_time : float (1)
        The minimum crossing time for the events generated (days)
    max_crossing_time : float (10)
        The max crossing time for the events generated (days)
    t_start : float (1)
        The night to start generating peaks (days)
    t_end : float (3652)
        The night to end generating peaks (days)
    n_events : int (10000)
        Number of microlensing events to generate
    seed : float (42)
        Random number seed
    nside : int (128)
        HEALpix nside, used to pick which stellar density map to load
    filtername : str ('r')
        The filter to use for the stellar density map
    """
    np.random.seed(seed)

    crossing_times = np.random.uniform(low=min_crossing_time, high=max_crossing_time, size=n_events)
    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)
    impact_paramters = np.random.uniform(low=0, high=1, size=n_events)

    mapDir = os.path.join(get_data_dir(), 'maps', 'TriMaps')
    data = np.load(os.path.join(mapDir, 'TRIstarDensity_%s_nside_%i.npz' % (filtername, nside)))
    starDensity = data['starDensity'].copy()
    # magnitude bins
    bins = data['bins'].copy()
    data.close()

    star_mag = 22
    bin_indx = np.where(bins[1:] >= star_mag)[0].min()
    density_used = starDensity[:, bin_indx].ravel()
    order = np.argsort(density_used)
    # I think the model might have a few outliers at the extreme, let's truncate it a bit
    density_used[order[-10:]] = density_used[order[-11]]

    # now, let's draw N from that distribution squared
    dist = density_used[order]**2
    cumm_dist = np.cumsum(dist)
    cumm_dist = cumm_dist/np.max(cumm_dist)
    uniform_draw = np.random.uniform(size=n_events)
    indexes = np.floor(np.interp(uniform_draw, cumm_dist, np.arange(cumm_dist.size)))
    hp_ids = order[indexes.astype(int)]
    gal_l, gal_b = hpid2RaDec(nside, hp_ids, nest=True)
    ra, dec = equatorialFromGalactic(gal_l, gal_b)

    # Set up the slicer to evaluate the catalog we just made
    slicer = slicers.UserPointsSlicer(ra, dec, latLonDeg=True, badval=0)
    # Add any additional information about each object to the slicer
    slicer.slicePoints['peak_time'] = peak_times
    slicer.slicePoints['crossing_time'] = crossing_times
    slicer.slicePoints['impact_parameter'] = impact_paramters

    return slicer
