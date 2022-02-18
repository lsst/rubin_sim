################################################################################################
# Metric to evaluate contemporaneous and complementary observations of
# microlensing events in the Roman Galactic Exoplanet Survey field from the
# Rubin as well as the Roman Observatory.
#
# Author - Rachel Street: rstreet@lco.global
################################################################################################
import os
import numpy as np
import healpy as hp
import rubin_sim.maf as maf
from astropy import units as u
from astropy_healpix import HEALPix
from astropy.coordinates import Galactic, TETE, SkyCoord
from astropy.time import Time
from datetime import datetime
from rubin_sim.data import get_data_dir
import readGalPlaneMaps

class MicrolensingEvent():

    def __init__(self):
        self.t0 = None
        self.tE = None
        self.rho = None
        self.startMJD = None
        self.endMJD = None
        self.romanSeason = None
        self.romanTimestamps = np.array([])

    def calcDuration(self):
        """Calculates the estimated MJD date range between which an event
        might be detectable.  For the purposes of this metric, this is
        assumed to be between (t0-tE) >= t <= (t0+tE)"""
        self.startMJD = self.t0 - self.tE
        self.endMJD = self.t0 + self.tE

class RGESSurvey():
    """Parameters describing the Roman Galactic Exoplanet Survey, as described
    in Johnson et al., 2021, AJ, 160, id.123
    https://arxiv.org/pdf/2006.10760.pdf

    Due to spacecraft pointing constraints, it can only observe the
    Galactic Bulge field between ∼Feb 12– Apr 24 and ∼Aug 19 – Oct 29, each
    year, so the survey will be conducted as seasons of nominal lenght
    of 72days.  This will take place between ∼2026 to 2030.

    As the mission is due for launch in 2025, these parameters are currently
    nominal and likely subject to later revision.
    """
    def __init__(self):
        self.location = {'l_center': 2.216, 'b_center': -3.14,
                            'l_width': 1.75, 'b_height': 1.75}
        self.seasonLength = 72.0    # days
        self.nSeasons = 6
        self.cadence = 15.0 / (60.0 * 24.0)         # days
        self.seasons = [
                            {'start': '2026-02-12T00:00:00',
                            'end': '2026-04-24T00:00:00'},
                            {'start': '2026-08-19T00:00:00',
                            'end': '2026-10-29T00:00:00'},
                            {'start': '2027-02-12T00:00:00',
                            'end': '2027-04-24T00:00:00'},
                            {'start': '2027-08-19T00:00:00',
                            'end': '2027-10-29T00:00:00'},
                            {'start': '2028-02-12T00:00:00',
                            'end': '2028-04-24T00:00:00'},
                            {'start': '2028-08-19T00:00:00',
                            'end': '2028-10-29T00:00:00'},
                            ]

        mjdSeasons = []
        for season in self.seasons:
            start = Time(season['start'], format='isot', scale='utc')
            end = Time(season['end'], format='isot', scale='utc')
            mjdSeasons.append( {'start': start.mjd, 'end': end.mjd} )
        self.seasons = mjdSeasons

    def calcHealpix(self,ahp):
        self.skycoord = SkyCoord(self.location['l_center']*u.deg,
                                 self.location['b_center']*u.deg,
                                 frame=Galactic())
        self.pixels = ahp.cone_search_skycoord(self.skycoord,
                                            self.location['l_width']*u.deg/2.0)

    def calcTimeStamps(self):
        """Calculate the timestamps of observations within the specified
        observing seasons"""

        self.timestamps = np.array([])

        for season in self.seasons:
            self.timestamps = np.concatenate( (self.timestamps,
                                            np.arange(season['start'],
                                                        season['end'],
                                                        self.cadence)) )

def simLensingEvents(nSimEvents,obsSeasons,nSeasons):
    """Based on code by Etienne Bachelet, adapted by Rachel Street"""

    events = []

    # Configure boundaries to the t0 (time of closest approach) of all
    # simulated events to lie between the start of the first RGES season and
    for i in range(0, nSimEvents, 1):
        event = MicrolensingEvent()

        # Generate a time of apparent closest approach, t0, between lens and
        # source star that lies between the start and end of one of the RGES
        # seasons.
        event.romanSeason = np.random.randint(0,high=nSeasons)
        event.t0 = np.random.uniform(obsSeasons[event.romanSeason]['start'],
                                    obsSeasons[event.romanSeason]['end'])

        # Generate tE: Einstein crossing time,
        iLensType = np.random.randint(0,high=2)
        if iLensType == 0:
            # Lenses with stellar host
            event.tE = abs(np.random.normal(30,15))
        else:
            # Lenses with compact object host
            event.tE = abs(np.random.normal(100,50))

        # Generate rho: angular source size in units of angular Einstein,
        # radius including a range from stellar dwarfs to giant stars
        rho = np.random.uniform(-4,-1.30)
        event.rho = 10**rho

        # Calculate the start and end MJD of this event:
        event.calcDuration()

        events.append( event )

    return events

class lensDetectRubinRomanMetric(maf.BaseMetric):
    """Metric to evaluate the fraction of microlensing events in the Roman
    Galactic Exoplanet Survey field that will be detected from the
    Rubin as well as the Roman Observatory.

    Parameters
    ----------
    fieldRA : float, RA in degrees of a given pointing
    fieldDec : float, Dec in degrees of a given pointing
    filter : str, filter bandpass used for a given observation
    observationStartMJD : float, MJD timestamp of the start of a given observation
    visitExposureTime : float, exposure time in seconds
    fiveSigmaDepth : float, limiting magnitude at 5sigma
    """

    def __init__(
        self,
        cols=[
            "fieldRA",
            "fieldDec",
            "filter",
            "observationStartMJD",
            "visitExposureTime",
            "fiveSigmaDepth",
        ],
        metricName="lensDetectRubinRomanMetric",
        **kwargs
    ):
        """Kwargs must contain:
        filters  list Filterset over which to compute the metric
        """

        self.ra_col = "fieldRA"
        self.dec_col = "fieldDec"
        self.filterCol = "filter"
        self.m5Col = "fiveSigmaDepth"
        self.mjdCol = "observationStartMJD"
        self.exptCol = "visitExposureTime"
        self.filters = ["u", "g", "r", "i", "z", "y"]
        self.magCuts = {
            "u": 22.7,
            "g": 24.1,
            "r": 23.7,
            "i": 23.1,
            "z": 22.2,
            "y": 21.4,
        }
        self.NSIDE = 64
        self.nSimEvents = 2000
        self.simEvents = []
        self.defineRGES()

        super().__init__(col=cols, metricName=metricName)

    def defineRGES(self):
        self.RGES = RGESSurvey()

    def calcRomanObsInWindow(self,event):

        seasonStart = self.RGES.seasons[event.romanSeason]['start']
        seasonEnd = self.RGES.seasons[event.romanSeason]['end']

        obsWindowStart = max( event.startMJD, seasonStart )
        obsWindowEnd = min( event.startMJD, seasonEnd )

        if obsWindowEnd > obsWindowStart:
            event.romanTimestamps = np.arange(obsWindowStart,
                                                obsWindowEnd,
                                                self.RGES.cadence)

    def run(self, dataSlice, slicePoint=None):

        # Minimum number of datapoints required within the event duration to
        # consider that event detected:
        minDataPoints = 10
        ahp = HEALPix(nside=self.NSIDE, order='ring', frame=TETE())
        self.defineRGES()
        self.RGES.calcHealpix(ahp)

        # Simulate a range of event parameters,
        # for events within the RGES footprint
        self.events = simLensingEvents(self.nSimEvents,
                                        self.RGES.seasons,
                                        self.RGES.nSeasons)

        # Select Rubin observations with adequate signal-to-noise,
        # combining data from all filters
        match_obs = []
        for i, f in enumerate(self.filters):
            idx1 = np.where(dataSlice[self.filterCol] == f)[0]
            idx2 = np.where(dataSlice[self.m5Col] >= self.magCuts[f])[0]
            match = list(set(idx1).intersection(set(idx2)))
            match_obs += match

        # Calculate the coordinates of where the matching observations
        # were acquired and convert this to a list of HEALpixels
        coords_icrs = SkyCoord(
            dataSlice[self.ra_col][match_obs],
            dataSlice[self.dec_col][match_obs],
            frame="icrs",
            unit=(u.deg, u.deg),
        )
        coords_gal = coords_icrs.transform_to(Galactic())
        pixels = ahp.skycoord_to_healpix(coords_gal)

        # Identify the observations from HEALpixels in the RGES footprint
        overlap_pixels = list(set(pixels.tolist()).intersection(set(self.RGES.pixels.tolist())))

        # Identify which observations from the dataSlice correspond to
        # the overlapping survey region.  This may produce multiple
        # indices in the array, referred to different observations
        match = np.array(match_obs)
        match_obs = []
        for p in overlap_pixels:
            ip = np.where(pixels == p)[0]
            match_obs += match[ip].tolist()

        # Extract the timestamps of matching Rubin observations of the
        # RGES footprint
        rubinTimestamps = dataSlice[self.mjdCol][match_obs]

        # Compute metric - number of events jointly detected by both surveys
        nJointDetections = 0
        for i in range(0, self.nSimEvents, 1):

            # Calculate the timestamps of Roman observations within the
            # event timeframe, taking account of the Roman seasons
            self.calcRomanObsInWindow(self.events[i])

            # Metric value: Test whether both observatories would provide
            # at least 10 datapoints within the event window
            if len(rubinTimestamps) > minDataPoints and \
                len(self.events[i].romanTimestamps) > minDataPoints:
                nJointDetections += 1

        return float(nJointDetections)/float(self.nSimEvents)*100.0

class complementaryObsMetric(maf.BaseMetric):
    """Metric to evaluate whether a Rubin OpSim will acquire observations of
    the Roman Galactic Exoplanet Survey region contemporaneously with (within
    24hrs of) Roman observations.

    Parameters
    ----------
    fieldRA : float, RA in degrees of a given pointing
    fieldDec : float, Dec in degrees of a given pointing
    filter : str, filter bandpass used for a given observation
    observationStartMJD : float, MJD timestamp of the start of a given observation
    visitExposureTime : float, exposure time in seconds
    fiveSigmaDepth : float, limiting magnitude at 5sigma
    """

    def __init__(
        self,
        cols=[
            "fieldRA",
            "fieldDec",
            "filter",
            "observationStartMJD",
            "visitExposureTime",
            "fiveSigmaDepth",
        ],
        metricName="complementaryObsMetric",
        **kwargs
    ):
        """Kwargs must contain:
        filters  list Filterset over which to compute the metric
        """

        self.ra_col = "fieldRA"
        self.dec_col = "fieldDec"
        self.filterCol = "filter"
        self.m5Col = "fiveSigmaDepth"
        self.mjdCol = "observationStartMJD"
        self.exptCol = "visitExposureTime"
        self.filters = ["u", "g", "r", "i", "z", "y"]
        self.magCuts = {
            "u": 22.7,
            "g": 24.1,
            "r": 23.7,
            "i": 23.1,
            "z": 22.2,
            "y": 21.4,
        }
        self.NSIDE = 64
        self.defineRGES()

        super().__init__(col=cols, metricName=metricName, metricDtype="object")

    def defineRGES(self):
        self.RGES = RGESSurvey()

    def countContemporaneousObs(self,rubinTimestamps,romanTimestamps):

        rubinTimestamps = np.sort(rubinTimestamps)
        romanTimestamps = np.sort(romanTimestamps)

        tt1,tt2 = np.meshgrid(rubinTimestamps, romanTimestamps)
        delta_times = abs(tt2-tt1)

        idx = np.where(delta_times < 1.0)
        obs = np.unique(np.concatenate( (tt1[idx], tt2[idx]) ))

        return len(idx[0])

    def countGapObs(self,rubinTimestamps,romanSeasons):

        rubinTimestamps = np.sort(rubinTimestamps)

        gapObs = np.array([])
        for i in range(0,len(romanSeasons)-1,1):
            season1 = romanSeasons[i]
            season2 = romanSeasons[i+1]
            ts = rubinTimestamps[(rubinTimestamps >= season1['end']) & \
                                 (rubinTimestamps <= season2['start'])]
            gapObs = np.concatenate( (gapObs,ts) )

        return len(gapObs)

    def run(self, dataSlice, slicePoint=None):

        metric_data = {}

        # Establish the parameters of the RGES survey
        ahp = HEALPix(nside=self.NSIDE, order='ring', frame=TETE())
        self.defineRGES()
        self.RGES.calcHealpix(ahp)

        # Calculate timestamps of Roman observations
        self.RGES.calcTimeStamps()

        # Select Rubin observations with adequate signal-to-noise,
        # combining data from all filters
        match_obs = []
        for i, f in enumerate(self.filters):
            idx1 = np.where(dataSlice[self.filterCol] == f)[0]
            idx2 = np.where(dataSlice[self.m5Col] >= self.magCuts[f])[0]
            match = list(set(idx1).intersection(set(idx2)))
            match_obs += match

        # Calculate the coordinates of where the matching observations
        # were acquired and convert this to a list of HEALpixels
        coords_icrs = SkyCoord(
            dataSlice[self.ra_col][match_obs],
            dataSlice[self.dec_col][match_obs],
            frame="icrs",
            unit=(u.deg, u.deg),
        )
        coords_gal = coords_icrs.transform_to(Galactic())
        pixels = ahp.skycoord_to_healpix(coords_gal)

        # Identify the observations from HEALpixels in the RGES footprint
        overlap_pixels = list(set(pixels.tolist()).intersection(set(self.RGES.pixels.tolist())))

        if len(overlap_pixels) > 0:
            # Identify which observations from the dataSlice correspond to
            # the overlapping survey region.  This may produce multiple
            # indices in the array, referred to different observations
            match = np.array(match_obs)
            match_obs = []
            for p in overlap_pixels:
                ip = np.where(pixels == p)[0]
                match_obs += match[ip].tolist()

            # Extract the timestamps of matching Rubin observations of the
            # RGES footprint
            rubinTimestamps = dataSlice[self.mjdCol][match_obs]

            # Metric value 1: Test whether observations are acquired from
            # both observatories within the same 24hr period
            metric_data['nContObs'] = self.countContemporaneousObs(rubinTimestamps,
                                                            self.RGES.timestamps)

            # Metric value 2: Test whether observations are acquired between
            # Roman observing seasons
            metric_data['nGapObs'] = self.countGapObs(rubinTimestamps,
                                                    self.RGES.seasons)

        else:
            metric_data['nContObs'] = 0
            metric_data['nGapObs'] = 0

        return metric_data
