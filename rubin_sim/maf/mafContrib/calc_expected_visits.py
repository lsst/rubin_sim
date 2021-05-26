# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 17:11:20 2018

@author: rstreet
"""

import rubin_sim.maf.db as db
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metricBundles as metricBundles
from rubin_sim.maf.metrics import BaseMetric
import numpy as np
from .calculate_lsst_field_visibility_astropy import calculate_lsst_field_visibility

__all__ = ['CalcExpectedVisitsMetric']


class CalcExpectedVisitsMetric(BaseMetric):
    """Function to calculate the maximum possible number of visits to a
    given pointing, given the expected cadence of observation and within
    the date ranges given, taking target visibility into account.

    Input:
    :param array ra:            RAs, J2000.0, sexigesimal format
    :param array dec:           Decs, J2000.0, sexigesimal format
    :param float cadence:       Interval between successive visits in the
                                same single filter in hours
    :param string start_date:   Start of observing window YYYY-MM-DD
    :param string start_date:   End of observation window YYYY-MM-DD

    Output:
    :param list of arrays n_visits:       Number of visits possible per night
                                          for each pointing
    :param list of arrays hrs_visibility: Hours of visibility per night
                                          for each pointing
    """

    def __init__(self, pointings,cadence,start_date,end_date,filter_id,
                 ra_col='fieldRA', dec_col='fieldDec',
                 metricName='CalcExpectedVisitsMetric',verbose=False):
        """Input:
            :param array ra:            RAs, J2000.0, sexigesimal format
            :param array dec:           Decs, J2000.0, sexigesimal format
            :param float cadence:       Interval between successive visits in the
                                        same single filter in hours
            :param string start_date:   Start of observing window YYYY-MM-DD
            :param string start_date:   End of observation window YYYY-MM-DD

            Output:
            :param list of arrays n_visits:       Number of visits possible per night
                                                  for each pointing
            :param list of arrays hrs_visibility: Hours of visibility per night
                                                  for each pointing
        """

        self.pointings = pointings
        self.cadence = cadence
        self.start_date = start_date
        self.end_date = end_date
        self.filter_id = filter_id
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.verbose = verbose

        columns = [ self.ra_col, self.dec_col ]

        super(CalcExpectedVisitsMetric,self).__init__(col=columns, metricName=metricName)

    def run(self, dataSlice, slicePoint=None):
        n_visits = []
        hrs_visibility = []

        if self.verbose:
            print('Calculating visbility for '+str(len(self.pointings))+' fields')

        for i in range(0,len(self.pointings),1):

            #(ra, dec) = pointings[i]
            ra = dataSlice[self.ra_col][0]
            dec = dataSlice[self.dec_col][0]

            if self.verbose:
                print(' -> RA '+str(ra)+', Dec '+str(dec))

            (total_time_visible, hrs_visible_per_night) = calculate_lsst_field_visibility(ra,dec,self.start_date,self.end_date,verbose=False)

            n_visits.append( (np.array(hrs_visible_per_night) / self.cadence).astype(int) )
            hrs_visibility.append( np.array(hrs_visible_per_night) )

        return n_visits,hrs_visibility
