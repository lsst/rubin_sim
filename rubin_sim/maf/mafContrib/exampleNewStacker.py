# Example of a stacker added to the repo.
# ljones@astro.washington.edu

import numpy as np
from rubin_sim.maf.stackers import BaseStacker
from rubin_sim.maf.stackers import wrapRADec

__all__ = ['YearlyDitherStacker']


class YearlyDitherStacker(BaseStacker):
    """ Add a dither of half the FOV depending on the year of the survey.
    """
    def __init__(self, expMJDCol='expMJD', raCol='fieldRA', decCol='fieldDec'):
        # Names of columns we want to add.
        self.colsAdded = ['yearlyDitherRA', 'yearlyDitherDec']
        # Names of columns we need from database.
        self.colsReq = [expMJDCol, raCol, decCol]
        # List of units for our new columns.
        self.units = ['rad', 'rad']
        # Set our dither offset.
        self.ditherOffset = 1.75/180.*np.pi
        # And save the column names.
        self.expMJDCol = expMJDCol
        self.raCol = raCol
        self.decCol = decCol

    def _run(self, simData):
        # What 'year' is each visit in?
        year = np.floor((simData[self.expMJDCol] - simData[self.expMJDCol][0]) / 365.25)
        # Define dither based on year.
        ditherRA = np.zeros(len(simData[self.raCol]))
        ditherDec = np.zeros(len(simData[self.decCol]))
        # In y1, y3, y5, y6, y8 & y10 ra dither = 0.
        # In y2 & y7, ra dither = +ditherOffset
        # In y4 & y9, ra dither = -ditherOffset
        condition = ((year == 2) | (year == 7))
        ditherRA[condition] = self.ditherOffset
        condition = ((year == 4) | (year == 9))
        ditherRA[condition] = -1.*self.ditherOffset
        # In y1, y2, y4, y6, y7 & y9, dec dither = 0
        # In y3 & y8, dec dither = -ditherOffset
        # In y5 & y10, dec dither = ditherOffset
        condition = ((year == 3) | (year == 8))
        ditherDec[condition] = -1.*self.ditherOffset
        condition = ((year == 5) | (year == 10))
        ditherDec[condition] = self.ditherOffset
        # Calculate actual RA/Dec and wrap into appropriate range.
        simData['yearlyDitherRA'], simData['yearlyDitherDec'] = wrapRADec(simData[self.raCol] + ditherRA,
                                                                          simData[self.decCol] + ditherDec)
        return simData


