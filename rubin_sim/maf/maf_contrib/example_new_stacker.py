# Example of a stacker added to the repo.
# ljones@astro.washington.edu

import numpy as np

from rubin_sim.maf.stackers import BaseStacker, wrapRADec

__all__ = ("YearlyDitherStacker",)


class YearlyDitherStacker(BaseStacker):
    """Add a dither of half the FOV depending on the year of the survey."""

    def __init__(self, exp_mjd_col="expMJD", ra_col="fieldRA", dec_col="fieldDec"):
        # Names of columns we want to add.
        self.cols_added = ["yearlyDitherRA", "yearlyDitherDec"]
        # Names of columns we need from database.
        self.cols_req = [exp_mjd_col, ra_col, dec_col]
        # List of units for our new columns.
        self.units = ["rad", "rad"]
        # Set our dither offset.
        self.dither_offset = 1.75 / 180.0 * np.pi
        # And save the column names.
        self.exp_mjd_col = exp_mjd_col
        self.ra_col = ra_col
        self.dec_col = dec_col

    def _run(self, sim_data):
        # What 'year' is each visit in?
        year = np.floor((sim_data[self.exp_mjd_col] - sim_data[self.exp_mjd_col][0]) / 365.25)
        # Define dither based on year.
        dither_ra = np.zeros(len(sim_data[self.ra_col]))
        dither_dec = np.zeros(len(sim_data[self.dec_col]))
        # In y1, y3, y5, y6, y8 & y10 ra dither = 0.
        # In y2 & y7, ra dither = +ditherOffset
        # In y4 & y9, ra dither = -ditherOffset
        condition = (year == 2) | (year == 7)
        dither_ra[condition] = self.dither_offset
        condition = (year == 4) | (year == 9)
        dither_ra[condition] = -1.0 * self.dither_offset
        # In y1, y2, y4, y6, y7 & y9, dec dither = 0
        # In y3 & y8, dec dither = -ditherOffset
        # In y5 & y10, dec dither = ditherOffset
        condition = (year == 3) | (year == 8)
        dither_dec[condition] = -1.0 * self.dither_offset
        condition = (year == 5) | (year == 10)
        dither_dec[condition] = self.dither_offset
        # Calculate actual RA/Dec and wrap into appropriate range.
        sim_data["yearlyDitherRA"], sim_data["yearlyDitherDec"] = wrapRADec(
            sim_data[self.ra_col] + dither_ra, sim_data[self.dec_col] + dither_dec
        )
        return sim_data
