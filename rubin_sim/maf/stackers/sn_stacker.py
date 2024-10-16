__all__ = ("CoaddStacker",)

import numpy as np

from rubin_sim.maf.stackers import BaseStacker


class CoaddStacker(BaseStacker):
    """Stacker to estimate m5 "coadded" per band and par night

    Parameters
    ----------
    mjd_col : `str`, optional
        Name of the MJD column
    ra_col : `str`, optional
        Name of the RA column
    dec_col : `str`, optional
        Name of the Dec column
    m5_col : `str`, optional
        Name of the m5 column
    filter_col : `str`, optional
        Name of the filter column
    night_col : `str`, optional
        Name of the night column
    num_exposures_col : `str`, optional
        Name of the number of exposures per visit column
    visit_time_col : `str`, optional
        Name of the total visit time column
    visit_exposure_time_col : `str`, optional
        Name of the on-sky exposure time column
    """

    cols_added = ["coadd"]

    def __init__(
        self,
        mjd_col="observationStartMJD",
        ra_col="fieldRA",
        dec_col="fieldDec",
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        night_col="night",
        num_exposures_col="numExposures",
        visit_time_col="visitTime",
        visit_exposure_time_col="visitExposureTime",
    ):
        self.cols_req = [
            mjd_col,
            ra_col,
            dec_col,
            m5_col,
            filter_col,
            night_col,
            num_exposures_col,
            visit_time_col,
            visit_exposure_time_col,
        ]
        self.ra_col = ra_col
        self.dec_col = dec_col
        self.night_col = night_col
        self.filter_col = filter_col
        self.m5_col = m5_col
        self.num_exposures_col = num_exposures_col
        self.visit_time_col = visit_time_col
        self.visit_exposure_time_col = visit_exposure_time_col

        self.units = ["int"]

    def _run(self, sim_data, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct
            # and does not need recalculating.
            return sim_data
        self.dtype = sim_data.dtype
        r = []
        for ra, dec, band in np.unique(sim_data[[self.ra_col, self.dec_col, self.filter_col]]):
            idx = np.abs(sim_data[self.ra_col] - ra) < 1.0e-5
            idx &= np.abs(sim_data[self.dec_col] - dec) < 1.0e-5
            idx &= sim_data[self.filter_col] == band

            sel = sim_data[idx]
            for night in np.unique(sel[self.night_col]):
                idxb = sel[self.night_col] == night
                r.append(tuple(self.fill(sel[idxb])))

        myarray = np.array(r, dtype=self.dtype)
        return myarray

    def fill(self, tab):
        """Estimation of new fields (m5 "coadded" values, ...)

        Parameters
        ---------------
        tab : array of (initial) data


        Returns
        -----------
        tuple with modified field values:
         - m5Col: "coadded" m5
        - numExposuresCol: sum of  numExposuresCol
        - visitTimeCol: sum of visitTimeCol
        - visitExposureTimeCol: sum of visitExposureTimeCol
        - all other input fields except band (Ra, Dec, night) : median(field)
        """

        r = []

        for colname in self.dtype.names:
            if colname not in [
                self.m5_col,
                self.num_exposures_col,
                self.visit_time_col,
                self.visit_exposure_time_col,
                self.filter_col,
            ]:
                if colname == "coadd":
                    r.append(1)
                else:
                    if tab[colname].dtype == "object":
                        r.append(tab[colname][0])
                    else:
                        r.append(np.median(tab[colname]))
            if colname == self.m5_col:
                r.append(self.m5_coadd(tab[self.m5_col]))
            if colname in [
                self.num_exposures_col,
                self.visit_time_col,
                self.visit_exposure_time_col,
            ]:
                r.append(np.sum(tab[colname]))
            if colname == self.filter_col:
                r.append(np.unique(tab[self.filter_col])[0])

        return r

    def m5_coadd(self, m5):
        """Estimation of "coadded" m5 values based on:
        flux_5sigma = 10**(-0.4*m5)
        sigmas = flux_5sigma/5.
        sigma_tot = 1./sqrt(np.sum(1/sigmas**2))
        flux_tot = 5.*sigma_tot

        Parameters
        ---------------
        m5 : set of m5 (five-sigma depths) values

        Returns
        -----------
        "coadded" m5 value
        """
        return 1.25 * np.log10(np.sum(10.0 ** (0.8 * m5)))
