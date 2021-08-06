import numpy as np
import rubin_sim.maf.metrics as metrics

__all__ = ['SNCadenceMetric']


class SNCadenceMetric(metrics.BaseMetric):
    """
    Metric to estimate the redshift limit for faint supernovae (x1,color) = (-2.0,0.2)

    Parameters
    ----------
    list : str, optional
        Name of the columns used to estimate the metric
    coadd :  bool, optional
        to make "coaddition" per night (uses snStacker)
        Default True
    lim_sn : class, optional
       Reference data used to estimate redshift values (interpolation)
    """

    def __init__(self, metricName='SNCadenceMetric',
                 mjdCol='observationStartMJD', RaCol='fieldRA', DecCol='fieldDec',
                 filterCol='filter', m5Col='fiveSigmaDepth', exptimeCol='visitExposureTime',
                 nightCol='night', obsidCol='observationId', nexpCol='numExposures',
                 vistimeCol='visitTime', coadd=True, lim_sn=None, **kwargs):

        self.mjdCol = mjdCol
        self.m5Col = m5Col
        self.filterCol = filterCol
        self.RaCol = RaCol
        self.DecCol = DecCol
        self.exptimeCol = exptimeCol
        self.seasonCol = 'season'
        self.nightCol = nightCol
        self.obsidCol = obsidCol
        self.nexpCol = nexpCol
        self.vistimeCol = vistimeCol

        cols = [self.nightCol, self.m5Col, self.filterCol, self.mjdCol, self.obsidCol,
                self.nexpCol, self.vistimeCol, self.exptimeCol, self.seasonCol]
        if coadd:
            cols += ['coadd']

        super(SNCadenceMetric, self).__init__(
            col=cols, metricName=metricName, **kwargs)

        self.filterNames = np.array(['u', 'g', 'r', 'i', 'z', 'y'])

        self.lim_sn = lim_sn

    def run(self, dataSlice, slicePoint=None):

        # Cut down to only include filters in correct wave range.

        goodFilters = np.in1d(dataSlice['filter'], self.filterNames)
        dataSlice = dataSlice[goodFilters]
        if dataSlice.size == 0:
            return None
        dataSlice.sort(order=self.mjdCol)

        r = []
        fieldRA = np.mean(dataSlice[self.RaCol])
        fieldDec = np.mean(dataSlice[self.DecCol])
        band = np.unique(dataSlice[self.filterCol])[0]

        sel = dataSlice
        bins = np.arange(np.floor(sel[self.mjdCol].min()), np.ceil(
            sel[self.mjdCol].max()), 1.)
        c, b = np.histogram(sel[self.mjdCol], bins=bins)
        if (c.mean() < 1.e-8) | np.isnan(c).any() | np.isnan(c.mean()):
            cadence = 0.
        else:
            cadence = 1. / c.mean()
        # time_diff = sel[self.mjdCol][1:]-sel[self.mjdCol][:-1]
        r.append((fieldRA, fieldDec, band,
                  np.mean(sel[self.m5Col]), cadence))

        res = np.rec.fromrecords(
            r, names=['fieldRA', 'fieldDec', 'band', 'm5_mean', 'cadence_mean'])

        zref = self.lim_sn.interp_griddata(res)

        if np.isnan(zref):
            zref = self.badval

        return zref
