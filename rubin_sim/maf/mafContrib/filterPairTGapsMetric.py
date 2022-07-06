import numpy as np
from ..metrics.baseMetric import BaseMetric

__all__ = ["FilterPairTGapsMetric"]


class FilterPairTGapsMetric(BaseMetric):
    """
    figure of merit to measure the coverage the time gaps in same and different filter pairs;
    FoM is defined as sum of Nv / standard deviation after a clip;
    Parameters:
        colname: list, ['observationStartMJD', 'filter', 'fiveSigmaDepth']
        fltpairs: filter pair, default ['uu', 'ug', 'ur', 'ui', 'uz','uy',
                                        'gg', 'gr', 'gi', 'gz', 'gy',
                                        'rr', 'ri', 'rz', 'ry',
                                        'ii', 'iz', 'iy',
                                        'zz', 'zy',
                                        'yy']
        mag_lim: list, fiveSigmaDepth threshold each filter, default {'u':18, 'g':18, 'r':18, 'i':18, 'z':18, 'y':18}
        bins_same: np.array, bins to get histogram for same-filter pair ;
        bins_diff: np.array, bins to get histogram for diff-filter pair ;
        Nv_clip: number of visits of pairs to clip, std is calculated below Nv_clip
        allgaps: boolean, all possible pairs if True, else consider only nearest

    Returns:
        result: sum of fom for all filterpairs,

    """

    def __init__(
        self,
        mjdCol="observationStartMJD",
        filterCol="filter",
        m5Col="fiveSigmaDepth",
        fltpairs=[
            "uu",
            "ug",
            "ur",
            "ui",
            "uz",
            "uy",
            "gg",
            "gr",
            "gi",
            "gz",
            "gy",
            "rr",
            "ri",
            "rz",
            "ry",
            "ii",
            "iz",
            "iy",
            "zz",
            "zy",
            "yy",
        ],
        mag_lim={"u": 18, "g": 18, "r": 18, "i": 18, "z": 18, "y": 18},
        bins_same=np.logspace(np.log10(5 / 60 / 60 / 24), np.log10(3650), 50),
        bins_diff=np.logspace(np.log10(5 / 60 / 60 / 24), np.log10(2), 50),
        Nv_clip={
            "uu": 30,
            "ug": 30,
            "ur": 30,
            "ui": 30,
            "uz": 30,
            "uy": 30,
            "gg": 69,
            "gr": 30,
            "gi": 30,
            "gz": 30,
            "gy": 30,
            "rr": 344,
            "ri": 30,
            "rz": 30,
            "ry": 30,
            "ii": 355,
            "iz": 30,
            "iy": 30,
            "zz": 282,
            "zy": 30,
            "yy": 288,
        },
        allgaps=True,
        **kwargs
    ):

        self.mjdCol = mjdCol
        self.filterCol = filterCol
        self.m5Col = m5Col
        self.fltpairs = fltpairs
        self.mag_lim = mag_lim
        self.bins_same = bins_same
        self.bins_diff = bins_diff

        self.allgaps = allgaps

        # number of visits to clip, default got from 1/10th of baseline_v2.0 WFD
        self.Nv_clip = Nv_clip

        super().__init__(col=[self.mjdCol, self.filterCol, self.m5Col], **kwargs)

    def _get_dT(self, dataSlice, f0, f1):

        # select
        idx0 = (dataSlice[self.filterCol] == f0) & (
            dataSlice[self.m5Col] > self.mag_lim[f0]
        )
        idx1 = (dataSlice[self.filterCol] == f1) & (
            dataSlice[self.m5Col] > self.mag_lim[f1]
        )

        timeCol0 = dataSlice[self.mjdCol][idx0]
        timeCol1 = dataSlice[self.mjdCol][idx1]

        # timeCol0 = timeCol0.reshape((len(timeCol0), 1))
        # timeCol1 = timeCol1.reshape((len(timeCol1), 1))

        # calculate time gaps matrix
        # diffmat = np.subtract(timeCol0, timeCol1.T)

        if self.allgaps:
            # collect all time gaps
            if f0 == f1:
                timeCol0 = timeCol0.reshape((len(timeCol0), 1))
                timeCol1 = timeCol1.reshape((len(timeCol1), 1))

                diffmat = np.subtract(timeCol0, timeCol1.T)
                diffmat = np.abs(diffmat)
                # get only triangle part
                dt_tri = np.tril(diffmat, -1)
                dT = dt_tri[dt_tri != 0]  # flatten lower triangle
            else:
                # dT = diffmat.flatten()
                dtmax = np.max(self.bins_diff)  # time gaps window for measure color
                dT = []
                for timeCol in timeCol0:
                    timeCol_inWindow = timeCol1[
                        (timeCol1 >= (timeCol - dtmax))
                        & (timeCol1 <= (timeCol + dtmax))
                    ]

                    dT.append(np.abs(timeCol_inWindow - timeCol))

                dT = np.concatenate(dT) if len(dT) > 0 else np.array(dT)
        else:
            # collect only nearest
            if f0 == f1:
                # get diagonal ones nearest nonzero, offset=1
                # dT = np.diagonal(diffmat, offset=1)
                dT = np.diff(timeCol0)
            else:
                timeCol0 = dataSlice[self.mjdCol][idx0]
                timeCol1 = dataSlice[self.mjdCol][idx1]

                timeCol0 = timeCol0.reshape((len(timeCol0), 1))
                timeCol1 = timeCol1.reshape((len(timeCol1), 1))

                # calculate time gaps matrix
                diffmat = np.subtract(timeCol0, timeCol1.T)
                # get tgaps both left and right
                # keep only negative ones
                masked_ar = np.ma.masked_where(
                    diffmat >= 0,
                    diffmat,
                )
                left_ar = np.max(masked_ar, axis=1)
                dT_left = -left_ar.data[~left_ar.mask]

                # keep only positive ones
                masked_ar = np.ma.masked_where(diffmat <= 0, diffmat)
                right_ar = np.min(masked_ar, axis=1)
                dT_right = right_ar.data[~right_ar.mask]

                dT = np.concatenate([dT_left.flatten(), dT_right.flatten()])

        return dT

    def run(self, dataSlice, slicePoint=None):
        # sort the dataSlice in order of time.
        dataSlice.sort(order=self.mjdCol)

        fom_dic = {}
        dT_dic = {}
        for fltpair in self.fltpairs:
            dT = self._get_dT(dataSlice, fltpair[0], fltpair[1])
            # calculate FoM
            # cut out at average
            # std below
            # FOM = Nv / std

            if fltpair[0] == fltpair[1]:
                bins = self.bins_same
            else:
                bins = self.bins_diff
                dT_dic[fltpair] = dT

            hist, _ = np.histogram(dT, bins=bins)

            if np.any(hist):
                hist_clip = np.clip(hist, a_min=0, a_max=self.Nv_clip[fltpair])
                fom = hist.sum() / hist_clip.std()
            else:
                fom = np.nan
            fom_dic[fltpair] = fom
            # print( fltpair, fom)

        return np.nansum(list(fom_dic.values()))
