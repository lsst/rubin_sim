__all__ = ("FilterPairTGapsMetric",)

import numpy as np

from ..metrics.base_metric import BaseMetric


class FilterPairTGapsMetric(BaseMetric):
    """Figure of merit to measure the coverage the time gaps in same
    and different filter pairs;

    FoM is defined as sum of Nv / standard deviation after a clip;

    Parameters
    ----------
    fltpairs : `list` [`str`], optional
        List of filter pair sets to search for.
    mag_lim : `list` [`float`]
        FiveSigmaDepth threshold each filter,
        default {'u':18, 'g':18, 'r':18, 'i':18, 'z':18, 'y':18}
    bins_same : `np.ndarray`, (N,)
        Bins to get histogram for same-filter pair.
    bins_diff : `np.ndarray`, (N,)
        Bins to get histogram for diff-filter pair.
    nv_clip : `int`, optional
        Number of visits of pairs to clip, std is calculated below nv_clip.
    allgaps : `bool``, optional
        All possible pairs if True, else consider only nearest

    Returns
    -------
    result : `float`
        sum of fom for all filterpairs,
    """

    def __init__(
        self,
        mjd_col="observationStartMJD",
        filter_col="filter",
        m5_col="fiveSigmaDepth",
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
        nv_clip={
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
        **kwargs,
    ):
        self.mjd_col = mjd_col
        self.filter_col = filter_col
        self.m5_col = m5_col
        self.fltpairs = fltpairs
        self.mag_lim = mag_lim
        self.bins_same = bins_same
        self.bins_diff = bins_diff

        self.allgaps = allgaps

        # number of visits to clip, default from 1/10th of baseline_v2.0 WFD
        self.nv_clip = nv_clip

        super().__init__(col=[self.mjd_col, self.filter_col, self.m5_col], **kwargs)

    def _get_d_t(self, data_slice, f0, f1):
        # select
        idx0 = (data_slice[self.filter_col] == f0) & (data_slice[self.m5_col] > self.mag_lim[f0])
        idx1 = (data_slice[self.filter_col] == f1) & (data_slice[self.m5_col] > self.mag_lim[f1])

        time_col0 = data_slice[self.mjd_col][idx0]
        time_col1 = data_slice[self.mjd_col][idx1]

        # time_col0 = time_col0.reshape((len(time_col0), 1))
        # time_col1 = time_col1.reshape((len(time_col1), 1))

        # calculate time gaps matrix
        # diffmat = np.subtract(time_col0, time_col1.T)

        if self.allgaps:
            # collect all time gaps
            if f0 == f1:
                time_col0 = time_col0.reshape((len(time_col0), 1))
                time_col1 = time_col1.reshape((len(time_col1), 1))

                diffmat = np.subtract(time_col0, time_col1.T)
                diffmat = np.abs(diffmat)
                # get only triangle part
                dt_tri = np.tril(diffmat, -1)
                d_t = dt_tri[dt_tri != 0]  # flatten lower triangle
            else:
                # d_t = diffmat.flatten()
                dtmax = np.max(self.bins_diff)
                # time gaps window for measure color
                d_t = []
                for time_col in time_col0:
                    time_col_in_window = time_col1[
                        (time_col1 >= (time_col - dtmax)) & (time_col1 <= (time_col + dtmax))
                    ]

                    d_t.append(np.abs(time_col_in_window - time_col))

                d_t = np.concatenate(d_t) if len(d_t) > 0 else np.array(d_t)
        else:
            # collect only nearest
            if f0 == f1:
                # get diagonal ones nearest nonzero, offset=1
                # d_t = np.diagonal(diffmat, offset=1)
                d_t = np.diff(time_col0)
            else:
                time_col0 = data_slice[self.mjd_col][idx0]
                time_col1 = data_slice[self.mjd_col][idx1]

                time_col0 = time_col0.reshape((len(time_col0), 1))
                time_col1 = time_col1.reshape((len(time_col1), 1))

                # calculate time gaps matrix
                diffmat = np.subtract(time_col0, time_col1.T)
                # get tgaps both left and right
                # keep only negative ones
                masked_ar = np.ma.masked_where(
                    diffmat >= 0,
                    diffmat,
                )
                left_ar = np.max(masked_ar, axis=1)
                d_t_left = -left_ar.data[~left_ar.mask]

                # keep only positive ones
                masked_ar = np.ma.masked_where(diffmat <= 0, diffmat)
                right_ar = np.min(masked_ar, axis=1)
                d_t_right = right_ar.data[~right_ar.mask]

                d_t = np.concatenate([d_t_left.flatten(), d_t_right.flatten()])

        return d_t

    def run(self, data_slice, slice_point=None):
        # sort the data_slice in order of time.
        data_slice.sort(order=self.mjd_col)

        fom_dic = {}
        d_t_dic = {}
        for fltpair in self.fltpairs:
            d_t = self._get_d_t(data_slice, fltpair[0], fltpair[1])
            # calculate FoM
            # cut out at average
            # std below
            # FOM = Nv / std

            if fltpair[0] == fltpair[1]:
                bins = self.bins_same
            else:
                bins = self.bins_diff
                d_t_dic[fltpair] = d_t

            hist, _ = np.histogram(d_t, bins=bins)

            if np.any(hist):
                hist_clip = np.clip(hist, a_min=0, a_max=self.nv_clip[fltpair])
                fom = hist.sum() / hist_clip.std()
            else:
                fom = np.nan
            fom_dic[fltpair] = fom
            # print( fltpair, fom)

        return np.nansum(list(fom_dic.values()))
