__all__ = ("int_binned_stat",)

import numpy as np


def int_binned_stat(ids, values, statistic=np.mean):
    """
    Like scipy.binned_statistic, but for unique integer ids.

    Parameters
    ----------
    ids : array-like of ints
        The integer ID for each value
    values : array-like
        The values to be combined
    statistic : function (np.mean)
        Function to run on the values that have matching ids.

    Returns
    -------
    unique ids, binned values
    """

    uids = np.unique(ids)
    order = np.argsort(ids)

    ordered_ids = ids[order]
    ordered_values = values[order]

    left = np.searchsorted(ordered_ids, uids, side="left")
    right = np.searchsorted(ordered_ids, uids, side="right")

    stat_results = []
    for le, ri in zip(left, right):
        stat_results.append(statistic(ordered_values[le:ri]))

    return uids, np.array(stat_results)
