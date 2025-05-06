__all__ = ("empty_info",)

import pandas as pd


def empty_info(as_df_row=False):
    """Return an empty info dictionary with keys already set

    Parameters
    ----------
    as_df_row : `bool`
        Return the result as a pandas DataFrame

    Returns
    -------
    dict or pandas.DataFrame with keys
    run_name : `str`
    metric: name : `str`
    metric: col : `str`
    observations_subset : `str`
    slicer: nside : `int`
    summary_name : `str`
    value : `float`
    caption : `str`
    """

    result = {}
    result["run_name"] = ""
    result["metric: name"] = ""
    result["metric: col"] = ""
    result["observations_subset"] = ""
    result["slicer: nside"] = 0
    result["summary_name"] = ""
    result["value"] = 0.0
    result["caption"] = ""

    if as_df_row:
        result = pd.Series(result).to_frame().T

    return result
