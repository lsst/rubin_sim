__all__ = ("empty_info",)

import pandas as pd


def empty_info(as_df_row=False, **kwargs):
    """Return an empty info dictionary with keys already set

    Parameters
    ----------
    as_df_row : `bool`
        Return the result as a pandas DataFrame

    Returns
    -------
    dict or pandas.DataFrame with keys
    data_source : `str`
    name : `str`
    col : `str`
    observations_subset : `str`
    population : `str`
    slicer: nside : `int`
    summary_name : `str`
    value : `float`
    caption : `str`
    """

    result = {}
    result["data_source"] = ""
    result["name"] = ""
    result["col"] = ""
    result["unit"] = ""
    result["times"] = ""
    result["observations_subset"] = ""
    result["population"] = ""
    result["nside"] = 0
    result["summary_name"] = ""
    result["value"] = 0.0
    result["table_name"] = ""
    result["caption"] = ""
    result["group"] = ""
    result["subgroup"] = ""
    result["plot_filename"] = ""
    result["data_filename"] = ""

    if as_df_row:
        result = pd.Series(result).to_frame().T

    return result
