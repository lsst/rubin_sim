__all__ = ("get_date_version",)

import time
from importlib import metadata


def get_date_version():
    """
    Get today's date and a dictionary with the MAF version information.
    This is written into configuration output files, to help track MAF runs.

    Returns
    -------
    str, dict
        String with today's date, Dictionary with version information.
    """

    version = metadata.version("rubin_sim")
    # today_date = time.strftime("%x")
    today_date = "-".join([time.strftime(x) for x in ["%Y", "%m", "%d"]])
    version_info = {
        "__version__": version,
        "__repo_version__": None,
        "__fingerprint__": None,
        "__dependency_versions__": None,
    }

    return today_date, version_info
