import time
from importlib import metadata

__all__ = ['getDateVersion']


def getDateVersion():
    """
    Get today's date and a dictionary with the MAF version information.
    This is written into configuration output files, to help track MAF runs.

    Returns
    -------
    str, dict
        String with today's date, Dictionary with version information.
    """

    version = metadata.version('rubin_sim')
    #today_date = time.strftime("%x")
    today_date = '-'.join([time.strftime(x) for x in ["%Y", "%m", "%d"]])
    versionInfo = {'__version__': version,
                   '__repo_version__': None,
                   '__fingerprint__': None,
                   '__dependency_versions__': None}

    return today_date, versionInfo
