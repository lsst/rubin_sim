import os
import inspect


__all__ = ["getPackageDir"]


def getPackageDir(package):
    """Return the path to a package"""

    cwd = os.getcwd()
    # If we are running at the top level, like the CI, do nothing
    # otherwise, find the full path
    if os.path.split(cwd)[-1] == "rubin_sim":
        result = ""
    else:
        result = os.path.dirname(os.path.dirname(inspect.getfile(package)))
    return result
