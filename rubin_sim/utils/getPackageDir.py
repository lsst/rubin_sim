import os
import inspect


__all__ = ["getPackageDir"]


def getPackageDir(package):
    """Return the path to a package"""

    # See if removing $PREFIX let's the CI run it.
    result = os.path.dirname(os.path.dirname(inspect.getfile(package))).replace(
        "$PREFIX", ""
    )
    return result
