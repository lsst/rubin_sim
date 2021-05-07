import os
import inspect


__all__  = ["getPackageDir"]

def getPackageDir(package):
    """Return the path to a package
    """
    return os.path.dirname(os.path.dirname(inspect.getfile(package)))
