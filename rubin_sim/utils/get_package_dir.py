import os
import inspect


__all__ = ["get_package_dir"]


def get_package_dir(package):
    """Return the path to a package"""
    return os.path.dirname(os.path.dirname(inspect.getfile(package)))
