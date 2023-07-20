__all__ = ("get_package_dir",)

import inspect
import os


def get_package_dir(package):
    """Return the path to a package"""
    return os.path.dirname(os.path.dirname(inspect.getfile(package)))
