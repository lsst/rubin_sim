from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("rubin_sim")
except PackageNotFoundError:
    # package is not installed
    pass
