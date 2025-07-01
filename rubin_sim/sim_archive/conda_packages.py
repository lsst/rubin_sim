import json
import os
import re
import sys
from functools import lru_cache

__all__ = [
    "get_conda_packages",
]


@lru_cache(maxsize=1)
def get_conda_packages() -> dict[str, str]:
    """Get products and their versions from the conda environment.

    Returns
    -------
    packages : `dict`
        Keys (type `str`) are product names; values (type `str`) are their
        versions.

    Notes
    -----
    Returns empty result if a conda environment is not in use or can not
    be queried.
    """
    if "CONDA_PREFIX" not in os.environ:
        return {}

    # conda list is very slow. Ten times faster to scan the directory
    # directly. This will only find conda packages and not pip installed
    # packages.
    meta_path = os.path.join(os.environ["CONDA_PREFIX"], "conda-meta")

    try:
        filenames = os.scandir(path=meta_path)
    except FileNotFoundError:
        return {}

    packages = {}
    for filename in filenames:
        if not filename.name.endswith(".json"):
            continue
        with open(filename) as f:
            try:
                data = json.load(f)
            except ValueError:
                continue
            try:
                packages[data["name"]] = data["version"]
            except KeyError:
                continue

    packages = dict(sorted(packages.items()))

    # Try to work out the conda environment name and include it as a fake
    # package. The "obvious" way of running "conda info --json" does give
    # access to the active_prefix but takes about 2 seconds to run.
    # As a compromise look for the env name in the path to the python
    # executable
    match = re.search(r"/envs/(.*?)/bin/", sys.executable)
    if match:
        packages["conda_env"] = match.group(1)

    return packages
