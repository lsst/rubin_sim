import importlib.util
import logging

from .make_snapshot import *

HAVE_LSST_RESOURCES = importlib.util.find_spec("lsst") and importlib.util.find_spec("lsst.resources")
if HAVE_LSST_RESOURCES:
    from .sim_archive import *  # isort:skip
    from .prenight import *
else:
    logging.error("rubin_sim.sim_archive requires lsst.resources.")
