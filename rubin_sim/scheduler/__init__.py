import warnings

from rubin_scheduler.scheduler import *

warnings.simplefilter("default")
warnings.warn("rubin_sim.scheduler is deprecated, switch to rubin_scheduler.scheduler", DeprecationWarning)
