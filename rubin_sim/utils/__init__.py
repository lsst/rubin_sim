import warnings

from rubin_scheduler.utils import *  #noqa: F403

warnings.simplefilter("default")
warnings.warn("rubin_sim.utils is deprecated, switch to rubin_scheduler.utils", DeprecationWarning)
