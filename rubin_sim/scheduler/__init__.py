import warnings

from rubin_scheduler.scheduler import *  # noqa: F403

warnings.simplefilter("default")
warnings.warn("rubin_sim.scheduler is deprecated, switch to rubin_scheduler.scheduler", DeprecationWarning)
