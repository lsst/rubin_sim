import warnings

from rubin_scheduler.skybrightness_pre import *

warnings.simplefilter("default")
warnings.warn(
    "rubin_sim.skybrightness_pre is deprecated, switch to rubin_scheduler.skybrightness_pre",
    DeprecationWarning,
)
