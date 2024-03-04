import warnings

from rubin_scheduler.site_models import *  # noqa: F403

warnings.simplefilter("default")
warnings.warn(
    "rubin_sim.site_models is deprecated, switch to rubin_scheduler.site_models", DeprecationWarning
)
