.. py:currentmodule:: rubin_sim

.. _rubin_sim:

#########
rubin_sim
#########

The rubin_sim module provides support for Rubin Observatory's LSST survey
scheduler, survey strategy analysis, and some basic simulation requirements.

The submodules include:

* `rubin_sim.utils </rs_utils>` provides some basic utilities we use throughout the rest of rubin_sim, but may be useful for other purposes.
* `rubin_sim.data </rs_data>` provides a minimal tool to track the location of the associated downloaded data (see rs_download_data).
* `rubin_sim.photUtils </rs_photUtils>` provides synthetic photometry and SNR tools.
* `rubin_sim.site_models </rs_site_models>` provides tools to interact with our models for seeing and weather, as well as almanacs of sunrise/sunset.
* `rubin_sim.skybrightness </rs_skybrightness>` can generate predicted skybrightness values for the Rubin site.
* `rubin_sim.skybrightness_pre </rs_skybrightness_pre>` provides pre-calculated versions of the skybrightness for the lifetime of LSST.
* `rubin_sim.scheduler </rs_scheduler>` provides the scheduling algorithms for Rubin and can generate (currently simulated) pointing histories.
* `rubin_sim.movingObjects </rs_movingObjects>` can generate ephemerides for Solar System small bodies for a simulated LSST pointing history.
* `rubin_sim.maf </rs_maf>` provides metric analysis tools for simulated pointing histories.

