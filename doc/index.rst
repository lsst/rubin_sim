.. py:currentmodule:: rubin_sim

.. _rubin_sim:

#########
rubin_sim
#########

The rubin_sim module provides support for Rubin Observatory's LSST survey
scheduler, survey strategy analysis, and some basic simulation requirements.

List of submodules:

* :doc:`rubin_sim.utils <rs_utils/index>` provides some basic utilities we use throughout the rest of rubin_sim, but may be useful for other purposes.
* :doc:`rubin_sim.data <rs_data/index>` provides a minimal tool to track the location of the associated downloaded data (see rs_download_data).
* :doc:`rubin_sim.phot_utils <rs_phot_utils/index>` provides synthetic photometry and SNR tools.
* :doc:`rubin_sim.satellite_constellations <rs_satellite_constellations/index>` tools for mega satellite constellations.
* :doc:`rubin_sim.selfcal <rs_selfcal/index>` generating stellar catalogs and running self-calibration.
* :doc:`rubin_sim.site_models <rs_site_models/index>` provides tools to interact with our models for seeing and weather, as well as almanacs of sunrise/sunset.
* :doc:`rubin_sim.skybrightness <rs_skybrightness/index>` can generate predicted skybrightness values for the Rubin site.
* :doc:`rubin_sim.skybrightness_pre <rs_skybrightness_pre/index>` provides pre-calculated versions of the skybrightness for the lifetime of LSST.
* :doc:`rubin_sim.scheduler <rs_scheduler/index>` provides the scheduling algorithms for Rubin and can generate (currently simulated) pointing histories.
* :doc:`rubin_sim.moving_objects <rs_moving_objects/index>` can generate ephemerides for Solar System small bodies for a simulated LSST pointing history.
* :doc:`rubin_sim.maf <rs_maf/index>` provides metric analysis tools for simulated pointing histories.


:doc:`Table of Contents <toc>`