.. py:currentmodule:: rubin_sim.skybrightness

.. _skybrightness:

#############
Skybrightness
#############

The ``rubin_sim.skybrightness`` module generates
predicted skybrightness values (in either magnitudes per
square arcsecond for any LSST bandpass or
as a SED over the relevant wavelengths).
It uses the ESO skybrightness model components
(includes upper and lower atmosphere emission lines, airglow continuum,
zodiacal light and scattered lunar light) and has additional
twilight components.
The model predictions have been tested against skybrightness
measurements at the LSST site.

More details about the rubin_sim version of the model and
its validation for Rubin are available in
`An optical to IR sky brightness model for the LSST by Yoachim et. al.
<https://www.osti.gov/biblio/1784946>`_.
