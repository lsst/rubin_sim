__all__ = ("SeeingModel",)

import os
import warnings

import numpy as np

from rubin_sim.data import get_data_dir
from rubin_sim.phot_utils import Bandpass


class SeeingModel:
    """LSST FWHM calculations for FWHM_effective and FWHM_geometric.

    Calculations of the delivered values are based on equations in Document-20160
    ("Atmospheric and Delivered Image Quality in OpSim" by Bo Xin, George Angeli, Zeljko Ivezic)
    An example of the calculation of delivered image seeing from DIMM FWHM_500 is available in
    https://smtn-002.lsst.io/#calculating-m5-values-in-the-lsst-operations-simulator

    Parameters
    ----------
    filter_list : `list` [`str`], opt
        List of the filter bandpasses for which to calculate delivered FWHM_effective and FWHM_geometric
        Default ['u', 'g', 'r', 'i', 'z', 'y']
    eff_wavelens : `list` [`float`] or None, opt
        Effective wavelengths for those bandpasses, in nanometers.
        If None, the SeeingModel will read the throughput curves from disk
        ($RUBIN_SIM_DATA_DIR/throughputs/baseline/total_[f].dat) and calculate effective wavelengths.
    telescope_seeing : `float`, opt
        The contribution to the delivered FWHM from the telescope, in arcseconds.
        Default 0.25"
    optical_design_seeing : `float`, opt
        The contribution to the seeing from the optical design, in arcseconds.
        Default 0.08 arcseconds
    camera_seeing : `float`, opt
        The contribution to the seeing from the camera, in arcseconds.
        Default 0.30 arcseconds
    raw_seeing_wavelength : `float`, opt
        The wavelength of the DIMM-delivered equivalent FWHM, in nanometers.
        Default 500nm.
    efd_seeing : `str`, opt
        The name of the DIMM FWHM measurements in the efd / conditions object.
        Default `FWHM_500`
    """

    def __init__(
        self,
        filter_list=["u", "g", "r", "i", "z", "y"],
        eff_wavelens=None,
        telescope_seeing=0.25,
        optical_design_seeing=0.08,
        camera_seeing=0.30,
        raw_seeing_wavelength=500,
        efd_seeing="FWHM_500",
    ):
        self.filter_list = filter_list
        if eff_wavelens is None:
            fdir = os.path.join(get_data_dir(), "throughputs/baseline")
            eff_wavelens = []
            for f in filter_list:
                bp = Bandpass()
                bp.read_throughput(os.path.join(fdir, "total_" + f + ".dat"))
                eff_wavelens.append(bp.calc_eff_wavelen()[1])
        self.eff_wavelens = np.array(eff_wavelens)
        self.telescope_seeing = telescope_seeing
        self.optical_design_seeing = optical_design_seeing
        self.raw_seeing_wavelength = raw_seeing_wavelength
        self.efd_seeing = efd_seeing
        self.camera_seeing = camera_seeing

        self._set_fwhm_zenith_system()

    def configure(self):
        """Deprecated. Configure through the init method."""
        warnings.warn("the configure method is deprecated")

    def config_info(self):
        """Deprecated. Report configuration parameters and version information."""
        warnings.warn("the config_info method is deprecated.")

    def _set_fwhm_zenith_system(self):
        """Calculate the system contribution to FWHM at zenith.

        This is simply the individual telescope, optics, and camera contributions
        combined in quadrature.
        """
        self.fwhm_system_zenith = np.sqrt(
            self.telescope_seeing**2 + self.optical_design_seeing**2 + self.camera_seeing**2
        )

    def __call__(self, fwhm_z, airmass):
        """Calculate the seeing values FWHM_eff and FWHM_geom at the given airmasses,
        for the specified effective wavelengths, given FWHM_zenith (typically FWHM_500).

        FWHM_geom represents the geometric size of the PSF; FWHM_eff represents the FWHM of a
        single gaussian which encloses the same number of pixels as N_eff (the number of pixels
        enclosed in the actual PSF -- this is the value to use when calculating SNR).

        FWHM_geom(") = 0.822 * FWHM_eff(") + 0.052"

        The FWHM_eff includes a contribution from the system and from the atmosphere.
        Both of these are expected to scale with airmass^0.6 and with (500(nm)/wavelength(nm))^0.3.
        FWHM_eff = 1.16 * sqrt(FWHM_sys**2 + 1.04*FWHM_atm**2)

        Parameters
        ----------
        fwhm_z: `float`, or efdData `dict`
            FWHM at zenith (arcsec).
        airmass: `float`, `np.array`, or targetDict `dict`
            Airmass (unitless).

        Returns
        -------
        FWHMeff, FWHMGeom : `dict` of {`numpy.ndarray`, `numpy.ndarray`}
            FWHMeff, FWHMgeom: both are the same shape numpy.ndarray.
            If airmass is a single value, FWHMeff & FWHMgeom are 1-d arrays,
            with the same order as eff_wavelen (i.e. eff_wavelen[0] = u, then FWHMeff[0] = u).
            If airmass is a numpy array, FWHMeff and FWHMgeom are 2-d arrays,
            in the order of <filter><airmass> (i.e. eff_wavelen[0] = u, 1-d array over airmass range).
        """
        if isinstance(fwhm_z, dict):
            fwhm_z = fwhm_z[self.efd_seeing]
        if isinstance(airmass, dict):
            airmass = airmass["airmass"]
        airmass_correction = np.power(airmass, 0.6)
        wavelen_correction = np.power(self.raw_seeing_wavelength / self.eff_wavelens, 0.3)
        if isinstance(airmass, np.ndarray):
            fwhm_system = self.fwhm_system_zenith * np.outer(
                np.ones(len(wavelen_correction)), airmass_correction
            )
            fwhm_atmo = fwhm_z * np.outer(wavelen_correction, airmass_correction)
        else:
            fwhm_system = self.fwhm_system_zenith * airmass_correction
            fwhm_atmo = fwhm_z * wavelen_correction * airmass_correction
        # Calculate combined FWHMeff.
        fwhm_eff = 1.16 * np.sqrt(fwhm_system**2 + 1.04 * fwhm_atmo**2)
        # Translate to FWHMgeom.
        fwhm_geom = self.fwhm_eff_to_fwhm_geom(fwhm_eff)
        return {"fwhmEff": fwhm_eff, "fwhmGeom": fwhm_geom}

    @staticmethod
    def fwhm_eff_to_fwhm_geom(fwhm_eff):
        """Calculate FWHM_geom from FWHM_eff.

        Parameters
        ----------
        fwhm_eff : `float` or `np.ndarray`

        Returns
        -------
        FWHM_geom : `float` or `np.ndarray`
        """
        return 0.822 * fwhm_eff + 0.052

    @staticmethod
    def fwhm_geom_to_fwhm_eff(fwhm_geom):
        """Calculate FWHM_eff from FWHM_geom.

        Parameters
        ----------
        fwhm_geom : `float` or `np.ndarray`

        Returns
        -------
        FWHM_eff : `float` or `np.ndarray`
        """
        return (fwhm_geom - 0.052) / 0.822
