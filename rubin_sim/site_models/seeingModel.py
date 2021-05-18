import numpy as np
import warnings
import os
from rubin_sim.photUtils import Bandpass
from rubin_sim.data import get_data_dir


__all__ = ["SeeingModel"]


class SeeingModel(object):
    """LSST FWHM calculations for FWHM_effective and FWHM_geometric.
    Calculations of the delivered values are based on equations in Document-20160
    ("Atmospheric and Delivered Image Quality in OpSim" by Bo Xin, George Angeli, Zeljko Ivezic)

    Parameters
    ----------
    XXX--add documentation

    """
    def __init__(self, filter_list=['u', 'g', 'r', 'i', 'z', 'y'],
                 eff_wavelens=None,
                 telescope_seeing=0.25, optical_design_seeing=0.08, camera_seeing=0.30,
                 raw_seeing_wavelength=500, efd_seeing='FWHM_500'):
        self.filter_list = filter_list
        if eff_wavelens is None:
            fdir = os.path.join(get_data_dir(), 'throughputs/baseline')
            eff_wavelens = []
            for f in filter_list:
                bp = Bandpass()
                bp.readThroughput(os.path.join(fdir, 'total_' + f + '.dat'))
                eff_wavelens.append(bp.calcEffWavelen()[1])
        self.eff_wavelens = np.array(eff_wavelens)
        self.telescope_seeing = telescope_seeing
        self.optical_design_seeing = optical_design_seeing
        self.raw_seeing_wavelength = raw_seeing_wavelength
        self.efd_seeing = efd_seeing
        self.camera_seeing = camera_seeing

        self._set_fwhm_zenith_system()

    def configure(self, config=None):
        """
        """
        warnings.warn('the configure method is deprecated')

    def config_info(self):
        """Report configuration parameters and version information.

        Returns
        -------
        OrderedDict
        """
        warnings.warn('the config_info method is deprecated.')
        return None

    def _set_fwhm_zenith_system(self):
        """Calculate the system contribution to FWHM at zenith.

        This is simply the individual telescope, optics, and camera contributions
        combined in quadrature.
        """
        self.fwhm_system_zenith = np.sqrt(self.telescope_seeing**2 +
                                          self.optical_design_seeing**2 +
                                          self.camera_seeing**2)

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
        fwhm_z: float, or efdData dict
            FWHM at zenith (arcsec).
        airmass: float, np.array, or targetDict
            Airmass (unitless).

        Returns
        -------
        dict of numpy.ndarray, numpy.ndarray
            FWHMeff, FWHMgeom: both are the same shape numpy.ndarray.
            If airmass is a single value, FWHMeff & FWHMgeom are 1-d arrays,
            with the same order as eff_wavelen (i.e. eff_wavelen[0] = u, then FWHMeff[0] = u).
            If airmass is a numpy array, FWHMeff and FWHMgeom are 2-d arrays,
            in the order of <filter><airmass> (i.e. eff_wavelen[0] = u, 1-d array over airmass range).
        """
        if isinstance(fwhm_z, dict):
            fwhm_z = fwhm_z[self.efd_seeing]
        if isinstance(airmass, dict):
            airmass = airmass['airmass']
        airmass_correction = np.power(airmass, 0.6)
        wavelen_correction = np.power(self.raw_seeing_wavelength / self.eff_wavelens, 0.3)
        if isinstance(airmass, np.ndarray):
            fwhm_system = self.fwhm_system_zenith * np.outer(np.ones(len(wavelen_correction)),
                                                             airmass_correction)
            fwhm_atmo = fwhm_z * np.outer(wavelen_correction, airmass_correction)
        else:
            fwhm_system = self.fwhm_system_zenith * airmass_correction
            fwhm_atmo = fwhm_z * wavelen_correction * airmass_correction
        # Calculate combined FWHMeff.
        fwhm_eff = 1.16 * np.sqrt(fwhm_system ** 2 + 1.04 * fwhm_atmo ** 2)
        # Translate to FWHMgeom.
        fwhm_geom = self.fwhmEff_to_fwhmGeom(fwhm_eff)
        return {'fwhmEff': fwhm_eff, 'fwhmGeom': fwhm_geom}

    @staticmethod
    def fwhmEff_to_fwhmGeom(fwhm_eff):
        """Calculate FWHM_geom from FWHM_eff.

        Parameters
        ----------
        fwhm_eff : float or np.ndarray

        Returns
        -------
        float or np.ndarray
        """
        return (0.822 * fwhm_eff + 0.052)

    @staticmethod
    def fwhmGeom_to_fwhmEff(fwhm_geom):
        """Calculate FWHM_eff from FWHM_geom.

        Parameters
        ----------
        fwhm_geom : float or np.ndarray

        Returns
        -------
        float or np.ndarray
        """
        return (fwhm_geom - 0.052)/0.822
