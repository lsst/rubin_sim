__all__ = ("PhotometricParameters", "DustValues")

import os

import numpy as np
from rubin_scheduler.data import get_data_dir

from .bandpass import Bandpass
from .sed import Sed


class DustValues:
    """Calculate extinction values

    Parameters
    ----------
    R_v : `float`
        Extinction law parameter (3.1).
    bandpassDict : `dict`
        A dict with keys of filtername and values of
        rubin_sim.phot_utils.Bandpass objects.
        Default of None will load the standard ugrizy bandpasses.
    ref_ev : `float`
        The reference E(B-V) value to use. Things in MAF assume 1.

    Note
    ----
    The value that dust_values calls "ax1" is equivalent  to r_x in any filter.
    And  r_x * ebv = A_x (the extinction due to dust in any bandpass).
    DustValues.r_x is also provided as a copy of DustValues.ax1 ..
    eventually ax1 may be deprecated in favor of r_x.
    """

    def __init__(self, r_v=3.1, bandpass_dict=None, ref_ebv=1.0):
        # Calculate dust extinction values
        self.ax1 = {}
        if bandpass_dict is None:
            bandpass_dict = {}
            root_dir = os.path.join(get_data_dir(), "throughputs", "baseline")
            for f in ["u", "g", "r", "i", "z", "y"]:
                bandpass_dict[f] = Bandpass()
                bandpass_dict[f].read_throughput(os.path.join(root_dir, f"total_{f}.dat"))

        for filtername in bandpass_dict:
            wavelen_min = bandpass_dict[filtername].wavelen.min()
            wavelen_max = bandpass_dict[filtername].wavelen.max()
            testsed = Sed()
            testsed.set_flat_sed(wavelen_min=wavelen_min, wavelen_max=wavelen_max, wavelen_step=1.0)
            self.ref_ebv = ref_ebv
            # Calculate non-dust-extincted magnitude
            flatmag = testsed.calc_mag(bandpass_dict[filtername])
            # Add dust
            a, b = testsed.setup_ccm_ab()
            testsed.add_dust(a, b, ebv=self.ref_ebv, r_v=r_v)
            # Calculate difference due to dust when EBV=1.0
            # (m_dust = m_nodust - Ax, Ax > 0)
            self.ax1[filtername] = testsed.calc_mag(bandpass_dict[filtername]) - flatmag
        # Add the R_x term, to start to transition toward this name.
        self.r_x = self.ax1.copy()


def make_dict(value, bandpass_names=("u", "g", "r", "i", "z", "y", "any")):
    newdict = {}
    for f in bandpass_names:
        newdict[f] = value
    return newdict


class DefaultPhotometricParameters:
    """
    This class will just contain a bunch of dict which store
    the default PhotometricParameters for LSST Bandpasses

    Users should not access this class (which is why it is
    not included in the __all__ declaration for this file).

    It is only used to initialize PhotometricParameters for
    a bandpass name.
    """

    # Obviously, some of these parameters (effarea, gain, platescale,
    # darkcurrent, and readnoise) will not change as a function of bandpass;
    # we are just making them dicts here to be consistent with
    # everything else (and to make it possible for
    # PhotometricParameters to access them using the bandpass name
    # passed to its constructor)
    #
    # Note: all dicts contain an 'any' key which will be the default
    # value if an unknown bandpass is asked for
    #
    # 'any' values should be kept consistent with r band

    bandpass_names = ["u", "g", "r", "i", "z", "y", "any"]

    # exposure time in seconds
    exptime_sec = 15.0
    exptime = make_dict(exptime_sec)

    # number of exposures
    nexp_n = 2
    nexp = make_dict(nexp_n)

    # effective area in cm^2
    effarea_cm2 = np.pi * (6.423 / 2.0 * 100) ** 2
    effarea = make_dict(effarea_cm2)

    # electrons per ADU
    gain_adu = 2.3
    gain = make_dict(gain_adu)

    # electrons per pixel per exposure
    readnoise_e = 8.8
    readnoise = make_dict(readnoise_e)

    # electrons per pixel per second
    darkcurrent_e = 0.2
    darkcurrent = make_dict(darkcurrent_e)

    # electrons per pixel per exposure
    othernoise_e = 0.0
    othernoise = make_dict(othernoise_e)

    # arcseconds per pixel
    platescale_as = 0.2
    platescale = make_dict(platescale_as)

    # systematic squared error in magnitudes
    # see Table 14 of the SRD document
    # https://docushare.lsstcorp.org/docushare/dsweb/Get/LPM-17
    sigma_sys = {
        "u": 0.0075,
        "g": 0.005,
        "r": 0.005,
        "i": 0.005,
        "z": 0.0075,
        "y": 0.0075,
        "any": 0.005,
    }


class PhotometricParameters:
    def __init__(
        self,
        exptime=None,
        nexp=None,
        effarea=None,
        gain=None,
        readnoise=None,
        darkcurrent=None,
        othernoise=None,
        platescale=None,
        sigma_sys=None,
        bandpass=None,
    ):
        """Store photometric parameters for SNR calculations.

        Parameters
        ----------
        exptime : `float`
            Exposure time in seconds (per exposure).
            None will default to value from DefaultPhotometricParameters.
        nexp : `int`
            Number of exposures per visit.
            None will default to value from DefaultPhotometricParameters.
        effarea : `float`
            Effective area in cm^2.
            None will default to value from DefaultPhotometricParameters.
        gain : `float`
            Electrons per ADU.
            None will default to value from DefaultPhotometricParameters.
        readnoise : `float`
            Electrons per pixel per exposure.
            None will default to value from DefaultPhotometricParameters.
        darkcurrent : `float`
            Electons per pixel per second.
            None will default to value from DefaultPhotometricParameters.
        othernoise : `float`
            Electrons per pixel per exposure.
            None will default to value from DefaultPhotometricParameters.
        platescale : `float`
            Arcseconds per pixel.
            None will default to value from DefaultPhotometricParameters.
        sigma_sys : `float`
            Systematic error in magnitudes.
            None will default to value from DefaultPhotometricParameters.
        bandpass : `str`
            The name of the bandpass for these parameters.

        Examples
        --------
        If `bandpass` is set to an LSST bandpass,
        the constructor will initialize
        PhotometricParameters to LSST default values for that bandpass,
        excepting any parameters that have been set by hand.  e.g.

        >>> myPhotParams = PhotometricParameters(nexp=3, bandpass='u')

        will initialize a PhotometricParameters object to `u` band defaults,
        except with 3 exposures instead of 2. A bandpass value of None
        will use defaults from LSST `r` band where appropriate.
        """
        # readnoise, darkcurrent and othernoise are measured in electrons.
        # This is taken from the specifications document LSE-30 on Docushare
        # Section 3.4.2.3 states that the total noise per pixel shall
        # be 12.7 electrons per visit which the defaults sum to
        # (remember to multply darkcurrent by the number of seconds
        # in an exposure=15). [9 e- per 15 second exposure]

        self._exptime = None
        self._nexp = None
        self._effarea = None
        self._gain = None
        self._platescale = None
        self._sigma_sys = None
        self._readnoise = None
        self._darkcurrent = None
        self._othernoise = None

        self._bandpass = bandpass
        defaults = DefaultPhotometricParameters()

        if bandpass is None:
            bandpass_key = "any"
            # This is so we do not set the self._bandpass member variable
            # without the user's explicit consent, but we can still access
            # default values from the PhotometricParameterDefaults
        else:
            bandpass_key = bandpass

        if bandpass_key in defaults.bandpass_names:
            self._exptime = defaults.exptime[bandpass_key]
            self._nexp = defaults.nexp[bandpass_key]
            self._effarea = defaults.effarea[bandpass_key]
            self._gain = defaults.gain[bandpass_key]
            self._platescale = defaults.platescale[bandpass_key]
            self._sigma_sys = defaults.sigma_sys[bandpass_key]
            self._readnoise = defaults.readnoise[bandpass_key]
            self._darkcurrent = defaults.darkcurrent[bandpass_key]
            self._othernoise = defaults.othernoise[bandpass_key]

        if exptime is not None:
            self._exptime = exptime

        if nexp is not None:
            self._nexp = nexp

        if effarea is not None:
            self._effarea = effarea

        if gain is not None:
            self._gain = gain

        if platescale is not None:
            self._platescale = platescale

        if sigma_sys is not None:
            self._sigma_sys = sigma_sys

        if readnoise is not None:
            self._readnoise = readnoise

        if darkcurrent is not None:
            self._darkcurrent = darkcurrent

        if othernoise is not None:
            self._othernoise = othernoise

        failure_message = ""
        failure_ct = 0

        if self._exptime is None:
            failure_message += "did not set exptime\n"
            failure_ct += 1

        if self._nexp is None:
            failure_message += "did not set nexp\n"
            failure_ct += 1

        if self._effarea is None:
            failure_message += "did not set effarea\n"
            failure_ct += 1

        if self._gain is None:
            failure_message += "did not set gain\n"
            failure_ct += 1

        if self._platescale is None:
            failure_message += "did not set platescale\n"
            failure_ct += 1

        if self._sigma_sys is None:
            failure_message += "did not set sigma_sys\n"
            failure_ct += 1

        if self._readnoise is None:
            failure_message += "did not set readnoise\n"
            failure_ct += 1

        if self._darkcurrent is None:
            failure_message += "did not set darkcurrent\n"
            failure_ct += 1

        if self._othernoise is None:
            failure_message += "did not set othernoise\n"
            failure_ct += 1

        if failure_ct > 0:
            raise RuntimeError("In PhotometricParameters:\n%s" % failure_message)

    @property
    def bandpass(self):
        """
        The name of the bandpass associated with these parameters.
        Can be None.
        """
        return self._bandpass

    @bandpass.setter
    def bandpass(self, value):
        raise RuntimeError(
            "You should not be setting bandpass on the fly; "
            + "Just instantiate a new case of PhotometricParameters"
        )

    @property
    def exptime(self):
        """
        exposure time in seconds
        """
        return self._exptime

    @exptime.setter
    def exptime(self, value):
        raise RuntimeError(
            "You should not be setting exptime on the fly; "
            + "Just instantiate a new case of PhotometricParameters"
        )

    @property
    def nexp(self):
        """
        number of exposures
        """
        return self._nexp

    @nexp.setter
    def nexp(self, value):
        raise RuntimeError(
            "You should not be setting nexp on the fly; "
            + "Just instantiate a new case of PhotometricParameters"
        )

    @property
    def effarea(self):
        """
        effective area in cm^2
        """
        return self._effarea

    @effarea.setter
    def effarea(self, value):
        raise RuntimeError(
            "You should not be setting effarea on the fly; "
            + "Just instantiate a new case of PhotometricParameters"
        )

    @property
    def gain(self):
        """
        electrons per ADU
        """
        return self._gain

    @gain.setter
    def gain(self, value):
        raise RuntimeError(
            "You should not be setting gain on the fly; "
            + "Just instantiate a new case of PhotometricParameters"
        )

    @property
    def platescale(self):
        """
        arcseconds per pixel
        """
        return self._platescale

    @platescale.setter
    def platescale(self, value):
        raise RuntimeError(
            "You should not be setting platescale on the fly; "
            + "Just instantiate a new case of PhotometricParameters"
        )

    @property
    def readnoise(self):
        """
        electrons per pixel per exposure
        """
        return self._readnoise

    @readnoise.setter
    def readnoise(self, value):
        raise RuntimeError(
            "You should not be setting readnoise on the fly; "
            + "Just instantiate a new case of PhotometricParameters"
        )

    @property
    def darkcurrent(self):
        """
        electrons per pixel per second
        """
        return self._darkcurrent

    @darkcurrent.setter
    def darkcurrent(self, value):
        raise RuntimeError(
            "You should not be setting darkcurrent on the fly; "
            + "Just instantiate a new case of PhotometricParameters"
        )

    @property
    def othernoise(self):
        """
        electrons per pixel per exposure
        """
        return self._othernoise

    @othernoise.setter
    def othernoise(self, value):
        raise RuntimeError(
            "You should not be setting othernoise on the fly; "
            + "Just instantiate a new case of PhotometricParameters"
        )

    @property
    def sigma_sys(self):
        """
        systematic error in magnitudes
        """
        return self._sigma_sys

    @sigma_sys.setter
    def sigma_sys(self, value):
        raise RuntimeError(
            "You should not be setting sigma_sys on the fly; "
            + "Just instantiate a new case of PhotometricParameters"
        )
