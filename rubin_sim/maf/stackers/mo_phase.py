"""Phase functions for moving objects.

Adapted from code written by Michael Kelley - mkelley @ github
(https://github.com/lsst-sssc/lsstcomet/blob/master/lsstcomet/phase.py)
[the HalleyMarcus phase curve is entirely from Michael Kelley's code]
"""

__all__ = ("phase__halley_marcus", "phase__log_linear", "phase_hg")

import numpy as np
from scipy.interpolate import splev, splrep

_halley_marcus_phase_curve = splrep(
    np.arange(181),
    np.array(
        [
            1.0000e00,
            9.5960e-01,
            9.2170e-01,
            8.8590e-01,
            8.5220e-01,
            8.2050e-01,
            7.9060e-01,
            7.6240e-01,
            7.3580e-01,
            7.1070e-01,
            6.8710e-01,
            6.6470e-01,
            6.4360e-01,
            6.2370e-01,
            6.0490e-01,
            5.8720e-01,
            5.7040e-01,
            5.5460e-01,
            5.3960e-01,
            5.2550e-01,
            5.1220e-01,
            4.9960e-01,
            4.8770e-01,
            4.7650e-01,
            4.6590e-01,
            4.5590e-01,
            4.4650e-01,
            4.3770e-01,
            4.2930e-01,
            4.2150e-01,
            4.1420e-01,
            4.0730e-01,
            4.0090e-01,
            3.9490e-01,
            3.8930e-01,
            3.8400e-01,
            3.7920e-01,
            3.7470e-01,
            3.7060e-01,
            3.6680e-01,
            3.6340e-01,
            3.6030e-01,
            3.5750e-01,
            3.5400e-01,
            3.5090e-01,
            3.4820e-01,
            3.4580e-01,
            3.4380e-01,
            3.4210e-01,
            3.4070e-01,
            3.3970e-01,
            3.3890e-01,
            3.3850e-01,
            3.3830e-01,
            3.3850e-01,
            3.3890e-01,
            3.3960e-01,
            3.4050e-01,
            3.4180e-01,
            3.4320e-01,
            3.4500e-01,
            3.4700e-01,
            3.4930e-01,
            3.5180e-01,
            3.5460e-01,
            3.5760e-01,
            3.6090e-01,
            3.6450e-01,
            3.6830e-01,
            3.7240e-01,
            3.7680e-01,
            3.8150e-01,
            3.8650e-01,
            3.9170e-01,
            3.9730e-01,
            4.0320e-01,
            4.0940e-01,
            4.1590e-01,
            4.2280e-01,
            4.3000e-01,
            4.3760e-01,
            4.4560e-01,
            4.5400e-01,
            4.6270e-01,
            4.7200e-01,
            4.8160e-01,
            4.9180e-01,
            5.0240e-01,
            5.1360e-01,
            5.2530e-01,
            5.3750e-01,
            5.5040e-01,
            5.6380e-01,
            5.7800e-01,
            5.9280e-01,
            6.0840e-01,
            6.2470e-01,
            6.4190e-01,
            6.5990e-01,
            6.7880e-01,
            6.9870e-01,
            7.1960e-01,
            7.4160e-01,
            7.6480e-01,
            7.8920e-01,
            8.1490e-01,
            8.4200e-01,
            8.7060e-01,
            9.0080e-01,
            9.3270e-01,
            9.6640e-01,
            1.0021e00,
            1.0399e00,
            1.0799e00,
            1.1223e00,
            1.1673e00,
            1.2151e00,
            1.2659e00,
            1.3200e00,
            1.3776e00,
            1.4389e00,
            1.5045e00,
            1.5744e00,
            1.6493e00,
            1.7294e00,
            1.8153e00,
            1.9075e00,
            2.0066e00,
            2.1132e00,
            2.2281e00,
            2.3521e00,
            2.4861e00,
            2.6312e00,
            2.7884e00,
            2.9592e00,
            3.1450e00,
            3.3474e00,
            3.5685e00,
            3.8104e00,
            4.0755e00,
            4.3669e00,
            4.6877e00,
            5.0418e00,
            5.4336e00,
            5.8682e00,
            6.3518e00,
            6.8912e00,
            7.4948e00,
            8.1724e00,
            8.9355e00,
            9.7981e00,
            1.0777e01,
            1.1891e01,
            1.3166e01,
            1.4631e01,
            1.6322e01,
            1.8283e01,
            2.0570e01,
            2.3252e01,
            2.6418e01,
            3.0177e01,
            3.4672e01,
            4.0086e01,
            4.6659e01,
            5.4704e01,
            6.4637e01,
            7.7015e01,
            9.2587e01,
            1.1237e02,
            1.3775e02,
            1.7060e02,
            2.1348e02,
            2.6973e02,
            3.4359e02,
            4.3989e02,
            5.6292e02,
            7.1363e02,
            8.8448e02,
            1.0533e03,
            1.1822e03,
            1.2312e03,
        ]
    ),
)


def phase__halley_marcus(phase):
    """Halley-Marcus composite dust phase function.
    This is appropriate for use when calculating the brightness of cometary coma.

    Parameters
    ----------
    phase : float or array
        Phase angle (degrees).

    Returns
    -------
    phi : float or array
        Phase function evaluated at ``phase``.

    """
    return splev(phase, _halley_marcus_phase_curve)


def phase__log_linear(phase, slope=0.04):
    """A logLinear phase function, roughly appropriate for cometary nuclei.
    An H-G phase function is likely a better approximation.

    Parameters
    ----------
    phase : float or array
        Phase angle (degrees)
    slope : float, optional
        The slope for the phase function. Default 0.04.

    Returns
    -------
    phi : float or array
        Phase function evaluated at phase
    """
    return 10 ** (-0.4 * slope * phase)


def phase_hg(phase, G=0.15):
    """The Bowell et al 1989 (Asteroids II) HG phase curve.
    https://ui.adsabs.harvard.edu/abs/1989aste.conf..524B/abstract

    Parameters
    ----------
    phase : float or array
        Phase angle (degrees)
    G : float, optional
        The G value for the formula. Default 0.15.

    Returns
    -------
    phi : float or array
        Phase function evaluated at phase
    """
    # see Muinonen et al 2010, eqn 6 (http://dx.doi.org/10.1016/j.icarus.2010.04.003)
    phi1 = np.exp(-3.33 * np.power(np.tan(np.radians(phase) / 2), 0.63))
    phi2 = np.exp(-1.87 * np.power(np.tan(np.radians(phase) / 2), 1.22))
    return (1 - G) * phi1 + G * phi2


def phase_hg12(phase, g12=0.1):
    pass
