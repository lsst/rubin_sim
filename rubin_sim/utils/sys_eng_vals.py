# Values from systems engineering that may need to be updated when new filter curves come out

__all__ = ("SysEngVals",)


class SysEngVals:
    """Object to store values calculated in sys-eng

    generated from notebook in:  https://github.com/lsst-pst/syseng_throughputs/tree/master/notebooks
    """

    def __init__(self):
        self.zp_t = {
            "u": 26.524237,
            "g": 28.508375,
            "r": 28.360838,
            "i": 28.171396,
            "z": 27.782264,
            "y": 26.817819,
        }
        self.tb = {"u": 0.022928, "g": 0.142565, "r": 0.124451, "i": 0.104526, "z": 0.073041, "y": 0.030046}
        self.gamma = {
            "u": 0.037534,
            "g": 0.038715,
            "r": 0.039034,
            "i": 0.039196,
            "z": 0.039320,
            "y": 0.039345,
        }
        self.k_atm = {
            "u": 0.470116,
            "g": 0.212949,
            "r": 0.126369,
            "i": 0.095764,
            "z": 0.068417,
            "y": 0.171009,
        }
        self.cm = {
            "u": 22.967681,
            "g": 24.582309,
            "r": 24.602134,
            "i": 24.541152,
            "z": 24.371077,
            "y": 23.840175,
        }
        self.d_cm_infinity = {
            "u": 0.543325,
            "g": 0.088310,
            "r": 0.043438,
            "i": 0.027510,
            "z": 0.018530,
            "y": 0.016283,
        }
        self.d_cm_double = {
            "u": 0.343781,
            "g": 0.043738,
            "r": 0.019756,
            "i": 0.011651,
            "z": 0.007261,
            "y": 0.006144,
        }
        self.sky_mag = {
            "u": 23.051983,
            "g": 22.253839,
            "r": 21.197579,
            "i": 20.462795,
            "z": 19.606305,
            "y": 18.601512,
        }
        self.exptime = 30
