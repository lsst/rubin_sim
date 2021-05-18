# Values from systems engineering that may need to be updated when new filter curves come out

__all__ = ['SysEngVals']


class SysEngVals(object):
    """Object to store values calculated in sys-eng

    generated from notebook in:  https://github.com/lsst-pst/syseng_throughputs/tree/master/notebooks
    """
    def __init__(self):
        self.Zp_t = {"u": 27.029555, "g": 28.380922, "r": 28.155692, "i": 27.856980, "z": 27.460876, "y": 26.680288}
        self.Tb = {"u": 0.036516, "g": 0.126775, "r": 0.103025, "i": 0.078245, "z": 0.054327, "y": 0.026472}
        self.gamma = {"u": 0.037809, "g": 0.038650, "r": 0.038948, "i": 0.039074, "z": 0.039219, "y": 0.039300}
        self.kAtm = {"u": 0.502296, "g": 0.213738, "r": 0.125886, "i": 0.096182, "z": 0.068623, "y": 0.169504}
        self.Cm = {"u": 23.390261, "g": 24.506791, "r": 24.489914, "i": 24.372551, "z": 24.202753, "y": 23.769195}
        self.dCm_infinity = {"u": 0.371939, "g": 0.098515, "r": 0.051961, "i": 0.036845, "z": 0.024581, "y": 0.018609}
        self.dCm_double = {"u": 0.220178, "g": 0.049343, "r": 0.024140, "i": 0.016277, "z": 0.010146, "y": 0.007225}
        self.skyMag = {"u": 22.960730, "g": 22.257758, "r": 21.196590, "i": 20.477419, "z": 19.599578, "y": 18.610405}

        self.exptime = 30.
