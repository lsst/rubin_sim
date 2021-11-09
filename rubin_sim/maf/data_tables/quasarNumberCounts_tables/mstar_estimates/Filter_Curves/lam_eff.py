import astropy.units as u

#LSST Filter wavelengths. Will be useful for the obscuration function we will need to set up.
lam_eff = {'LSSTu': 3751.36*u.AA,
           'LSSTg': 4741.64*u.AA,
           'LSSTr': 6173.23*u.AA,
           'LSSTi': 7501.62*u.AA,
           'LSSTz': 8679.19*u.AA,
           'LSSTy': 9711.53*u.AA,
          }
