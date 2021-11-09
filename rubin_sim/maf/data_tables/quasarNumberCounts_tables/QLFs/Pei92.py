#!/usr/bin/env python

from astropy.table import Table
import astropy.units as u
from scipy.interpolate import interp1d
import numpy as np
import os

class P92_Extinction(object):

    def __init__(self, red_type):

        self.red_type = red_type
        if red_type not in ["MW", "LMC", "SMC"]:
            print("Model {} not recognized.".format(red_type))
            return

        #Start by reading the RV tables.
        self.R_V = dict()
        RV_file = open(os.path.dirname(__file__)+"/Pei92_RV.txt")
        for line in RV_file:
            x = line.split()
            self.R_V[x[0]] = float(x[1])
        RV_file.close()

        #Now, read the extinction tables.
        extinction_curves = Table.read(os.path.dirname(__file__)+"/Pei92_Extinction_curves.txt", format='ascii')

        #For the requested model, caculate Xi and make the interpolation function.
        lam     = 1./extinction_curves['ilam_'+red_type] * u.um
        Erat    = extinction_curves['E_rat_'+red_type]
        self.xi_func = interp1d(lam.value, (Erat+self.R_V[red_type])/(1+self.R_V[red_type]), fill_value='extrapolate')

        #Also, load the best-fit model.
        self.xi_fit_table = Table.read(os.path.dirname(__file__)+"/Pei92_xi_fit_{}.txt".format(red_type), format='ascii')

        self.a = self.xi_fit_table['a']
        self.b = self.xi_fit_table['b']
        self.n = self.xi_fit_table['n']
        self.lam_fit = self.xi_fit_table['lam']

        return

    def xi(self,lam):
        return self.xi_func(lam.to(u.um).value)

    def xi_fit(self, lam):
        if isinstance(lam, np.ndarray):
            lam2D = np.tile(lam.to(u.micron).value, [6, 1]).T
            x = lam2D/self.lam_fit
            return np.sum(self.a/(x**self.n+x**-self.n+self.b), axis=1)
        else:
            x = lam.to(u.micron).value/self.lam_fit
            return np.sum(self.a/(x**self.n+x**-self.n+self.b))
