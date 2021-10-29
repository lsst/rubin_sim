import numpy as np
from scipy import interpolate
import astropy.units as u

import os
import re

class Fast_Nqso(object):

    def __init__(self, band, qlf_module, qlf_model, SED_model="Richards06", area=4.*np.pi*u.sr):

        #Save the input values.
        self.band = band
        self.qlf_module  = qlf_module
        self.qlf_model   = qlf_model
        self.SED_model   = SED_model
        self.area        = area
        #self.area_factor = (area/(4.*np.pi*u.sr)).to(1.).value
        self.area_factor = area.to(u.deg**2).value

        #Start by reading the long table. These are numbers for the entire sky.
        cat = open("../../rubin_sim/maf/data_tables/Long_Table.{0}.{1}.{2}.{3}.txt".format(self.band, self.qlf_module, self.qlf_model, self.SED_model))
        mags = np.array([float(x) for x in cat.readline().split()])
        zs   = np.array([float(x) for x in cat.readline().split()])
        mz_data = np.loadtxt(cat)
        cat.close()

        #Make the table cumulative.
        c_mz_data = np.zeros((mz_data.shape[0]+1, mz_data.shape[1]+1))
        c_mz_data[1:,1:] = mz_data
        c_mz_data = np.cumsum(c_mz_data, axis=0)
        c_mz_data = np.cumsum(c_mz_data, axis=1)

        #Now, make a 2D interpolation. The last element seems to be nan always because of an error in the code that made the mstar_z files. So, we just don't use that last element.
        self.Nqso_cumulative = interpolate.interp2d(zs[:-1], mags[:-1], c_mz_data[:-1,:-1], kind='cubic')
        #Nqso_cumulative = interpolate.RectBivariateSpline(zs, mags, c_mz_data)

        return

    def Nqso(self, zmin, zmax, m_bright, m_faint):
        """
        Function that returns the number of quasars in a redshift and magnitude range for a given area.
        """

        N11 = self.Nqso_cumulative(zmin, m_bright)
        N12 = self.Nqso_cumulative(zmin, m_faint )
        N21 = self.Nqso_cumulative(zmax, m_bright)
        N22 = self.Nqso_cumulative(zmax, m_faint )

        return (N22 - N21 - N12 + N11)*self.area_factor
