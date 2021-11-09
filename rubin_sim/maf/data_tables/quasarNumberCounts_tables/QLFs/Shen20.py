import numpy as np
from astropy.constants import L_sun, c
import astropy.units as u
import os
from scipy.special import erf

# import os
# import re
# root_path = re.search("(.*/AGN_Photoz_LSST_OpSim)/*",os.getcwd()).group(1)

# import sys
# sys.path.append(root_path+"/QLFs/")
from Pei92 import P92_Extinction

class QLF(object):

    """
    Class that implements the Quasar Luminosity Function from Shen et al. (2020, arXiv:2001.02696), shortenned to S20 hereafter.

    Each of the functional form parameters, as well as the space density, are implemented as methods. Each method is documented with comments throughout its definition.

    Parameters
    ----------
    model: string
        A or B. Determins the Shen et al. QLF parametrization to use. Default is model B.

    """

    def __init__(self, model="B"):

        #Save the model.
        self.model = model

        #Check which model is being used.
        if model=="A":
            icol = 1
        elif model=="B":
            icol = 4
        else:
            print("Wrong model")
            return

        #Read the QLF model parameters from Shen20.dat, which is just Table 4 of S20.
        S20_T4 = open(os.path.dirname(__file__)+"/Shen20.dat")
        for line in S20_T4:
            x = line.split()
            exec("self.{0} = {1}".format(x[0], x[icol]))
        S20_T4.close()

        #Reference redshift parameter (see section 4.2 of S20).
        self.z_ref = 2

        #Dust to gas ratio assumed: A_B/NH
        self.dgr_local = 8.47e-22 * u.cm**2

        #Units of Lstar and phi_star. Since the methods log_Lstar and log_phi_star return the base 10 logarithm of them, we need to maintain the units in these variables such that we can write
        #
        # Lstar = 10.**(self.log_Lstar(z)) * Lstar_units
        #
        # phi_star = 10.**(self.log_phi_star(z)) * phi_star_units
        #
        # to get those parameters in the correct units.
        self.Lstar_units = L_sun
        self.phi_star_units = u.dex**-1 * u.Mpc**-3

        #Set the reddening model.
        self.red_model = P92_Extinction("MW")

        #Coefficients to calculate the bolometric correction for B-band.
        self.c_B = np.array([3.759, 9.830])
        self.k_B = np.array([-0.361, -0.0063])

        #Coefficients to calculate the bolometric correction dispersion for B-band.
        self.sig1_B , self.sig2_B , self.logL0_B , self.sig3_B  = -0.383, 0.405, 42.39, 2.378
        self.sig1_15, self.sig2_15, self.logL0_15, self.sig3_15 = -0.338, 0.407, 42.16, 2.193
        self.sig1_SX, self.sig2_SX, self.logL0_SX, self.sig3_SX = 0.080, 0.180, 44.16, 1.496

        #Coefficients to calculate the bolometric correction for the 2-10keV band.
        self.c_HX = np.array([4.073, 12.60])
        self.k_HX = np.array([-0.026, 0.278])

        return

    def log_Lstar(self, z):
        """
        This method returns the base 10 logarithm of bolometric value of L_star in units of solar luminosities (L_sun) at redshift z.

        Based on equation (14) of S20.

        """
        x = (1.+z)/(1.+self.z_ref)
        log_Lstar = 2.*self.c0/(x**self.c1+x**self.c2)
        return log_Lstar


    def gamma1(self, z):
        """
        This method returns the value of the gamma_1(z) parameter at redshift z.
        The Chebyshev polynomials are defined later as the method _T(n,x) for n=0, 1 and 2.

        Based on equations (14 and 16) of S20.

        """
        if self.model=="A":
            x = (1+z)
            return self.a0*self._T(0,x) + self.a1*self._T(1,x) + self.a2*self._T(2,x)
        elif self.model=="B":
            x = (1.+z)/(1.+self.z_ref)
            return self.a0 * x**self.a1
        else:
            return np.nan

    def gamma2(self, z):
        """
        This method returns the value of the gamma_2(z) parameter at redshift z.

        Based on equation (14) of S20.

        """
        x = (1.+z)/(1.+self.z_ref)
        return 2*self.b0/(x**self.b1+x**self.b2)

    def log_phi_star(self, z):
        """
        This method returns the base 10 logarithm of phi_star(z) in units of mag^-1 cMpc^-3.

        Based on equation (14) of S20.

        """
        x = (1+z)
        return self.d0*self._T(0,x) + self.d1*self._T(1,x)


    def phi_bol(self, L, z):
        """
        This method returns the space density of quasars with bolometric luminosity L, in units of mag^-1 cMpc^-3.

        Based on equation (11) of S20.

        """
        Lstar = 10.**(self.log_Lstar(z)) * self.Lstar_units
        x = L/Lstar
        phi_star = 10.**(self.log_phi_star(z)) * self.phi_star_units
        return phi_star/(x**self.gamma1(z)+x**self.gamma2(z))

    def phi_bol_Lfrac(self, Lfrac, z):
        """
        This method returns the space density of quasars with bolometric luminosity L, in units of mag^-1 cMpc^-3.

        Based on equation (11) of S20.

        """
        phi_star = 10.**(self.log_phi_star(z)) * self.phi_star_units
        return phi_star/(Lfrac**self.gamma1(z)+Lfrac**self.gamma2(z))

    def dndL(self, Lfrac, z):
        """
        This method returns the number density of quasars with bolometric luminosity L, in units of Lsun^-1 cMpc^-3.

        Based on equation (12) of S20.

        Parameters
        ----------
        Lfrac: float
            Bolometric luminosity fraction L/L*.

        z: float
            Redshift

        """
        Lstar = 10.**(self.log_Lstar(z)) * self.Lstar_units
        phi_star = 10.**(self.log_phi_star(z)) * self.phi_star_units
        phi_star_prime = phi_star/np.log(10.) * u.dex
        alpha = -(self.gamma1(z)+1)
        beta  = -(self.gamma2(z)+1)
        return (phi_star_prime/Lstar)/(Lfrac**(-alpha)+Lfrac**(-beta))

    def dndLfrac(self, Lfrac, z):
        """
        This method returns the number density of quasars with bolometric luminosity fraction Lfrac=L/L*, in units of Lsun^-1 cMpc^-3.

        Based on equation (12) of S20, modified to make the derivative be with respect to L/L* instead of L.

        This is just a simple rewrite of equation (12) to avoid calculating L* in every iteration if we know the value of L/L* instead of L. This is useful for integrating over L/L* instead of over L.

        Parameters
        ----------
        Lfrac: float
            Bolometric luminosity fraction L/L*.

        z: float
            Redshift

        """
        phi_star = 10.**(self.log_phi_star(z)) * self.phi_star_units
        phi_star_prime = phi_star/np.log(10.) * u.dex
        alpha = -(self.gamma1(z)+1)
        beta  = -(self.gamma2(z)+1)
        return phi_star_prime/(Lfrac**(-alpha)+Lfrac**(-beta))


    def L1450(self, Lbol):
        """
        This method returns the L1450 monochromatic luminosity of a quasar of bolometric luminosity Lbol using equation (5) and the coefficients in Table 1 of S20.

        While the typical use of equation (5) is to determine Lbol given an observable monochromatic luminosity, here we use the conversion to go from Lbol to L1450. A direct application of this function is used in the accompanying script mstar.vandenberk.py, where we want to estimate the observed fluxes of a type 1 quasar with bolometric luminosity equal to L*.

        Note that the monochromatic luminosity is defined as in Table 1 of S20, so the units are the same as those in Lbol. In other words, this method return nu*L_nu, not L_nu.

        """
        #Coefficients from Table 1 for the UV luminosity.
        c1, k1, c2, k2 = 1.862, -0.361, 4.870, -0.0063
        #Implementation of equation (5).
        x = Lbol/(1e10*L_sun)
        bc = c1*x**k1 + c2*x**k2
        return Lbol/bc

    def L_B(self, Lbol):
        """
        This method returns the B-band luminosity of a quasar of bolometric luminosity Lbol using equation (5) of S20.

        While the typical use of equation (5) is to determine Lbol given an observable monochromatic luminosity, here we use the conversion to go from Lbol to LB. A direct application of this function is used in the accompanying script mstar.vandenberk.py, where we want to estimate the observed fluxes of a type 1 quasar with bolometric luminosity equal to L*.

        """
        #Implementation of equation (5).
        x = Lbol/(1e10*L_sun)
        bc = self.c_B[0]*x**self.k_B[0] + self.c_B[1]*x**self.k_B[1]
        return Lbol/bc

    def L_x(self, Lbol):
        """
        This method returns the hard X-rayluminosity of a quasar of bolometric luminosity Lbol using equation (5) of S20.

        """
        #Implementation of equation (5).
        x = Lbol/(1e10*L_sun)
        bc = self.c_HX[0]*x**self.k_HX[0] + self.c_HX[1]*x**self.k_HX[1]
        return Lbol/bc

    def L_x_Lfrac(self, Lfrac, Lstar_10):
        """
        This method returns the hard X-rayluminosity of a quasar of bolometric luminosity Lbol using equation (2) of H07.

        """
        #Implementation of equation (5).
        x = Lfrac * Lstar_10
        bc = self.c_HX[0]*x**self.k_HX[0] + self.c_HX[1]*x**self.k_HX[1]
        return Lfrac*Lstar_10*1e10*L_sun/bc

    def xi(self, lam):
        return self.red_model.xi_fit(lam)

    def fNH(self, log_NH_2D, lLfrac, Lstar_10=None, z=None):

        #Get the hard x-ray luminosity for each Lfrac in units of 10^44 erg/s. This will be useful later.
        #Lfrac = 10.**lLfrac
        lLfrac_use = np.where(lLfrac>10.0, 10., lLfrac)
        lLfrac_use = np.where(lLfrac_use<-10.0, -10., lLfrac_use)
        Lfrac = 10**(lLfrac_use)
        Lx = self.L_x_Lfrac(Lfrac, Lstar_10)
        lLx_44 = np.log10(Lx/(u.erg/u.s)).value - 43.75
        lLx_44 = np.where(lLfrac >  10,  np.inf, lLx_44)
        lLx_44 = np.where(lLfrac < -10, -np.inf, lLx_44)

        f_CTK = 1.0
        eps = 1.7
        psi_max = 0.84
        psi_min = 0.2
        if z<2.0:
            psi44 = 0.43*(1+z)**0.48
        else:
            psi44 = 0.43*(1+2)**0.48
        psi = psi44 - 0.24*lLx_44
        psi = np.where(psi<psi_min, psi_min, psi)
        psi = np.where(psi>psi_max, psi_max, psi)

        f_20_21_1 = 1.0 - (2.0+eps)/(1.0+eps) * psi
        f_21_22_1 = 1.0/(1.0+eps) * psi
        f_22_23_1 = 1.0/(1.0+eps) * psi
        f_23_24_1 = eps/(1.0+eps) * psi
        f_24_26_1 = f_CTK/2.0 * psi

        f_20_21_2 = 2.0/3.0 - (3.0+2.0*eps)/(3.0+3.0*eps) * psi
        f_21_22_2 = 1.0/3.0 - eps/(3.0+3.0*eps) * psi
        f_22_23_2 = 1.0/(1.0+eps) * psi
        f_23_24_2 = eps/(1.0+eps) * psi
        f_24_26_2 = f_CTK/2.0 * psi

        psi_lim = (1.0+eps)/(3.0+eps)
        f_20_21 = np.where(psi<psi_lim, f_20_21_1, f_20_21_2)
        f_21_22 = np.where(psi<psi_lim, f_21_22_1, f_21_22_2)
        f_22_23 = np.where(psi<psi_lim, f_22_23_1, f_22_23_2)
        f_23_24 = np.where(psi<psi_lim, f_23_24_1, f_23_24_2)
        f_24_26 = np.where(psi<psi_lim, f_24_26_1, f_24_26_2)

        f_20_21 /= (1.0+f_CTK*psi)
        f_21_22 /= (1.0+f_CTK*psi)
        f_22_23 /= (1.0+f_CTK*psi)
        f_23_24 /= (1.0+f_CTK*psi)
        f_24_26 /= (1.0+f_CTK*psi)

        # f_NH = np.zeros((len(log_NH), len(lLx_44)))
        # f_20_21 = np.tile(f_20_21, [len(log_NH),1])
        # f_21_22 = np.tile(f_21_22, [len(log_NH),1])
        # f_22_23 = np.tile(f_22_23, [len(log_NH),1])
        # f_23_24 = np.tile(f_23_24, [len(log_NH),1])
        # f_24_26 = np.tile(f_24_26, [len(log_NH),1])
        #
        # log_NH_2D = np.tile(log_NH, [len(lLx_44),1]).T
        f_NH = np.zeros(log_NH_2D.shape)
        f_NH = np.where((log_NH_2D>=20.0) & (log_NH_2D<21.0) , f_20_21, f_NH)
        f_NH = np.where((log_NH_2D>=21.0) & (log_NH_2D<22.0) , f_21_22, f_NH)
        f_NH = np.where((log_NH_2D>=22.0) & (log_NH_2D<23.0) , f_22_23, f_NH)
        f_NH = np.where((log_NH_2D>=23.0) & (log_NH_2D<24.0) , f_23_24, f_NH)
        f_NH = np.where((log_NH_2D>=24.0) & (log_NH_2D<=26.0), f_24_26, f_NH)

        return f_NH


    def get_sigma(self, Lfrac, Lstar_10, lam):
        """
        This function calculates the dispersion of the bolometric correction.

        Parameters
        ----------

        Lfrac: numpy array
            Values of L/Lstar for which to calculate Lfrac_lam = L_lam/L_lam(Lstar)

        Lstar_10: float
            Value of Lstar in units of 10^10 Lsun.

        lam: float with astropy.units of wavelength.
            Wavelength at which we want to evaluate the dispersion.

        """
        lLstar_erg_s = np.log10(Lstar_10) + 10 + np.log10(L_sun.to(u.erg/u.s).value)
        lL_erg_s = np.log10(Lfrac) + lLstar_erg_s
        lam_15 = 15.0*u.um
        lam_B  = 4400.*u.AA
        lam_SX = (c/(0.5 * 2.418e17*u.Hz))
        if lam < 4400.*u.AA:
            s1 = self.get_sigma_pre(lL_erg_s, "15")
            s2 = self.get_sigma_pre(lL_erg_s, "BB")
            lam1 = lam_15
            lam2 = lam_B
        else:
            s1 = self.get_sigma_pre(lL_erg_s, "BB")
            s2 = self.get_sigma_pre(lL_erg_s, "SX")
            lam1 = lam_B
            lam2 = lam_SX
        sigma = s1 + (s2-s1)*np.log10((lam1/lam).to(1.).value) / np.log10((lam1/lam2).to(1.).value)
        sigma = np.where(sigma<0.010, 0.010, sigma)
        return sigma


    def get_sigma_pre(self, lL_erg_s, mode):
        """
        This function calculates the dispersion of the bolometric correction in different bands.

        Parameters
        ----------

        lL_erg_s: numpy array
            Values of log L in erg/s

        mode: string
            Can be BB, SX or 15.
        """
        if mode=="BB":
            erf_arg = lL_erg_s - self.logL0_B
            erf_arg /= ((2.)**0.5 * self.sig3_B)
            sig1 = self.sig1_B
            sig2 = self.sig2_B
        elif mode=="SX":
            erf_arg = lL_erg_s - self.logL0_SX
            erf_arg /= ((2.)**0.5 * self.sig3_SX)
            sig1 = self.sig1_SX
            sig2 = self.sig2_SX
        elif mode=="15":
            erf_arg = lL_erg_s - self.logL0_15
            erf_arg /= ((2.)**0.5 * self.sig3_15)
            sig1 = self.sig1_15
            sig2 = self.sig2_15
        return sig2 + sig1*0.5*(1.+erf(erf_arg))


    def dgr(self, z):
        return  self.dgr_local * 10.**(0.35 + 0.93*np.exp(-0.43*z)-1.05)/10.**(0.35+0.93-1.05)

    def _T(self, n, x):
        """
        This method returns the nth order Chebyshev polynomial Tn(x) evaluated at the input value x, with n=1, 2 or 3.

        See eqn. (14) fo S20.
        """
        if n==0:
            return 1.0
        elif n==1:
            return x
        elif n==2:
            return 2*x**2-1
        else:
            raise ValueError("Chebyshev polynomial not implemented for order n={}. Returning 0.".format(n))
            return 0
