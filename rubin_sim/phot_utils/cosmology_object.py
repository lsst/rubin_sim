"""
31 October 2014

The class CosmologyObject provides an interface for the methods in astropy.cosmology that
we anticipate using the most.

The methods in astropy.cosmology are accessed by instantiating a cosmology object and calling
methods that belong to that object.  CosmologyObject interfaces with this by declaring a member
variable self.activeCosmology.  Methods provided by CosmologyObject call the equivalent
astropy.cosmology methods on self.activeCosmology.  activeCosmology is set by calling
CosmologyObject.initializeCosmology(args...) with the appropriate cosmological Parameters.
Passing in no parametrs loads the Millennium Simulation cosmology (Springel et al 2005, Nature 435, 629
or arXiv:astro-ph/0504097).

The difficulty with all of this that, between the version of astropy shipped with anaconda (v0.2.5) and
the most modern version (v0.4), the API for astropy.cosmology has changed in two ways.

One difference is that methods like comoving_distance have gone from returning floats to returning
astropy.Quantity's which come with both a value and units.  To deal with this, CosmologyObject
checks dir(cosmology.comoving_distance()) etc.  If 'units' is defined, CosmologyObject sets
member variables such as self.distanceUnits, self.hUnits, and self.modulusUnits defining the units
in which we want to return those quantities.  When you call the wrapper for comoving_distance,
CosmologyObject will make sure that the output is returned in the units we expect (Mpc).
The expected units are set in CosmologyObject.set_units()

The other API difference is in how 'default_cosmology' is stored.  astropy.cosmology allows
the user to set a default cosmology that the system stores so that the user does not have to
constantly redeclare the same cosmology object at different points in the code.  Unfortunately,
the naming conventions for the methods to set and retrieve this default cosmology have changed
between recent versions of astropy.  CosmologyObject deals with this change in API using
CosmologyObject.set_current() (called automatically by CosmologyObject's __init__)
and CosmologyObject.get_current(), which returns a cosmology object containing the activeCosmology
contained in CosmologyObject.

A user who wants to interact with the naked
astropy.cosmology methods can run something like

uu = CosmologyObject() #which sets activeCosmology to the Millennium Simulation cosmology
myUniverse = uu.getCurrent()

myUniverse now contains a cosmology object which is equivalent to the activeCosmology.  Direct
calls to the astropy.cosmology methods of the form

dd = myUniverse.comoving_distance(1.0) #comoving distance to redshift z=1

will now work.


The methods in CosmologyObject have been tested on astropy v0.2.5 and v0.4.2
"""
import numpy
import astropy.cosmology as cosmology
import astropy.units as units

flatnessthresh = 1.0e-12

__all__ = ["CosmologyObject"]


class CosmologyObject(object):
    def __init__(self, h0=73.0, om0=0.25, ok0=None, w0=None, wa=None):
        """
        Initialize the cosmology wrapper with the parameters specified
        (e.g. does not account for massive neutrinos)

        param [in] h0 is the Hubble parameter at the present epoch in km/s/Mpc

        param [in] om0 is the current matter density Parameter (fraction of critical density)

        param [in] ok0 is the current curvature density parameter

        param [in] w0 is the current dark energy equation of state w0 Parameter

        param[in] wa is the current dark energy equation of state wa Parameter

        The total dark energy equation of state as a function of z is
        w = w0 + wa z/(1+z)

        Currently, this wrapper class expects you to specify either a LambdaCDM (flat or non-flat) cosmology
        or a w0, wa (flat or non-flat) cosmology.

        The default cosmology is taken as the cosmology used
        in the Millennium Simulation (Springel et al 2005, Nature 435, 629 or
        arXiv:astro-ph/0504097)

        om0 = 0.25
        Ob0  = 0.045 (baryons; not currently used in this code)
        h0 = 73.0
        ok0 = 0.0, (implying ode0 approx 0.75)
        w0 = -1.0
        wa = 0.0

        where
        om0 + ok0 + ode0 + Ogamma0 + onu0 = 1.0

        sigma_8 = 0.9 (rms mass flucutation in an 8 h^-1 Mpc sphere;
                       not currently used in this code)

        ns = 1 (index of the initial spectrum of linear mas perturbations;
                not currently used in this code)

        """

        self.active_cosmology = None

        if w0 is not None and wa is None:
            wa = 0.0

        is_cosmological_constant = False
        if (w0 is None and wa is None) or (w0 == -1.0 and wa == 0.0):
            is_cosmological_constant = True

        is_flat = False
        if ok0 is None or (numpy.abs(ok0) < flatnessthresh):
            is_flat = True

        if is_cosmological_constant and is_flat:
            universe = cosmology.FlatLambdaCDM(H0=h0, Om0=om0)
        elif is_cosmological_constant:
            tmpmodel = cosmology.FlatLambdaCDM(H0=h0, Om0=om0)
            ode0 = 1.0 - om0 - tmpmodel.Ogamma0 - tmpmodel.Onu0 - ok0
            universe = cosmology.LambdaCDM(H0=h0, Om0=om0, Ode0=ode0)
        elif is_flat:
            universe = cosmology.Flatw0waCDM(H0=h0, Om0=om0, w0=w0, wa=wa)
        else:
            tmpmodel = cosmology.Flatw0waCDM(H0=h0, Om0=om0, w0=w0, wa=wa)
            ode0 = 1.0 - om0 - tmpmodel.Ogamma0 - tmpmodel.Onu0 - ok0

            universe = cosmology.w0waCDM(H0=h0, Om0=om0, Ode0=ode0, w0=w0, wa=wa)

        self.set_current(universe)

    def set_current(self, universe):
        """
        Take the cosmology indicated by 'universe' and set it as the current/default
        cosmology (depending on the API of the version of astropy being run)

        universe is also assigned to self.active_cosmology, which is the cosmology that
        this wrapper's methods use for calculations.
        """

        if "default_cosmology" in dir(cosmology):
            cosmology.default_cosmology.set(universe)
        elif "set_current" in dir(cosmology):
            cosmology.set_current(universe)
        else:
            raise RuntimeError(
                "CosmologyObject.set_current does not know how to handle this version of astropy"
            )

        self.active_cosmology = universe
        self.set_units()

    def set_units(self):
        """
        This method specifies the units in which various outputs from the wrapper are expected
        (this is because the latest version of astropy.cosmology outputs quantities such as
        the Hubble parameter and luminosity distance with units attached; the version of
        astropy.cosmology that comes within anaconda does not do this as of 30 October 2014)
        """

        H = self.active_cosmology.H(0.0)
        if "unit" in dir(H):
            self.h_units = units.Unit("km / (Mpc s)")
        else:
            self.h_units = None

        dd = self.active_cosmology.comoving_distance(0.0)
        if "unit" in dir(dd):
            self.distance_units = units.Mpc
        else:
            self.distance_units = None

        mm = self.active_cosmology.distmod(1.0)
        if "unit" in dir(mm):
            self.modulus_units = units.mag
        else:
            self.modulus_units = None

    def get_current(self):
        """
        Return the cosmology currently stored as the current cosmology

        This is for users who want direct access to all of astropy.cosmology's methods,
        not just those wrapped by this class.

        documentation for astropy.cosmology can be found at the URL below (be sure to check which version of
        astropy you are running; as of 30 October 2014, the anaconda distributed with the stack
        comes with version 0.2.5)

        https://astropy.readthedocs.org/en/v0.2.5/cosmology/index.html
        """

        return self.active_cosmology

    def H(self, redshift=0.0):
        """
        return the Hubble Parameter in km/s/Mpc at the specified redshift

        effectively wrapps astropy.cosmology.FLRW.H()
        """

        H = self.active_cosmology.H(redshift)

        if "value" in dir(H):
            if H.unit == self.h_units:
                return H.value
            else:
                return H.to(self.h_units).value
        else:
            return H

    def omega_matter(self, redshift=0.0):
        """
        return the matter density Parameter (fraction of critical density) at the specified redshift

        effectively wraps astropy.cosmology.FLRW.Om()
        """

        return self.active_cosmology.Om(redshift)

    def omega_dark_energy(self, redshift=0.0):
        """
        return the dark energy density Parameter (fraction of critical density) at the specified redshift

        effectively wraps astropy.cosmology.FLRW.Ode()
        """

        return self.active_cosmology.Ode(redshift)

    def omega_photons(self, redshift=0.0):
        """
        return the photon density Parameter (fraction of critical density) at the specified redshift

        effectively wraps astropy.cosmology.FLRW.Ogamma()
        """

        return self.active_cosmology.Ogamma(redshift)

    def omega_neutrinos(self, redshift=0.0):
        """
        return the neutrino density Parameter (fraction of critical density) at the specified redshift

        assumes neutrinos are massless

        effectively wraps astropy.cosmology.FLRW.Onu()
        """

        return self.active_cosmology.Onu(redshift)

    def omega_curvature(self, redshift=0.0):
        """
        return the effective curvature density Parameter (fraction of critical density) at the
        specified redshift.

        Positive means the universe is open.

        Negative means teh universe is closed.

        Zero means the universe is flat.

        effectively wraps astropy.cosmology.FLRW.Ok()
        """

        return self.active_cosmology.Ok(redshift)

    def w(self, redshift=0.0):
        """
        return the dark energy equation of state at the specified redshift

        effecitvely wraps astropy.cosmology.FLRW.w()
        """

        return self.active_cosmology.w(redshift)

    def comoving_distance(self, redshift=0.0):
        """
        return the comoving distance to the specified redshift in Mpc

        note, this comoving distance is X in the FRW metric

        ds^2 = -c^2 dt^2 + a^2 dX^2 + a^2 sin^2(X) dOmega^2

        i.e. the curvature of the universe is folded into the sin()/sinh() function.
        This distande just integrates dX = c dt/a

        effectively wraps astropy.cosmology.FLRW.comoving_distance()
        """
        dd = self.active_cosmology.comoving_distance(redshift)

        if "value" in dir(dd):
            if dd.unit == self.distance_units:
                return dd.value
            else:
                return dd.to(self.distance_units).value
        else:
            return dd

    def luminosity_distance(self, redshift=0.0):
        """
        the luminosity distance to the specified redshift in Mpc

        accounts for spatial curvature

        effectively wraps astropy.cosmology.FLRW.luminosity_distance()
        """

        dd = self.active_cosmology.luminosity_distance(redshift)

        if "value" in dir(dd):
            if dd.unit == self.distance_units:
                return dd.value
            else:
                return dd.to(self.distance_units).value
        else:
            return dd

    def angular_diameter_distance(self, redshift=0.0):
        """
        angular diameter distance to the specified redshift in Mpc

        effectively wraps astropy.cosmology.FLRW.angular_diameter_distance()
        """

        dd = self.active_cosmology.angular_diameter_distance(redshift)

        if "value" in dir(dd):
            if dd.unit == self.distance_units:
                return dd.value
            else:
                return dd.to(self.distance_units).value
        else:
            return dd

    def distance_modulus(self, redshift=0.0):
        """
        distance modulus to the specified redshift

        effectively wraps astropy.cosmology.FLRW.distmod()
        """

        mm = self.active_cosmology.distmod(redshift)
        if "unit" in dir(mm):
            if mm.unit == self.modulus_units:
                mod = mm.value
            else:
                mod = mm.to(self.modulus_units).value
        else:
            mod = mm

        # The astropy.cosmology.distmod() method has no problem returning a negative
        # distance modulus (or -inf if redshift==0.0)
        # Given that this makes no sense, the code below forces all distance moduli
        # to be greater than zero.
        #
        # a Runtime Warning will be raised (because distmod will try to take the
        # logarithm of luminosityDistance = 0, but the code will still run
        if isinstance(mod, float):
            if mod < 0.0:
                return 0.0
            else:
                return mod
        else:
            return numpy.where(mod > 0.0, mod, 0.0)
