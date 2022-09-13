import unittest
import numpy as np
import scipy
from rubin_sim.phot_utils import CosmologyObject
from rubin_sim.phot_utils.utils import comoving_distance_integrand, cosmological_omega


class CosmologyUnitTest(unittest.TestCase):
    def setUp(self):
        self.speed_of_light = 2.9979e5  # in km/sec

    def tearDown(self):
        del self.speed_of_light

    def test_flat_lcdm(self):
        """
        Test the evolution of H and Omega_i as a function of redshift for
        flat Lambda CDM models
        """
        h0 = 50.0
        for om0 in np.arange(start=0.1, stop=0.91, step=0.4):
            universe = CosmologyObject(h0=h0, om0=om0)

            og0 = universe.omega_photons(redshift=0.0)
            onu0 = universe.omega_neutrinos(redshift=0.0)

            self.assertAlmostEqual(universe.omega_matter(redshift=0.0), om0, 10)
            self.assertAlmostEqual(
                1.0 - om0 - universe.omega_dark_energy(redshift=0.0), og0 + onu0, 6
            )
            self.assertAlmostEqual(universe.H(redshift=0.0), h0, 10)
            self.assertEqual(universe.omega_curvature(), 0.0)

            om0 = universe.omega_matter(redshift=0.0)

            for zz in np.arange(start=0.0, stop=4.1, step=2.0):

                (
                    hcontrol,
                    om_control,
                    ode_control,
                    og_control,
                    onu_control,
                    ok_control,
                ) = cosmological_omega(zz, h0, om0, og0=og0, onu0=onu0)

                self.assertAlmostEqual(om_control, universe.omega_matter(redshift=zz), 6)
                self.assertAlmostEqual(
                    ode_control, universe.omega_dark_energy(redshift=zz), 6
                )
                self.assertAlmostEqual(
                    og_control, universe.omega_photons(redshift=zz), 6
                )
                self.assertAlmostEqual(
                    onu_control, universe.omega_neutrinos(redshift=zz), 6
                )
                self.assertAlmostEqual(hcontrol, universe.H(redshift=zz), 6)

            del universe

    def test_flat_w0_wa(self):
        """
        Test the evolution of H and Omega_i as a function of redshift for
        flat models with w = w0 + wa * z / (1 + z)
        """

        h0 = 96.0
        for om0 in np.arange(start=0.1, stop=0.95, step=0.4):
            for w0 in np.arange(start=-1.1, stop=-0.89, step=0.2):
                for wa in np.arange(start=-0.1, stop=0.11, step=0.2):

                    universe = CosmologyObject(h0=h0, om0=om0, w0=w0, wa=wa)

                    og0 = universe.omega_photons(redshift=0.0)
                    onu0 = universe.omega_neutrinos(redshift=0.0)

                    self.assertAlmostEqual(universe.omega_matter(redshift=0.0), om0, 10)
                    self.assertAlmostEqual(
                        1.0 - om0 - universe.omega_dark_energy(redshift=0.0),
                        og0 + onu0,
                        6,
                    )
                    self.assertAlmostEqual(universe.H(redshift=0.0), h0, 10)
                    self.assertEqual(universe.omega_curvature(), 0.0)

                    om0 = universe.omega_matter(redshift=0.0)

                    for zz in np.arange(start=0.0, stop=4.1, step=2.0):

                        w_control = w0 + wa * (1.0 - 1.0 / (1.0 + zz))
                        self.assertAlmostEqual(w_control, universe.w(redshift=zz), 6)

                        (
                            hcontrol,
                            om_control,
                            ode_control,
                            og_control,
                            onu_control,
                            ok_control,
                        ) = cosmological_omega(
                            zz, h0, om0, og0=og0, onu0=onu0, w0=w0, wa=wa
                        )

                        self.assertAlmostEqual(
                            om_control, universe.omega_matter(redshift=zz), 6
                        )
                        self.assertAlmostEqual(
                            ode_control, universe.omega_dark_energy(redshift=zz), 6
                        )
                        self.assertAlmostEqual(
                            og_control, universe.omega_photons(redshift=zz), 6
                        )
                        self.assertAlmostEqual(
                            onu_control, universe.omega_neutrinos(redshift=zz), 6
                        )
                        self.assertAlmostEqual(hcontrol, universe.H(redshift=zz), 6)

                    del universe

    def test_flat_w0(self):
        """
        Test the evolution of H and Omega_i as a function of redshift for flat
        models with constant w
        """

        h0 = 96.0
        for om0 in np.arange(start=0.1, stop=0.95, step=0.4):
            for w0 in np.arange(start=-1.5, stop=-0.49, step=1.0):

                universe = CosmologyObject(h0=h0, om0=om0, w0=w0)

                og0 = universe.omega_photons(redshift=0.0)
                onu0 = universe.omega_neutrinos(redshift=0.0)

                self.assertAlmostEqual(universe.omega_matter(redshift=0.0), om0, 10)
                self.assertAlmostEqual(
                    1.0 - om0 - universe.omega_dark_energy(redshift=0.0), og0 + onu0, 6
                )
                self.assertAlmostEqual(universe.H(redshift=0.0), h0, 10)
                self.assertEqual(universe.omega_curvature(), 0.0)

                om0 = universe.omega_matter(redshift=0.0)

                for zz in np.arange(start=0.0, stop=4.1, step=2.0):

                    self.assertAlmostEqual(w0, universe.w(redshift=zz), 6)

                    (
                        hcontrol,
                        om_control,
                        ode_control,
                        og_control,
                        onu_control,
                        ok_control,
                    ) = cosmological_omega(
                        zz, h0, om0, og0=og0, onu0=onu0, w0=w0, wa=0.0
                    )

                    self.assertAlmostEqual(
                        om_control, universe.omega_matter(redshift=zz), 6
                    )
                    self.assertAlmostEqual(
                        ode_control, universe.omega_dark_energy(redshift=zz), 6
                    )
                    self.assertAlmostEqual(
                        og_control, universe.omega_photons(redshift=zz), 6
                    )
                    self.assertAlmostEqual(
                        onu_control, universe.omega_neutrinos(redshift=zz), 6
                    )
                    self.assertAlmostEqual(hcontrol, universe.H(redshift=zz), 6)

                del universe

    def test_non_flat_lcdm(self):
        """
        Test the evolution of H and Omega_i as a function of redshift for non-flat
        Lambda CDM models
        """
        w0 = -1.0
        wa = 0.0
        h0 = 77.0

        for om0 in np.arange(start=0.15, stop=0.96, step=0.4):
            for ok0 in np.arange(start=-0.1, stop=0.11, step=0.2):

                universe = CosmologyObject(h0=h0, om0=om0, ok0=ok0, w0=w0, wa=wa)

                og0 = universe.omega_photons(redshift=0.0)
                onu0 = universe.omega_neutrinos(redshift=0.0)

                self.assertAlmostEqual(universe.omega_matter(redshift=0.0), om0, 10)
                self.assertAlmostEqual(universe.omega_curvature(redshift=0.0), ok0, 10)
                self.assertAlmostEqual(
                    1.0 - ok0 - om0 - universe.omega_dark_energy(redshift=0.0),
                    og0 + onu0,
                    6,
                )
                self.assertAlmostEqual(universe.H(redshift=0.0), h0, 10)

                om0 = universe.omega_matter(redshift=0.0)
                ode0 = universe.omega_dark_energy(redshift=0.0)
                ok0 = universe.omega_curvature(redshift=0.0)

                for zz in np.arange(start=0.0, stop=4.0, step=2.0):

                    (
                        hcontrol,
                        om_control,
                        ode_control,
                        og_control,
                        onu_control,
                        ok_control,
                    ) = cosmological_omega(zz, h0, om0, og0=og0, onu0=onu0, ode0=ode0)

                    self.assertAlmostEqual(
                        om_control, universe.omega_matter(redshift=zz), 6
                    )
                    self.assertAlmostEqual(
                        ode_control, universe.omega_dark_energy(redshift=zz), 6
                    )
                    self.assertAlmostEqual(
                        og_control, universe.omega_photons(redshift=zz), 6
                    )
                    self.assertAlmostEqual(
                        onu_control, universe.omega_neutrinos(redshift=zz), 6
                    )
                    self.assertAlmostEqual(
                        ok_control, universe.omega_curvature(redshift=zz), 6
                    )
                    self.assertAlmostEqual(hcontrol, universe.H(redshift=zz), 6)

                del universe

    def test_non_flat_w0_wa(self):
        """
        Test the evolution of H and Omega_i as a function of redshift for non-flat
        models with w = w0 + wa * z / (1+z)
        """

        h0 = 60.0

        for om0 in np.arange(start=0.15, stop=0.76, step=0.3):
            for ok0 in np.arange(start=-0.1, stop=0.11, step=0.2):
                for w0 in np.arange(start=-1.1, stop=-0.89, step=0.2):
                    for wa in np.arange(start=-0.1, stop=0.15, step=0.2):

                        universe = CosmologyObject(
                            h0=h0, om0=om0, ok0=ok0, w0=w0, wa=wa
                        )

                        og0 = universe.omega_photons(redshift=0.0)
                        onu0 = universe.omega_neutrinos(redshift=0.0)

                        self.assertAlmostEqual(
                            universe.omega_matter(redshift=0.0), om0, 10
                        )
                        self.assertAlmostEqual(
                            ok0, universe.omega_curvature(redshift=0.0), 10
                        )
                        self.assertAlmostEqual(
                            1.0 - om0 - ok0 - universe.omega_dark_energy(redshift=0.0),
                            og0 + onu0,
                            10,
                        )
                        self.assertAlmostEqual(universe.H(redshift=0.0), h0, 10)

                        om0 = universe.omega_matter(redshift=0.0)
                        ode0 = universe.omega_dark_energy(redshift=0.0)

                        for zz in np.arange(start=0.0, stop=4.0, step=2.0):

                            w_control = w0 + wa * (1.0 - 1.0 / (1.0 + zz))
                            self.assertAlmostEqual(
                                w_control, universe.w(redshift=zz), 6
                            )

                            (
                                hcontrol,
                                om_control,
                                ode_control,
                                og_control,
                                onu_control,
                                ok_control,
                            ) = cosmological_omega(
                                zz, h0, om0, og0=og0, onu0=onu0, w0=w0, wa=wa, ode0=ode0
                            )

                            self.assertAlmostEqual(
                                om_control, universe.omega_matter(redshift=zz), 6
                            )
                            self.assertAlmostEqual(
                                ode_control, universe.omega_dark_energy(redshift=zz), 6
                            )
                            self.assertAlmostEqual(
                                og_control, universe.omega_photons(redshift=zz), 6
                            )
                            self.assertAlmostEqual(
                                onu_control, universe.omega_neutrinos(redshift=zz), 6
                            )
                            self.assertAlmostEqual(
                                ok_control, universe.omega_curvature(redshift=zz), 6
                            )
                            self.assertAlmostEqual(hcontrol, universe.H(redshift=zz), 6)

                        del universe

    def test_non_flat_w0(self):
        """
        Test the evolution of H and Omega_i as a function of redshift for non-flat
        models with constant w
        """

        h0 = 60.0

        for om0 in np.arange(start=0.15, stop=0.76, step=0.3):
            for ok0 in np.arange(start=0.1, stop=0.11, step=0.2):
                for w0 in np.arange(start=-1.1, stop=-0.89, step=0.2):

                    universe = CosmologyObject(h0=h0, om0=om0, ok0=ok0, w0=w0)

                    og0 = universe.omega_photons(redshift=0.0)
                    onu0 = universe.omega_neutrinos(redshift=0.0)

                    self.assertAlmostEqual(universe.omega_matter(redshift=0.0), om0, 10)
                    self.assertAlmostEqual(
                        ok0, universe.omega_curvature(redshift=0.0), 10
                    )
                    self.assertAlmostEqual(
                        1.0 - om0 - ok0 - universe.omega_dark_energy(redshift=0.0),
                        og0 + onu0,
                        10,
                    )

                    self.assertAlmostEqual(universe.H(redshift=0.0), h0, 10)

                    om0 = universe.omega_matter(redshift=0.0)
                    ode0 = universe.omega_dark_energy(redshift=0.0)

                    for zz in np.arange(start=0.0, stop=4.0, step=2.0):

                        self.assertAlmostEqual(w0, universe.w(redshift=zz), 6)

                        (
                            hcontrol,
                            om_control,
                            ode_control,
                            og_control,
                            onu_control,
                            ok_control,
                        ) = cosmological_omega(
                            zz, h0, om0, og0=og0, onu0=onu0, w0=w0, wa=0.0, ode0=ode0
                        )

                        self.assertAlmostEqual(
                            om_control, universe.omega_matter(redshift=zz), 6
                        )
                        self.assertAlmostEqual(
                            ode_control, universe.omega_dark_energy(redshift=zz), 6
                        )
                        self.assertAlmostEqual(
                            og_control, universe.omega_photons(redshift=zz), 6
                        )
                        self.assertAlmostEqual(
                            onu_control, universe.omega_neutrinos(redshift=zz), 6
                        )
                        self.assertAlmostEqual(
                            ok_control, universe.omega_curvature(redshift=zz), 6
                        )
                        self.assertAlmostEqual(hcontrol, universe.H(redshift=zz), 6)

                    del universe

    def test_comoving_distance(self):
        """
        Test comoving distance calculation

        Note: this is comoving distance defined as X in the FRW metric

        ds^2 = -c^2 dt^2 + a^2 dX^2 + sin^2(X) dOmega^2

        where spatial curvature is accounted for in the sin function
        """

        h0 = 73.0
        for om0 in np.arange(start=0.15, stop=0.56, step=0.2):
            for ok0 in np.arange(start=-0.1, stop=0.11, step=0.2):
                for w0 in np.arange(start=-1.1, stop=-0.85, step=0.2):
                    for wa in np.arange(start=-0.1, stop=0.115, step=0.02):

                        universe = CosmologyObject(
                            h0=h0, om0=om0, ok0=ok0, w0=w0, wa=wa
                        )
                        og0 = universe.omega_photons()
                        onu0 = universe.omega_neutrinos()
                        ode0 = universe.omega_dark_energy()

                        for zz in np.arange(start=0.1, stop=4.2, step=2.0):
                            comoving_control = universe.comoving_distance(redshift=zz)

                            comoving_test = (
                                self.speed_of_light
                                * scipy.integrate.quad(
                                    comoving_distance_integrand,
                                    0.0,
                                    zz,
                                    args=(
                                        h0,
                                        om0,
                                        ode0,
                                        og0,
                                        onu0,
                                        w0,
                                        wa,
                                    ),
                                )[0]
                            )

                            self.assertAlmostEqual(
                                comoving_control / comoving_test, 1.0, 4
                            )

    def test_luminosity_distance(self):
        """
        Test the calculation of the luminosity distance
        """

        h0 = 73.0

        for om0 in np.arange(start=0.15, stop=0.56, step=0.2):
            for ok0 in np.arange(start=-0.1, stop=0.11, step=0.2):
                for w0 in np.arange(start=-1.1, stop=-0.85, step=0.2):
                    for wa in np.arange(start=-0.1, stop=0.11, step=0.2):

                        universe = CosmologyObject(
                            h0=h0, om0=om0, ok0=ok0, w0=w0, wa=wa
                        )

                        sqrtk_curvature = (
                            np.sqrt(np.abs(universe.omega_curvature()))
                            * universe.H()
                            / self.speed_of_light
                        )

                        og0 = universe.omega_photons()
                        onu0 = universe.omega_neutrinos()
                        ode0 = universe.omega_dark_energy()

                        for zz in np.arange(start=0.1, stop=4.2, step=2.0):
                            luminosity_control = universe.luminosity_distance(
                                redshift=zz
                            )

                            comoving_distance = (
                                self.speed_of_light
                                * scipy.integrate.quad(
                                    comoving_distance_integrand,
                                    0.0,
                                    zz,
                                    args=(
                                        h0,
                                        om0,
                                        ode0,
                                        og0,
                                        onu0,
                                        w0,
                                        wa,
                                    ),
                                )[0]
                            )

                            if universe.omega_curvature() < 0.0:
                                nn = sqrtk_curvature * comoving_distance
                                nn = np.sin(nn)
                                luminosity_test = (1.0 + zz) * nn / sqrtk_curvature
                            elif universe.omega_curvature() > 0.0:
                                nn = sqrtk_curvature * comoving_distance
                                nn = np.sinh(nn)
                                luminosity_test = (1.0 + zz) * nn / sqrtk_curvature
                            else:
                                luminosity_test = (1.0 + zz) * comoving_distance
                            self.assertAlmostEqual(
                                luminosity_control / luminosity_test, 1.0, 4
                            )

    def test_angular_diameter_distance(self):
        """
        Test the calculation of the angular diameter distance
        """

        h0 = 56.0
        universe = CosmologyObject()
        for om0 in np.arange(start=0.15, stop=0.56, step=0.2):
            for ok0 in np.arange(start=-0.1, stop=0.11, step=0.2):
                for w0 in np.arange(start=-1.1, stop=-0.85, step=0.2):
                    for wa in np.arange(start=-0.1, stop=0.11, step=0.2):

                        universe = CosmologyObject(
                            h0=h0, om0=om0, ok0=ok0, w0=w0, wa=wa
                        )

                        sqrtk_curvature = (
                            np.sqrt(np.abs(universe.omega_curvature()))
                            * universe.H()
                            / self.speed_of_light
                        )

                        og0 = universe.omega_photons()
                        onu0 = universe.omega_neutrinos()
                        ode0 = universe.omega_dark_energy()

                        for zz in np.arange(start=0.1, stop=4.2, step=2.0):
                            angular_control = universe.angular_diameter_distance(
                                redshift=zz
                            )

                            comoving_distance = (
                                self.speed_of_light
                                * scipy.integrate.quad(
                                    comoving_distance_integrand,
                                    0.0,
                                    zz,
                                    args=(
                                        h0,
                                        om0,
                                        ode0,
                                        og0,
                                        onu0,
                                        w0,
                                        wa,
                                    ),
                                )[0]
                            )

                            if universe.omega_curvature() < 0.0:
                                nn = sqrtk_curvature * comoving_distance
                                nn = np.sin(nn)
                                angular_test = nn / sqrtk_curvature
                            elif universe.omega_curvature() > 0.0:
                                nn = sqrtk_curvature * comoving_distance
                                nn = np.sinh(nn)
                                angular_test = nn / sqrtk_curvature
                            else:
                                angular_test = comoving_distance
                            angular_test /= 1.0 + zz
                            self.assertAlmostEqual(
                                angular_control / angular_test, 1.0, 4
                            )

    def test_distance_modulus(self):
        """
        Test the calculation of the distance modulus out to a certain redshift
        """
        h0 = 73.0

        universe = CosmologyObject()
        for om0 in np.arange(start=0.15, stop=0.56, step=0.2):
            for ok0 in np.arange(start=-0.1, stop=0.11, step=0.2):
                for w0 in np.arange(start=-1.1, stop=-0.85, step=0.2):
                    for wa in np.arange(start=-0.1, stop=0.11, step=0.2):

                        universe = CosmologyObject(
                            h0=h0, om0=om0, ok0=ok0, w0=w0, wa=wa
                        )

                        sqrtk_curvature = (
                            np.sqrt(np.abs(universe.omega_curvature()))
                            * universe.H()
                            / self.speed_of_light
                        )

                        og0 = universe.omega_photons()
                        onu0 = universe.omega_neutrinos()
                        ode0 = universe.omega_dark_energy()

                        for zz in np.arange(start=0.1, stop=4.2, step=2.0):
                            modulus_control = universe.distance_modulus(redshift=zz)

                            comoving_distance = (
                                self.speed_of_light
                                * scipy.integrate.quad(
                                    comoving_distance_integrand,
                                    0.0,
                                    zz,
                                    args=(
                                        h0,
                                        om0,
                                        ode0,
                                        og0,
                                        onu0,
                                        w0,
                                        wa,
                                    ),
                                )[0]
                            )

                            if universe.omega_curvature() < 0.0:
                                nn = sqrtk_curvature * comoving_distance
                                nn = np.sin(nn)
                                luminosity_distance = (1.0 + zz) * nn / sqrtk_curvature
                            elif universe.omega_curvature() > 0.0:
                                nn = sqrtk_curvature * comoving_distance
                                nn = np.sinh(nn)
                                luminosity_distance = (1.0 + zz) * nn / sqrtk_curvature
                            else:
                                luminosity_distance = (1.0 + zz) * comoving_distance

                            modulus_test = 5.0 * np.log10(luminosity_distance) + 25.0
                            self.assertAlmostEqual(
                                modulus_control / modulus_test, 1.0, 4
                            )

    def test_distance_modulus_at_zero(self):
        """
        Test to make sure that the distance modulus is set to zero if the distance modulus method
        returns a negative number
        """
        universe = CosmologyObject()
        ztest = [0.0, 1.0, 2.0, 0.0, 3.0]
        mm = universe.distance_modulus(redshift=ztest)
        self.assertEqual(mm[0], 0.0)
        self.assertEqual(mm[3], 0.0)
        self.assertEqual(
            mm[1], 5.0 * np.log10(universe.luminosity_distance(ztest[1])) + 25.0
        )
        self.assertEqual(
            mm[2], 5.0 * np.log10(universe.luminosity_distance(ztest[2])) + 25.0
        )
        self.assertEqual(
            mm[4], 5.0 * np.log10(universe.luminosity_distance(ztest[4])) + 25.0
        )

    def test_get_current(self):
        """
        Test to make sure that get_current returns the activeCosmology
        """

        for om0 in np.arange(start=0.2, stop=0.5, step=0.29):
            for ok0 in np.arange(start=-0.2, stop=0.2, step=0.39):
                for w0 in np.arange(start=-1.2, stop=-0.7, step=0.49):
                    for wa in np.arange(start=-0.2, stop=0.2, step=0.39):
                        universe = CosmologyObject(om0=om0, ok0=ok0, w0=w0, wa=wa)
                        test_universe = universe.get_current()

                        for zz in np.arange(start=1.0, stop=2.1, step=1.0):
                            self.assertEqual(
                                universe.omega_matter(redshift=zz), test_universe.Om(zz)
                            )
                            self.assertEqual(
                                universe.omega_dark_energy(redshift=zz),
                                test_universe.Ode(zz),
                            )
                            self.assertEqual(
                                universe.omega_photons(redshift=zz),
                                test_universe.Ogamma(zz),
                            )
                            self.assertEqual(
                                universe.omega_neutrinos(redshift=zz),
                                test_universe.Onu(zz),
                            )
                            self.assertEqual(
                                universe.omega_curvature(redshift=zz),
                                test_universe.Ok(zz),
                            )


if __name__ == "__main__":
    unittest.main()
