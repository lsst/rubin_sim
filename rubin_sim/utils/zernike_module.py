__all__ = ("_FactorialGenerator", "ZernikePolynomialGenerator")

import numbers

import numpy as np


class _FactorialGenerator:
    """
    A class that generates factorials
    and stores them in a dict to be referenced
    as needed.
    """

    def __init__(self):
        self._values = {0: 1, 1: 1}
        self._max_i = 1

    def evaluate(self, num):
        """
        Return the factorial of num
        """
        if num < 0:
            raise RuntimeError("Cannot handle negative factorial")

        i_num = int(np.round(num))
        if i_num in self._values:
            return self._values[i_num]

        val = self._values[self._max_i]
        for ii in range(self._max_i, num):
            val *= ii + 1
            self._values[ii + 1] = val

        self._max_i = num
        return self._values[num]


class ZernikePolynomialGenerator:
    """
    A class to generate and evaluate the Zernike
    polynomials.  Definitions of Zernike polynomials
    are taken from
    https://en.wikipedia.org/wiki/Zernike_polynomials
    """

    def __init__(self):
        self._factorial = _FactorialGenerator()
        self._coeffs = {}
        self._powers = {}

    def _validate_nm(self, n, m):
        """
        Make sure that n, m are a valid pair of indices for
        a Zernike polynomial.

        n is the radial order

        m is the angular order
        """
        if not isinstance(n, int) and not isinstance(n, np.int64):
            raise RuntimeError("Zernike polynomial n must be int")
        if not isinstance(m, int) and not isinstance(m, np.int64):
            raise RuntimeError("Zernike polynomial m must be int")

        if n < 0:
            raise RuntimeError("Radial Zernike n cannot be negative")
        if m < 0:
            raise RuntimeError("Radial Zernike m cannot be negative")
        if n < m:
            raise RuntimeError("Radial Zerniki n must be >= m")

        n = int(n)
        m = int(m)

        return (n, m)

    def _make_polynomial(self, n, m):
        """
        Make the radial part of the n, m Zernike
        polynomial.

        n is the radial order

        m is the angular order

        Returns 2 numpy arrays: coeffs and powers.

        The radial part of the Zernike polynomial is

        sum([coeffs[ii]*power(r, powers[ii])
             for ii in range(len(coeffs))])
        """

        n, m = self._validate_nm(n, m)

        # coefficients taken from
        # https://en.wikipedia.org/wiki/Zernike_polynomials

        n_coeffs = 1 + (n - m) // 2
        local_coeffs = np.zeros(n_coeffs, dtype=float)
        local_powers = np.zeros(n_coeffs, dtype=float)
        for k in range(0, n_coeffs):
            if k % 2 == 0:
                sgn = 1.0
            else:
                sgn = -1.0

            num_fac = self._factorial.evaluate(n - k)
            k_fac = self._factorial.evaluate(k)
            d1_fac = self._factorial.evaluate(((n + m) // 2) - k)
            d2_fac = self._factorial.evaluate(((n - m) // 2) - k)

            local_coeffs[k] = sgn * num_fac / (k_fac * d1_fac * d2_fac)
            local_powers[k] = n - 2 * k

        self._coeffs[(n, m)] = local_coeffs
        self._powers[(n, m)] = local_powers

    def _evaluate_radial_number(self, r, nm_tuple):
        """
        Evaluate the radial part of a Zernike polynomial.

        r is a scalar value

        nm_tuple is a tuple of the form (radial order, angular order)
        denoting the polynomial to evaluate

        Return the value of the radial part of the polynomial at r
        """
        if r > 1.0:
            return np.NaN

        r_term = np.power(r, self._powers[nm_tuple])
        return (self._coeffs[nm_tuple] * r_term).sum()

    def _evaluate_radial_array(self, r, nm_tuple):
        """
        Evaluate the radial part of a Zernike polynomial.

        r is a numpy array of radial values

        nm_tuple is a tuple of the form (radial order, angular order)
        denoting the polynomial to evaluate

        Return the values of the radial part of the polynomial at r
        (returns np.NaN if r>1.0)
        """
        if len(r) == 0:
            return np.array([], dtype=float)

        # since we use np.where to handle cases of
        # r==0, use np.errstate to temporarily
        # turn off the divide by zero and
        # invalid double scalar RuntimeWarnings
        with np.errstate(divide="ignore", invalid="ignore"):
            log_r = np.log(r)
            log_r = np.where(np.isfinite(log_r), log_r, -1.0e10)
            r_power = np.exp(np.outer(log_r, self._powers[nm_tuple]))

            results = np.dot(r_power, self._coeffs[nm_tuple])
            return np.where(r < 1.0, results, np.NaN)

    def _evaluate_radial(self, r, n, m):
        """
        Evaluate the radial part of a Zernike polynomial

        r is a radial value or an array of radial values

        n is the radial order of the polynomial

        m is the angular order of the polynomial

        Return the value(s) of the radial part of the polynomial at r
        (returns np.NaN if r>1.0)
        """

        is_array = False
        if not isinstance(r, numbers.Number):
            is_array = True

        nm_tuple = self._validate_nm(n, m)

        if (nm_tuple[0] - nm_tuple[1]) % 2 == 1:
            if is_array:
                return np.zeros(len(r), dtype=float)
            return 0.0

        if nm_tuple not in self._coeffs:
            self._make_polynomial(nm_tuple[0], nm_tuple[1])

        if is_array:
            return self._evaluate_radial_array(r, nm_tuple)

        return self._evaluate_radial_number(r, nm_tuple)

    def evaluate(self, r, phi, n, m):
        """
        Evaluate a Zernike polynomial in polar coordinates

        r is the radial coordinate (a scalar or an array)

        phi is the angular coordinate in radians (a scalar or an array)

        n is the radial order of the polynomial

        m is the angular order of the polynomial

        Return the value(s) of the polynomial at r, phi
        (returns np.NaN if r>1.0)
        """
        radial_part = self._evaluate_radial(r, n, np.abs(m))
        if m >= 0:
            return radial_part * np.cos(m * phi)
        return radial_part * np.sin(m * phi)

    def norm(self, n, m):
        """
        Return the normalization of the n, m Zernike
        polynomial

        n is the radial order

        m is the angular order
        """
        nm_tuple = self._validate_nm(n, np.abs(m))
        if nm_tuple[1] == 0:
            eps = 2.0
        else:
            eps = 1.0
        return eps * np.pi / (nm_tuple[0] * 2 + 2)

    def evaluate_xy(self, x, y, n, m):
        """
        Evaluate a Zernike polynomial at a point in
        Cartesian space.

        x and y are the Cartesian coordinaes (either scalars
        or arrays)

        n is the radial order of the polynomial

        m is the angular order of the polynomial

        Return the value(s) of the polynomial at x, y
        (returns np.NaN if sqrt(x**2+y**2)>1.0)
        """
        # since we use np.where to handle r==0 cases,
        # use np.errstate to temporarily turn off the
        # divide by zero and invalid double scalar
        # RuntimeWarnings
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.sqrt(x**2 + y**2)
            cos_phi = np.where(r > 0.0, x / r, 0.0)
            arccos_phi = np.arccos(cos_phi)
            phi = np.where(y >= 0.0, arccos_phi, 0.0 - arccos_phi)
        return self.evaluate(r, phi, n, m)
