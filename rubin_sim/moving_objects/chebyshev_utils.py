"""Utilities to estimate and evaluate Chebyshev coefficients of a function.

Implementation of Newhall, X. X. 1989, Celestial Mechanics, 45, p. 305-310
"""

__all__ = ("chebeval", "chebfit", "make_cheb_matrix", "make_cheb_matrix_only_x")

import numpy as np

# Evaluation routine.


def chebeval(x, p, interval=(-1.0, 1.0), do_velocity=True, mask=False):
    """Evaluate a Chebyshev series and first derivative at points x.

    If p is of length n + 1, this function returns:
    y_hat(x) = p_0 * T_0(x*) + p_1 * T_1(x*) + ... + p_n * T_n(x*)
    where T_n(x*) are the orthogonal Chebyshev polynomials of the
    first kind, defined on the interval [-1, 1] and p_n are the
    coefficients. The scaled variable x* is defined on the [-1, 1]
    interval such that (x*) = (2*x - a - b)/(b - a), and x is defined
    on the [a, b] interval.

    Parameters
    ----------
    x : `scalar` or `np.ndarray`
        Points at which to evaluate the polynomial.
    p :  `np.ndarray`
        Chebyshev polynomial coefficients, as returned by chebfit.
    interval : 2-element list/tuple
        Bounds the x-interval on which the Chebyshev coefficients were fit.
    do_velocity : `bool`
        If True, compute the first derivative at points x.
    mask : `bool`
        If True, return Nans when the x goes beyond 'interval'.
        If False, extrapolate fit beyond 'interval' limits.

    Returns
    -------
    y, v : `float` or `np.ndarray`, `float` or `np.ndarray` (or None)
        Y (position) and velocity values (if computed)
    """
    if len(interval) != 2:
        raise RuntimeError("interval must have length 2")

    interval_begin = float(interval[0])
    interval_end = float(interval[-1])
    t = 2.0 * np.array(x, dtype=np.float64) - interval_begin - interval_end
    t /= interval_end - interval_begin

    y = 0.0
    v = 0.0
    y0 = np.ones_like(t)
    y1 = t
    v0 = np.zeros_like(t)
    v1 = np.ones_like(t)
    v2 = 4.0 * t
    t = 2.0 * t
    N = len(p)

    if do_velocity:
        for i in np.arange(0, N, 2):
            if i == N - 1:
                y1 = 0.0
                v1 = 0.0
            j = min(i + 1, N - 1)

            y += p[i] * y0 + p[j] * y1
            v += p[i] * v0 + p[j] * v1

            y2 = t * y1 - y0
            y3 = t * y2 - y1
            v2 = t * v1 - v0 + 2 * y1
            v3 = t * v2 - v1 + 2 * y2

            y0 = y2
            y1 = y3
            v0 = v2
            v1 = v3

        if mask:
            mask = np.where((x < interval_begin) | (x > interval_end), True, False)
            y = np.where(mask, np.nan, y)
            v = np.where(mask, np.nan, v)
        return y, 2 * v / (interval_end - interval_begin)
    else:
        for i in np.arange(0, N, 2):
            if i == N - 1:
                y1 = 0.0
            j = min((i + 1), (N - 1))
            y += p[i] * y0 + p[j] * y1
            y0 = t * y1 - y0
            y1 = t * y0 - y1
        if mask:
            mask = np.where((x < interval_begin) | (x > interval_end), True, False)
            y = np.where(mask, np.nan, y)
        return y, None


# Fitting routines.


def make_cheb_matrix(n_points, n_poly, weight=0.16):
    """Compute C1^(-1)C2 using Newhall89 approach.

    Utility function for fitting chebyshev polynomials to
    x(t) and dx/dt(t) forcing equality at the end points.
    This function computes the matrix (C1^(-1)C2).
    Multiplying this matrix by the x and dx/dt values to be fit
    produces the chebyshev coefficient.
    This function need only be called once for a given polynomial degree and
    number of points.

    The matrices returned are of shape(n_points+1)x(n_poly).
    The coefficients fitting the n_points+1 points, X, are found by:
    A = xMultiplier * x  +  dxMultiplier * dxdt
    if derivative information is known, or
    A = xMultiplier * x
    if no derivative information is known.
    The xMultiplier matrices are different,
    depending on whether derivative information is known.
    Use function make_cheb_matrix_only_x if derviative is not known.
    See Newhall, X. X. 1989, Celestial Mechanics, 45, p. 305-310 for details.

    Parameters
    ----------
    n_points : `int`
        Number of point to be fits. Must be greater than 2.
    n_poly :  `int`
        Number of polynomial terms. Polynomial degree + 1
    weight : `float`, optional
        Weight to allow control of relative effectos of position and velocity
        values. Newhall80 found best results are obtained with
        velocity weighted at 0.4 relative to position,
        giving W the form (1.0, 0.16, 1.0, 0.16,...)

    Returns
    -------
    c1c2: `np.ndarray`
        xMultiplier, C1^(-1)C2 even rows of shape (n_points+1)x(n_poly) to
        be multiplied by x values.
    c1c2: `np.ndarray`
        dxMultiplier, C1^(-1)C2 odd rows of shape (n_points+1)x(n_poly) to
        be multiplied by dx/dy values
    """
    tmat = np.zeros([n_points, n_poly])
    tdot = np.zeros([n_points, n_poly])

    cj = np.zeros([n_poly])
    xj = np.linspace(1, -1, n_points)

    for i in np.arange(0, n_poly):
        cj[:] = 0
        cj[i] = 1
        y, v = chebeval(xj, cj)
        tmat[:, i] = y
        tdot[:, i] = v

    # make matrix T*W
    tw = np.zeros([n_poly, n_points, 2])
    tw[:, :, 0] = tmat.transpose()
    tw[:, :, 1] = tdot.transpose() * weight

    # make matrix T*WT
    twt = np.dot(tw[:, :, 0], tmat) + np.dot(tw[:, :, 1], tdot)
    tw = tw.reshape(n_poly, 2 * n_points)

    # insert matrix T*W in matrix C2
    c2 = np.zeros([n_poly + 4, 2 * n_points])
    c2[0:n_poly] = tw
    c2[n_poly, 0] = 1
    c2[n_poly + 1, 1] = 1
    c2[n_poly + 2, -2] = 1
    c2[n_poly + 3, -1] = 1

    # insert matrix T*WT in matrix C1
    c1 = np.zeros([n_poly + 4, n_poly + 4])
    c1[0:n_poly, 0:n_poly] = twt
    c1[n_poly + 0, 0:n_poly] = tmat[0]
    c1[n_poly + 1, 0:n_poly] = tdot[0]
    c1[n_poly + 2, 0:n_poly] = tmat[-1]
    c1[n_poly + 3, 0:n_poly] = tdot[-1]

    c1[0:n_poly, n_poly:] = c1[n_poly:, 0:n_poly].transpose()

    c1inv = np.linalg.inv(c1)
    c1c2 = np.dot(c1inv, c2)
    c1c2 = c1c2.reshape(n_poly + 4, n_points, 2)
    c1c2 = c1c2[:, ::-1, :]
    c1c2 = c1c2.reshape(n_poly + 4, 2 * n_points)

    # separate even rows for x, and odd rows for dx/dt
    return c1c2[0:n_poly, 0::2], c1c2[0:n_poly, 1::2]


def make_cheb_matrix_only_x(n_points, n_poly):
    """Compute C1^(-1)C2 using Newhall89 approach without dx/dt

    Compute xMultiplier using only the equality constraint of the x-values
    at the endpoints.
    To be used when first derivatives are not available.
    If chebyshev approximations are strung together piecewise only the x-values
    and not the first derivatives will be continuous at the boundaries.
    Multiplying this matrix by the x-values to be fit produces the chebyshev
    coefficients. This function need only be called once for a given
    polynomial degree and
    number of points.
    See Newhall, X. X. 1989, Celestial Mechanics, 45, p. 305-310.

    Parameters
    ----------
    n_points : `int`
        Number of point to be fits. Must be greater than 2.
    n_poly : `int`
        Number of polynomial terms. Polynomial degree + 1

    Returns
    -------
    c1c2: `np.ndarray`
        xMultiplier, Even rows of C1^(-1)C2 w/ shape (n_points+1)x(n_poly)
        to be multiplied by x values
    """

    tmat = np.zeros([n_points, n_poly])
    cj = np.zeros([n_poly])
    xj = np.linspace(1, -1, n_points)
    for i in range(0, n_poly):
        cj[:] = 0
        cj[i] = 1
        tmat[:, i], v = chebeval(xj, cj)

    # Augment matrix T to get matrix C2
    c2 = np.zeros([n_poly + 2, n_points])
    c2[0:n_poly] = tmat.transpose()
    c2[n_poly, 0] = 1
    c2[n_poly + 1, n_points - 1] = 1

    # Augment matrix T*WT to get the matrix C1
    c1 = np.zeros([n_poly + 2, n_poly + 2])
    c1[0:n_poly, 0:n_poly] = np.dot(tmat.transpose(), tmat)
    c1[n_poly + 0, 0:n_poly] = tmat[0]
    c1[n_poly + 1, 0:n_poly] = tmat[-1]
    c1[0:n_poly, n_poly:] = c1[n_poly:, 0:n_poly].transpose()

    c1inv = np.linalg.inv(c1)
    # C1^(-1) C2
    c1c2 = np.dot(c1inv, c2)

    c1c2 = c1c2.reshape(n_poly + 2, n_points)
    c1c2 = c1c2[:, ::-1]
    return c1c2[0:n_poly]


def chebfit(t, x, dxdt=None, x_multiplier=None, dx_multiplier=None, n_poly=7):
    """Fit Chebyshev polynomial constrained at endpoints using
    Newhall89 approach.

    Return Chebyshev coefficients and statistics from fit
    to array of positions (x) and optional velocities (dx/dt).
    If both the function and its derivative are specified, then the value and
    derivative of the interpolating polynomial at the
    endpoints will be exactly equal to the input endpoint values.
    Many approximations may be piecewise strung together and the function value
    and its first derivative will be continuous across boundaries.
    If derivatives are not provided, only the function value will be
    continuous across boundaries.

    If x_multiplier and dx_multiplier are not provided or
    are an inappropriate shape for t and x, they will be recomputed.
    See Newhall, X. X. 1989, Celestial Mechanics, 45, p. 305-310
    for details.

    Parameters
    ----------
    t : `np.ndarray`
        Array of regularly sampled independent variable (e.g. time)
    x : `np.ndarray`
        Array of regularly sampled dependent variable (e.g. declination)
    dxdt : `np.ndarray', optional
        Optionally, array of first derivatives of x with respect to t,
        at the same grid points. (e.g. sky velocity ddecl/dt)
    x_multiplier : `np.ndarray`, optional
        Optional 2D Matrix with rows of C1^(-1)C2 corresponding to x.
        Use make_cheb_matrix to compute
    dx_multiplier : `np.ndarray`, optional
        Optional 2D Matrix with rows of C1^(-1)C2 corresponding to dx/dt.
        Use make_cheb_matrix to compute
    n_poly : `int`, optional
        Number of polynomial terms. Degree + 1.  Must be >=2 and <=2*n_points,
        when derivative information is specified, or <=n_points, when no
        derivative information is specified. Default = 7.

    Returns
    -------
    a_n : `np.ndarray`
        Array of chebyshev coefficients with length=n_poly.
    residuals : `np.ndarray`
        Array of residuals of the tabulated function x minus the
        approximated function.
    rms : `float`
        The rms of the residuals in the fit.
    maxresid : `float`
        The maximum of the residals to the fit.
    """
    n_points = len(t)
    if len(x) != n_points:
        raise ValueError("length of x (%s) != length of t (%s)" % (len(x), n_points))
    if dxdt is None:
        if n_poly > n_points:
            raise RuntimeError(
                "Without velocity constraints, n_poly (%d) must be less than %s" % (n_poly, n_points)
            )
        if n_poly < 2:
            raise RuntimeError("Without velocity constraints, n_poly (%d) must be greater than 2" % n_poly)
    else:
        if n_poly > 2 * n_points:
            raise RuntimeError(
                "n_poly (%d) must be less than %s (%d)" % (n_poly, "2 * n_points", 2 * (n_points))
            )
        if n_poly < 4:
            raise RuntimeError("n_poly (%d) must be greater than 4" % n_poly)

    # Recompute C1invX2 if x_multiplier and dx_multiplier are None or
    # they are not appropriate for sizes of input positions and velocities.

    if x_multiplier is None:
        redo_x = True
    else:
        redo_x = (x_multiplier.shape[1] != n_points) | (x_multiplier.shape[0] != n_poly)

    if dx_multiplier is None:
        redo_v = True
    else:
        redo_v = (dx_multiplier.shape[1] != n_points) | (dx_multiplier.shape[0] != n_poly)

    if (dxdt is None) & redo_x:
        x_multiplier = make_cheb_matrix_only_x(n_points, n_poly)

    if (dxdt is not None) & (redo_v | redo_x):
        x_multiplier, dx_multiplier = make_cheb_matrix(n_points, n_poly)

    if x.size != n_points:
        raise RuntimeError("Not enough elements in X")

    t_interval = np.array([t[0], t[-1]]) - t[0]
    t_scaled = t - t[0]

    # Compute the X portion of the coefficients
    a_n = np.dot(x_multiplier, x)

    # Compute statistics
    # for x and dxdt if it is available
    if dxdt is not None:
        a_n = a_n + np.dot(dx_multiplier, dxdt * (t_interval[1] - t_interval[0]) / 2.0)
        x_approx, dx_approx = chebeval(t_scaled, a_n, interval=t_interval)
    else:
        # Statistics for x only
        x_approx, _ = chebeval(t_scaled, a_n, interval=t_interval, do_velocity=False)

    residuals = x - x_approx
    se = np.sum(residuals**2)
    rms = np.sqrt(se / (n_points - 1))
    maxresid = np.max(np.abs(residuals))

    return a_n, residuals, rms, maxresid
