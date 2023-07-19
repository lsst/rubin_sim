__all__ = (
    "thetaphi2xyz",
    "even_points",
    "elec_potential",
    "ang_potential",
    "fib_sphere_grid",
    "iterate_potential_random",
    "iterate_potential_smart",
    "even_points_xyz",
    "elec_potential_xyz",
    "xyz2thetaphi",
)

import numpy as np
from scipy.optimize import minimize

from rubin_sim.utils import _angular_separation


def thetaphi2xyz(theta, phi):
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z


def xyz2thetaphi(x, y, z):
    phi = np.arccos(z)
    theta = np.arctan2(y, x)
    return theta, phi


def elec_potential(x0):
    """
    Compute the potential energy for electrons on a sphere

    Parameters
    ----------
    x0 : array
       First half of x0 or theta values, secnd half phi

    Returns
    -------
    Potential energy
    """

    theta = x0[0 : int(x0.size / 2)]
    phi = x0[int(x0.size / 2) :]

    x, y, z = thetaphi2xyz(theta, phi)
    # Distance squared
    dsq = 0.0

    indices = np.triu_indices(x.size, k=1)

    for coord in [x, y, z]:
        coord_i = np.tile(coord, (coord.size, 1))
        coord_j = coord_i.T
        d = (coord_i[indices] - coord_j[indices]) ** 2
        dsq += d

    U = np.sum(1.0 / np.sqrt(dsq))
    return U


def potential_single(coord0, x, y, z):
    """
    Find the potential contribution from a single point.
    """

    x0 = coord0[0]
    y0 = coord0[1]
    z0 = coord0[2]
    # Enforce point has to be on a sphere
    rsq = x0**2 + y0**2 + z0**2
    r = np.sqrt(rsq)
    x0 = x0 / r
    y0 = y0 / r
    z0 = z0 / r

    dsq = (x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2
    U = np.sum(1.0 / np.sqrt(dsq))
    return U


def xyz2_u(x, y, z):
    """
    compute the potential
    """
    dsq = 0.0

    indices = np.triu_indices(x.size, k=1)

    for coord in [x, y, z]:
        coord_i = np.tile(coord, (coord.size, 1))
        coord_j = coord_i.T
        dsq += (coord_i[indices] - coord_j[indices]) ** 2

    d = np.sqrt(dsq)
    U = np.sum(1.0 / d)
    return U


def iterate_potential_smart(x0, stepfrac=0.1):
    """
    Calculate the change in potential by shifting points in theta and phi directions
    # wow, that sure didn't work at all.
    """

    theta = x0[0 : x0.size / 2]
    phi = x0[x0.size / 2 :]
    x, y, z = thetaphi2xyz(theta, phi)
    u_input = xyz2_u(x, y, z)

    # Now to loop over each point, and find where it's potenital minimum would be, and move it
    # half-way there.
    xyz_new = np.zeros((x.size, 3), dtype=float)
    mask = np.ones(x.size, dtype=bool)
    for i in np.arange(x.size):
        mask[i] = 0
        fit = minimize(potential_single, [x[i], y[i], z[i]], args=(x[mask], y[mask], z[mask]))
        mask[i] = 1
        xyz_new[i] = fit.x / np.sqrt(np.sum(fit.x**2))

    xyz_input = np.array((x, y, z)).T
    diff = xyz_input - xyz_new

    # Move half way in x-y-z space
    xyz_out = xyz_input + stepfrac * diff
    # Project back onto sphere
    xyz_out = xyz_out.T / np.sqrt(np.sum(xyz_out**2, axis=1))
    u_new = xyz2_u(xyz_out[0, :], xyz_out[1, :], xyz_out[2, :])
    theta, phi = xyz2thetaphi(xyz_out[0, :], xyz_out[1, :], xyz_out[2, :])
    return np.concatenate((theta, phi)), u_new


def iterate_potential_random(x0, stepsize=0.05):
    """
    Given a bunch of theta,phi values, shift things around to minimize potential
    """

    theta = x0[0 : int(x0.size / 2)]
    phi = x0[int(x0.size / 2) :]

    x, y, z = thetaphi2xyz(theta, phi)
    # Distance squared
    dsq = 0.0

    indices = np.triu_indices(x.size, k=1)

    for coord in [x, y, z]:
        coord_i = np.tile(coord, (coord.size, 1))
        coord_j = coord_i.T
        d = (coord_i[indices] - coord_j[indices]) ** 2
        dsq += d

    d = np.sqrt(dsq)

    u_input = 1.0 / d

    # offset everything by a random ammount
    x_new = x + np.random.random(theta.size) * stepsize
    y_new = y + np.random.random(theta.size) * stepsize
    z_new = z + np.random.random(theta.size) * stepsize

    r = (x_new**2 + y_new**2 + z_new**2) ** 0.5
    # put back on the sphere
    x_new = x_new / r
    y_new = y_new / r
    z_new = z_new / r

    dsq_new = 0
    for coord, coord_new in zip([x, y, z], [x_new, y_new, z_new]):
        coord_i_new = np.tile(coord_new, (coord_new.size, 1))
        coord_j = coord_i_new.T
        d_new = (coord_i_new[indices] - coord_j[indices]) ** 2
        dsq_new += d_new
    u_new = 1.0 / np.sqrt(dsq_new)

    u_diff = np.sum(u_new) - np.sum(u_input)
    if u_diff > 0:
        return x0, 0.0
    else:
        theta, phi = xyz2thetaphi(x_new, y_new, z_new)
        return np.concatenate((theta, phi)), u_diff


def ang_potential(x0):
    """
    If distance is computed along sphere rather than through 3-space.
    """
    theta = x0[0 : int(x0.size / 2)]
    phi = np.pi / 2 - x0[int(x0.size / 2) :]

    indices = np.triu_indices(theta.size, k=1)

    theta_i = np.tile(theta, (theta.size, 1))
    theta_j = theta_i.T
    phi_i = np.tile(phi, (phi.size, 1))
    phi_j = phi_i.T
    d = _angular_separation(theta_i[indices], phi_i[indices], theta_j[indices], phi_j[indices])
    U = np.sum(1.0 / d)
    return U


def fib_sphere_grid(npoints):
    """
    Use a Fibonacci spiral to distribute points uniformly on a sphere.

    based on https://people.sc.fsu.edu/~jburkardt/py_src/sphere_fibonacci_grid/sphere_fibonacci_grid_points.py

    Returns theta and phi in radians
    """

    phi = (1.0 + np.sqrt(5.0)) / 2.0

    i = np.arange(npoints, dtype=float)
    i2 = 2 * i - (npoints - 1)
    theta = (2.0 * np.pi * i2 / phi) % (2.0 * np.pi)
    sphi = i2 / npoints
    phi = np.arccos(sphi)
    return theta, phi


def even_points(npts, use_fib_init=True, method="CG", potential_func=elec_potential, maxiter=None):
    """
    Distribute npts over a sphere and minimize their potential, making them
    "evenly" distributed

    Starting with the Fibonacci spiral speeds things up by ~factor of 2.
    """

    if use_fib_init:
        # Start with fibonacci spiral guess
        theta, phi = fib_sphere_grid(npts)
    else:
        # Random on a sphere
        theta = np.random.rand(npts) * np.pi * 2.0
        phi = np.arccos(2.0 * np.random.rand(npts) - 1.0)

    x = np.concatenate((theta, phi))
    # XXX--need to check if this is the best minimizer
    min_fit = minimize(potential_func, x, method="CG", options={"maxiter": maxiter})

    x = min_fit.x
    theta = x[0 : int(x.size / 2)]
    phi = x[int(x.size / 2) :]
    # Looks like I get the same energy values as https://en.wikipedia.org/wiki/Thomson_problem
    return theta, phi


def elec_potential_xyz(x0):
    x0 = x0.reshape(3, int(x0.size / 3))
    x = x0[0, :]
    y = x0[1, :]
    z = x0[2, :]
    dsq = 0.0

    r = np.sqrt(x**2 + y**2 + z**2)
    x = x / r
    y = y / r
    z = z / r
    indices = np.triu_indices(x.size, k=1)

    for coord in [x, y, z]:
        coord_i = np.tile(coord, (coord.size, 1))
        coord_j = coord_i.T
        d = (coord_i[indices] - coord_j[indices]) ** 2
        dsq += d

    U = np.sum(1.0 / np.sqrt(dsq))
    return U


def elec_p_xyx_loop(x0):
    """do this with a brutal loop that can be numba ified"""
    x0 = x0.reshape(3, int(x0.size / 3))
    x = x0[0, :]
    y = x0[1, :]
    z = x0[2, :]
    U = 0.0

    r = np.sqrt(x**2 + y**2 + z**2)
    x = x / r
    y = y / r
    z = z / r

    npts = x.size
    for i in range(npts - 1):
        for j in range(i + 1, npts):
            dsq = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 + (z[i] - z[j]) ** 2
            U += 1.0 / np.sqrt(dsq)
    return U


def x02sphere(x0):
    x0 = x0.reshape(3, int(x0.size / 3))
    x = x0[0, :]
    y = x0[1, :]
    z = x0[2, :]

    r = np.sqrt(x**2 + y**2 + z**2)
    x = x / r
    y = y / r
    z = z / r

    return np.concatenate((x, y, z))


def even_points_xyz(
    npts,
    use_fib_init=True,
    method="CG",
    potential_func=elec_p_xyx_loop,
    maxiter=None,
    callback=None,
    verbose=True,
):
    """
    Distribute npts over a sphere and minimize their potential, making them
    "evenly" distributed

    Starting with the Fibonacci spiral speeds things up by ~factor of 2.
    """

    if use_fib_init:
        # Start with fibonacci spiral guess
        theta, phi = fib_sphere_grid(npts)
    else:
        # Random on a sphere
        theta = np.random.rand(npts) * np.pi * 2.0
        phi = np.arccos(2.0 * np.random.rand(npts) - 1.0)

    x = np.concatenate(thetaphi2xyz(theta, phi))

    if verbose:
        print("initial potential=", elec_potential_xyz(x))
    # XXX--need to check if this is the best minimizer
    min_fit = minimize(potential_func, x, method="CG", options={"maxiter": maxiter}, callback=callback)

    if verbose:
        print("final potential=", elec_potential_xyz(min_fit.x))
        print("niteration=", min_fit.nit)

    x = x02sphere(min_fit.x)

    # Looks like I get the same energy values as https://en.wikipedia.org/wiki/Thomson_problem
    return x
