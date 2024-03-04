__all__ = ("LsqrSolver",)

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

# Modified from /astro/store/scratch/tmp/yoachim/Research/LSST/Parallel_Solver
# and then from https://github.com/lsst-sims/legacy_sims_selfcal


class LsqrSolver:
    """
    Class to solve self-calibration

    Parameters
    ----------
    observations : `np.array`
        A numpy array of the observations.
        Should have columns id, patch_id, observed_mag, mag_uncert.
    atol : `float`
        Tolerance passed to lsqr.
    btol : `float`
        Tolerance passed to lsqr.
    iter_lim : `int`
        Iteration limit passed to lsqr.
    show : `bool`
        Should the lsqr solver print some iteration logs (False).
    """

    def __init__(
        self,
        observations,
        atol=1e-8,
        btol=1e-8,
        iter_lim=None,
        show=False,
    ):
        self.atol = atol
        self.btol = btol
        self.iter_lim = iter_lim
        self.observations = observations
        self.show = show

    def run(self):
        """clean data, solve matrix, write solution out."""
        self.clean_data()
        self.solve_matrix()

    def clean_data(self):
        """
        Remove observations that can't contribute to a solution.
        Index remaining stars and patches so they are continuous.
        """
        n_start = 1.0
        n_end = 0.0
        while n_start != n_end:
            n_start = self.observations.size
            self.observations.sort(order="id")
            # Remove observations if the star was only observed once
            good = np.where(
                (self.observations["id"] - np.roll(self.observations["id"], 1))
                * (self.observations["id"] - np.roll(self.observations["id"], -1))
                == 0
            )
            self.observations = self.observations[good]

            # Remove patches with only one star
            self.observations.sort(order="patch_id")

            good = np.where(
                (self.observations["patch_id"] - np.roll(self.observations["patch_id"], 1))
                * (self.observations["patch_id"] - np.roll(self.observations["patch_id"], -1))
                == 0
            )
            self.observations = self.observations[good]
            n_end = self.observations.size

        self.observations.sort(order="patch_id")

        self.patches = np.unique(self.observations["patch_id"])
        n_patches = np.size(self.patches)
        self.n_patches = n_patches
        self.n_patches = n_patches
        patches_index = np.arange(n_patches)
        left = np.searchsorted(self.observations["patch_id"], self.patches)
        right = np.searchsorted(self.observations["patch_id"], self.patches, side="right")
        for i in range(np.size(left)):
            self.observations["patch_id"][left[i] : right[i]] = patches_index[i]

        # Convert id to continuous running index to keep matrix
        # as small as possible
        self.observations.sort(order="id")

        self.stars = np.unique(self.observations["id"])
        n_stars = np.size(self.stars)
        self.n_stars = np.size(self.stars)
        stars_index = np.arange(1, n_stars + 1)
        left = np.searchsorted(self.observations["id"], self.stars)
        right = np.searchsorted(self.observations["id"], self.stars, side="right")
        for i in range(np.size(left)):
            self.observations["id"][left[i] : right[i]] = stars_index[i]

    def solve_matrix(self):
        n_obs = np.size(self.observations)
        # construct sparse matrix
        # A = lil_matrix((nPatches+nStars,np.size(observations['patch_id'])))
        row = np.arange(n_obs)
        row = np.append(row, row)
        col = np.append(
            self.observations["patch_id"],
            self.observations["id"] + np.max(self.observations["patch_id"]),
        )
        # data = np.append(np.ones(nObs),1./observations['mag_uncert'])
        data = 1.0 / self.observations["mag_uncert"]
        data = np.append(data, data)
        # maybe do this in place earlier?
        # then just delete parts of observations earlier to save total memory
        b = self.observations["observed_mag"] / self.observations["mag_uncert"]

        # blast away data now that we have the matrix constructed
        del self.observations

        A = coo_matrix((data, (row, col)), shape=(n_obs, self.n_patches + self.n_stars))
        A = A.tocsr()
        # solve Ax = b
        self.solution = lsqr(A, b, show=self.show, atol=self.atol, btol=self.btol, iter_lim=self.iter_lim)

    def return_solution(self):
        """
        Returns
        -------
        patches, stars: `np.array`, `np.array`
            Two arrays containing patch zeropoints and star best-fit mags.
        """
        patches = np.empty(self.patches.size, dtype=list(zip(["patch_id", "zp"], [int, float])))
        patches["patch_id"] = self.patches
        patches["zp"] = self.solution[0][0 : self.n_patches]

        stars = np.empty(self.stars.size, dtype=list(zip(["id", "fit_mag"], [int, float])))
        stars["id"] = self.stars  # should match the input ID.
        stars["fit_mag"] = self.solution[0][self.n_patches :]

        return patches, stars
