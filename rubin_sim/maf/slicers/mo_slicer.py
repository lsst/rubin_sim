import numpy as np
import pandas as pd

from .base_slicer import BaseSlicer
from rubin_sim.maf.plots.mo_plotters import MetricVsH, MetricVsOrbit

from rubin_sim.moving_objects.orbits import Orbits

__all__ = ["MoObjSlicer"]


class MoObjSlicer(BaseSlicer):
    """Slice moving object _observations_, per object and optionally clone/per H value.

    Iteration over the MoObjSlicer will go as:
    * iterate over each orbit;
    * if Hrange is not None, for each orbit, iterate over Hrange.

    Parameters
    ----------
    h_range : numpy.ndarray or None
        The H values to clone the orbital parameters over. If Hrange is None, will not clone orbits.
    """

    def __init__(self, h_range=None, verbose=True, badval=0):
        super(MoObjSlicer, self).__init__(verbose=verbose, badval=badval)
        self.Hrange = h_range
        self.slicer_init = {"h_range": h_range, "badval": badval}
        # Set default plot_funcs.
        self.plot_funcs = [
            MetricVsH(),
            MetricVsOrbit(xaxis="q", yaxis="e"),
            MetricVsOrbit(xaxis="q", yaxis="inc"),
        ]

    def setup_slicer(self, orbit_file, delim=None, skiprows=None, obs_file=None):
        """Set up the slicer and read orbit_file and obs_file from disk.

        Sets self.orbits (with orbit parameters), self.allObs, and self.obs
        self.orbit_file and self.obs_file

        Parameters
        ----------
        orbit_file : str
            The file containing the orbit information.
            This is necessary, in order to be able to generate plots.
        obs_file : str, optional
            The file containing the observations of each object, optional.
            If not provided (default, None), then the slicer will not be able to 'slice', but can still plot.
        """
        self.read_orbits(orbit_file, delim=delim, skiprows=skiprows)
        if obs_file is not None:
            self.read_obs(obs_file)
        else:
            self.obsFile = None
            self.allObs = None
            self.obs = None
        # Add these filenames to the slicer init values, to preserve in output files.
        self.slicer_init["orbit_file"] = self.orbit_file
        self.slicer_init["obs_file"] = self.obsFile

    def read_orbits(self, orbit_file, delim=None, skiprows=None):
        # Use sims_movingObjects to read orbit files.
        orb = Orbits()
        orb.readOrbits(orbit_file, delim=delim, skiprows=skiprows)
        self.orbit_file = orbit_file
        self.orbits = orb.orbits
        # Then go on as previously. Need to refactor this into 'setup_slicer' style.
        self.nSso = len(self.orbits)
        self.slicePoints = {}
        self.slicePoints["orbits"] = self.orbits
        # And set the slicer shape/size.
        if self.Hrange is not None:
            self.shape = [self.nSso, len(self.Hrange)]
            self.slicePoints["H"] = self.Hrange
        else:
            self.shape = [self.nSso, 1]
            self.slicePoints["H"] = self.orbits["H"]
        # Set the rest of the slicePoint information once
        self.nslice = self.shape[0] * self.shape[1]

    def read_obs(self, obsFile):
        """Read observations of the solar system objects (such as created by sims_movingObjects).

        Parameters
        ----------
        obsFile: str
            The file containing the observation information.
        """
        # For now, just read all the observations (should be able to chunk this though).
        self.allObs = pd.read_csv(obsFile, delim_whitespace=True, comment="#")
        self.obsFile = obsFile
        # We may have to rename the first column from '#objId' to 'objId'.
        if self.allObs.columns.values[0].startswith("#"):
            newcols = self.allObs.columns.values
            newcols[0] = newcols[0].replace("#", "")
            self.allObs.columns = newcols
        if "velocity" not in self.allObs.columns.values:
            self.allObs["velocity"] = np.sqrt(
                self.allObs["dradt"] ** 2 + self.allObs["ddecdt"] ** 2
            )
        if "visitExpTime" not in self.allObs.columns.values:
            self.allObs["visitExpTime"] = (
                np.zeros(len(self.allObs["objId"]), float) + 30.0
            )
        # If we created intermediate data products by pandas, we may have an inadvertent 'index'
        #  column. Since this creates problems later, drop it here.
        if "index" in self.allObs.columns.values:
            self.allObs.drop("index", axis=1, inplace=True)
        self.subset_obs()

    def subset_obs(self, pandas_constraint=None):
        """
        Choose a subset of all the observations, such as those in a particular time period.
        """
        if pandas_constraint is None:
            self.obs = self.allObs
        else:
            self.obs = self.allObs.query(pandas_constraint)

    def _slice_obs(self, idx):
        """Return the observations of a given ssoId.

        For now this works for any ssoId; in the future, this might only work as ssoId is
        progressively iterated through the series of ssoIds (so we can 'chunk' the reading).

        Parameters
        ----------
        idx : integer
            The integer index of the particular SSO in the orbits dataframe.
        """
        # Find the matching orbit.
        orb = self.orbits.iloc[idx]
        # Find the matching observations.
        if self.obs["objId"].dtype == "object":
            obs = self.obs.query('objId == "%s"' % (orb["objId"]))
        else:
            obs = self.obs.query("objId == %d" % (orb["objId"]))
        # Return the values for H to consider for metric.
        if self.Hrange is not None:
            Hvals = self.Hrange
        else:
            Hvals = np.array([orb["H"]], float)
        # Note that ssoObs / obs is a recarray not Dataframe!
        # But that the orbit IS a Dataframe.
        return {"obs": obs.to_records(), "orbit": orb, "Hvals": Hvals}

    def __iter__(self):
        """
        Iterate through each of the ssoIds.
        """
        self.idx = 0
        return self

    def __next__(self):
        """
        Returns result of self._getObs when iterating over moSlicer.
        """
        if self.idx >= self.nSso:
            raise StopIteration
        idx = self.idx
        self.idx += 1
        return self._slice_obs(idx)

    def __getitem__(self, idx):
        # This may not be guaranteed to work if/when we implement chunking of the obsfile.
        return self._slice_obs(idx)

    def __eq__(self, other_slicer):
        """
        Evaluate if two slicers are equal.
        """
        result = False
        if isinstance(other_slicer, MoObjSlicer):
            if other_slicer.orbit_file == self.orbit_file:
                if other_slicer.obsFile == self.obsFile:
                    if np.array_equal(
                        other_slicer.slicePoints["H"], self.slicePoints["H"]
                    ):
                        result = True
        return result
