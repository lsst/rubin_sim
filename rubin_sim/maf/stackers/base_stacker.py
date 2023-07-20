__all__ = ("StackerRegistry", "BaseStacker")

import inspect
import warnings

import numpy as np
from six import with_metaclass


class StackerRegistry(type):
    """
    Meta class for Stackers, to build a registry of stacker classes.
    """

    def __init__(cls, name, bases, dict):
        super(StackerRegistry, cls).__init__(name, bases, dict)
        if not hasattr(cls, "registry"):
            cls.registry = {}
        if not hasattr(cls, "source_dict"):
            cls.source_dict = {}
        modname = inspect.getmodule(cls).__name__
        if modname.startswith("rubin_sim.maf.stackers"):
            modname = ""
        else:
            if len(modname.split(".")) > 1:
                modname = ".".join(modname.split(".")[:-1]) + "."
            else:
                modname = modname + "."
        stackername = modname + name
        if stackername in cls.registry:
            raise Exception(
                "Redefining stacker %s! (there are >1 stackers with the same name)" % (stackername)
            )
        if stackername != "BaseStacker":
            cls.registry[stackername] = cls
        cols_added = cls.cols_added
        for col in cols_added:
            cls.source_dict[col] = cls

    def get_class(cls, stackername):
        return cls.registry[stackername]

    def help(cls, doc=False):
        for stackername in sorted(cls.registry):
            if not doc:
                print(stackername)
            if doc:
                print("---- ", stackername, " ----")
                print(cls.registry[stackername].__doc__)
                stacker = cls.registry[stackername]()
                print(" Columns added to SimData: ", ",".join(stacker.cols_added))
                print(" Default columns required: ", ",".join(stacker.cols_req))


class BaseStacker(with_metaclass(StackerRegistry, object)):
    """Base MAF Stacker: add columns generated at run-time to the simdata array."""

    # List of the names of the columns generated by the Stacker.
    cols_added = []

    def __init__(self):
        """
        Instantiate the stacker.
        This method should be overriden by the user. This serves as an example of
        the variables required by the framework.
        """
        # Add the list of new columns generated by the stacker as class attributes (colsAdded - above).
        # List of the names of the columns required from the database (to generate the Stacker columns).
        self.cols_req = []
        # Optional: specify the new column types.
        self.cols_added_dtypes = None
        # Optional: provide a list of units for the columns defined in colsAdded.
        self.units = [None]

    def __hash__(self):
        return None

    def __eq__(self, other_stacker):
        """
        Evaluate if two stackers are equivalent.
        """
        # If the class names are different, they are not 'the same'.
        if self.__class__.__name__ != other_stacker.__class__.__name__:
            return False
        # Otherwise, this is the same stacker class, but may be instantiated differently.
        # We have to delve a little further, and compare the kwargs & attributes for each stacker.
        state_now = dir(self)
        for key in state_now:
            if not key.startswith("_") and key != "registry" and key != "run" and key != "next":
                if not hasattr(other_stacker, key):
                    return False
                # If the attribute is from numpy, assume it's an array and test it
                if type(getattr(self, key)).__module__ == np.__name__:
                    if not np.array_equal(getattr(self, key), getattr(other_stacker, key)):
                        return False
                else:
                    if getattr(self, key) != getattr(other_stacker, key):
                        return False
        return True

    def __ne__(self, other_stacker):
        """
        Evaluate if two stackers are not equal.
        """
        if self == other_stacker:
            return False
        else:
            return True

    def _add_stacker_cols(self, sim_data):
        """
        Add the new Stacker columns to the sim_data array.
        If columns already present in sim_data, just allows 'run' method to overwrite.
        Returns sim_data array with these columns added (so 'run' method can set their values).
        """
        if not hasattr(self, "cols_added_dtypes") or self.cols_added_dtypes is None:
            self.cols_added_dtypes = [float for col in self.cols_added]
        # Create description of new recarray.
        newdtype = sim_data.dtype.descr
        cols_present = [False] * len(self.cols_added)
        for i, (col, dtype) in enumerate(zip(self.cols_added, self.cols_added_dtypes)):
            if col in sim_data.dtype.names:
                if sim_data[col][0] is not None:
                    cols_present[i] = True
                    warnings.warn(
                        "Warning - column %s already present in sim_data, may be overwritten "
                        "(depending on stacker)." % (col)
                    )
            else:
                newdtype += [(col, dtype)]
        new_data = np.empty(sim_data.shape, dtype=newdtype)
        # Add references to old data.
        for col in sim_data.dtype.names:
            new_data[col] = sim_data[col]
        # Were all columns present and populated with something not None? If so, then consider 'all there'.
        if sum(cols_present) == len(self.cols_added):
            cols_present = True
        else:
            cols_present = False
        return new_data, cols_present

    def run(self, sim_data, override=False):
        """
        Example: Generate the new stacker columns, given the simdata columns from the database.
        Returns the new simdata structured array that includes the new stacker columns.
        """
        # Add new columns
        if len(sim_data) == 0:
            return sim_data
        sim_data, cols_present = self._add_stacker_cols(sim_data)
        # If override is set, it means go ahead and recalculate stacker values.
        if override:
            cols_present = False
        # Run the method to calculate/add new data.
        try:
            return self._run(sim_data, cols_present)
        except TypeError:
            warnings.warn(
                "Please update the stacker %s so that the _run method matches the current API. "
                "This will give you the option to skip re-running stackers if the columns are "
                "already present." % (self.__class__.__name__)
            )
            return self._run(sim_data)

    def _run(self, sim_data, cols_present=False):
        """Run the stacker. This is the method to subclass.

        Parameters
        ----------
        sim_data: np.NDarray
            The observation data, provided by the MAF framework.
        cols_present: bool, optional
            Flag to indicate whether the columns to be added are already present in the data.
            This will also be provided by the MAF framework -- but your _run method can use the value.
            If it is 'True' and you do trust the existing value, the _run method can simply return sim_data.

        Returns
        -------
        np.NDarray
            The simdata, with the columns added or updated (or simply already present).
        """
        # By moving the calculation of these columns to a separate method, we add the possibility of using
        #  stackers with pandas dataframes. The _addStackerCols method won't work with dataframes, but the
        #  _run methods are quite likely to (depending on their details), as they are just populating columns.
        raise NotImplementedError(
            "Not Implemented: " "the child stackers should implement their own _run methods"
        )
