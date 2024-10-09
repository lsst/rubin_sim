__all__ = ("Orbits",)

import warnings

import numpy as np
import pandas as pd


class Orbits:
    """Orbits reads, checks for required values, and stores orbit
    parameters for moving objects.

    self.orbits stores the orbital parameters, as a pandas dataframe.
    self.dataCols defines the columns required,
    although obj_id, H, g, and sed_filename are optional.
    """

    def __init__(self):
        self.orbits = None
        self.orb_format = None

        # Specify the required columns/values in the self.orbits dataframe.
        # Which columns are required depends on self.orb_format.
        self.data_cols = {}
        self.data_cols["COM"] = [
            "obj_id",
            "q",
            "e",
            "inc",
            "Omega",
            "argPeri",
            "tPeri",
            "epoch",
            "H",
            "g",
            "sed_filename",
        ]
        self.data_cols["KEP"] = [
            "obj_id",
            "a",
            "e",
            "inc",
            "Omega",
            "argPeri",
            "meanAnomaly",
            "epoch",
            "H",
            "g",
            "sed_filename",
        ]
        self.data_cols["CAR"] = [
            "obj_id",
            "x",
            "y",
            "z",
            "xdot",
            "ydot",
            "zdot",
            "epoch",
            "H",
            "g",
            "sed_filename",
        ]

    def __len__(self):
        return len(self.orbits)

    def __getitem__(self, i):
        orb = Orbits()
        orb.set_orbits(self.orbits.iloc[i])
        return orb

    def __iter__(self):
        for i, orbit in self.orbits.iterrows():
            orb = Orbits()
            orb.set_orbits(orbit)
            yield orb

    def __eq__(self, other_orbits):
        if isinstance(other_orbits, Orbits):
            if self.orb_format != other_orbits.orb_format:
                return False
            for col in self.data_cols[self.orb_format]:
                if not np.all(self.orbits[col].values == other_orbits.orbits[col].values):
                    return False
                else:
                    return True
        else:
            return False

    def __neq__(self, other_orbits):
        if self == other_orbits:
            return False
        else:
            return True

    def set_orbits(self, orbits):
        """Set and validate orbital parameters contain all required values.

        Sets self.orbits and self.orb_format.
        If objid is not present in orbits,
        a sequential series of integers will be used.
        If H is not present in orbits,
        a default value of 20 will be used.
        If g is not present in orbits,
        a default value of 0.15 will be used.
        If sed_filename is not present in orbits,
        either C or S type will be assigned according to the
        semi-major axis value.

        Parameters
        ----------
        orbits : `pd.DataFrame`, `pd.Series` or `np.ndarray`
           Array-like object containing orbital parameter information.
        """
        # Do we have a single item or multiples?
        if isinstance(orbits, pd.Series):
            # Passed a single SSO in Series, convert to a DataFrame.
            orbits = pd.DataFrame([orbits])
        elif isinstance(orbits, np.ndarray):
            # Passed a numpy array, convert to DataFrame.
            orbits = pd.DataFrame.from_records(orbits)
        elif isinstance(orbits, np.record):
            # This was a single object in a numpy array
            orbits = pd.DataFrame.from_records([orbits], columns=orbits.dtype.names)
        elif isinstance(orbits, pd.DataFrame):
            # This was a pandas dataframe ..
            # but we probably want to drop the index and recount.
            orbits.reset_index(drop=True, inplace=True)

        if "index" in orbits:
            del orbits["index"]

        n_sso = len(orbits)

        # Error if orbits is empty
        # (this avoids hard-to-interpret error messages from pyoorb).
        if n_sso == 0:
            raise ValueError("Length of the orbits dataframe was 0.")

        # Discover which type of orbital parameters we have on disk.
        self.orb_format = None
        if "FORMAT" in orbits:
            if ~(orbits["FORMAT"] == orbits["FORMAT"].iloc[0]).all():
                raise ValueError("All orbital elements in the set should have the same FORMAT.")
            self.orb_format = orbits["FORMAT"].iloc[0]
            # Backwards compatibility .. a bit.
            # CART is deprecated, so swap it to CAR.
            if self.orb_format == "CART":
                self.orb_format = "CAR"
            del orbits["FORMAT"]
            # Check that the orbit format is approximately right.
            if self.orb_format == "COM":
                if "q" not in orbits:
                    raise ValueError('The stated format was COM, but "q" not present in orbital elements?')
            if self.orb_format == "KEP":
                if "a" not in orbits:
                    raise ValueError('The stated format was KEP, but "a" not present in orbital elements?')
            if self.orb_format == "CAR":
                if "x" not in orbits:
                    raise ValueError('The stated format was CAR but "x" not present in orbital elements?')
        if self.orb_format is None:
            # Try to figure out the format, if it wasn't provided.
            if "q" in orbits:
                self.orb_format = "COM"
            elif "a" in orbits:
                self.orb_format = "KEP"
            elif "x" in orbits:
                self.orb_format = "CAR"
            else:
                raise ValueError(
                    "Can't determine orbital type, as neither q, a or x in input orbital elements.\n"
                    "Was attempting to base orbital element quantities on header row, "
                    "with columns: \n%s" % orbits.columns
                )

        # Check that the orbit epoch is within a 'reasonable' range,
        # to detect possible column mismatches.
        general_epoch = orbits["epoch"].head(1).values[0]
        # Look for epochs between 1800 and 2200 -
        # this is primarily to check if people used MJD (and not JD).
        expect_min_epoch = -21503.0
        expect_max_epoch = 124594.0
        if general_epoch < expect_min_epoch or general_epoch > expect_max_epoch:
            raise ValueError(
                "The epoch detected for this orbit is odd - %f. "
                "Expecting a value between %.1f and %.1f (MJD!)"
                % (general_epoch, expect_min_epoch, expect_max_epoch)
            )

        # If these columns are not available in the input data,
        # auto-generate them.
        if "obj_id" not in orbits:
            obj_id = np.arange(0, n_sso, 1)
            orbits = orbits.assign(obj_id=obj_id)
        if "H" not in orbits:
            orbits = orbits.assign(H=20.0)
        if "g" not in orbits:
            orbits = orbits.assign(g=0.15)
        if "sed_filename" not in orbits:
            orbits = orbits.assign(sed_filename=self.assign_sed(orbits))

        # Make sure we gave all the columns we need.
        for col in self.data_cols[self.orb_format]:
            if col not in orbits:
                raise ValueError(
                    "Missing required orbital element %s for orbital format type %s" % (col, self.orb_format)
                )

        # Check to see if we have duplicates.
        if len(orbits["obj_id"].unique()) != n_sso:
            warnings.warn(
                "There are duplicates in the orbit obj_id values" + " - was this intended? (continuing)."
            )
        # All is good.
        self.orbits = orbits

    def assign_sed(self, orbits, random_seed=None):
        """Assign either a C or S type SED,
        depending on the semi-major axis of the object.
        P(C type) = 0 (a<2); 0.5*a - 1 (2<a<4); 1 (a > 4),
        based on figure 23 from Ivezic et al 2001 (AJ, 122, 2749).

        Parameters
        ----------
        orbits : `pd.DataFrame`, `pd.Series` or `np.ndarray`
           Array-like object containing orbital parameter information.

        Returns
        -------
        sedvals : `np.ndarray`
            Array containing the SED type for each object in 'orbits'.
        """
        # using fig. 23 from Ivezic et al. 2001 (AJ, 122, 2749),
        # we can approximate the sed types with a simple linear form:
        #  p(C) = 0 for a<2
        #  p(C) = 0.5*a-1  for 2<a<4
        #  p(C) = 1 for a>4
        # where a is semi-major axis, and p(C) is the probability that
        # an asteroid is C type, with p(S)=1-p(C) for S types.
        if "a" in orbits:
            a = orbits["a"]
        elif "q" in orbits:
            a = orbits["q"] / (1 - orbits["e"])
        elif "x" in orbits:
            # This isn't right, but it's a placeholder to make it work for now.
            a = np.sqrt(orbits["x"] ** 2 + orbits["y"] ** 2 + orbits["z"] ** 2)
        else:
            raise ValueError("Need either a or q (plus e) in orbit data frame.")

        if not hasattr(self, "_rng"):
            if random_seed is not None:
                self._rng = np.random.RandomState(random_seed)
            else:
                self._rng = np.random.RandomState(42)

        chance = self._rng.random_sample(len(orbits))
        prob_c = 0.5 * a - 1.0
        # if chance <= prob_c:
        sedvals = np.where(chance <= prob_c, "C.dat", "S.dat")
        return sedvals

    def read_orbits(self, orbit_file, delim=None, skiprows=None):
        """Read orbits from a file.

        This generates a pandas dataframe containing columns matching dataCols,
        for the appropriate orbital parameter format.
        (currently accepts COM, KEP or CAR formats).

        After reading and standardizing the column names,
        calls self.set_orbits to validate the
        orbital parameters.
        Expects angles in orbital element formats to be in degrees.

        Note that readOrbits uses pandas.read_csv to read the data file
        with the orbital parameters.
        Thus, it should have column headers specifying the column names ..
        unless skiprows = -1 or there is just no header line at all.
        in which case it is assumed to be a standard DES format file,
        with no header line.

        Parameters
        ----------
        orbit_file : `str`
            The name of the input file with orbital parameter information.
        delim : `str`, optional
            The delimiter for the input orbit file.
            Default is None, will use delim_whitespace=True.
        skiprows : `int`, optional
            The number of rows to skip before reading the header information.
            Default is None, which will trigger a search of the file for
            the header columns.
        """
        names = None

        # If skiprows is set, then we will assume the user has
        # handled this so that the first line read has the header information.
        # But, if skiprows is not set, then we have to do some checking to
        # see if there is header information and which row it might start in.
        if skiprows is None:
            skiprows = -1
            # Figure out whether the header is in the first line,
            # or if there are rows to skip.
            # We need to do a bit of juggling to do this before pandas
            # reads the whole orbit file.
            with open(orbit_file, "r") as fp:
                headervalues = None
                for line in fp:
                    values = line.split()
                    try:
                        # If it is a valid orbit line,
                        # we expect column 3 to be a number.
                        float(values[3])
                        # And if it worked, we're done here (it's an orbit) -
                        # go on to parsing header values.
                        break
                    except (ValueError, IndexError):
                        # This wasn't a valid number or there wasn't
                        # anything in the third value.
                        # So this is either the header line or it's a
                        # comment line before the header columns.
                        skiprows += 1
                        headervalues = values

            if headervalues is not None:  # (and skiprows > -1)
                # There is a header, but we also need to check if there
                # is a comment key at the start of the proper header line.
                # (Because this varies as well).
                linestart = headervalues[0]
                if linestart == "#" or linestart == "!!" or linestart == "##":
                    names = headervalues[1:]
                else:
                    names = headervalues
                # Add 1 to skiprows, so that we skip the header column line.
                skiprows += 1

        # So now skiprows is a value.
        # If it is -1, then there is no header information.
        if skiprows == -1:
            # No header; assume it's a typical DES file -
            # we'll assign the column names based on the FORMAT.
            names_com = (
                "obj_id",
                "FORMAT",
                "q",
                "e",
                "i",
                "node",
                "argperi",
                "t_p",
                "H",
                "epoch",
                "INDEX",
                "N_PAR",
                "MOID",
                "COMPCODE",
            )
            names_kep = (
                "obj_id",
                "FORMAT",
                "a",
                "e",
                "i",
                "node",
                "argperi",
                "meanAnomaly",
                "H",
                "epoch",
                "INDEX",
                "N_PAR",
                "MOID",
                "COMPCODE",
            )
            names_car = (
                "obj_id",
                "FORMAT",
                "x",
                "y",
                "z",
                "xdot",
                "ydot",
                "zdot",
                "H",
                "epoch",
                "INDEX",
                "N_PAR",
                "MOID",
                "COMPCODE",
            )
            # First use names_com, and then change if required.
            orbits = pd.read_csv(orbit_file, sep=r"\s+", header=None, names=names_com)

            if orbits["FORMAT"][0] == "KEP":
                orbits.columns = names_kep
            elif orbits["FORMAT"][0] == "CAR":
                orbits.columns = names_car

        else:
            if delim is None:
                orbits = pd.read_csv(orbit_file, sep=r"\s+", skiprows=skiprows, names=names)
            else:
                orbits = pd.read_csv(orbit_file, sep=delim, skiprows=skiprows, names=names)

        # Drop some columns that are typically present in DES files
        # but that we don't need.
        if "INDEX" in orbits:
            del orbits["INDEX"]
        if "N_PAR" in orbits:
            del orbits["N_PAR"]
        if "MOID" in orbits:
            del orbits["MOID"]
        if "COMPCODE" in orbits:
            del orbits["COMPCODE"]
        if "tmp" in orbits:
            del orbits["tmp"]

        # Normalize the column names to standard values and
        # identify the orbital element types.
        sso_cols = orbits.columns.values.tolist()

        # These are the alternative possibilities for various column headers
        # (depending on file version, origin, etc.)
        # that might need remapping to our standardized names.
        alt_names = {}
        alt_names["obj_id"] = [
            "obj_id",
            "objid",
            "!!ObjID",
            "!!OID",
            "!!S3MID",
            "OID",
            "S3MID" "objid(int)",
            "full_name",
            "#name",
        ]
        alt_names["q"] = ["q"]
        alt_names["a"] = ["a"]
        alt_names["e"] = ["e", "ecc"]
        alt_names["inc"] = ["inc", "i", "i(deg)", "incl"]
        alt_names["Omega"] = [
            "Omega",
            "omega",
            "node",
            "om",
            "node(deg)",
            "BigOmega",
            "Omega/node",
            "longNode",
        ]
        alt_names["argPeri"] = [
            "argPeri",
            "argperi",
            "omega/argperi",
            "w",
            "argperi(deg)",
            "peri",
        ]
        alt_names["tPeri"] = ["tPeri", "t_p", "timeperi", "t_peri", "T_peri"]
        alt_names["epoch"] = ["epoch", "t_0", "Epoch", "epoch_mjd"]
        alt_names["H"] = ["H", "magH", "magHv", "Hv", "H_v"]
        alt_names["g"] = ["g", "phaseV", "phase", "gV", "phase_g", "G"]
        alt_names["meanAnomaly"] = ["meanAnomaly", "meanAnom", "M", "ma"]
        alt_names["sed_filename"] = ["sed_filename", "sed"]
        alt_names["xdot"] = ["xdot", "xDot"]
        alt_names["ydot"] = ["ydot", "yDot"]
        alt_names["zdot"] = ["zdot", "zDot"]

        # Update column names that match any of the alternatives above.
        for name, alternatives in alt_names.items():
            intersection = list(set(alternatives) & set(sso_cols))
            if len(intersection) > 1:
                raise ValueError(
                    "Received too many possible matches to %s in orbit file %s" % (name, orbit_file)
                )
            if len(intersection) == 1:
                idx = sso_cols.index(intersection[0])
                sso_cols[idx] = name
        # Assign the new column names back to the orbits dataframe.
        orbits.columns = sso_cols

        # Failing on negative inclinations.
        if "inc" in orbits.keys():
            if np.min(orbits["inc"]) < 0:
                negative_incs = np.where(orbits["inc"].values < 0)[0]
                negative_ids = orbits["obj_id"].values[negative_incs]
                ValueError("Negative orbital inclinations not supported. Problem obj_ids=%s" % negative_ids)

        # Validate and assign orbits to self.
        self.set_orbits(orbits)

    def update_orbits(self, neworb):
        """Update existing orbits with new values,
        leaving OrbitIds, H, g, and sed_filenames in place.

        Example use: transform orbital parameters (using PyOrbEphemerides)
        and then replace original values.
        Example use: propagate orbital parameters (using PyOrbEphemerides)
        and then replace original values.

        Parameters
        ----------
        neworb: `pd.DataFrame`
        """
        col_orig = ["obj_id", "sed_filename"]
        new_order = ["obj_id"] + [n for n in neworb.columns] + ["sed_filename"]
        updated_orbits = neworb.join(self.orbits[col_orig])[new_order]
        self.set_orbits(updated_orbits)
