import glob
import os

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.time import Time
from rubin_scheduler.data import get_data_dir
from sbpy.data import Orbit

from rubin_sim.phot_utils import Bandpass, Sed


def read_orbits(orbit_file, delim=None, skiprows=None, orb_format=None):
    """breaking out from rubin_sim.moving_objects.Orbit"""

    names = None

    data_cols = {}
    data_cols["COM"] = [
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
    data_cols["KEP"] = [
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
    data_cols["CAR"] = [
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
            raise ValueError("Received too many possible matches to %s in orbit file %s" % (name, orbit_file))
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

    # Validate and assign orbits
    if "index" in orbits:
        del orbits["index"]

    n_sso = len(orbits)

    # Error if orbits is empty
    # (this avoids hard-to-interpret error messages from pyoorb).
    if n_sso == 0:
        raise ValueError("Length of the orbits dataframe was 0.")

    # Discover which type of orbital parameters we have on disk.
    orb_format = None
    if "FORMAT" in orbits:
        if ~(orbits["FORMAT"] == orbits["FORMAT"].iloc[0]).all():
            raise ValueError("All orbital elements in the set should have the same FORMAT.")
        orb_format = orbits["FORMAT"].iloc[0]
        # Backwards compatibility .. a bit.
        # CART is deprecated, so swap it to CAR.
        if orb_format == "CART":
            orb_format = "CAR"
        del orbits["FORMAT"]
        # Check that the orbit format is approximately right.
        if orb_format == "COM":
            if "q" not in orbits:
                raise ValueError('The stated format was COM, but "q" not present in orbital elements?')
        if orb_format == "KEP":
            if "a" not in orbits:
                raise ValueError('The stated format was KEP, but "a" not present in orbital elements?')
        if orb_format == "CAR":
            if "x" not in orbits:
                raise ValueError('The stated format was CAR but "x" not present in orbital elements?')
    if orb_format is None:
        # Try to figure out the format, if it wasn't provided.
        if "q" in orbits:
            orb_format = "COM"
        elif "a" in orbits:
            orb_format = "KEP"
        elif "x" in orbits:
            orb_format = "CAR"
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
    # if "sed_filename" not in orbits:
    #    orbits = orbits.assign(sed_filename=self.assign_sed(orbits))

    # Make sure we gave all the columns we need.
    for col in data_cols[orb_format]:
        if col not in orbits:
            raise ValueError(
                "Missing required orbital element %s for orbital format type %s" % (col, orb_format)
            )

    return orbits


def read_filters(
    filter_dir=None,
    bandpass_root="total_",
    bandpass_suffix=".dat",
    filterlist=("u", "g", "r", "i", "z", "y"),
    v_dir=None,
    v_filter="harris_V.dat",
):
    """Read (LSST) and Harris (V) filter throughput curves.

    Only the defaults are LSST specific;
    this can easily be adapted for any survey.

    Parameters
    ----------
    filter_dir : `str`, optional
        Directory containing the filter throughput curves ('total*.dat')
        Default set by 'LSST_THROUGHPUTS_BASELINE' env variable.
    bandpass_root : `str`, optional
        Rootname of the throughput curves in filterlist.
        E.g. throughput curve names are bandpass_root + filterlist[i]
        + bandpass_suffix
        Default `total_` (appropriate for LSST throughput repo).
    bandpass_suffix : `str`, optional
        Suffix for the throughput curves in filterlist.
        Default '.dat' (appropriate for LSST throughput repo).
    filterlist : `list`, optional
        List containing the filter names to use to calculate colors.
        Default ('u', 'g', 'r', 'i', 'z', 'y')
    v_dir : `str`, optional
        Directory containing the V band throughput curve.
        Default None = $RUBIN_SIM_DATA_DIR/movingObjects
    v_filter : `str`, optional
        Name of the V band filter curve.
        Default harris_V.dat.
    """
    if filter_dir is None:
        filter_dir = os.path.join(get_data_dir(), "throughputs/baseline")
    if v_dir is None:
        v_dir = os.path.join(get_data_dir(), "movingObjects")
    # Read filter throughput curves from disk.
    bps = {}
    for f in filterlist:
        bps[f] = Bandpass()
        bps[f].read_throughput(os.path.join(filter_dir, bandpass_root + f + bandpass_suffix))
    return bps


def calc_colors(
    bps, f1=["u", "g", "i", "z", "y"], f2=["r", "r", "r", "r", "r"], sedname="C.dat", sed_dir=None
):
    """Calculate the colors for a given SED.

    If the sedname is not already in the dictionary self.colors,
    this reads the SED from disk and calculates all V-[filter] colors
    for all filters in self.filterlist.
    The result is stored in self.colors[sedname][filter], so will not
    be recalculated if the SED + color is reused for another object.

    Parameters
    ----------
    sedname : `str`, optional
        Name of the SED. Default 'C.dat'.
    sed_dir : `str`, optional
        Directory containing the SEDs of the moving objects.
        Default None = $RUBIN_SIM_DATA_DIR/movingObjects,

    Returns
    -------
    colors : `dict` {'filter': color}}
        Dictionary of the colors in self.filterlist.
    """

    if sed_dir is None:
        sed_dir = os.path.join(get_data_dir(), "movingObjects")
    mo_sed = Sed()
    mo_sed.read_sed_flambda(os.path.join(sed_dir, sedname))

    result = {}

    for filtername1, filtername2 in zip(f1, f2):
        result["%s-%s" % (filtername1, filtername2)] = mo_sed.calc_mag(bps[filtername1]) - mo_sed.calc_mag(
            bps[filtername2]
        )

    return result


if __name__ == "__main__":

    orbit_files = glob.glob(os.path.join(get_data_dir(), "orbits") + "/*.txt")

    out_path = os.path.join(get_data_dir(), "sorcha")

    units = {"a": u.au, "incl": u.deg, "Omega": u.deg, "w": u.deg, "M": u.deg, "H": u.mag, "q": u.au}

    bps = read_filters()

    for filename in orbit_files:
        rsorb = read_orbits(filename)

        # rename some columns to try and match astropy
        rsorb.rename(
            columns={"inc": "incl", "argPeri": "w", "meanAnomaly": "M", "objId": "targetname", "tPeri": "Tp"},
            inplace=True,
        )
        rsorb["G"] = 0

        table = Table()
        table = table.from_df(rsorb, units=units)
        table["epoch"] = Time(table["epoch"], format="mjd")
        if "Tp" in table.keys():
            table["Tp"] = Time(rsorb["Tp"], format="mjd")
        table["id"] = np.arange(table["epoch"].size)

        orbit = Orbit.from_table(table)

        orbit_kep = orbit.oo_transform("KEP")

        # Catch if openOrb failed silently!
        a_failed_indx = np.where(orbit_kep["a"] == 0)[0]
        if np.size(a_failed_indx) > 0:
            for indx in np.arange(len(orbit_kep)):
                orbit_kep._table[indx] = orbit[indx].oo_transform("KEP")._table[0]

        outfilename = os.path.basename(filename).split(".")[0] + "_kep.csv"
        # orbit_kep.to_file(outfilename, format="ascii.csv", overwrite=True)
        sorcha_style = pd.DataFrame()
        sorcha_style["ObjID"] = np.arange(table["epoch"].size)
        sorcha_style["a"] = orbit_kep["a"].value
        sorcha_style["e"] = orbit_kep["e"].value
        sorcha_style["inc"] = orbit_kep["incl"].value
        sorcha_style["node"] = orbit_kep["Omega"].value
        sorcha_style["argPeri"] = orbit_kep["w"].value
        sorcha_style["ma"] = orbit_kep["M"].value
        sorcha_style["epochMJD_TDB"] = orbit_kep["epoch"].value
        sorcha_style["FORMAT"] = "KEP"

        ack = np.where(sorcha_style["e"] >= 1)[0]
        good_orbits = np.where(sorcha_style["e"] < 1)[0]

        # Going to fudge wacky orbits that have e > 1
        if len(ack) > 0:
            if len(ack) > 1:
                sorcha_style = sorcha_style.iloc[good_orbits]
            else:
                for indx in ack:
                    sorcha_style.loc[indx, "e"] = 0.9999
                    sorcha_style.loc[indx, "ma"] = 359.0

        sorcha_style.to_csv(os.path.join(out_path, outfilename), index=False, sep=" ")

        # Now need to write a params file

        params = pd.DataFrame()
        params["ObjID"] = table["id"]

        params["H_r"] = 4.0
        params["GS"] = 0.15

        sed_names = np.unique(rsorb["sed_filename"])

        params["u-r"] = 0.0
        params["g-r"] = 0.0
        params["i-r"] = 0.0
        params["z-r"] = 0.0
        params["y-r"] = 0.0

        for sed_name in sed_names:
            colors = calc_colors(bps, sedname=sed_name)

            indx = np.where(rsorb["sed_filename"].values == sed_name)[0]

            params.loc[indx, "u-r"] = colors["u-r"]
            params.loc[indx, "g-r"] = colors["g-r"]
            params.loc[indx, "i-r"] = colors["i-r"]
            params.loc[indx, "z-r"] = colors["z-r"]
            params.loc[indx, "y-r"] = colors["y-r"]

        params = params.iloc[good_orbits]
        outfile = os.path.join(out_path, os.path.basename(filename).split(".")[0] + "_param.csv")
        params.to_csv(outfile, index=False, sep=" ")
