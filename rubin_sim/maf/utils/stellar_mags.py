__all__ = ("stellar_mags",)

import numpy as np


def calc_wd_colors():
    """
    Calculate a few example WD colors. Values to go in stellar_mags().
    Here in case values need to be regenerated
    (different stars, bandpasses change, etc.)
    """

    try:
        import os

        from lsst.utils import getPackageDir

        from rubin_sim.photUtils import Bandpass, Sed
    except ImportError:
        "Need to setup sims_photUtils to generate WD magnitudes."

    names = ["HeWD_25200_80", "WD_11000_85", "WD_3000_85"]
    fns = [
        "bergeron_He_24000_80.dat_25200.gz",
        "bergeron_10500_85.dat_11000.gz",
        "bergeron_2750_85.dat_3000.gz",
    ]
    wd_dir = os.path.join(getPackageDir("sims_sed_library"), "starSED/wDs/")
    files = [os.path.join(wd_dir, filename) for filename in fns]

    # Read in the LSST bandpasses
    bp_names = ["u", "g", "r", "i", "z", "y"]
    bps = []
    through_path = os.path.join(getPackageDir("throughputs"), "baseline")
    for key in bp_names:
        bp = np.loadtxt(
            os.path.join(through_path, "filter_" + key + ".dat"),
            dtype=list(zip(["wave", "trans"], [float] * 2)),
        )
        temp_b = Bandpass()
        temp_b.setBandpass(bp["wave"], bp["trans"])
        bps.append(temp_b)

    # Read in the SEDs and compute mags
    mags = []
    for filename in files:
        star = Sed()
        star.readSED_flambda(filename)
        single_mags = [star.calcMag(band) for band in bps]
        mags.append([single_mags[i - 1] - single_mags[i] for i in range(1, 6)])

    for maglist, fn, name in zip(mags, fns, names):
        format = (name, fn) + tuple(maglist)
        print("['%s', '%s', %f, %f, %f, %f, %f]" % format)


def stellar_mags(stellar_type, rmag=19.0):
    """
    Calculates the expected magnitudes in LSST filters for a
    typical star of the given spectral type.

    Based on mapping of Kuruz models to spectral types here:
    http://www.stsci.edu/hst/observatory/crds/k93models.html


    Parameters
    ----------
    stellar_type : str
        Spectral type of a star (O,B,A,F,G,K,M), or for white dwarf colors,
        one of 'HeWD_25200_80, 'WD_11000_85', 'WD_3000_85'
    rmag : float
        The expected r-band magnitude of the star.

    Returns
    -------
    dict of floats
        The expected magnitudes in LSST filters.
    """

    # If this is the first time running the function, set up the data array
    if not hasattr(stellar_mags, "data"):
        names = ["stellar_type", "Model Name", "u-g", "g-r", "r-i", "i-z", "z-y"]
        types = [("U", 20), ("U", 35), float, float, float, float, float]
        data = np.core.records.fromrecords(
            [
                (
                    "O",
                    "kp00_50000[g50]",
                    -0.4835688497,
                    -0.5201721327,
                    -0.3991733698,
                    -0.3106800468,
                    -0.2072290744,
                ),
                (
                    "B",
                    "kp00_30000[g40]",
                    -0.3457202828,
                    -0.4834762052,
                    -0.3812792176,
                    -0.2906072887,
                    -0.1927230035,
                ),
                (
                    "A",
                    "kp00_9500[g40]",
                    0.8823182684,
                    -0.237288029,
                    -0.2280783991,
                    -0.1587960264,
                    -0.03043824335,
                ),
                (
                    "F",
                    "kp00_7250[g45]",
                    0.9140316091,
                    0.1254277486,
                    -0.03419150003,
                    -0.0802010739,
                    -0.03802756413,
                ),
                (
                    "G",
                    "kp00_6000[g45]",
                    1.198219095,
                    0.3915608688,
                    0.09129426676,
                    0.002604263747,
                    -0.004659443668,
                ),
                (
                    "K",
                    "kp00_5250[g45]",
                    1.716635024,
                    0.6081567546,
                    0.1796910856,
                    0.06492278686,
                    0.0425155827,
                ),
                (
                    "M",
                    "kp00_3750[g45]",
                    2.747842719,
                    1.287599638,
                    0.5375622482,
                    0.4313486709,
                    0.219308065,
                ),
                (
                    "HeWD_25200_80",
                    "bergeron_He_24000_80.dat_25200.gz",
                    -0.218959,
                    -0.388374,
                    -0.326946,
                    -0.253573,
                    -0.239460,
                ),
                (
                    "WD_11000_85",
                    "bergeron_10500_85.dat_11000.gz",
                    0.286146,
                    -0.109115,
                    -0.178500,
                    -0.185833,
                    -0.186913,
                ),
                (
                    "WD_3000_85",
                    "bergeron_2750_85.dat_3000.gz",
                    3.170620,
                    1.400062,
                    0.167195,
                    0.127024,
                    -0.378069,
                ),
            ],
            dtype=list(zip(names, types)),
        )
        # Switch to a dict for faster look-up
        stellar_mags.data = {}
        for row in data:
            stellar_mags.data["%s" % row["stellar_type"]] = row

    results = {}
    # good = np.where(stellar_mags.data['stellar_type'] == stellar_type)
    if stellar_type not in stellar_mags.data:
        message = "Received stellar_type %s" % stellar_type
        message += " but expected one of %s" % ", ".join(stellar_mags.data.keys())
        raise ValueError(message)

    results["r"] = rmag
    results["i"] = rmag - stellar_mags.data[stellar_type]["r-i"]
    results["z"] = results["i"] - stellar_mags.data[stellar_type]["i-z"]
    results["y"] = results["z"] - stellar_mags.data[stellar_type]["z-y"]
    results["g"] = stellar_mags.data[stellar_type]["g-r"] + results["r"]
    results["u"] = stellar_mags.data[stellar_type]["u-g"] + results["g"]
    return results
