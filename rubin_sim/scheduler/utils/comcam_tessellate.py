__all__ = ("comcam_tessellate",)

import numpy as np


def comcam_tessellate(side_length=0.7, overlap=0.11):
    """Tesselate the sphere with a square footprint

    XXX--TODO:  This really sucks at the poles, should add some different pole cap behavior.

    Parameters
    ----------
    side_length : float (0.7)
        The length of a side of the square (degrees)
    overlap : float (0.11)
        How much overlap to have in pointings

    Returns
    -------
    fields : numpy array
       With 'RA' and 'dec' keys that have the field positions in radians
    """

    # Convert to radians for all internal work
    side_length = np.radians(side_length)
    overlap = np.radians(overlap)

    rotation_step = side_length / 2.0

    step_size = side_length - overlap
    n_dec_steps = np.ceil(np.pi / 2.0 / step_size)
    dec_stripes = np.linspace(0, np.pi / 2.0, n_dec_steps)

    # Lists to hold the RA and dec coords
    ras = []
    decs = []
    running_step = 0.0
    test_decs = np.array([-step_size, 0.0, step_size])
    for dec in dec_stripes:
        # Find the largest circumfrance the squares will have to cover
        circum = np.max(2.0 * np.pi * np.cos(test_decs + dec))
        n_ra_steps = np.ceil(circum / step_size)
        new_ras = np.linspace(0, 2.0 * np.pi, n_ra_steps)
        running_step += rotation_step / np.cos(dec)
        new_ras = (new_ras + running_step) % (2.0 * np.pi)
        ras.extend(new_ras.tolist())
        new_decs = np.empty(new_ras.size)
        new_decs.fill(dec)
        decs.extend(new_decs.tolist())

    # That was the northern hemisphere, copy to do the south
    north = np.where(np.array(decs) > 0)
    decs.extend((-1.0 * np.array(decs))[north].tolist())
    new_ras = -1.0 * np.array(ras)[north]
    new_ras = new_ras % (2.0 * np.pi)
    ras.extend(new_ras.tolist())

    names = ["RA", "dec"]
    types = [float, float]
    fields = np.empty(len(ras), dtype=list(zip(names, types)))
    fields["RA"] = np.array(ras)
    fields["dec"] = np.array(decs)

    return fields
