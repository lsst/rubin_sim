# Tuples of RA,dec in degrees
elaiss1 = (9.45, -44.0)
XMM_LSS = (35.708333, -4 - 45 / 60.0)
ECDFS = (53.125, -28.0 - 6 / 60.0)
COSMOS = (150.1, 2.0 + 10.0 / 60.0 + 55 / 3600.0)
edfs_a = (58.90, -49.315)
edfs_b = (63.6, -47.60)


def ddf_locations():
    """Return the DDF locations as as dict. RA and dec in degrees."""
    result = {}
    result["elaiss1"] = elaiss1
    result["XMM_LSS"] = XMM_LSS
    result["ECDFS"] = ECDFS
    result["COSMOS"] = COSMOS
    result["edfs_a"] = edfs_a
    result["edfs_b"] = edfs_b

    return result
