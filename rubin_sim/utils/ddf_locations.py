__all__ = ("ddf_locations",)

# Tuples of RA,dec in degrees
ELAISS1 = (9.45, -44.0)
XMM_LSS = (35.708333, -4 - 45 / 60.0)
ECDFS = (53.125, -28.0 - 6 / 60.0)
COSMOS = (150.1, 2.0 + 10.0 / 60.0 + 55 / 3600.0)
EDFS_A = (58.90, -49.315)
EDFS_B = (63.6, -47.60)


def ddf_locations():
    """Return the DDF locations as as dict. RA and dec in degrees."""
    result = {}
    result["ELAISS1"] = ELAISS1
    result["XMM_LSS"] = XMM_LSS
    result["ECDFS"] = ECDFS
    result["COSMOS"] = COSMOS
    result["EDFS_a"] = EDFS_A
    result["EDFS_b"] = EDFS_B

    return result
