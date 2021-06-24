

# Tuples of RA,dec in degrees
ELAISS1 = (9.45, -44.)
XMM_LSS = (35.708333, -4-45/60.)
ECDFS = (53.125, -28.-6/60.)
COSMOS = (150.1, 2.+10./60.+55/3600.)
EDFS_a = (58.90, -49.315)
EDFS_b = (63.6, -47.60)


def ddf_locations():
    """Return the DDF locations as as dict. RA and dec in degrees.
    """
    result = {}
    result['ELAISS1'] = ELAISS1
    result['XMM_LSS'] = XMM_LSS
    result['ECDFS'] = ECDFS
    result['COSMOS'] = COSMOS
    result['EDFS_a'] = EDFS_a
    result['EDFS_b'] = EDFS_b

    return result
