import numpy as np
import healpy as hp

from rubin_sim.utils import xyz_from_ra_dec, xyz_angular_radius, \
    _buildTree, _xyz_from_ra_dec
from rubin_sim.site_models import FieldsDatabase

from .baseStacker import BaseStacker

__all__ = ['OpsimFieldStacker', 'WFDlabelStacker']


class OpSimFieldStacker(BaseStacker):
    """Add the fieldId of the closest OpSim field for each RA/Dec pointing.

    Parameters
    ----------
    raCol : str, opt
        Name of the RA column. Default fieldRA.
    decCol : str, opt
        Name of the Dec column. Default fieldDec.

    """
    colsAdded = ['opsimFieldId']

    def __init__(self, raCol='fieldRA', decCol='fieldDec', degrees=True):
        self.colsReq = [raCol, decCol]
        self.units = ['#']
        self.raCol = raCol
        self.decCol = decCol
        self.degrees = degrees
        fields_db = FieldsDatabase()
        # Returned RA/Dec coordinates in degrees
        fieldid, ra, dec = fields_db.get_id_ra_dec_arrays("select * from Field;")
        asort = np.argsort(fieldid)
        self.tree = _buildTree(np.radians(ra[asort]),
                               np.radians(dec[asort]))

    def _run(self, simData, cols_present=False):
        if cols_present:
            # Column already present in data; assume it is correct and does not need recalculating.
            return simData

        if self.degrees:
            coord_x, coord_y, coord_z = xyz_from_ra_dec(simData[self.raCol],
                                                        simData[self.decCol])
            field_ids = self.tree.query_ball_point(list(zip(coord_x, coord_y, coord_z)),
                                                   xyz_angular_radius())

        else:
            # use _xyz private method (sending radians)
            coord_x, coord_y, coord_z = _xyz_from_ra_dec(simData[self.raCol],
                                                         simData[self.decCol])
            field_ids = self.tree.query_ball_point(list(zip(coord_x, coord_y, coord_z)),
                                                   xyz_angular_radius())

        simData['opsimFieldId'] = np.array([ids[0] for ids in field_ids]) + 1
        return simData


class WFDlabelStacker(BaseStacker):
    """Add a single new column 'WFD' which flags whether a visit is in the hp_footprint
    (and not tagged as a DD visit).  Calculate hp_footprint to set the WFD footprint.
    """
    colsAdded = ['proposalID']

    def __init__(self, hp_footprint):
        self.colsRequired = ['note', 'fieldRA', 'fieldDec']
        self.colsAddedDtypes = [int]
        self.units = [None]
        self.footprint = hp_footprint
        self.nside = hp.nside2npix(len(self.footprint))

    def _run(self, simData, cols_present=False):
        # Set up DD names.
        d = set()
        for p in simData['note'].unique():
            if p.startswith('DD'):
                d.add(define_ddname(p))
        # Define dictionary of proposal tags.
        propTags = {'Other': 0, 'WFD': 1}
        for i, field in enumerate(d):
            propTags[field] = i + 2
        # Identify Healpixels associated with each visit.
        vec = hp.dir2vec(simData['fieldRA'], so,Data['fieldDec'], lonlat=True)
        vec = vec.swapaxes(0, 1)
        radius = np.radians(1.75)  # fov radius
        propId = np.zeros(len(simData), int)
        for i, (v, note) in enumerate(zip(vec, simData['note'])):
            # Identify the healpixels which would be inside this pointing
            pointing_healpix = hp.query_disc(nside, v, radius, inclusive=False)
            # The wfd_footprint consists of values of 0/1 if out/in WFD footprint
            in_wfd = self.footprint[pointing_healpix].sum()
            # So in_wfd = the number of healpixels which were in the WFD footprint
            # .. in the # in / total # > limit (0.4) then "yes" it's in WFD
            propId[i] = np.where(in_wfd / len(pointing_healpix) > 0.4, propTags['WFD'], 0)
            # BUT override - if the visit was taken for DD, use that flag instead.
            if note.startswith('DD'):
                propId[i] = propTags[define_ddname(note)]
        simData['proposalID'] = propId
        return simData