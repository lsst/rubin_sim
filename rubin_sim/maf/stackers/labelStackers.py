import numpy as np
import healpy as hp

from rubin_sim.utils import xyz_from_ra_dec, xyz_angular_radius, \
    _buildTree, _xyz_from_ra_dec
from rubin_sim.site_models import FieldsDatabase

from .baseStacker import BaseStacker

__all__ = ['OpSimFieldStacker', 'WFDlabelStacker']


class OpSimFieldStacker(BaseStacker):
    """Add the fieldId of the closest OpSim field for each RA/Dec pointing.

    Parameters
    ----------
    raCol : str, optional
        Name of the RA column. Default fieldRA.
    decCol : str, optional
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
    """Modify the 'proposalId' column to flag whether a visit was inside the 'footprint'.

    Parameters
    ----------
    footprint: np.NDarray, optional
        The healpix map indicating the desired footprint region.
        If this is not defined (default None), then the entire sky is used as the footprint.
    fp_threshold: float, optional
        The threshold for the fraction of the visit area which falls within the footprint in order
        to be counted as 'in' the footprint. Default 0.4.
    raCol: str, optional
        The name of the RA column. Default fieldRA.
    decCol: str, optional
        The name of the Dec column. Default fieldDec.
    noteCol: str, optional
        The name of the 'note' column in the database. Default 'note'. This is used to identify visits
        which were part of a DD sequence.

    This stacker modifies the proposalID column in the opsim database, to be labelled with '1' if the
    visit falls within the healpix footprint map and is not tagged as a DD visit. If it falls outside
    the footprint, the visit is tagged as '0'. If it was part of a DD sequence, the visit is tagged with
    an ID which is unique to that DD field.
    Generally this would be likely to be used to tag visits as belonging to WFD - but not necessarily!
    Any healpix footprint is valid.
    """
    colsAdded = ['proposalId']

    def __init__(self, footprint=None, fp_threshold=0.4,
                 raCol='fieldRA', decCol='fieldDec', noteCol='note'):
        self.raCol = raCol
        self.decCol = decCol
        self.noteCol = noteCol
        self.colsRequired = [self.raCol, self.decCol, self.noteCol]
        self.colsAddedDtypes = [int]
        self.units = [None]
        self.fp_threshold = fp_threshold
        if footprint is None:
            # If footprint was not defined, just set it to cover the entire sky, at nside=64
            footprint = np.ones(hp.nside2npix(64))
        self.footprint = footprint
        self.nside = hp.npix2nside(len(self.footprint))

    def define_ddname(self, note):
        field = note.replace('u,', '')
        field = field.split(',')[0].replace(',', '')
        return field

    def _run(self, simData, cols_present=False):
        # Even if cols_present is true, recalculate.
        # Set up DD names.
        d = set()
        for p in np.unique(simData[self.noteCol]):
            if p.startswith('DD'):
                d.add(self.define_ddname(p))
        # Define dictionary of proposal tags.
        propTags = {'Other': 0, 'WFD': 1}
        for i, field in enumerate(d):
            propTags[field] = i + 2
        # Identify Healpixels associated with each visit.
        vec = hp.dir2vec(simData[self.raCol], simData[self.decCol], lonlat=True)
        vec = vec.swapaxes(0, 1)
        radius = np.radians(1.75)  # fov radius
        propId = np.zeros(len(simData), int)
        for i, (v, note) in enumerate(zip(vec, simData[self.noteCol])):
            # Identify the healpixels which would be inside this pointing
            pointing_healpix = hp.query_disc(self.nside, v, radius, inclusive=False)
            # The wfd_footprint consists of values of 0/1 if out/in WFD footprint
            hp_in_fp = self.footprint[pointing_healpix].sum()
            # So in_fp= the number of healpixels which were in the specified footprint
            # .. in the # in / total # > limit (0.4) then "yes" it's in the footprint
            in_fp = hp_in_fp / len(pointing_healpix)
            if note.startswith('DD'):
                propId[i] = propTags[self.define_ddname(note)]
            else:
                if in_fp >= self.fp_threshold:
                    propId[i] = propTags['WFD']
                else:
                    propId[i] = propTags['Other']
        simData['proposalId'] = propId
        return simData