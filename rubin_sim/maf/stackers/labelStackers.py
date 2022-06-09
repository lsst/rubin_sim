import numpy as np
import healpy as hp

from rubin_sim.utils import (
    xyz_from_ra_dec,
    xyz_angular_radius,
    _buildTree,
    _xyz_from_ra_dec,
)
from rubin_sim.site_models import FieldsDatabase

from .baseStacker import BaseStacker

__all__ = ["WFDlabelStacker"]


class WFDlabelStacker(BaseStacker):
    """Add an 'areaId' column to flag whether a visit was inside the 'footprint'.

    Parameters
    ----------
    footprint : `np.NDarray`, optional
        The healpix map indicating the desired footprint region.
        If this is not defined (default None), then the entire sky is used as the footprint.
    fp_threshold : `float`, optional
        The threshold for the fraction of the visit area which falls within the footprint in order
        to be counted as 'in' the footprint. Default 0.4.
    areaIdName : `str`, optional
        Value to place in the areaId column for visits which match this area.
    raCol : `str`, optional
        The name of the RA column. Default fieldRA.
    decCol : `str`, optional
        The name of the Dec column. Default fieldDec.
    noteCol : `str`, optional
        The name of the 'note' column in the database. Default 'note'. This is used to identify visits
        which were part of a DD sequence.
    excludeDD : `bool`, optional
        Exclude (True) or include (False) visits which are part of a DD sequence within this 'area'.

    This stacker adds an areaId column in the opsim database, to be labelled with 'areaIdName' if the
    visit falls within the healpix footprint map and (optionally) is not tagged as a DD visit.
    If it falls outside the footprint, the visit is tagged as "NULL".
    If it was part of a DD sequence, the visit is tagged with an ID which is unique to that DD field,
    if 'excludeDD' is True.
    Generally this would be likely to be used to tag visits as belonging to WFD - but not necessarily!
    Any healpix footprint is valid.
    """

    colsAdded = ["areaId"]

    def __init__(
        self,
        footprint=None,
        fp_threshold=0.4,
        area_id_name="WFD",
        raCol="fieldRA",
        decCol="fieldDec",
        noteCol="note",
        excludeDD=True,
    ):
        self.raCol = raCol
        self.decCol = decCol
        self.noteCol = noteCol
        self.colsReq = [self.raCol, self.decCol, self.noteCol]
        self.colsAddedDtypes = [(str, 15)]
        self.units = [""]
        self.fp_threshold = fp_threshold
        self.area_id_name = area_id_name
        self.excludeDD = excludeDD
        if footprint is None:
            # If footprint was not defined, just set it to cover the entire sky, at nside=64
            footprint = np.ones(hp.nside2npix(64))
        self.footprint = footprint
        self.nside = hp.npix2nside(len(self.footprint))

    def define_ddname(self, note):
        field = note.replace("u,", "")
        field = field.split(",")[0].replace(",", "")
        return field

    def _run(self, simData, cols_present=False):
        # Even if cols_present is true, recalculate.
        # Set up DD names.
        d = set()
        for p in np.unique(simData[self.noteCol]):
            if p.startswith("DD"):
                d.add(self.define_ddname(p))
        # Identify Healpixels associated with each visit.
        vec = hp.dir2vec(simData[self.raCol], simData[self.decCol], lonlat=True)
        vec = vec.swapaxes(0, 1)
        radius = np.radians(1.75)  # fov radius
        areaId = np.zeros(len(simData), self.colsAddedDtypes[0])
        for i, (v, note) in enumerate(zip(vec, simData[self.noteCol])):
            # Identify the healpixels which would be inside this pointing
            pointing_healpix = hp.query_disc(self.nside, v, radius, inclusive=False)
            # The wfd_footprint consists of values of 0/1 if out/in WFD footprint
            hp_in_fp = self.footprint[pointing_healpix].sum()
            # So in_fp= the number of healpixels which were in the specified footprint
            # .. in the # in / total # > limit (0.4) then "yes" it's in the footprint
            in_fp = hp_in_fp / len(pointing_healpix)
            if note.startswith("DD") and self.excludeDD:
                areaId[i] = self.define_ddname(note)
            else:
                if in_fp >= self.fp_threshold:
                    areaId[i] = self.area_id_name
                else:
                    areaId[i] = "NULL"
        simData["areaId"] = areaId
        return simData
