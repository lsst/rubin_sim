import numpy as np
from .healpixSlicer import HealpixSlicer
from functools import wraps
import matplotlib.path as mplPath
from rubin_sim.maf.utils.mafUtils import gnomonic_project_toxy
from rubin_sim.maf.plots import HealpixSDSSSkyMap

__all__ = ['HealpixSDSSSlicer']

class HealpixSDSSSlicer(HealpixSlicer):
    """For use with SDSS stripe 82 square images """
    def __init__(self, nside=128, lonCol ='RA1' , latCol='Dec1', verbose=True,
                 useCache=True, radius=17./60., leafsize=100, **kwargs):
        """Using one corner of the chip as the spatial key and the diagonal as the radius.  """
        super(HealpixSDSSSlicer,self).__init__(verbose=verbose,
                                            lonCol=lonCol, latCol=latCol,
                                            radius=radius, leafsize=leafsize,
                                            useCache=useCache,nside=nside )
        self.cornerLables = ['RA1', 'Dec1', 'RA2','Dec2','RA3','Dec3','RA4','Dec4']
        self.plotFuncs = [HealpixSDSSSkyMap,]

    def setupSlicer(self, simData, maps=None):
        """
        Use simData[self.lonCol] and simData[self.latCol]
        (in radians) to set up KDTree.
        """
        self._runMaps(maps)
        self._buildTree(simData[self.lonCol], simData[self.latCol], self.leafsize)
        self._setRad(self.radius)
        self.corners = simData[self.cornerLables]
        @wraps(self._sliceSimData)
        def _sliceSimData(islice):
            """Return indexes for relevant opsim data at slicepoint
            (slicepoint=lonCol/latCol value .. usually ra/dec)."""
            sx, sy, sz = self._treexyz(self.slicePoints['ra'][islice], self.slicePoints['dec'][islice])
            # Query against tree.
            initIndices = self.opsimtree.query_ball_point((sx, sy, sz), self.rad)
            # Loop through all the images and check if the slicepoint is inside the corners of the chip
            # XXX--should check if there's a better/faster way to do this.
            # Maybe in the setupSlicer loop through each image, and use the contains_points method to test all the
            # healpixels simultaneously?  Then just have a dict with keys = healpix id and values = list of indices?
            # That way _sliceSimData is just doing a dict look-up and we can get rid of the spatialkey kwargs.


            indices=[]
            # Gnomic project all the corners that are near the slice point, centered on slice point
            x1,y1 = gnomonic_project_toxy(self.corners['RA1'][initIndices], self.corners['Dec1'][initIndices],
                                          self.slicePoints['ra'][islice], self.slicePoints['dec'][islice])
            x2,y2 = gnomonic_project_toxy(self.corners['RA2'][initIndices], self.corners['Dec2'][initIndices],
                                          self.slicePoints['ra'][islice], self.slicePoints['dec'][islice])
            x3,y3 = gnomonic_project_toxy(self.corners['RA3'][initIndices], self.corners['Dec3'][initIndices],
                                          self.slicePoints['ra'][islice], self.slicePoints['dec'][islice])
            x4,y4 = gnomonic_project_toxy(self.corners['RA4'][initIndices], self.corners['Dec4'][initIndices],
                                          self.slicePoints['ra'][islice], self.slicePoints['dec'][islice])

            for i,ind in enumerate(initIndices):
                # Use matplotlib to make a polygon on
                bbPath = mplPath.Path(np.array([[x1[i], y1[i]],
                                                [x2[i], y2[i]],
                                                [x3[i], y3[i]],
                                                [x4[i], y4[i]],
                                                [x1[i], y1[i]]] ))
                # Check if the slicepoint is inside the image corners and append to list if it is
                if bbPath.contains_point((0.,0.)) == 1:
                    indices.append(ind)

            return {'idxs':indices,
                    'slicePoint':{'sid':self.slicePoints['sid'][islice],
                                  'ra':self.slicePoints['ra'][islice],
                                  'dec':self.slicePoints['dec'][islice]}}
        setattr(self, '_sliceSimData', _sliceSimData)
