import os
import numpy as np
from scipy import interpolate
import healpy as hp

from rubin_sim.data import get_data_dir
from .baseMetric import BaseMetric
from .exgalM5 import ExgalM5

__all__ = ['QSONumberCountsMetric']

class QSONumberCountsMetric(BaseMetric):
    """
    Calculate the number of quasars expected with SNR>=5 according to the Shen et al. (2020) QLF -
    model A in the redshift range zmin < z < zmax. The 5 sigma depths are obtained using the ExgalM5 metric.
    Only quasars fainter than the saturation magnitude are counted.

    By default, zmin is 0.3 and zmax is the minimum between 6.7 and the redshift at which the Lyman break
    matches the effective wavelength of the band. For bands izy, zmax is 6.7. This default choice is to
    match Table 10.2 for i-band quasar counts in the LSST Science book.
    """
    def __init__(self, lsstFilter, m5Col='fiveSigmaDepth', units='mag', extinction_cut=1.0,
                 filterCol='filter', metricName='QSONumberCountsMetric', qlf_module='Shen20',
                 qlf_model='A', SED_model="Richards06", zmin=0.3, zmax=None, **kwargs):
        #Declare the effective wavelengths. 
        self.effwavelen = {'u':367.0, 'g':482.5, 'r':622.2, 'i':754.5, 'z':869.1, 'y':971.0}

        #Dust Extinction limit. Regions with larger extinction and dropped from the counting.
        self.extinction_cut = extinction_cut

        #Save the filter information.
        self.filterCol = filterCol
        self.lsstFilter = lsstFilter

        #Save zmin and zmax, or set zmax to the default value.
        # The default zmax is the lower number between 6.7 and the redshift at which the
        # Lyman break (91.2nm) hits the effective wavelength of the filter.
        # Note that this means that for i, z and y the default value for zmax is 6.7
        self.zmin = zmin
        if zmax is None:
            zmax = np.min([6.7, self.effwavelen[self.lsstFilter]/91.2-1.0])
        self.zmax = zmax

        #This calculation uses the ExgalM5 metric. So declare that here.
        self.exgalM5 = ExgalM5(m5Col=m5Col, units=units)

        #Save the input parameters that relate to the QLF model. 
        self.qlf_module = qlf_module
        self.qlf_model  = qlf_model
        self.SED_model  = SED_model
        
        #Read the long tables, which the number of quasars expected for a given band,
        # qlf_module and qlf_model in a range of redshifts and magnitudes.
        table_name = "Long_Table.LSST{0}.{1}.{2}.{3}.txt".format(self.lsstFilter,
                                                                 self.qlf_module,
                                                                 self.qlf_model,
                                                                 self.SED_model)
        data_dir = os.path.join(get_data_dir(), 'maf', 'quasarNumberCounts')
        filename = os.path.join(data_dir, table_name)
        with open(filename, 'r') as f:
            mags = np.array([float(x) for x in f.readline().split()])
            zs = np.array([float(x) for x in f.readline().split()])
        mz_data = np.loadtxt(filename, skiprows=2)

        #Make the long table cumulative.
        c_mz_data = np.zeros((mz_data.shape[0]+1, mz_data.shape[1]+1))
        c_mz_data[1:,1:] = mz_data
        c_mz_data = np.cumsum(c_mz_data, axis=0)
        c_mz_data = np.cumsum(c_mz_data, axis=1)        
        
        #Create a 2D interpolation object for the long table. 
        self.Nqso_cumulative = interpolate.interp2d(zs[:-1], mags[:-1], c_mz_data[:-1,:-1], kind='cubic')

        super().__init__(col=[m5Col, filterCol, 'saturation_mag'],
                         metricName=metricName,
                         maps=self.exgalM5.maps,
                         units=units, **kwargs)

    def run(self, dataSlice, slicePoint=None):
        # exclude areas with high extinction
        if slicePoint['ebv'] > self.extinction_cut:
            return self.badval
        
        # For the dataslice, get the 5 sigma limiting magnitude.
        dS = dataSlice[dataSlice[self.filterCol] == self.lsstFilter]
        mlim5 = self.exgalM5.run(dS, slicePoint)
        
        # Get the slicer pixel area.
        nside = slicePoint['nside']
        pix_area = hp.nside2pixarea(nside, degrees=True)

        # tranform that limiting magnitude into an expected number of quasars.
        # If there is more than one, take the faintest.
        m_bright = np.max(dS['saturation_mag'])
        N11 = self.Nqso_cumulative(self.zmin, m_bright)
        N12 = self.Nqso_cumulative(self.zmin, mlim5 )
        N21 = self.Nqso_cumulative(self.zmax, m_bright)
        N22 = self.Nqso_cumulative(self.zmax, mlim5 )

        Nqso =  (N22 - N21 - N12 + N11)*pix_area
        return Nqso