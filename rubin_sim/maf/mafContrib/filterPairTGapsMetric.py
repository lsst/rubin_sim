import numpy as np
#from lsst.sims.maf.utils import radec2pix

# if rubin_sim installed
from rubin_sim.maf.utils.mafUtils import radec2pix
import rubin_sim.maf.metrics as metrics

class filterPairTGapsMetric(metrics.BaseMetric):
    """
    Parameters:
        colname: list, ['observationStartMJD', 'filter', 'fiveSigmaDepth']
        fltpairs: filter pair, default ['uu', 'ug', 'ur', 'ui', 'uz','uy',
                                        'gg', 'gr', 'gi', 'gz', 'gy',
                                        'rr', 'ri', 'rz', 'ry', 
                                        'ii', 'iz', 'iy', 
                                        'zz', 'zy',
                                        'yy']
        mag_lim: list, fiveSigmaDepth threshold each filter, default {'u':18, 'g':18, 'r':18, 'i':18, 'z':18, 'y':18}
        bins_same: np.array, bins to get histogram for same-filter pair ;
        bins_diff: np.array, bins to get histogram for diff-filter pair ;
        save_dT: boolean, save time gaps array as result if True
        allgaps: boolean, all possible pairs if True, else consider only nearest 
        
    Returns:
        result: dictionary, 
        reduce_tgaps: median
        
    """

    def __init__(self, colname=['observationStartMJD', 'filter', 'fiveSigmaDepth'], 
                 fltpairs= ['uu', 'ug', 'ur', 'ui', 'uz','uy', 'gg', 'gr', 'gi', 'gz', 'gy', \
                             'rr', 'ri', 'rz', 'ry', 'ii', 'iz', 'iy',  'zz', 'zy', 'yy'],
                 mag_lim={'u':18, 'g':18, 'r':18, 'i':18, 'z':18, 'y':18}, 
                 bins_same=np.logspace(np.log10( 5/60/60/24), np.log10(3650), 50), 
                 bins_diff=np.logspace(np.log10(5/60/60/24), np.log10(2), 50), 
                 save_dT=False, allgaps=True,  **kwargs):
        
        self.colname = colname
        self.fltpairs = fltpairs
        self.mag_lim = mag_lim
        self.bins_same = bins_same
        self.bins_diff = bins_diff

        self.save_dT = save_dT
        self.allgaps = allgaps
        
        # number of visits to clip, got from average baseline_v2.0 WFD
        self.Nv_clip = {'uu': 30, 'ug': 4, 'ur': 4, 'ui': 4, 'uz': 4, 'uy': 4, \
                        'gg': 69, 'gr': 4, 'gi': 4, 'gz': 4, 'gy': 4, \
                        'rr': 344, 'ri': 4, 'rz': 4, 'ry': 4, \
                        'ii': 355, 'iz': 4, 'iy': 4, \
                        'zz': 282, 'zy': 4, \
                        'yy': 288}
                        
        super().__init__(col=self.colname, **kwargs)
    
    def _get_dT(self, dataSlice, f0, f1):
        
        # select 
        idx0 = ( dataSlice['filter'] == f0 ) & ( dataSlice['fiveSigmaDepth'] > self.mag_lim[f0])
        idx1 = ( dataSlice['filter'] == f1 ) & ( dataSlice['fiveSigmaDepth'] > self.mag_lim[f1])
        
        timeCol0 = dataSlice['observationStartMJD'][idx0]
        timeCol1 = dataSlice['observationStartMJD'][idx1]

        #timeCol0 = timeCol0.reshape((len(timeCol0), 1))
        #timeCol1 = timeCol1.reshape((len(timeCol1), 1))
        
        # calculate time gaps matrix
        #diffmat = np.subtract(timeCol0, timeCol1.T)
        
        if self.allgaps:
            # collect all time gaps
            if f0==f1:
                timeCol0 = timeCol0.reshape((len(timeCol0), 1))
                timeCol1 = timeCol1.reshape((len(timeCol1), 1))

                diffmat = np.subtract(timeCol0, timeCol1.T)
                diffmat = np.abs( diffmat )
                # get only triangle part
                dt_tri = np.tril(diffmat, -1)
                dT = dt_tri[dt_tri!=0]    # flatten lower triangle 
            else:
                # dT = diffmat.flatten()  
                dtmax = np.max(self.bins_diff) # time gaps window for measure color
                dT = []
                for timeCol in timeCol0:
                    timeCol_inWindow = timeCol1[(timeCol1>=(timeCol-dtmax)) & (timeCol1<=(timeCol + dtmax))]
                    
                    dT.append( np.abs(timeCol_inWindow-timeCol ) )
                
                dT = np.concatenate(dT) if len(dT)>0 else np.array(dT)
        else:
            # collect only nearest 
            if f0==f1:
                # get diagonal ones nearest nonzero, offset=1
                #dT = np.diagonal(diffmat, offset=1)
                dT = np.diff(timeCol0)
            else:
                timeCol0 = dataSlice['observationStartMJD'][idx0]
                timeCol1 = dataSlice['observationStartMJD'][idx1]

                timeCol0 = timeCol0.reshape((len(timeCol0), 1))
                timeCol1 = timeCol1.reshape((len(timeCol1), 1))

                # calculate time gaps matrix
                diffmat = np.subtract(timeCol0, timeCol1.T)
                # get tgaps both left and right
                # keep only negative ones
                masked_ar = np.ma.masked_where(diffmat>=0, diffmat, )
                left_ar = np.max(masked_ar, axis=1)
                dT_left = -left_ar.data[~left_ar.mask]
                
                # keep only positive ones
                masked_ar = np.ma.masked_where(diffmat<=0, diffmat)
                right_ar = np.min(masked_ar, axis=1)
                dT_right = right_ar.data[~right_ar.mask]
                
                dT = np.concatenate([dT_left.flatten(), dT_right.flatten()])
        
        return dT
    
    def run(self, dataSlice, slicePoint=None):
                        
        # sort the dataSlice in order of time.  
        dataSlice.sort(order='observationStartMJD')
        
        fieldRA = np.mean(dataSlice['fieldRA']) 
        fieldDec = np.mean(dataSlice['fieldDec'])
        pixId = radec2pix(nside=16, ra=np.radians(fieldRA), dec=np.radians(fieldDec))
        
        fom_dic = {}
        dT_dic = {}
        for fltpair in self.fltpairs:
            dT = self._get_dT(dataSlice, fltpair[0], fltpair[1])
            # calculate FoM
            # cut out at average
            # std below 
            # FOM = Nv  / std
            
            if fltpair[0]==fltpair[1]:
                bins = self.bins_same
            else:
                bins = self.bins_diff
                dT_dic[fltpair] = dT
                
            hist, _ = np.histogram(dT, bins=bins)
            
            if np.any(hist):
                hist_clip = np.clip(hist, a_min=0, a_max=self.Nv_clip[fltpair])
                fom = hist.sum() / hist_clip.std()
            else:
                fom = np.nan
            fom_dic[fltpair] = fom
            
            print(pixId, fltpair, fom)
            
        if self.save_dT:
            result = {
                'pixId': pixId,
                'dT_dic': dT_dic,
                'fom_dic': fom_dic,
                'dataSlice': dataSlice
                  }
        else:
            result = {
                'pixId': pixId,
                'fom_dic': fom_dic,
                  }
        return result

    def reduce_tgaps_FoM(self, metric):
        return np.nansum( list(metric['fom_dic'].values()) )


