__all__ = ("TheLastMetric")

"""
TheLastMetric
An information-based metric comparing the recoverability of redshift information
of simulated OpSims runs. It returns a single number as the Figure of Merit for an OpSim.

Dependencies:
- jax
- pzflow

Catalog file:
https://github.com/dirac-institute/CMNN_Photoz_Estimator/blob/master/mock_catalog.dat

Demonstration:
https://colab.research.google.com/drive/1aJjgYS9XvWlyK_qIKbYXz2Rh4IbCwfh7?usp=sharing

Reference:
Alex M, et al. An information-based metric for observing strategy optimization,
demonstrated in the context of photometric redshifts with applications to cosmology
https://arxiv.org/abs/2104.08229
"""

# Install required libraries (uncomment if needed)
# !pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# !pip install astropy pzflow corner


import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
#import seaborn as sn

import healpy as hp
import rubin_sim.maf.metrics as metrics
import rubin_sim.maf.slicers as slicers
import rubin_sim.maf.metric_bundles as metricBundles
import rubin_sim.maf.db as db

import jax.numpy as jnp
import pickle
import corner
from astropy.table import Table
from pzflow import Flow, FlowEnsemble
from pzflow.distributions import Uniform
from pzflow.bijectors import Chain, StandardScaler, NeuralSplineCoupling, ColorTransform, InvSoftplus, RollingSplineCoupling, ShiftBounds
from collections import namedtuple

import scipy.stats as sps
import datetime

class TheLastMetric(metrics.BaseMetric):
    """
    Parameters:
        colname: list, ['observationStartMJD', 'filter', 'fiveSigmaDepth']
        nside: nside of healpix
        input_data_dir: string, directory to input catalog data, zphot.cat, test.cat
        flowfile: pickle, pre-trained flow model

    Returns:
        tlm: the last metric value, tlm = flow.log_prob + entropy,
        reduce_tlm: reduce to single number, mean

    """

    def __init__(self, colname=['observationStartMJD', 'fieldRA', 'fieldDec', 'filter', 'fiveSigmaDepth'],
                 nside=16, opsdb='/drive/Shareddrives/MAF/TheLastMetric/baseline_v2.0_10yrs.db',
                 catalog = './mock_catalog.dat',
                 outDir='./outDir',
                 input_data_dir = '/drive/Shareddrives/MAF/TheLastMetric/',
                 flowfile = None,
                  **kwargs):
        self.colname = colname
        self.nside = nside
        self.opsdb = opsdb
        self.outDir = outDir

        self.catalog = catalog

        self.input_data_dir = input_data_dir
        self.flowfile = flowfile
        super().__init__(col=self.colname, **kwargs)


    def make_test_and_train(self, verbose=True, runid='1', filtmask=[1,1,1,1,1,1], yfilt=0, outDir='output/',
                            catalog='mock_catalog.dat', roman_exp=0,
                            test_m5=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 30, 30, 30], train_m5=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 30, 30, 30],
                            test_mcut=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 30, 30, 30], train_mcut=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 30, 30, 30],
                            force_idet=True, force_gridet=True,
                            test_N=5000, train_N=20000, cmnn_minNc=3):

        '''
        from https://github.com/dirac-institute/CMNN_Photoz_Estimator/tree/master
        Create the test and training set based on user specifications.

        Inputs described in cmnn_run.main.

        Outputs: output/run_<runid>/test.cat and train.cat
        '''

        if verbose:
            print('Starting cmnn_catalog.make_test_and_train: ',datetime.datetime.now())

        # read galaxy data from the catalog
        # recall use of the yfilt parameter is
        #   yfilt = 0 : use PanSTARRs y-band (default, column 7)
        #   yfilt = 1 : use Euclid y-band (column 8)
        all_id = np.loadtxt(catalog, dtype='float', usecols=(0))
        all_tz = np.loadtxt(catalog, dtype='float', usecols=(1))
        if yfilt == 0:
            all_tm = np.loadtxt(catalog, dtype='float', usecols=(2, 3, 4, 5, 6, 7, 9, 10, 11))
        elif yfilt == 1:
            all_tm = np.loadtxt(catalog, dtype='float', usecols=(2, 3, 4, 5, 6, 8, 9, 10, 11))

        # convert user-specified magnitude limits to numpy arrays
        np_test_m5    = np.asarray(test_m5, dtype='float')
        np_train_m5   = np.asarray(train_m5, dtype='float')
        np_test_mcut  = np.asarray(test_mcut, dtype='float')
        np_train_mcut = np.asarray(train_mcut, dtype='float')

        # gamma sets the impact of sky brightness in magnitude error estimates
        gamma = np.asarray( [0.037, 0.038, 0.039, 0.039, 0.04, 0.04, 0.04, 0.04, 0.04], dtype='float' )

        # apply user-specified m5 depths to calculate magnitude errors for all galaxies
        all_test_me = np.sqrt((0.04 - gamma) * (np.power(10.0, 0.4 * (all_tm[:] - np_test_m5))) + \
                              gamma * (np.power(10.0, 0.4*(all_tm[:] - np_test_m5))**2))
        all_train_me = np.sqrt((0.04 - gamma) * (np.power(10.0, 0.4 * (all_tm[:] - np_train_m5))) + \
                               gamma * (np.power(10.0, 0.4 * (all_tm[:] - np_train_m5))**2))

        # apply the uncertainty floor of 0.005 mag
        for f in range(9):
            tex = np.where( all_test_me[:,f] < 0.0050)[0]
            all_test_me[tex,f] = float(0.0050)
            trx = np.where( all_train_me[:,f] < 0.0050)[0]
            all_train_me[trx,f] = float(0.0050)

        # use the errors to calculate apparent observed magnitudes
        all_test_m = all_tm + all_test_me * np.random.normal(size=(len(all_tm), 9))
        all_train_m = all_tm + all_train_me * np.random.normal(size=(len(all_tm), 9))

        # apply 17 mag as the saturation limit for all filters
        for f in range(9):
            tx = np.where(all_tm[:,f] < 17.0000)[0]
            all_test_me[tx] = np.nan
            all_test_m[tx] = np.nan
            all_train_me[tx] = np.nan
            all_train_m[tx] = np.nan
            del tx

        # do not allow "upscattering" of > 0.2 mag
        for f in range(9):
            tx = np.where(all_tm[:,f] > np_test_m5[f] + 0.20)[0]
            all_test_me[tx] = np.nan
            all_test_m[tx] = np.nan
            del tx
            tx = np.where(all_tm[:,f] > np_train_m5[f] + 0.20)[0]
            all_train_me[tx] = np.nan
            all_train_m[tx] = np.nan
            del tx

        # apply the user-specified magnitude cuts
        for f in range(9):
            te_x = np.where(all_test_m[:,f] > np_test_mcut[f])[0]
            if len(te_x) > 0:
                all_test_m[te_x, f] = np.nan
                all_test_me[te_x, f] = np.nan
                if (force_idet == True) & (f == 3):
                    all_test_m[te_x, :] = np.nan
                    all_test_me[te_x, :] = np.nan
                if (force_gridet == True) & ((f == 1) | (f == 2) | (f == 3)):
                    all_test_m[te_x, :] = np.nan
                    all_test_me[te_x, :] = np.nan
            tr_x = np.where(all_train_m[:,f] > np_train_mcut[f])[0]
            if len(tr_x) > 0:
                all_train_m[tr_x, f] = np.nan
                all_train_me[tr_x, f] = np.nan
                if (force_idet == True) & (f == 3):
                    all_train_m[tr_x, :] = np.nan
                    all_train_me[tr_x, :] = np.nan
                if (force_gridet == True) & ((f == 1) | (f == 2) | (f == 3)):
                    all_train_m[tr_x, :] = np.nan
                    all_train_me[tr_x, :] = np.nan
            del te_x,tr_x

        # Roman special experiements
        #   0 : fifth color is z-y; do nothing
        #   1 : fifth color is z-J; put J into y
        #   2 : fifth color is z-H; put H into y
        #   3 : fifth color is z-K; put K into y
        #   4 : sixth color is y-J; do nothing
        #   5 : sixth color is y-H; put H into J
        #   6 : sixth color is y-K; put K into J
        if roman_exp == 1:
            all_test_m[:, 5] = all_test_m[:, 6]
            all_test_me[:, 5] = all_test_me[:, 6]
            all_train_m[:, 5] = all_train_m[:, 6]
            all_train_me[:, 5] = all_train_me[:, 6]
        if roman_exp == 2:
            all_test_m[:, 5] = all_test_m[:, 7]
            all_test_me[:, 5] = all_test_me[:, 7]
            all_train_m[:, 5] = all_train_m[:, 7]
            all_train_me[:, 5] = all_train_me[:, 7]
        if roman_exp == 3:
            all_test_m[:, 5] = all_test_m[:, 8]
            all_test_me[:, 5] = all_test_me[:, 8]
            all_train_m[:, 5] = all_train_m[:, 8]
            all_train_me[:, 5] = all_train_me[:, 8]
        if roman_exp == 5:
            all_test_m[:, 6] = all_test_m[:, 7]
            all_test_me[:, 6] = all_test_me[:, 7]
            all_train_m[:, 6] = all_train_m[:, 7]
            all_train_me[:, 6] = all_train_me[:, 7]
        if roman_exp == 6:
            all_test_m[:, 6] = all_test_m[:, 8]
            all_test_me[:, 6] = all_test_me[:, 8]
            all_train_m[:, 6] = all_train_m[:, 8]
            all_train_me[:, 6] = all_train_me[:, 8]

        # apply filtmask
        for f, fm in enumerate(filtmask):
            if fm == 0:
                all_test_m[:, f] = np.nan
                all_test_me[:, f] = np.nan
                all_train_m[:, f] = np.nan
                all_train_me[:, f] = np.nan

        # calculate colors, color errors, and number of colors
        all_test_c = np.zeros((len(all_tm), 8), dtype='float')
        all_test_ce = np.zeros((len(all_tm), 8), dtype='float')
        all_train_c = np.zeros((len(all_tm), 8), dtype='float')
        all_train_ce = np.zeros((len(all_tm), 8), dtype='float')
        for c in range(8):
            all_test_c[:, c]   = all_test_m[:, c] - all_test_m[:, c+1]
            all_train_c[:, c]  = all_train_m[:, c] - all_train_m[:, c+1]
            all_test_ce[:, c]  = np.sqrt( all_test_me[:, c]**2  + all_test_me[:, c+1]**2 )
            all_train_ce[:, c] = np.sqrt( all_train_me[:, c]**2 + all_train_me[:, c+1]**2 )
        all_test_Nc = np.nansum(all_test_c/all_test_c, axis=1)
        all_train_Nc = np.nansum(all_train_c/all_train_c, axis=1)

        # create test and training sets
        te_x = np.where( all_test_Nc >= cmnn_minNc )[0]
        tr_x = np.where( all_train_Nc >= cmnn_minNc )[0]

        if (len(te_x) < test_N) | (len(tr_x) < train_N):
            print('Error. Desired number of test/training galaxies higher than what is available.')
            print('  test number desired, available: %i %i' % (test_N, len(te_x)))
            print('  train number desired, available: %i %i' % (train_N, len(tr_x)))
            print('Exit (inputs too constraining to build test/train set).')
            exit()

        else:
            te_rx = np.random.choice(te_x, size=test_N, replace=False)
            test_fout = open(outDir + '/run_'+runid+'_test.cat', 'w')
            for i in te_rx:
                test_fout.write('%10i %10.8f ' % (all_id[i], all_tz[i]))
                for f in range(9):
                    test_fout.write('%9.6f %9.6f ' % (all_test_m[i, f], all_test_me[i, f]))
                for c in range(8):
                    test_fout.write('%9.6f %9.6f ' % (all_test_c[i, c], all_test_ce[i, c]))
                test_fout.write('\n')
            test_fout.close()
            del te_rx,test_fout

            tr_rx = np.random.choice(tr_x, size=train_N, replace=False)
            train_fout = open(outDir + '/run_'+runid+'_train.cat','w')
            for i in tr_rx:
                train_fout.write('%10i %10.8f ' % (all_id[i], all_tz[i]))
                for f in range(9):
                    train_fout.write('%9.6f %9.6f ' % (all_train_m[i, f], all_train_me[i, f]))
                for c in range(8):
                    train_fout.write('%9.6f %9.6f ' % (all_train_c[i, c], all_train_ce[i, c]))
                train_fout.write('\n')
            train_fout.close()
            del tr_rx,train_fout

            if verbose:
                print('Wrote ', outDir+'run_'+runid+'_test.cat' + outDir + 'run_'+runid+'_train.cat')
                print('Finished cmnn_catalog.make_test_and_train: ',datetime.datetime.now())
        train_cat = outDir + 'run_'+runid + '_train.cat'
        test_cat = outDir + 'run_'+runid + '_test.cat'
        return train_cat, test_cat



    def get_coaddM5(self, outDir='./outDir', colname=['observationStartMJD', 'filter', 'fiveSigmaDepth'],
                     nside=16, opsdb = '/drive/Shareddrives/MAF/TheLastMetric/baseline_v2.0_10yrs.db'
                     ):

        """run a ExGalCoadd to get the median of M5"""
        resultsDb = db.ResultsDb(out_dir=outDir)
        # metric, slicer, constraint
        metric = ExGalCoaddM5Metric( colname=colname, )

        slicer = slicers.HealpixSlicer(nside=nside)

        # bundle
        metricSky = metricBundles.MetricBundle(metric, slicer, sqlstr)

        # group bundle
        bundleDict = {'metricSky':metricSky}

        # table names='observations' for v2.0
        #if 'observations' in opsdb.get_table_names():
        #    dbTable='observations'
        #else:
        #    dbTable = 'SummaryAllProps'

        dbTable = 'observations'

        #group = metricBundles.MetricBundleGroup(bundleDict, opsdb,
        #                                        outDir=outDir,
        #                                        resultsDb=resultsDb,
        #                                        dbTable=dbTable )

        group = metricBundles.MetricBundleGroup(bundleDict, opsdb,
                                                out_dir=outDir,
                                                results_db=resultsDb,
                                                db_table=dbTable )

        group.run_all()

        data = metricSky.metric_values.data[ ~metricSky.metric_values.mask]
        df = pd.DataFrame.from_records(data)
        coaddM5 = df[df['ebv']<0.2].dropna().median()[['coaddm5_u', 'coaddm5_g','coaddm5_r',
                                         'coaddm5_i', 'coaddm5_z','coaddm5_y',]].values

        return coaddM5


    def run(self, dataSlice, slice_point=None):
        # sort the dataSlice in order of time.
        # slicePoint {'ra':, 'dec':, 'sid':}
        #dataSlice.sort(order='observationStartMJD')
        #result = {}
        #result['fieldRA'] = slicePoint['fieldRA']
        #print(slicePoint)
        #slice_point['Nv'] = len(dataSlice)

        #-----------------
        # load catalog data

        #input_data_dir = '/drive/Shareddrives/MAF/TheLastMetric/'

        #names_z=('ID', 'z_true', 'z_phot', 'dz_phot', 'NN', 'N_train')

        #names_phot=('ID', 'z_true',
        #    'u', 'err_u', 'g', 'err_g', 'r', 'err_r', 'i', 'err_i', 'z', 'err_z', 'y', 'err_y',
        #    'u-g', 'err_u-g', 'g-r', 'err_g-r', 'r-i', 'err_r-i', 'i-z', 'err_i-z', 'z-y', 'err_z-y')

        #z_cat = Table.read(self.input_data_dir + 'zphot.cat',
        #                       format='ascii',
        #                       names=names_z)

        #phot_cat = Table.read(self.input_data_dir + 'test.cat',
        #                       format='ascii',
        #                       names=names_phot)

        #phot_cat = Table.from_pandas(phot_cat.to_pandas().dropna())

        #cat = phot_cat.to_pandas().merge(z_cat.to_pandas())

        ## --------cut by median m5--------------

        # -----------------------

        # get M5 median
        coaddM5 = self.get_coaddM5(outDir = self.outDir, colname = self.colname,
                     nside = self.nside, opsdb=self.opsdb)

        # append cut for J, H, K used in make_test_and_train
        coaddM5 = np.append(coaddM5, [30, 30, 30])

        print('coaddM5', coaddM5)
        _, test_cat_file = self.make_test_and_train(runid='1', catalog=self.catalog,
                                               train_mcut=coaddM5,
                                               test_mcut=coaddM5)

        names_phot=('ID', 'z_true',
            'u', 'err_u', 'g', 'err_g', 'r', 'err_r', 'i', 'err_i', 'z', 'err_z', 'y', 'err_y',
            'u-g', 'err_u-g', 'g-r', 'err_g-r', 'r-i', 'err_r-i', 'i-z', 'err_i-z', 'z-y', 'err_z-y')

        usecols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, #14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29,  #30, 31, 32, 33, 34, 35
            ]

        df_cat = pd.read_csv(test_cat_file, delim_whitespace=True, names=names_phot, usecols=usecols).dropna()
        print('loaded df_cat')

        data_columns = ["z_true"]
        drop_cols = ['ID', 'z_true', 'u', 'g',  'i', 'z', 'y',
                'err_u', 'err_g', 'err_r', 'err_i', 'err_z', 'err_y',
                'err_u-g', 'err_g-r', 'err_r-i', 'err_i-z', 'err_z-y']

        conditional_columns = df_cat.drop(drop_cols, axis = 1)
        # data_columns: z_true
        # conditional_columns: r, u-g, g-r, r-i, i-z, z-y

        if self.flowfile!=None:
            flow = FlowEnsemble(file=self.flowfile)
        else:
            # train flow model
            ndcol = len(data_columns)
            ncond = len(conditional_columns)
            ndim = ndcol+ncond

            K = 16
            bijector = Chain(
                StandardScaler(np.atleast_1d(1.6), np.atleast_1d(0.32)),
                NeuralSplineCoupling(B=5, n_conditions=6, K=K)
              )

            latent = Uniform(input_dim=ndcol, B=7) # this has changed.

            info = f"Models z_true conditioned on galaxy colors and r mag from {test_cat_file}. K = 16"

            flow = FlowEnsemble(data_columns = data_columns,
                                       conditional_columns = conditional_columns,
                                       bijector = bijector,
                                       latent = latent,
                                       info = info,
                                       N = 10
                                        )

            loss = flow.train(df_cat, #conditional_columns],
                                           convolve_errs=False,
                                   epochs=150, verbose=True);

        #IDS = phot_cat['ID']
        #z_cat_no_nan = z_cat.to_pandas()[z_cat.to_pandas()['ID'].isin(IDS)]

        b = sps.mstats.mquantiles(df_cat['z_true'], np.linspace(0,1, 101, endpoint=True))
        # print(len(z_cats_no_nan[which_os]['z_true']))
        # print(b)

        #b_centers = 0.5*(b[1:] + b[:-1])

        # Computing the entropy H(z)
        pz = sps.rv_histogram(np.histogram(df_cat['z_true'], bins=b))
        entropy = pz.entropy()

        # mutual information lower bound
        milb = flow.log_prob(df_cat, returnEnsemble=True, err_samples=10)# + entropy

        tlm = milb.mean(axis=0) + entropy
        print("calculated tlm")

        #with open('test_tlm.pkl', 'wb') as outfile:
        #    pickle.dump(tlm, outfile)

        return { #'dataSlice': dataSlice,
                'entropy': entropy,
                'milb': milb,
                'tlm': tlm
                }

    def reduce_tlm(self, metric):
        """get number of visits"""
        return metric['tlm'].mean()



class ExGalCoaddM5Metric(metrics.BaseMetric):
    """
    Calculate coadd M5 for all filters used to cut simulated catalog in theLastMetric
    Parameters:
        colname: list, ['observationStartMJD', 'filter', 'fiveSigmaDepth']
        nside: nside of healpix

    Returns:
        return coaddedM5   # if Ebv<0.2 mag

    """

    def __init__(self, colname=['observationStartMJD', 'fieldRA',
                                'fieldDec', 'filter', 'fiveSigmaDepth'],
                 maps=['DustMap'],
                  **kwargs):
        self.colname = colname
        #self.nside = nside

        super().__init__(col=self.colname, maps=['DustMap'], **kwargs)

    def run(self, data_slice, slice_point=None):
        # sort the dataSlice in order of time.
        # slicePoint {'ra':, 'dec':, 'sid':}
        data_slice.sort(order='observationStartMJD')
        #result = {}
        #result['fieldRA'] = slicePoint['fieldRA']
        #print(slicePoint)
        slice_point['Nv'] = len(data_slice)

        for f in np.unique(data_slice['filter']):
            data_slice_f = data_slice[data_slice['filter']==f]
            slice_point[f'Nv_{f}'] = len(data_slice_f)
            slice_point[f'coaddm5_{f}'] = metrics.Coaddm5Metric().run(data_slice_f)

        return slice_point

#    def reduce_m5(self, metric):
#        """return coaddM5"""
#        return metric['coaddm5']

    def reduce_ebv(self, metric):
        return metric['ebv']



if __name__=='__main__':

	outDir = './outDir'
	# outDir = '/drive/Shareddrives/MAF/TheLastMetric/outDir'
	resultsDb = db.ResultsDb(out_dir=outDir)
	
	colname = ['observationStartMJD', 'filter', 'fiveSigmaDepth']
	sqlstr = 'night<400'
	
	nside = 16
	
	opsdb = '/drive/Shareddrives/MAF/TheLastMetric/baseline_v2.0_10yrs.db'
	# opsdb = './baseline_v2.0_10yrs.db'
	#opsdb = db.Database(dbpath_v30+'baseline_v3.0_10yrs.db')
	
	# metric, slicer, constraint
	metric = TheLastMetric( colname=colname, nside=nside, opsdb=opsdb)
	
	#slicer = slicers.HealpixSlicer(nside=nside)
	
	slicer = slicers.UniSlicer()
	# bundle
	metricSky = metricBundles.MetricBundle(metric, slicer, sqlstr)
	
	# group bundle
	bundleDict = {'metricSky':metricSky}
	
	# table names='observations' for v2.0
	#if 'observations' in opsdb.get_table_names():
	#    dbTable='observations'
	#else:
	#    dbTable = 'SummaryAllProps'
	
	dbTable = 'observations'
	
	#group = metricBundles.MetricBundleGroup(bundleDict, opsdb,
	#                                        outDir=outDir,
	#                                        resultsDb=resultsDb,
	#                                        dbTable=dbTable )
	
	group = metricBundles.MetricBundleGroup(bundleDict, opsdb,
	                                        out_dir=outDir,
	                                        results_db=resultsDb,
	                                        db_table=dbTable )
	
	group.run_all()

