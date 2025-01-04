__all__ = ("TheLastMetric", "theLastMetricBatch")

import datetime
import os

import numpy as np
import pandas as pd
import scipy.stats as sps
from pzflow import FlowEnsemble
from pzflow.bijectors import Chain, NeuralSplineCoupling, StandardScaler
from pzflow.distributions import Uniform

from rubin_sim.data import get_data_dir

from ..batches import col_map_dict
from ..metric_bundles import MetricBundle, MetricBundleGroup
from ..metrics import BaseMetric, ExgalM5WithCuts
from ..slicers import HealpixSlicer, UniSlicer

# To run with GPU, install jax with cuda
# pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


class TheLastMetric(BaseMetric):
    """Calculate an information-based metric comparing the recoverability
    of redshift information based on exposure metadata.

    Parameters
    ----------
    catalog : `str` or None
        Path to the mock catalog. Default None uses
        $RUBIN_SIM_DATA_DIR/maf/mlg_mock/mock_catalog.dat
    flowfile : `str` or None
        Path to the XXXX
    m5_col : `str`
        Name of the column with m5 information
    filter_col : `str`
        Name of the column with filter information
    additional_cols : `list` [`str`]
        Additional columns necessary to run the metric.
    metric_name : `str`
        Name to apply to the metric (for labels and outputs).

    Returns
    -------
    entropy, milb, tlm
        tlm: the last metric value, tlm = flow.log_prob + entropy

    Notes
    -----
    An information-based metric comparing the recoverability of
    redshift information of simulated OpSims runs.
    It returns a single number as the Figure of Merit for an OpSim.

    Catalog file:
    https://github.com/dirac-institute/CMNN_Photoz_Estimator/blob/master/mock_catalog.dat

    Demonstration:
    https://colab.research.google.com/drive/1aJjgYS9XvWlyK_qIKbYXz2Rh4IbCwfh7?usp=sharing

    Reference:
    Alex M, et al. An information-based metric for observing strategy
    optimization, demonstrated in the context of photometric redshifts
    with applications to cosmology
    https://arxiv.org/abs/2104.08229
    """

    def __init__(
        self,
        catalog=None,
        flowfile=None,
        m5_col="fiveSigmaDepth",
        filter_col="filter",
        additional_cols=["fieldRA", "fieldDec", "rotSkyPos"],
        metric_name="TheLastMetric",
        scratch_dir=None,
        **kwargs,
    ):
        self.m5_col = m5_col
        self.filter_col = filter_col

        if catalog is not None:
            self.catalog = catalog
        else:
            self.catalog = os.path.join(get_data_dir(), "maf", "mlg_mock", "mock_catalog.dat")

        self.flowfile = flowfile

        cols = [self.m5_col, self.filter_col] + additional_cols
        super().__init__(col=cols, metric_name=metric_name, **kwargs)

        if scratch_dir is None:
            scratch_dir = os.getcwd()
        self.scratch_dir = scratch_dir

    def make_test_and_train(
        self,
        verbose=False,
        filtmask=[1, 1, 1, 1, 1, 1],
        yfilt=0,
        roman_exp=0,
        test_m5=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 30, 30, 30],
        train_m5=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 30, 30, 30],
        test_mcut=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 30, 30, 30],
        train_mcut=[26.1, 27.4, 27.5, 26.8, 26.1, 24.9, 30, 30, 30],
        force_idet=True,
        force_gridet=True,
        test_N=5000,
        train_N=20000,
        cmnn_minNc=3,
    ):
        """Create the test and training set based on user specifications.

        This is based on work in
        https://github.com/dirac-institute/CMNN_Photoz_Estimator
        Parameters are further described in the argparser for
        `cmnn_run` in that repository.

        The input catalog (default is mock_catalog.dat) includes galaxy
        measurements in u g r i z y J H K -- Rubin supplemented with Roman.
        This method sets m5 and mcut values in each bandpass, for creating
        testing and training catalogs from the input catalog.

        Parameters
        ----------
        filtmask : `list` [`int`]
            Set mask for use of filters u g r i z y J H K
        yfilt : `int`
            0 - use PanStarrs y band, 1 - use Euclid y-band
        roman_exp : `int`
            Set the 5th or 6th color to use Roman mags (0 = no Roman?)
        test_m5 : `list` [`float`]
            5-sigma magnitude limits (depths) for test-set galaxies
        train_m5 : `list` [`float`]
            5-sigma magnitude limits (depths) for training-set galaxies
        test_mcut : `list` [`float`]
            Magnitude cut-off to apply to the test-set galaxies
        train_mcut : `list` [`float`]
            Magnitude cut-off to apply to the training-set galaxies
        force_idet : `bool`
            True - force i-band detection for all galaxies
        force_gridet : `bool`
            True - force g+r+i-band detection for all galaxies
        test_N : `int`
            Number of test-set galaxies
        train_N : `int`
            Number of training-set galaxies
        cmnn_minNc : `int`
            Minimum number of colors for galaxies (2 to 8)

        Returns
        -------
        train_cat, test_cat : `np.ndarray` (N, M), `np.ndarray` (N, M)
            Training and testing sub-selected catalogs.
        """

        if verbose:
            print("Starting cmnn_catalog.make_test_and_train: ", datetime.datetime.now())

        # read galaxy data from the catalog
        # recall use of the yfilt parameter is
        #   yfilt = 0 : use PanSTARRs y-band (default, column 7)
        #   yfilt = 1 : use Euclid y-band (column 8)
        all_id = np.loadtxt(self.catalog, dtype="float", usecols=(0))
        all_tz = np.loadtxt(self.catalog, dtype="float", usecols=(1))
        if yfilt == 0:
            all_tm = np.loadtxt(self.catalog, dtype="float", usecols=(2, 3, 4, 5, 6, 7, 9, 10, 11))
        elif yfilt == 1:
            all_tm = np.loadtxt(self.catalog, dtype="float", usecols=(2, 3, 4, 5, 6, 8, 9, 10, 11))

        # convert user-specified magnitude limits to numpy arrays
        np_test_m5 = np.asarray(test_m5, dtype="float")
        np_train_m5 = np.asarray(train_m5, dtype="float")
        np_test_mcut = np.asarray(test_mcut, dtype="float")
        np_train_mcut = np.asarray(train_mcut, dtype="float")

        # gamma sets the impact of sky brightness in magnitude error estimates
        # Gamma can be double-checked against the values in
        # rubin_scheduler.utils.SysEngVals, and is here extended further red
        gamma = np.asarray([0.037, 0.038, 0.039, 0.039, 0.04, 0.04, 0.04, 0.04, 0.04], dtype="float")

        # apply user-specified m5 depths to calculate magnitude errors
        # for all galaxies
        all_test_me = np.sqrt(
            (0.04 - gamma) * (np.power(10.0, 0.4 * (all_tm[:] - np_test_m5)))
            + gamma * (np.power(10.0, 0.4 * (all_tm[:] - np_test_m5)) ** 2)
        )
        all_train_me = np.sqrt(
            (0.04 - gamma) * (np.power(10.0, 0.4 * (all_tm[:] - np_train_m5)))
            + gamma * (np.power(10.0, 0.4 * (all_tm[:] - np_train_m5)) ** 2)
        )

        # apply the uncertainty floor of 0.005 mag
        for f in range(9):
            tex = np.where(all_test_me[:, f] < 0.0050)[0]
            all_test_me[tex, f] = float(0.0050)
            trx = np.where(all_train_me[:, f] < 0.0050)[0]
            all_train_me[trx, f] = float(0.0050)

        # use the errors to calculate apparent observed magnitudes
        all_test_m = all_tm + all_test_me * np.random.normal(size=(len(all_tm), 9))
        all_train_m = all_tm + all_train_me * np.random.normal(size=(len(all_tm), 9))

        # apply 17 mag as the saturation limit for all filters
        for f in range(9):
            tx = np.where(all_tm[:, f] < 17.0000)[0]
            all_test_me[tx] = np.nan
            all_test_m[tx] = np.nan
            all_train_me[tx] = np.nan
            all_train_m[tx] = np.nan
            del tx

        # do not allow "upscattering" of > 0.2 mag
        for f in range(9):
            tx = np.where(all_tm[:, f] > np_test_m5[f] + 0.20)[0]
            all_test_me[tx] = np.nan
            all_test_m[tx] = np.nan
            del tx
            tx = np.where(all_tm[:, f] > np_train_m5[f] + 0.20)[0]
            all_train_me[tx] = np.nan
            all_train_m[tx] = np.nan
            del tx

        # apply the user-specified magnitude cuts
        for f in range(9):
            te_x = np.where(all_test_m[:, f] > np_test_mcut[f])[0]
            if len(te_x) > 0:
                all_test_m[te_x, f] = np.nan
                all_test_me[te_x, f] = np.nan
                if (force_idet == True) & (f == 3):
                    all_test_m[te_x, :] = np.nan
                    all_test_me[te_x, :] = np.nan
                if (force_gridet == True) & ((f == 1) | (f == 2) | (f == 3)):
                    all_test_m[te_x, :] = np.nan
                    all_test_me[te_x, :] = np.nan
            tr_x = np.where(all_train_m[:, f] > np_train_mcut[f])[0]
            if len(tr_x) > 0:
                all_train_m[tr_x, f] = np.nan
                all_train_me[tr_x, f] = np.nan
                if (force_idet == True) & (f == 3):
                    all_train_m[tr_x, :] = np.nan
                    all_train_me[tr_x, :] = np.nan
                if (force_gridet == True) & ((f == 1) | (f == 2) | (f == 3)):
                    all_train_m[tr_x, :] = np.nan
                    all_train_me[tr_x, :] = np.nan
            del te_x, tr_x

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
        all_test_c = np.zeros((len(all_tm), 8), dtype="float")
        all_test_ce = np.zeros((len(all_tm), 8), dtype="float")
        all_train_c = np.zeros((len(all_tm), 8), dtype="float")
        all_train_ce = np.zeros((len(all_tm), 8), dtype="float")
        for c in range(8):
            all_test_c[:, c] = all_test_m[:, c] - all_test_m[:, c + 1]
            all_train_c[:, c] = all_train_m[:, c] - all_train_m[:, c + 1]
            all_test_ce[:, c] = np.sqrt(all_test_me[:, c] ** 2 + all_test_me[:, c + 1] ** 2)
            all_train_ce[:, c] = np.sqrt(all_train_me[:, c] ** 2 + all_train_me[:, c + 1] ** 2)
        all_test_Nc = np.nansum(all_test_c / all_test_c, axis=1)
        all_train_Nc = np.nansum(all_train_c / all_train_c, axis=1)

        # create test and training sets
        te_x = np.where(all_test_Nc >= cmnn_minNc)[0]
        tr_x = np.where(all_train_Nc >= cmnn_minNc)[0]

        if (len(te_x) < test_N) | (len(tr_x) < train_N):
            print("Error. Desired number of test/training galaxies higher than what is available.")
            print("  test number desired, available: %i %i" % (test_N, len(te_x)))
            print("  train number desired, available: %i %i" % (train_N, len(tr_x)))
            return None, None

        else:
            # I would like this to return the actual catalogs, not filenames
            # Can you build these as a subselection from the arrays and just
            # pass them back?
            te_rx = np.random.choice(te_x, size=test_N, replace=False)
            test_cat = os.path.join(self.scratch_dir, "run_test.cat")
            with open(test_cat, "w") as test_fout:
                for i in te_rx:
                    test_fout.write("%10i %10.8f " % (all_id[i], all_tz[i]))
                    for f in range(9):
                        test_fout.write("%9.6f %9.6f " % (all_test_m[i, f], all_test_me[i, f]))
                    for c in range(8):
                        test_fout.write("%9.6f %9.6f " % (all_test_c[i, c], all_test_ce[i, c]))
                    test_fout.write("\n")
            del te_rx

            tr_rx = np.random.choice(tr_x, size=train_N, replace=False)
            train_cat = os.path.join(self.scratch_dir, "run_train.cat")
            with open(train_cat, "w") as train_fout:
                for i in tr_rx:
                    train_fout.write("%10i %10.8f " % (all_id[i], all_tz[i]))
                    for f in range(9):
                        train_fout.write("%9.6f %9.6f " % (all_train_m[i, f], all_train_me[i, f]))
                    for c in range(8):
                        train_fout.write("%9.6f %9.6f " % (all_train_c[i, c], all_train_ce[i, c]))
                    train_fout.write("\n")
            del tr_rx

            if verbose:
                print(f"Wrote {test_cat} and {train_cat}")
                print("Finished cmnn_catalog.make_test_and_train: ", datetime.datetime.now())

        return train_cat, test_cat

    def get_median_coaddM5(self, dataSlice):
        """Run ExgalM5WithCuts over the sky, return median per filter."""
        lsst_filters = ["u", "g", "r", "i", "z", "y"]
        metric = ExgalM5WithCuts(m5_col=self.m5_col, filter_col=self.filter_col,
            depth_cut=20, lsst_filter=lsst_filters)
        slicer = HealpixSlicer(nside=64, use_cache=False)
        exgal_bundle = MetricBundle(metric=metric, slicer=slicer, constraint="")

        g = MetricBundleGroup(
            {"exgal": exgal_bundle},
            db_con=None,
            out_dir=".",
            save_early=False,
            results_db=None,
            verbose=False,
        )
        g.run_current(constraint="", sim_data=dataSlice)

        median_coadd_depths = {}
        for i, f in enumerate(lsst_filters):
            median_coadd_depths[f] = np.median(exgal_bundle.metric_values[:, i].compressed())

        return median_coadd_depths

    def run(self, dataSlice, slice_point=None):

        # Get median m5 depths across the DESC footprint
        median_coadd_depths = self.get_median_coaddM5(dataSlice)
        coaddM5 = list(median_coadd_depths.values())

        # append cut for J, H, K used in make_test_and_train
        coaddM5 = np.append(coaddM5, [30, 30, 30])

        _, test_cat_file = self.make_test_and_train(train_mcut=coaddM5, test_mcut=coaddM5)
        if test_cat_file is None:
            return self.badval

        names_phot = (
            "ID",
            "z_true",
            "u",
            "err_u",
            "g",
            "err_g",
            "r",
            "err_r",
            "i",
            "err_i",
            "z",
            "err_z",
            "y",
            "err_y",
            "u-g",
            "err_u-g",
            "g-r",
            "err_g-r",
            "r-i",
            "err_r-i",
            "i-z",
            "err_i-z",
            "z-y",
            "err_z-y",
        )

        usecols = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,  # 14, 15, 16, 17, 18, 19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,  # 30, 31, 32, 33, 34, 35
        ]

        df_cat = pd.read_csv(test_cat_file, delim_whitespace=True, names=names_phot, usecols=usecols).dropna()
        print("loaded df_cat")

        data_columns = ["z_true"]
        drop_cols = [
            "ID",
            "z_true",
            "u",
            "g",
            "i",
            "z",
            "y",
            "err_u",
            "err_g",
            "err_r",
            "err_i",
            "err_z",
            "err_y",
            "err_u-g",
            "err_g-r",
            "err_r-i",
            "err_i-z",
            "err_z-y",
        ]

        conditional_columns = df_cat.drop(drop_cols, axis=1)
        # data_columns: z_true
        # conditional_columns: r, u-g, g-r, r-i, i-z, z-y

        if self.flowfile is not None:
            flow = FlowEnsemble(file=self.flowfile)
        else:
            # train flow model
            ndcol = len(data_columns)

            K = 16
            bijector = Chain(
                StandardScaler(np.atleast_1d(1.6), np.atleast_1d(0.32)),
                NeuralSplineCoupling(B=5, n_conditions=6, K=K),
            )

            latent = Uniform(input_dim=ndcol, B=7)

            info = f"Models z_true conditioned on galaxy colors and r mag from {test_cat_file}. K = 16"

            flow = FlowEnsemble(
                data_columns=data_columns,
                conditional_columns=conditional_columns,
                bijector=bijector,
                latent=latent,
                info=info,
                N=10,
            )

            _ = flow.train(df_cat, convolve_errs=False, epochs=150, verbose=True)

        b = sps.mstats.mquantiles(df_cat["z_true"], np.linspace(0, 1, 101, endpoint=True))

        # Computing the entropy H(z)
        pz = sps.rv_histogram(np.histogram(df_cat["z_true"], bins=b))
        entropy = pz.entropy()

        # mutual information lower bound
        milb = flow.log_prob(df_cat, returnEnsemble=True, err_samples=10)

        tlm = milb.mean(axis=0) + entropy
        # These should all be saved into npz file although format
        # may have trouble - let's see, if the rest works.
        return {"entropy": entropy, "milb": milb, "tlm": tlm}

    def reduce_tlm_mean(self, metric):
        """Return mean tlm as single number - will save into resultsDb"""
        return metric["tlm"].mean()


# I will move this over to the "batches" directory -- it'll probably
# be incorporated with some other batches, but this is how we specify
# how to run a given metric
def theLastMetricBatch(
    colmap=None,
    run_name="opsim",
    extra_sql=None,
    extra_info_label=None,
    display_group="Cosmology",
    subgroup="TheLastMetric",
    out_dir=".",
):
    """Set up TheLastMetric
    within a night.

    Parameters
    ----------
    colmap : `dict` or None, optional
        A dictionary with a mapping of column names.
    run_name : `str`, optional
        The name of the simulated survey.
    extra_sql : `str` or None, optional
        Additional sql constraint to apply to all metrics.
    extra_info_label : `str` or None, optional
        Additional info_label to apply to all results.
    display_group : `str` or None, optional
        In show_maf pages, the division where the metric will appear.
    subgroup : `str` or None, optional
        In show_maf pages, the section within the division for the metrics.
    out_dir : `str`
        To be used (temporarily) for the scratch directory for the metric
        BUT the metric should just pass back the test and train catalogs
        from the function "make_test_and_train"
        instead of writing and then reading them from disk

    Returns
    -------
    metric_bundleDict : `dict` of `maf.MetricBundle`
    """

    if colmap is None:
        colmap = col_map_dict()

    info_label = extra_info_label
    if extra_sql is not None and len(extra_sql) > 0:
        if info_label is None:
            info_label = extra_sql

    bundleList = []

    display_dict = {
        "group": display_group,
        "subgroup": subgroup,
        "caption": None,
        "order": 0,
    }

    metric = TheLastMetric(
        m5_col=colmap["fiveSigmaDepth"],
        filter_col=colmap["filter"],
        additional_cols=[colmap["ra"], colmap["dec"], "rotSkyPos"],
        scratch_dir=out_dir,
    )
    slicer = UniSlicer()

    display_dict["caption"] = (
        "The mean value of TheLastMetric, calculated using the median "
        "extinction-corrected m5 depths over the DESC footprint."
    )
    bundle = MetricBundle(
        metric,
        slicer,
        extra_sql,
        info_label=info_label,
        display_dict=display_dict,
        run_name=run_name,
    )
    bundleList.append(bundle)

    return bundleList
