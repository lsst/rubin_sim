__all__ = ("StaticProbesFoMEmulatorMetric",)

import healpy as hp
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

from rubin_sim.maf.metrics.base_metric import BaseMetric


class StaticProbesFoMEmulatorMetric(BaseMetric):
    """This calculates the Figure of Merit for the combined
    static probes (3x2pt, i.e., Weak Lensing, LSS, Clustering).

    This metric should be run as a summary metric on ExgalM5_with_cuts.
    This FoM takes into account the effects of the following systematics:
    - multiplicative shear bias
    - intrinsic alignments
    - galaxy bias
    - baryonic physics effects
    - photometric redshift uncertainties
    Default values for these systematics are provided

    The Emulator is uses a Gaussian Process to effectively interpolate between
    a grid of FoM values.

    """

    def __init__(
        self, nside=128, shear_m=0.003, sigma_z=0.05, sig_delta_z=0.001, sig_sigma_z=0.003, col=None, **kwargs
    ):
        self.nside = nside
        super().__init__(col=col, **kwargs)
        if col is None:
            self.col = "metricdata"
        self.shear_m = shear_m
        self.sigma_z = sigma_z
        self.sig_delta_z = sig_delta_z
        self.sig_sigma_z = sig_sigma_z

        # FoM is calculated at the following values
        self.parameters = dict(
            area=np.array(
                [
                    7623.22,
                    14786.3,
                    9931.47,
                    8585.43,
                    17681.8,
                    15126.9,
                    9747.99,
                    8335.08,
                    9533.42,
                    18331.3,
                    12867.8,
                    17418.9,
                    19783.1,
                    12538.8,
                    15260.0,
                    16540.7,
                    19636.8,
                    11112.7,
                    10385.5,
                    16140.2,
                    18920.1,
                    17976.2,
                    11352.0,
                    9214.77,
                    16910.7,
                    11995.6,
                    16199.8,
                    14395.1,
                    8133.86,
                    13510.5,
                    19122.3,
                    15684.5,
                    12014.8,
                    14059.7,
                    10919.3,
                    13212.7,
                ]
            ),
            depth=np.array(
                [
                    25.3975,
                    26.5907,
                    25.6702,
                    26.3726,
                    26.6691,
                    24.9882,
                    25.0814,
                    26.4247,
                    26.5088,
                    25.5596,
                    25.3288,
                    24.8035,
                    24.8792,
                    25.609,
                    26.2385,
                    25.0351,
                    26.7692,
                    26.5693,
                    25.8799,
                    26.3009,
                    25.5086,
                    25.4219,
                    25.8305,
                    26.2953,
                    26.0183,
                    25.26,
                    25.7903,
                    25.1846,
                    26.7264,
                    26.0507,
                    25.6996,
                    25.2256,
                    24.9383,
                    26.1144,
                    25.9464,
                    26.1878,
                ]
            ),
            shear_m=np.array(
                [
                    0.00891915,
                    0.0104498,
                    0.0145972,
                    0.0191916,
                    0.00450246,
                    0.00567828,
                    0.00294841,
                    0.00530922,
                    0.0118632,
                    0.0151849,
                    0.00410151,
                    0.0170622,
                    0.0197331,
                    0.0106615,
                    0.0124445,
                    0.00994507,
                    0.0136251,
                    0.0143491,
                    0.0164314,
                    0.016962,
                    0.0186608,
                    0.00945903,
                    0.0113246,
                    0.0155225,
                    0.00800846,
                    0.00732104,
                    0.00649453,
                    0.00243976,
                    0.0125932,
                    0.0182587,
                    0.00335859,
                    0.00682287,
                    0.0177269,
                    0.0035219,
                    0.00773304,
                    0.0134886,
                ]
            ),
            sigma_z=np.array(
                [
                    0.0849973,
                    0.0986032,
                    0.0875521,
                    0.0968222,
                    0.0225239,
                    0.0718278,
                    0.0733675,
                    0.0385274,
                    0.0425549,
                    0.0605867,
                    0.0178555,
                    0.0853407,
                    0.0124119,
                    0.0531027,
                    0.0304032,
                    0.0503145,
                    0.0132213,
                    0.0941765,
                    0.0416444,
                    0.0668198,
                    0.063227,
                    0.0291332,
                    0.0481633,
                    0.0595606,
                    0.0818742,
                    0.0472518,
                    0.0270185,
                    0.0767401,
                    0.0219945,
                    0.0902663,
                    0.0779705,
                    0.0337666,
                    0.0362358,
                    0.0692429,
                    0.0558841,
                    0.0150457,
                ]
            ),
            sig_delta_z=np.array(
                [
                    0.0032537,
                    0.00135316,
                    0.00168787,
                    0.00215043,
                    0.00406031,
                    0.00222358,
                    0.00334993,
                    0.00255186,
                    0.00266499,
                    0.00159226,
                    0.00183664,
                    0.00384965,
                    0.00427765,
                    0.00314377,
                    0.00456113,
                    0.00347868,
                    0.00487938,
                    0.00418152,
                    0.00469911,
                    0.00367598,
                    0.0028009,
                    0.00234161,
                    0.00194964,
                    0.00200982,
                    0.00122739,
                    0.00310886,
                    0.00275168,
                    0.00492736,
                    0.00437241,
                    0.00113931,
                    0.00104864,
                    0.00292328,
                    0.00452082,
                    0.00394114,
                    0.00150756,
                    0.003613,
                ]
            ),
            sig_sigma_z=np.array(
                [
                    0.00331909,
                    0.00529541,
                    0.00478151,
                    0.00437497,
                    0.00443062,
                    0.00486333,
                    0.00467423,
                    0.0036723,
                    0.00426963,
                    0.00515357,
                    0.0054553,
                    0.00310132,
                    0.00305971,
                    0.00406327,
                    0.00594293,
                    0.00348709,
                    0.00562526,
                    0.00396025,
                    0.00540537,
                    0.00500447,
                    0.00318595,
                    0.00460592,
                    0.00412137,
                    0.00336418,
                    0.00524988,
                    0.00390092,
                    0.00498349,
                    0.0056667,
                    0.0036384,
                    0.00455861,
                    0.00554822,
                    0.00381061,
                    0.0057615,
                    0.00357705,
                    0.00590572,
                    0.00422393,
                ]
            ),
            FOM=np.array(
                [
                    11.708,
                    33.778,
                    19.914,
                    19.499,
                    41.173,
                    17.942,
                    12.836,
                    26.318,
                    25.766,
                    28.453,
                    28.832,
                    14.614,
                    23.8,
                    21.51,
                    27.262,
                    20.539,
                    39.698,
                    19.342,
                    17.103,
                    25.889,
                    25.444,
                    32.048,
                    24.611,
                    23.747,
                    32.193,
                    18.862,
                    34.583,
                    14.54,
                    23.31,
                    25.812,
                    39.212,
                    25.078,
                    14.339,
                    24.12,
                    24.648,
                    29.649,
                ]
            ),
        )

    def run(self, data_slice, slice_point=None):
        import george
        from george import kernels

        # Chop off any outliers
        good_pix = np.where(data_slice[self.col] > 0)[0]

        # Calculate area and med depth from
        area = hp.nside2pixarea(self.nside, degrees=True) * np.size(good_pix)
        median_depth = np.median(data_slice[self.col][good_pix])

        # Standardizing data
        df_unscaled = pd.DataFrame(self.parameters)
        x_params = ["area", "depth", "shear_m", "sigma_z", "sig_delta_z", "sig_sigma_z"]
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X = df_unscaled.drop("FOM", axis=1)
        X = pd.DataFrame(scaler_x.fit_transform(df_unscaled.drop("FOM", axis=1)), columns=x_params)
        Y = pd.DataFrame(
            scaler_y.fit_transform(np.array(df_unscaled["FOM"]).reshape(-1, 1)),
            columns=["FOM"],
        )

        # Building Gaussian Process based emulator
        kernel = kernels.ExpSquaredKernel(metric=[1, 1, 1, 1, 1, 1], ndim=6)
        gp = george.GP(kernel, mean=Y["FOM"].mean())
        gp.compute(X)

        def neg_ln_lik(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(Y["FOM"])

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(Y["FOM"])

        result = minimize(neg_ln_lik, gp.get_parameter_vector(), jac=grad_neg_ln_like)

        gp.set_parameter_vector(result.x)

        # Survey parameters to predict FoM at
        to_pred = np.array(
            [
                [
                    area,
                    median_depth,
                    self.shear_m,
                    self.sigma_z,
                    self.sig_delta_z,
                    self.sig_sigma_z,
                ]
            ]
        )
        to_pred = scaler_x.transform(to_pred)

        pred_sfom = gp.predict(Y["FOM"], to_pred, return_cov=False)
        pred_fom = scaler_y.inverse_transform(pred_sfom.reshape(1, -1))

        return np.max(pred_fom)
