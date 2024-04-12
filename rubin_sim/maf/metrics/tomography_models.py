__all__ = ("DENSITY_TOMOGRAPHY_MODEL",)

import numpy as np

"""A dictionary of TOMOGRAPHY models for use with the
cosmology summary metrics (TomographicClusteringSigma8bias).

This dictionary is derived from work shown in 
https://github.com/ixkael/ObsStrat/blob/
meanz_uniformity_maf/code/meanz_uniformity/
romanrubinmock_for_sigma8tomoghy.ipynb
"""

# The dictionary DENSITY_TOMOGRAPHY_MODEL contains the current model.
# It is intended for use with various DESC cosmology metrics, in
# particular the TomographicClusteringSigma8bias metric.
#
# The first set of keys are the years (year1, ..., year10),
# since this would change typical depth and galaxy catalog cuts.
# In what follows we have 5 tomographic bins.
#
# The second nested dictionary has the following:
# `sigma8square_model` :
# The fiducial sigma8^2 value used in CCL for the theory predictions.
# `poly1d_coefs_loglog` :
# a polynomial (5th degree) describing the angular power spectra
# (in log log space) in the 5 tomographic bins considered,
# thus has shape (5, 6)
# `lmax` :
# the lmax limits to sum the Cells over when calculating sigma8
# for each tomographic bin. thus is it of shape (5, )
# `dlogN_dm5` :
# the derivatives of logN wrt m5 calculated in  Qianjun & Jeff's simulations.
# It is an array of 5 dictionaries (5 = the tomographic bins)
#
# Each dictionary must have keys that are the lsst bands.
# If some are missing they are ignored in the linear model.
# These badnpasses are the ones which will be fed to
# NestedLinearMultibandModelMetric.
# Everything else above is going into the modeling.
#
# The notebook I used to make this dictionary is
# https://github.com/ixkael/ObsStrat/blob/meanz_uniformity_maf/code/
# meanz_uniformity/romanrubinmock_for_sigma8tomography.ipynb


DENSITY_TOMOGRAPHY_MODEL = {
    "year1": {
        "sigma8square_model": 0.8**2.0,
        "poly1d_coefs_loglog": np.array(
            [
                [
                    -7.94546823e-04,
                    2.62494452e-02,
                    -2.27597655e-01,
                    4.67266497e-01,
                    3.38850627e-01,
                    -1.14120412e01,
                ],
                [
                    3.98978657e-04,
                    6.69012313e-03,
                    -1.35005872e-01,
                    3.72800783e-01,
                    3.97123211e-01,
                    -1.29908869e01,
                ],
                [
                    1.03940293e-03,
                    -5.18759967e-03,
                    -6.85178825e-02,
                    2.69468206e-01,
                    4.60031701e-01,
                    -1.41217016e01,
                ],
                [
                    1.42986845e-03,
                    -1.27862974e-02,
                    -2.26263084e-02,
                    1.85536763e-01,
                    5.12886349e-01,
                    -1.48861030e01,
                ],
                [
                    1.73667255e-03,
                    -1.89601890e-02,
                    1.80214246e-02,
                    9.50637447e-02,
                    5.58980246e-01,
                    -1.55053201e01,
                ],
            ]
        ),
        "lmax": np.array([64, 94, 124, 146, 164]),
        "dlogN_dm5": [
            {
                "cst": 0.0,
                "u": 0.043962974958662145,
                "g": 0.02066177878696687,
                "r": 0.0018693130274623476,
                "i": 0.04276675705247136,
                "z": 0.08638732823394031,
                "y": 0.0657505127473751,
                "ugrizy": 0.2472461161462007,
            },
            {
                "cst": 0.0,
                "u": 0.056364745687503465,
                "g": 0.034752650841693544,
                "r": -0.0166573314175779,
                "i": 0.03094286532175563,
                "z": 0.06502345706021054,
                "y": 0.05768411667472194,
                "ugrizy": 0.22095351609058403,
            },
            {
                "cst": 0.0,
                "u": 0.038530883579453466,
                "g": 0.04538858565704199,
                "r": 0.033787326976188484,
                "i": -0.040733526061764155,
                "z": 0.025969868799574303,
                "y": 0.03702233496870881,
                "ugrizy": 0.1250655075680508,
            },
            {
                "cst": 0.0,
                "u": -0.02161852795176933,
                "g": 0.0222336809395544,
                "r": 0.05908765697086321,
                "i": -0.058749555322422424,
                "z": 0.0188797224519434,
                "y": 0.035785840616014225,
                "ugrizy": 0.046115493661878317,
            },
            {
                "cst": 0.0,
                "u": -0.07624090622672856,
                "g": -0.060659802088035716,
                "r": -0.01676696950407968,
                "i": -0.04141021293176227,
                "z": -0.08739685057046752,
                "y": -0.033876500857633024,
                "ugrizy": -0.2762112449739721,
            },
        ],
    },
    "year2": {
        "sigma8square_model": 0.8**2.0,
        "poly1d_coefs_loglog": np.array(
            [
                [
                    -7.72116403e-04,
                    2.59654127e-02,
                    -2.26957718e-01,
                    4.69432713e-01,
                    3.39213385e-01,
                    -1.13989048e01,
                ],
                [
                    4.22801028e-04,
                    6.36607807e-03,
                    -1.34108644e-01,
                    3.74841312e-01,
                    3.96986022e-01,
                    -1.29600665e01,
                ],
                [
                    1.07354523e-03,
                    -5.73388904e-03,
                    -6.60135335e-02,
                    2.67377905e-01,
                    4.62220942e-01,
                    -1.40738990e01,
                ],
                [
                    1.46948747e-03,
                    -1.34766061e-02,
                    -1.89609170e-02,
                    1.80340598e-01,
                    5.17052663e-01,
                    -1.48679098e01,
                ],
                [
                    1.79954923e-03,
                    -1.99827467e-02,
                    2.28427111e-02,
                    9.06204330e-02,
                    5.69080814e-01,
                    -1.55353679e01,
                ],
            ]
        ),
        "lmax": np.array([64, 95, 123, 147, 166]),
        "dlogN_dm5": [
            {
                "cst": 0.0,
                "u": 0.055492947394854004,
                "g": 0.012699380270242842,
                "r": 0.007530749024132768,
                "i": 0.03513585435458864,
                "z": 0.07561295665384882,
                "y": 0.0736592982969794,
                "ugrizy": 0.24374481489942412,
            },
            {
                "cst": 0.0,
                "u": 0.046332184492532485,
                "g": 0.029608214159172044,
                "r": -0.006075560506575478,
                "i": 0.03333244846260029,
                "z": 0.06027754177286897,
                "y": 0.06506093356954053,
                "ugrizy": 0.2084760154143301,
            },
            {
                "cst": 0.0,
                "u": 0.03334437333732201,
                "g": 0.032573705405151664,
                "r": 0.01704700362928206,
                "i": -0.02614338304111813,
                "z": 0.03924255373249242,
                "y": 0.041759198393945374,
                "ugrizy": 0.13123919055353145,
            },
            {
                "cst": 0.0,
                "u": 0.0030051697578186262,
                "g": 0.03783532078713637,
                "r": 0.06049801557289073,
                "i": -0.061409368487154156,
                "z": 0.03808865014429153,
                "y": 0.02731197586031816,
                "ugrizy": 0.10545655695602359,
            },
            {
                "cst": 0.0,
                "u": -0.02882987071992067,
                "g": -0.013641440195713459,
                "r": 0.03418483342474093,
                "i": -0.018501441200469277,
                "z": -0.050114709867968066,
                "y": 0.021268476475702833,
                "ugrizy": -0.049794732397725236,
            },
        ],
    },
    "year3": {
        "sigma8square_model": 0.8**2.0,
        "poly1d_coefs_loglog": np.array(
            [
                [
                    -7.47335451e-04,
                    2.55922442e-02,
                    -2.25366760e-01,
                    4.68243951e-01,
                    3.39807736e-01,
                    -1.14004586e01,
                ],
                [
                    4.37952430e-04,
                    6.13359097e-03,
                    -1.33135431e-01,
                    3.74413364e-01,
                    3.97397465e-01,
                    -1.29291594e01,
                ],
                [
                    1.08534795e-03,
                    -5.92068724e-03,
                    -6.51754697e-02,
                    2.66734891e-01,
                    4.63243671e-01,
                    -1.40474513e01,
                ],
                [
                    1.47812824e-03,
                    -1.35845536e-02,
                    -1.88487898e-02,
                    1.82317694e-01,
                    5.17486035e-01,
                    -1.48384185e01,
                ],
                [
                    1.82831353e-03,
                    -2.03855872e-02,
                    2.40741011e-02,
                    9.27502779e-02,
                    5.73850795e-01,
                    -1.55657849e01,
                ],
            ]
        ),
        "lmax": np.array([64, 94, 123, 147, 168]),
        "dlogN_dm5": [
            {
                "cst": 0.0,
                "u": 0.04571336657551873,
                "g": 0.01002695291717099,
                "r": 0.006381748317232131,
                "i": 0.03346386406073478,
                "z": 0.07000026154199343,
                "y": 0.07122103158922388,
                "ugrizy": 0.2276375817331448,
            },
            {
                "cst": 0.0,
                "u": 0.05405844464453291,
                "g": 0.03578384485292078,
                "r": -0.006971742709660895,
                "i": 0.03055814323956491,
                "z": 0.0608644383751681,
                "y": 0.06285483616682519,
                "ugrizy": 0.2191177469764914,
            },
            {
                "cst": 0.0,
                "u": 0.03701180196157078,
                "g": 0.02283319745010296,
                "r": 0.023135003462698395,
                "i": -0.020108849022374618,
                "z": 0.03313299331060819,
                "y": 0.042722067239431505,
                "ugrizy": 0.12769514475136706,
            },
            {
                "cst": 0.0,
                "u": 0.005411661204687025,
                "g": 0.03102947466687921,
                "r": 0.04758492397282435,
                "i": -0.05756589965508176,
                "z": 0.038139167579015414,
                "y": 0.028170175608809505,
                "ugrizy": 0.09819722293690358,
            },
            {
                "cst": 0.0,
                "u": -0.007052479985258033,
                "g": 0.0072989635471763055,
                "r": 0.04502123045630644,
                "i": -0.0073176602744123255,
                "z": -0.025917249340435374,
                "y": 0.035858309542520166,
                "ugrizy": 0.042230607766813,
            },
        ],
    },
    "year4": {
        "sigma8square_model": 0.8**2.0,
        "poly1d_coefs_loglog": np.array(
            [
                [
                    -7.41568743e-04,
                    2.55297105e-02,
                    -2.25339653e-01,
                    4.69342516e-01,
                    3.39385437e-01,
                    -1.13829013e01,
                ],
                [
                    4.47724999e-04,
                    5.99613130e-03,
                    -1.32666973e-01,
                    3.74708889e-01,
                    3.97389350e-01,
                    -1.29064781e01,
                ],
                [
                    1.09566923e-03,
                    -6.07450334e-03,
                    -6.45513085e-02,
                    2.66522400e-01,
                    4.63653700e-01,
                    -1.40182124e01,
                ],
                [
                    1.48898237e-03,
                    -1.37812174e-02,
                    -1.77616026e-02,
                    1.80633787e-01,
                    5.19149830e-01,
                    -1.48277616e01,
                ],
                [
                    1.84812024e-03,
                    -2.07364151e-02,
                    2.60418879e-02,
                    8.93719930e-02,
                    5.77645934e-01,
                    -1.55728084e01,
                ],
            ]
        ),
        "lmax": np.array([63, 94, 123, 148, 169]),
        "dlogN_dm5": [
            {
                "cst": 0.0,
                "u": 0.054293436011708406,
                "g": 0.009952046153107954,
                "r": 0.0054797065106343325,
                "i": 0.02576260292778629,
                "z": 0.06525553880320857,
                "y": 0.06423711415381689,
                "ugrizy": 0.20811399377168,
            },
            {
                "cst": 0.0,
                "u": 0.049744205203719596,
                "g": 0.034776420860264834,
                "r": -0.008954957603949931,
                "i": 0.03315665664175325,
                "z": 0.06433009373174128,
                "y": 0.06440730378199838,
                "ugrizy": 0.22417139228019037,
            },
            {
                "cst": 0.0,
                "u": 0.04304128101754591,
                "g": 0.017432851239669426,
                "r": 0.023084985340137268,
                "i": -0.01232102883069664,
                "z": 0.03998793202306738,
                "y": 0.044236619617342786,
                "ugrizy": 0.1392270140374441,
            },
            {
                "cst": 0.0,
                "u": 0.009135822126354363,
                "g": 0.036515151515151424,
                "r": 0.04077785494592716,
                "i": -0.056015513609789944,
                "z": 0.04361845936433841,
                "y": 0.029657965796579696,
                "ugrizy": 0.10045949490109377,
            },
            {
                "cst": 0.0,
                "u": -0.001320660763385228,
                "g": 0.009954114661184083,
                "r": 0.05948333012553189,
                "i": 8.127849262724018e-05,
                "z": -0.02151012665531872,
                "y": 0.0434040047114252,
                "ugrizy": 0.08833836494032939,
            },
        ],
    },
    "year5": {
        "sigma8square_model": 0.8**2.0,
        "poly1d_coefs_loglog": np.array(
            [
                [
                    -7.35347360e-04,
                    2.54490508e-02,
                    -2.25123057e-01,
                    4.69736089e-01,
                    3.39586115e-01,
                    -1.13755955e01,
                ],
                [
                    4.53471097e-04,
                    5.90580865e-03,
                    -1.32288337e-01,
                    3.74594679e-01,
                    3.97577407e-01,
                    -1.29025614e01,
                ],
                [
                    1.10564516e-03,
                    -6.23727532e-03,
                    -6.37987371e-02,
                    2.65929839e-01,
                    4.64485800e-01,
                    -1.40110253e01,
                ],
                [
                    1.49887589e-03,
                    -1.39372292e-02,
                    -1.70839226e-02,
                    1.80323328e-01,
                    5.19727794e-01,
                    -1.47989678e01,
                ],
                [
                    1.87614685e-03,
                    -2.11800406e-02,
                    2.81877828e-02,
                    8.68286284e-02,
                    5.79267039e-01,
                    -1.55754670e01,
                ],
            ]
        ),
        "lmax": np.array([63, 95, 123, 148, 169]),
        "dlogN_dm5": [
            {
                "cst": 0.0,
                "u": 0.049239020937134025,
                "g": 0.01383328060395141,
                "r": -0.0007656653768851196,
                "i": 0.028065747821641045,
                "z": 0.06050907785720702,
                "y": 0.06377977731569703,
                "ugrizy": 0.20957610639279134,
            },
            {
                "cst": 0.0,
                "u": 0.055603049706020896,
                "g": 0.03635061259366789,
                "r": -0.0031966661282920375,
                "i": 0.03213295163976452,
                "z": 0.06832536553023698,
                "y": 0.06457919740205376,
                "ugrizy": 0.23517375380435723,
            },
            {
                "cst": 0.0,
                "u": 0.03475293260889745,
                "g": 0.017981923705511698,
                "r": 0.018654370753291738,
                "i": -0.016107465625364203,
                "z": 0.03550081300116625,
                "y": 0.048124231680847056,
                "ugrizy": 0.13850667560597368,
            },
            {
                "cst": 0.0,
                "u": 0.01511037258240387,
                "g": 0.033244669574385696,
                "r": 0.04459876391783935,
                "i": -0.057050518892029986,
                "z": 0.04070290636155964,
                "y": 0.022294059577929574,
                "ugrizy": 0.09371570861821565,
            },
            {
                "cst": 0.0,
                "u": 0.009871915026554134,
                "g": 0.0110669656320889,
                "r": 0.05829720129030573,
                "i": 0.0026049915531623793,
                "z": -0.022746170354716745,
                "y": 0.04106604585185824,
                "ugrizy": 0.09054948216340614,
            },
        ],
    },
    "year6": {
        "sigma8square_model": 0.8**2.0,
        "poly1d_coefs_loglog": np.array(
            [
                [
                    -7.29402173e-04,
                    2.53591187e-02,
                    -2.24744974e-01,
                    4.69511416e-01,
                    3.39616928e-01,
                    -1.13734220e01,
                ],
                [
                    4.57098537e-04,
                    5.86340062e-03,
                    -1.32257952e-01,
                    3.75362916e-01,
                    3.97431135e-01,
                    -1.28861252e01,
                ],
                [
                    1.10975980e-03,
                    -6.30581393e-03,
                    -6.34236372e-02,
                    2.65250360e-01,
                    4.64594967e-01,
                    -1.39905973e01,
                ],
                [
                    1.50470306e-03,
                    -1.40535883e-02,
                    -1.63140359e-02,
                    1.78531058e-01,
                    5.20616617e-01,
                    -1.47817593e01,
                ],
                [
                    1.87684022e-03,
                    -2.11967381e-02,
                    2.82475769e-02,
                    8.70224981e-02,
                    5.80844971e-01,
                    -1.55660655e01,
                ],
            ]
        ),
        "lmax": np.array([63, 94, 123, 147, 169]),
        "dlogN_dm5": [
            {
                "cst": 0.0,
                "u": 0.04731277533039653,
                "g": 0.007373242386006846,
                "r": 0.008974537732206396,
                "i": 0.021952456469413933,
                "z": 0.0631987046576748,
                "y": 0.06698033256378032,
                "ugrizy": 0.20345569174304204,
            },
            {
                "cst": 0.0,
                "u": 0.05541230971237573,
                "g": 0.03307274442076383,
                "r": -0.011785139084349847,
                "i": 0.03544678226556244,
                "z": 0.06044332758921493,
                "y": 0.06316287559760261,
                "ugrizy": 0.2248884735953701,
            },
            {
                "cst": 0.0,
                "u": 0.03224724233127598,
                "g": 0.018519321739837606,
                "r": 0.02629040528053873,
                "i": -0.01879121908695766,
                "z": 0.043381666839462575,
                "y": 0.0465917194286579,
                "ugrizy": 0.1360573901416739,
            },
            {
                "cst": 0.0,
                "u": 0.022608143953455426,
                "g": 0.029168555210221787,
                "r": 0.04195762939125768,
                "i": -0.051460395769436555,
                "z": 0.03947548537932443,
                "y": 0.0191135636135187,
                "ugrizy": 0.10394405843289192,
            },
            {
                "cst": 0.0,
                "u": 0.008432569998479788,
                "g": 0.015244199381057874,
                "r": 0.0564324179119396,
                "i": 0.0034085056574326596,
                "z": -0.0231010021253421,
                "y": 0.049696845834184995,
                "ugrizy": 0.10479317991975469,
            },
        ],
    },
    "year7": {
        "sigma8square_model": 0.8**2.0,
        "poly1d_coefs_loglog": np.array(
            [
                [
                    -7.18947651e-04,
                    2.52048402e-02,
                    -2.24128689e-01,
                    4.69247373e-01,
                    3.39996105e-01,
                    -1.13691749e01,
                ],
                [
                    4.66194938e-04,
                    5.71142011e-03,
                    -1.31505217e-01,
                    3.74462215e-01,
                    3.97920687e-01,
                    -1.28754286e01,
                ],
                [
                    1.11096134e-03,
                    -6.32112700e-03,
                    -6.34065660e-02,
                    2.65513757e-01,
                    4.64794474e-01,
                    -1.39828608e01,
                ],
                [
                    1.50696925e-03,
                    -1.40997435e-02,
                    -1.60608483e-02,
                    1.78259845e-01,
                    5.21324331e-01,
                    -1.47778620e01,
                ],
                [
                    1.89997066e-03,
                    -2.16155858e-02,
                    3.08453940e-02,
                    8.11771421e-02,
                    5.82883411e-01,
                    -1.55574865e01,
                ],
            ]
        ),
        "lmax": np.array([63, 94, 123, 148, 170]),
        "dlogN_dm5": [
            {
                "cst": 0.0,
                "u": 0.04691474328780031,
                "g": 0.010732781728918056,
                "r": 0.010261971860717898,
                "i": 0.022190117396450985,
                "z": 0.05969082742663293,
                "y": 0.06903541851329315,
                "ugrizy": 0.2089634228500948,
            },
            {
                "cst": 0.0,
                "u": 0.0586609387003185,
                "g": 0.02926172329832247,
                "r": -0.00867144813333158,
                "i": 0.03238973478100006,
                "z": 0.06403490756697708,
                "y": 0.061258549036290445,
                "ugrizy": 0.21531623412108158,
            },
            {
                "cst": 0.0,
                "u": 0.037673080027243955,
                "g": 0.024329841045213924,
                "r": 0.024028335574766118,
                "i": -0.004173864313724429,
                "z": 0.04098643494507197,
                "y": 0.04706774658791262,
                "ugrizy": 0.16192259431907577,
            },
            {
                "cst": 0.0,
                "u": 0.014161985933060993,
                "g": 0.02818398259140731,
                "r": 0.03724357335900577,
                "i": -0.05102903641687596,
                "z": 0.04037709323317632,
                "y": 0.019008759510214236,
                "ugrizy": 0.08457753101369372,
            },
            {
                "cst": 0.0,
                "u": 0.021186578579099866,
                "g": 0.01743166214581306,
                "r": 0.06093757433043147,
                "i": -0.004343243303855121,
                "z": -0.026688237135998293,
                "y": 0.03848277935815396,
                "ugrizy": 0.1090269688631813,
            },
        ],
    },
    "year8": {
        "sigma8square_model": 0.8**2.0,
        "poly1d_coefs_loglog": np.array(
            [
                [
                    -7.20978717e-04,
                    2.52386810e-02,
                    -2.24297998e-01,
                    4.69472361e-01,
                    3.39895949e-01,
                    -1.13578618e01,
                ],
                [
                    4.67843599e-04,
                    5.69127210e-03,
                    -1.31454331e-01,
                    3.74573469e-01,
                    3.97784625e-01,
                    -1.28655285e01,
                ],
                [
                    1.11711699e-03,
                    -6.42150038e-03,
                    -6.29326395e-02,
                    2.65065150e-01,
                    4.65281866e-01,
                    -1.39739921e01,
                ],
                [
                    1.51463261e-03,
                    -1.42349222e-02,
                    -1.53268735e-02,
                    1.77158327e-01,
                    5.22084943e-01,
                    -1.47629374e01,
                ],
                [
                    1.89029193e-03,
                    -2.14244957e-02,
                    2.94913874e-02,
                    8.48496470e-02,
                    5.82823818e-01,
                    -1.55582137e01,
                ],
            ]
        ),
        "lmax": np.array([63, 94, 123, 148, 170]),
        "dlogN_dm5": [
            {
                "cst": 0.0,
                "u": 0.04786137493054727,
                "g": 0.009655534393294174,
                "r": -0.0004343582618342657,
                "i": 0.022915274064171095,
                "z": 0.059461135879046355,
                "y": 0.0620954964843791,
                "ugrizy": 0.19278674763853387,
            },
            {
                "cst": 0.0,
                "u": 0.061847229214042274,
                "g": 0.03327254021698463,
                "r": -0.003037483153393612,
                "i": 0.032612371741557,
                "z": 0.06330901593731424,
                "y": 0.06700098294204117,
                "ugrizy": 0.22947625091188906,
            },
            {
                "cst": 0.0,
                "u": 0.033831383811744026,
                "g": 0.019055419055419107,
                "r": 0.02176497146894746,
                "i": -0.005173853421466327,
                "z": 0.04318093054632704,
                "y": 0.05310544905099789,
                "ugrizy": 0.14598595925075922,
            },
            {
                "cst": 0.0,
                "u": 0.014811863932028239,
                "g": 0.029658558690905158,
                "r": 0.03239827338991352,
                "i": -0.0542981307448116,
                "z": 0.03624916138976098,
                "y": 0.017430252211788642,
                "ugrizy": 0.0816796281398052,
            },
            {
                "cst": 0.0,
                "u": 0.008756393562914043,
                "g": 0.010614732066291235,
                "r": 0.05656537211028906,
                "i": 0.001649522457501062,
                "z": -0.02285214586340533,
                "y": 0.04088282647966966,
                "ugrizy": 0.10694124411573667,
            },
        ],
    },
    "year9": {
        "sigma8square_model": 0.8**2.0,
        "poly1d_coefs_loglog": np.array(
            [
                [
                    -7.23432734e-04,
                    2.52943590e-02,
                    -2.24707794e-01,
                    4.70555869e-01,
                    3.39390877e-01,
                    -1.13448965e01,
                ],
                [
                    4.68298841e-04,
                    5.69359050e-03,
                    -1.31559784e-01,
                    3.75124408e-01,
                    3.97498952e-01,
                    -1.28555902e01,
                ],
                [
                    1.11687532e-03,
                    -6.40919987e-03,
                    -6.30785495e-02,
                    2.65647380e-01,
                    4.64932561e-01,
                    -1.39661254e01,
                ],
                [
                    1.51771172e-03,
                    -1.42863837e-02,
                    -1.50790664e-02,
                    1.76934887e-01,
                    5.22416665e-01,
                    -1.47607945e01,
                ],
                [
                    1.90353259e-03,
                    -2.16360592e-02,
                    3.05135359e-02,
                    8.36649036e-02,
                    5.84231771e-01,
                    -1.55657045e01,
                ],
            ]
        ),
        "lmax": np.array([63, 94, 123, 148, 170]),
        "dlogN_dm5": [
            {
                "cst": 0.0,
                "u": 0.044390298549263456,
                "g": 0.011808056276693719,
                "r": 0.006910724770876357,
                "i": 0.018412740395619607,
                "z": 0.05811594349550721,
                "y": 0.06252677636632357,
                "ugrizy": 0.18925712972055841,
            },
            {
                "cst": 0.0,
                "u": 0.06119330455885738,
                "g": 0.024830399951491312,
                "r": -0.006768194405460846,
                "i": 0.030065619863371994,
                "z": 0.06255158775650581,
                "y": 0.07147566736996605,
                "ugrizy": 0.22306568518819123,
            },
            {
                "cst": 0.0,
                "u": 0.03789658719578499,
                "g": 0.027326499247561357,
                "r": 0.022781496880732575,
                "i": -0.007626568451506756,
                "z": 0.04268771512656232,
                "y": 0.04633870731223419,
                "ugrizy": 0.1571925190810045,
            },
            {
                "cst": 0.0,
                "u": 0.01457970225016212,
                "g": 0.027698293355913416,
                "r": 0.03530620058462831,
                "i": -0.04816414607472254,
                "z": 0.033660844914206074,
                "y": 0.022692062998703265,
                "ugrizy": 0.08856228929221631,
            },
            {
                "cst": 0.0,
                "u": 0.01976694240068723,
                "g": 0.01451018240142195,
                "r": 0.04892114102580124,
                "i": 0.004035038256796502,
                "z": -0.024440454014018773,
                "y": 0.03989380894491724,
                "ugrizy": 0.10691935463909878,
            },
        ],
    },
    "year10": {
        "sigma8square_model": 0.8**2.0,
        "poly1d_coefs_loglog": np.array(
            [
                [
                    -7.16061620e-04,
                    2.51744502e-02,
                    -2.24107456e-01,
                    4.69700181e-01,
                    3.39695775e-01,
                    -1.13492481e01,
                ],
                [
                    4.69488083e-04,
                    5.67705574e-03,
                    -1.31512276e-01,
                    3.75205008e-01,
                    3.97691903e-01,
                    -1.28530855e01,
                ],
                [
                    1.11838359e-03,
                    -6.43543666e-03,
                    -6.29190558e-02,
                    2.65259815e-01,
                    4.65238973e-01,
                    -1.39566261e01,
                ],
                [
                    1.52245292e-03,
                    -1.43703455e-02,
                    -1.45911836e-02,
                    1.75990818e-01,
                    5.22698913e-01,
                    -1.47457598e01,
                ],
                [
                    1.92138091e-03,
                    -2.19184749e-02,
                    3.20303750e-02,
                    8.07319417e-02,
                    5.85683971e-01,
                    -1.55499003e01,
                ],
            ]
        ),
        "lmax": np.array([63, 95, 123, 148, 170]),
        "dlogN_dm5": [
            {
                "cst": 0.0,
                "u": 0.05535070282190964,
                "g": 0.011193545843687215,
                "r": 0.006737582346111497,
                "i": 0.025833960371206527,
                "z": 0.06136534095644072,
                "y": 0.06324681820708049,
                "ugrizy": 0.2074608064455781,
            },
            {
                "cst": 0.0,
                "u": 0.05615851073325299,
                "g": 0.029973080589983655,
                "r": -0.008700108117848502,
                "i": 0.030847889011579494,
                "z": 0.06273045722713866,
                "y": 0.0650054579514396,
                "ugrizy": 0.2228857063055601,
            },
            {
                "cst": 0.0,
                "u": 0.03889599549031938,
                "g": 0.02232895038144942,
                "r": 0.02262813226603151,
                "i": -0.0020788337259347776,
                "z": 0.04185161853072303,
                "y": 0.046998102923906174,
                "ugrizy": 0.1611649091746466,
            },
            {
                "cst": 0.0,
                "u": 0.01329424791323612,
                "g": 0.025204737395854025,
                "r": 0.031181310573597302,
                "i": -0.046678413257774214,
                "z": 0.04040377859096919,
                "y": 0.020907507991371956,
                "ugrizy": 0.08643457382953187,
            },
            {
                "cst": 0.0,
                "u": 0.010769393019973022,
                "g": 0.018559094751972344,
                "r": 0.04731613285883742,
                "i": -0.002550702324093634,
                "z": -0.025182273951223154,
                "y": 0.04311795497390813,
                "ugrizy": 0.08119502007816377,
            },
        ],
    },
}
