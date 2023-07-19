###############################################################################################################################
# The goal here is to implement Eq. 9.4 from the LSST community WP, which defines our FoM, and create plots.
#
# Humna Awan: humna.awan@rutgers.edu
#
###############################################################################################################################
__all__ = (
    "get_fsky",
    "get_theory_spectra",
    "get_outdir_name",
    "return_cls",
    "calc_os_bias_err",
    "get_fom",
    "os_bias_overplots",
    "os_bias_overplots_diff_dbs",
)

import datetime
import os
from collections import OrderedDict

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from rubin_sim.maf.maf_contrib.lss_obs_strategy.constants_for_pipeline import power_law_const_a


###############################################################################################################################
# calculate fsky for a bundle
def get_fsky(outdir, band="i", print_fsky=True):
    """

    Calculate the fraction of the sky observed in a survey. The data must have been saved as
    .npz files in the given output directory. The method only looks at the mask of the data array.

    Filenames should be in the format: <whatever>_<band>_<dither_strategy>.npz

    Parameters
    -------------------
    outdir : str
        name of the output directory where the data-to-look-at is.
    band: str
        band to consider. Default: 'i'
    print_fsky: `bool`
        set to True if want to print( out fsky. Default: True

    """
    filenames = [f for f in os.listdir(outdir) if any([f.endswith("npz")])]
    fsky = {}
    for filename in filenames:
        # print('Reading in %s for fsky'%filename)
        dither_strategy = filename.split("%s_" % band)[1].split(".npz")[0]
        data = np.load("%s/%s" % (outdir, filename))
        # total number of pixels in the sky
        tot_pix = float(len(data["mask"]))
        in_survey_pix = float(len(np.where(data["mask"] == False)[0]))
        fsky[dither_strategy] = in_survey_pix / tot_pix
        if print_fsky:
            print("%s fsky: %s\n" % (dither_strategy, fsky[dither_strategy]))
    return fsky


###############################################################################################################################
def get_theory_spectra(mock_data_path, mag_cut=25.6, plot_spectra=True, nside=256):
    """

    Return the data for the five redshift bins, read from the files containing the
    with BAO galaxy power spectra from Hu Zhan.

    Parameters
    -------------------
    mock_data_path : `str`
        path to the folder with the theory spectra
    mag_cut: `float`
        r-band magnitude cut as the identifer in the filename from Hu.
        allowed options: 24.0, 25.6, 27.5. Default: 25.6
    plot_spectra : `bool`
        set to True if want to plot out the skymaps. Default: True
    nside : `int`
        HEALpix resolution parameter. Default: 256

    Returns
    -------
    ell : `np.ndarray`
        array containing the ells
    w_bao_cls : `dict`
        keys = zbin_tags; data = spectra (pixelized for specified nside)
    surf_num_density : `float`
        surface number density in 1/Sr
    """
    # read in the galaxy power spectra with the BAO
    filename = "%s/cls015-200z_r%s.bins" % (mock_data_path, mag_cut)
    print("\nReading in %s for theory cls." % filename)
    shot_noise_data = np.genfromtxt(
        filename
    )  # last column = surface number density for each bin in 1/(sq arcmin)

    filename = "%s/cls015-200z_r%s" % (mock_data_path, mag_cut)
    print("Reading in %s for theory cls." % filename)
    all_data = np.genfromtxt(filename)

    # set up to read the data
    ell = []
    w_bao_cls = OrderedDict()
    w_bao_cls["0.15<z<0.37"] = []
    w_bao_cls["0.37<z<0.66"] = []
    w_bao_cls["0.66<z<1.0"] = []
    w_bao_cls["1.0<z<1.5"] = []
    w_bao_cls["1.5<z<2.0"] = []
    surf_num_density = OrderedDict()

    # read in the cls
    for i in range(len(all_data)):
        ell.append(all_data[i][0])
        w_bao_cls["0.15<z<0.37"].append(all_data[i][1])
        w_bao_cls["0.37<z<0.66"].append(all_data[i][3])
        w_bao_cls["0.66<z<1.0"].append(all_data[i][5])
        w_bao_cls["1.0<z<1.5"].append(all_data[i][7])
        w_bao_cls["1.5<z<2.0"].append(all_data[i][9])

    # read in the surface number density and convert
    for j, key in enumerate(w_bao_cls.keys()):
        surf_num_density[key] = np.array(shot_noise_data[j][5] * 1.18 * 10**7)  # 1/arcmin^2 to 1/Sr

    # want Cl*W^2 where W is the pixel window function
    wl_nside = hp.sphtfunc.pixwin(nside=nside)

    # account for Hu's ells not starting with 0 but with ell=2
    wl_nside = wl_nside[2:]
    lmax = len(wl_nside)
    ell = np.array(ell[0:lmax])
    # pixelize the spectra
    for key in w_bao_cls:
        w_bao_cls[key] = np.array(w_bao_cls[key][0:lmax]) * (wl_nside**2)

    if plot_spectra:
        plt.clf()
        for key in w_bao_cls:
            plt.plot(
                ell,
                w_bao_cls[key] * ell * (ell + 1) / (2 * np.pi),
                linewidth=1.5,
                label=key,
            )
        plt.legend(bbox_to_anchor=(1, 1))
        plt.xlabel("$\ell$")
        plt.ylabel("$\ell(\ell+1)C_\ell/2\pi$")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        plt.show()

    return ell, w_bao_cls, surf_num_density


###############################################################################################################################
# return the outputDir name where the relevant c_ells are.
def get_outdir_name(
    band,
    nside,
    pixel_radius,
    yr_cutoff,
    zbin,
    mag_cut_i,
    run_name,
    poisson_noise,
    zero_pt,
):
    """

    Return the output directory name where the cls for deltaN/N would be, given the input parameters.
    We assume that dust extinction is always added and counts are always normalized.

    Returns: [outdir, yr_tag, zbin_tag, poisson_tag, zero_pt_tag]

    Parameters
    -------------------
      * band: str: band to get the output directory name for.
                   Options: 'u', 'g', 'r', 'i'
      * nside: int: HEALpix resolution parameter.
      * pixel_radius: int: number of pixels to mask along the shallow border
      * yr_cutoff: int: year cut to restrict analysis to only a subset of the survey.
                         Must range from 1 to 9, or None for the full survey analysis (10 yrs).
      * zbin: str: options include '0.<z<0.15', '0.15<z<0.37', '0.37<z<0.66, '0.66<z<1.0',
                          '1.0<z<1.5', '1.5<z<2.0', '2.0<z<2.5', '2.5<z<3.0','3.0<z<3.5', '3.5<z<4.0',
                          'all' for no redshift restriction (i.e. 0.<z<4.0)
      * mag_cut_i: float: upper limit on i-band magnitude when calculating the galaxy counts. will deal
                         with color correction automatically (for u,g,r bands ..) depending on the band
      * run_name: str: run name tag to identify the output of specified OpSim output.
                      Since new OpSim outputs have different columns, the run_name for enigma_1189 **must**
                      be 'enigma1189'; can be anything for other outputs, e.g. 'minion1016'
      * poisson_noise: bool: set to True to consider the case where poisson noise is added to  galaxy
                                   counts after border masking (and the incorporation of calibration errors).
      * zero_pt: bool: set to True to consider the case where photometric calibration errors were incorporated.

    """
    # check to make sure redshift bin is ok.
    allowed_zbins = power_law_const_a.keys()
    if zbin not in allowed_zbins:
        raise ValueError("ERROR: Invalid redshift bin. Input bin can only be among %s\n" % (allowed_zbins))

    # set up the tags.
    dust_tag = "withDustExtinction"

    zero_pt_tag = ""
    if zero_pt:
        zero_pt_tag = "with0ptErrors"
    else:
        zero_pt_tag = "no0ptErrors"

    if yr_cutoff is not None:
        yr_tag = "%syearCut" % yr_cutoff
    else:
        yr_tag = "fullSurveyPeriod"

    zbin_tag = zbin
    if zbin == "all":
        zbin_tag = "allRedshiftData"

    if poisson_noise:
        poisson_tag = "withPoissonNoise"
    else:
        poisson_tag = "noPoissonNoise"

    norm_tag = "normalizedGalaxyCounts"

    # account for color corrections.
    mag_cut = {}
    mag_cut["i"] = mag_cut_i
    mag_cut["r"] = float("%.1f" % (mag_cut["i"] + 0.4))
    mag_cut["g"] = float("%.1f" % (mag_cut["r"] + 0.4))
    mag_cut["u"] = float("%.1f" % (mag_cut["g"] + 0.4))

    outdir = "artificialStructure_%s_nside%s_%spixelRadiusForMasking_%s_%s" % (
        poisson_tag,
        nside,
        pixel_radius,
        zero_pt_tag,
        dust_tag,
    )
    outdir += "_%s<%s_%s_%s_%s_%s_directory/" % (
        band,
        mag_cut[band],
        run_name,
        yr_tag,
        zbin_tag,
        norm_tag,
    )

    return [outdir, yr_tag, zbin_tag, poisson_tag, zero_pt_tag]


###############################################################################################################################
# create a function that returns cls for a given band, nside, pixel_radius
def return_cls(path, outdir, band, specified_dith=None):
    """

    Get the cls from .npy files in path+outdir folder for a specified filter band.

    Returns the data in the form of a dictonary with dither strategies as keys.

    Parameters
    -------------------
      * path: str: path to the main directory where directories for the outputs from
                   artificialStructure are saved.
      * outdir: str: name of the directory where the cls are situated.
      * band: str: band to consider. Options: 'u', 'g', 'r', 'i', 'z', 'y'
      * specified_dith: list of str: list of the names (strings) of the dither strategies to consider, e.g.
                                    if want to plot only NoDither, specified_dith_only= ['NoDither']
    """
    if specified_dith is None:
        consider_all_npy = True
    else:
        consider_all_npy = False
    # get the filename
    filenames = [f for f in os.listdir("%s%s" % (path, outdir)) if any([f.endswith("npy")])]
    # read in the cls
    c_ells = {}
    for filename in filenames:
        if consider_all_npy:
            dither_strategy = filename.split("%s_" % band)[1].split(".npy")[0]
            c_ells[dither_strategy] = np.load("%s%s/%s" % (path, outdir, filename))
        else:
            for dith in specified_dith:
                if filename.__contains__(dith):
                    dither_strategy = filename.split("%s_" % band)[1].split(".npy")[0]
                    c_ells[dither_strategy] = np.load("%s%s/%s" % (path, outdir, filename))
    if specified_dith is not None:
        for dith in specified_dith:
            if dith not in c_ells:
                raise ValueError("Cls for %s are not found in %s." % (dith, outdir))
    return c_ells


###############################################################################################################################
def calc_os_bias_err(c_ells):
    """

    Calculate the OS bias (as an average across the specified bands) and the uncertainty in the
    bias (as the std across the cls from thes specified bands).

    Returns two dictionaries: [bias, bias_err]

    Parameters
    ----------
    c_ells: dictionary
        bands as keys, mapping the cls corresponding to the bands.

    """
    bias, bias_err = {}, {}
    band_keys = list(c_ells.keys())
    for dith in c_ells[band_keys[0]]:  # loop over each dith strategy
        temp_avg, temp_err = [], []
        for data_index in range(len(c_ells[band_keys[0]][dith])):  # loop over each C_ell-value
            row = []
            for band in band_keys:
                row.append(c_ells[band][dith][data_index])  # compiles the C_ell value for each band
            temp_avg.append(np.mean(row))
            temp_err.append(np.std(row))
        bias[dith] = temp_avg
        bias_err[dith] = temp_err
    return [bias, bias_err]


###############################################################################################################################
def get_stat_floor(
    ell_arr,
    zbin,
    w_bao_cls_zbin,
    surf_num_density_zbin,
    dither_strategy,
    fsky,
    with_shot_noise=True,
):
    # returns the sqrt of cosmic variance = statistical floor
    prefactor = np.sqrt(2.0 / ((2 * ell_arr + 1) * fsky))
    if with_shot_noise:
        return prefactor * (w_bao_cls_zbin + (1.0 / surf_num_density_zbin))
    else:
        return prefactor * (w_bao_cls_zbin)


###############################################################################################################################
def get_fom(
    ell_min,
    ell_max,
    ell_for_bias_err,
    bias_err,
    ell_stat_floor,
    floor_with_shot_noise,
    floor_wo_shot_noise,
):
    """

    Calculate the FoM based on the bias uncertaity and statistical floor. Returns the FoM (float).

    Parameters
    -------------------
      * ell_min: int: minimum ell-value to consider over which the FoM is calculated.
      * ell_max: int: maximum ell-value to consider over which the FoM is calculated.
      * ell_for_bias_err: array: ell-array corresponding to the bias_err array.
      * bias_err: array: array containing the bias_err for each ell.
      * ell_stat_floor: array: ell-array corresponding to the get_stat_floor array.
      * floor_with_shot_noise: array: array containing the statistical floor (for each ell)  with shot noise contribution.
      * floor_wo_shot_noise: array: array containing the statistical floor (for each ell) without shot noise contribution.

    """
    # need to adjust for the ell's not always starting with 0.
    osbias = np.array(bias_err)[ell_min - ell_for_bias_err[0] : ell_max - ell_for_bias_err[0] + 1]
    floor_with_shot_noise = floor_with_shot_noise[
        int(ell_min - ell_stat_floor[0]) : int(ell_max - ell_stat_floor[0] + 1)
    ]
    floor_wo_shot_noise = floor_wo_shot_noise[
        int(ell_min - ell_stat_floor[0]) : int(ell_max - ell_stat_floor[0] + 1)
    ]

    l_good = np.arange(ell_min, ell_max + 1)
    num_sq = np.sum(floor_wo_shot_noise**2)
    denom_sq = np.sum(floor_with_shot_noise**2 + osbias**2)

    return np.sqrt(num_sq / denom_sq)


###############################################################################################################################
def os_bias_overplots(
    out_dir,
    data_paths,
    lim_mags_i,
    legend_labels,
    fsky_dith_dict,
    fsky_best,
    mock_data_path,
    run_name,
    theory_lim_mag,
    specified_dith_only=None,
    run_name_filetag=None,
    ell_min=100,
    ell_max=300,
    lmax=500,
    filters=["u", "g", "r", "i"],
    nside=256,
    pixel_radius=14,
    yr_cutoff=None,
    zbin="0.66<z<1.0",
    poisson_noise=False,
    zero_pt=True,
    plot_interms=False,
    color_dict=None,
    ylim_min=None,
    ylim_max=None,
    show_plot=False,
    suptitle=None,
    file_append=None,
):
    """

    Calculate/plot the OS bias uncertainty and the statistical floor for the specified redshift bin.

    Could vary the dither strategies, but the data should be from the same OpSim run. The title of
    of each panel in final plot will be <dither strategy>, and each panel can have OS bias
    uncertainity from many different galaxy catalogs. Panel legends will specify the redshift bin
    and the magnitude cut.

    Parameters
    -------------------
      * out_dir: str: output directory where the output plots will be saved; a folder named
                      'os_bias_overplots' will be created in the directory, if its not there already.
      * data_paths: list of strings: list of strings containing the paths where the artificialStructure data will be
                                     found for the filters specified.
      * lim_mags_i: list of floats: list of the i-band magnitude cuts to get the data for.
      * legend_labels: list of strings: list of the 'tags' for each case; will be used in the legends. e.g. if
                                          lim_mags_i=[24.0, 25.3], legend_labels could be ['i<24.0', 'i<25.3']
                                          or ['r<24.4', 'i<25.7'] if want to label stuff with r-band limits.
      * fsky_dith_dict: dict: dictionary containing the fraction of sky covered; keys should be the dither strategies.
                                The function get_fsky outputs the right dictionary.
      * fsky_best: float: best fsky for the survey to compare everything relative to.
      * mock_data_path: str: path to the mock data to consider
      * run_name: str: run name tag to identify the output of specified OpSim output.
      * theory_lim_mag: float: magnitude cut as the identifer in the filename from Hu.
                               Allowed options: 24.0, 25.6, 27.5
      * specified_dith_only: list of string: list of the names (strings) of the dither strategies to consider, e.g.
                                               if want to plot only NoDither, specified_dith_only= ['NoDither']. If
                                               nothing is specified, all the dither strategies will be considered
                                               (based on the npy files available for the runs). Default: None
      * run_name_filtag: str: run name file tag. Default: None
      * filters: list of strings: list containing the bands (in strings) to be used to calculate the OS bias
                                  and its error. should contain AT LEAST two bands.
                                  e.g. if filters=['g', 'r'], OS bias (at every ell) will be calculated as the
                                  mean across g and r cls, while the bias error (at every ell) will be calculated
                                  as the std. dev. across g and r cls.
                                  Default: ['u', 'g', 'r', 'i']
      * nside: int: HEALpix resolution parameter. Default: 256
      * pixel_radius: int: number of pixels to mask along the shallow border. Default: 14
      * yr_cutoff: int: year cut to restrict analysis to only a subset of the survey.
                        Must range from 1 to 9, or None for the full survey analysis (10 yrs).
                        Default: None
      * zbin: str: options include '0.15<z<0.37', '0.37<z<0.66, '0.66<z<1.0', '1.0<z<1.5', '1.5<z<2.0'
                   Default: '0.66<z<1.0'
      * poisson_noise: bool: set to True to consider the case where poisson noise is added to galaxy counts
                                   after border masking (and the incorporation of calibration errors).
                                   Default: False
      * zero_pt: bool: set to True to consider the case where 0pt calibration errors were incorporated.
                          Default: True
      * plot_interms: bool: set to True to plot intermediate plots, e.g. BAO data. Default: False
      * color_dict: dict: color dictionary; keys should be the indentifiers provided. Default: None
                    **** Please note that in-built colors are for only a few indentifiers:
                        'r<24.0'], 'r<25.7','r<27.5', 'r<22.0', 'i<24.0', 'i<25.3','i<27.5', 'i<22.' ******
      * ylim_min: float: lower y-lim for the final plots. Defaut: None
      * ylim_max: float: upper y-lim for the final plots. Defaut: None
      * show_plot: bool: set to True if want to display the plot (aside from saving it). Default: False
      * suptitle: str: title to plot. Default: None
      * file_append: str: optional string to append to the saved plot

    """
    # check to make sure redshift bin is ok.
    allowed_zbins = list(power_law_const_a.keys()) + ["all"]
    if zbin not in allowed_zbins:
        raise ValueError("Invalid redshift bin. Input bin can only be among %s\n" % (allowed_zbins))

    # check to make sure we have at least two bands to calculate the bias uncertainty.
    if len(filters) < 2:
        raise ValueError(
            "Need at least two filters to calculate bias uncertainty. Currently given only: %s\n" % filters
        )

    # all is ok. proceed.
    tot_cases = len(data_paths)

    # get the outdir address for each 'case' and each band
    outdir_all = {}
    for i in range(tot_cases):
        outdir = {}
        for band in filters:
            out, yr_tag, zbin_tag, poisson_tag, zero_pt_tag = get_outdir_name(
                band,
                nside,
                pixel_radius,
                yr_cutoff,
                zbin,
                lim_mags_i[i],
                run_name,
                poisson_noise,
                zero_pt,
            )
            outdir[band] = "%s/cls_DeltaByN/" % out
        outdir_all[legend_labels[i]] = outdir

    # get the cls and calculate the OS bias and error.
    osbias_all, osbias_err_all = {}, {}
    for i in range(tot_cases):
        outdir = outdir_all[legend_labels[i]]

        c_ells = {}
        for band in filters:
            c_ells[band] = return_cls(
                data_paths[i], outdir[band], band, specified_dith=specified_dith_only
            )  # get the c_ells
        osbias, osbias_err = calc_os_bias_err(c_ells)
        osbias_all[legend_labels[i]] = osbias
        osbias_err_all[legend_labels[i]] = osbias_err

    # get the data to calculate the statistical floor.
    ell, w_bao_cls, surf_num_density = get_theory_spectra(
        mock_data_path=mock_data_path,
        mag_cut=theory_lim_mag,
        plot_spectra=plot_interms,
        nside=nside,
    )
    ########################################################################################################
    # set the directory
    outdir = "os_bias_overplots"
    if not os.path.exists("%s/%s" % (out_dir, outdir)):
        os.makedirs("%s/%s" % (out_dir, outdir))

    in_built_colors = {}
    in_built_colors["r<24.0"] = "r"
    in_built_colors["r<25.7"] = "b"
    in_built_colors["r<27.5"] = "g"
    in_built_colors["r<22.0"] = "m"
    in_built_colors["i<24.0"] = "r"
    in_built_colors["i<25.3"] = "b"
    in_built_colors["i<27.5"] = "g"
    in_built_colors["i<22.0"] = "m"

    if color_dict is None:
        colors = in_built_colors
    else:
        colors = color_dict

    # figure out max how many panels to create
    diths_to_consider = []
    if specified_dith_only is not None:
        diths_to_consider = specified_dith_only
        max_entries = len(diths_to_consider)
    else:
        diths_to_consider = colors.keys()
        max_entries = 0
        for identifier in legend_labels:
            max_entries = max(max_entries, len(list(osbias_err_all[identifier].keys())))

    ncols = 2
    if max_entries == 1:
        ncols = 1
    nrows = int(np.ceil(max_entries / ncols))

    # set up the figure
    plt.clf()
    fig, ax = plt.subplots(nrows, ncols)
    fig.subplots_adjust(hspace=0.4)
    row, col = 0, 0
    # run over the keys
    for dith in diths_to_consider:
        # ----------------------------------------------------------------------------------------
        for i in range(tot_cases):
            # ----------------------------------------------------------------------------------------
            osbias_err = osbias_err_all[legend_labels[i]]
            if dith in osbias_err.keys():
                # ----------------------------------------------------------------------------------------
                if nrows == 1:
                    if ncols == 1:
                        axis = ax
                    else:
                        axis = ax[col]
                else:
                    axis = ax[row, col]

                # ----------------------------------------------------------------------------------------
                # calcuate the floors with and without shot noise
                if zbin not in w_bao_cls:
                    raise ValueError("Invalid redshift bin: %s" % zbin)
                else:
                    # get the floor with shot noise
                    floor_with_eta = get_stat_floor(
                        ell_arr=ell,
                        zbin=zbin,
                        w_bao_cls_zbin=w_bao_cls[zbin],
                        surf_num_density_zbin=surf_num_density[zbin],
                        dither_strategy=dith,
                        fsky=fsky_dith_dict[dith],
                        with_shot_noise=True,
                    )
                    # get the "best" floor for the fom calculation
                    floor_no_eta = get_stat_floor(
                        ell_arr=ell,
                        zbin=zbin,
                        w_bao_cls_zbin=w_bao_cls[zbin],
                        surf_num_density_zbin=surf_num_density[zbin],
                        dither_strategy=dith,
                        fsky=fsky_best,
                        with_shot_noise=False,
                    )
                # calculate the fo_m
                l = np.arange(np.size(osbias_err[dith]))
                fo_m = get_fom(
                    ell_min,
                    ell_max,
                    l,
                    osbias_err[dith],
                    ell,
                    floor_with_eta,
                    floor_no_eta,
                )
                # ----------------------------------------------------------------------------------------
                if i == 0:
                    # plot the floor with shot noise
                    axis.plot(
                        ell,
                        floor_with_eta,
                        color="k",
                        lw=2.0,
                        label="$\Delta$C$_\ell$: $%s$" % zbin,
                    )

                # set up the legend
                add_leg = ""
                if i == 0:
                    add_leg = "$\mathrm{\sigma_{C_{\ell,OS}}}$: "
                else:
                    add_leg = "         "

                # plot the bias error
                splits = legend_labels[i].split("<")
                axis.plot(
                    l,
                    osbias_err[dith],
                    color=colors[legend_labels[i]],
                    label=r"%s$%s<%s$ ; fo_m: %.6f" % (add_leg, splits[0], splits[1], fo_m),
                )
                # ----------------------------------------------------------------------------------------
                # set up the details of the plot
                if (ylim_min is not None) & (ylim_max is not None):
                    axis.set_ylim(ylim_min, ylim_max)
                else:
                    ax.set_ylim(0, 0.00001)
                axis.set_title(dith)
                axis.set_xlabel(r"$\ell$")
                axis.set_xlim(0, lmax)
                if tot_cases > 4:
                    leg = axis.legend(
                        labelspacing=0.001,
                    )
                else:
                    leg = axis.legend()
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.0)
        row += 1
        if row > nrows - 1:
            row = 0
            col += 1
    # ----------------------------------------------------------------------------------------
    if suptitle is not None:
        plt.suptitle(suptitle)
    # turn stuff off if have odd number of panels
    if (max_entries % 2 != 0) and (max_entries > 1):  # i.e. have odd number of diths
        ax[nrows - 1, ncols - 1].axis("off")

    width = 20
    fig.set_size_inches(width, int(nrows * 30 / 7.0))

    # set up the filename
    dith_tag = ""
    if specified_dith_only is not None:
        dith_tag = "_%sdiths" % max_entries
    if run_name_filetag is None:
        run_name_filetag = run_name
    date_tag = datetime.date.isoformat(datetime.date.today())
    bias_type_tag = "".join(str(x) for x in filters)
    ell_tag = "_%s<ell<%s" % (ell_min, ell_max)
    if file_append is None:
        file_append = ""

    filename = "%s_OSbiaserr_%s_%s%s_" % (
        date_tag,
        bias_type_tag,
        run_name_filetag,
        dith_tag,
    )
    if tot_cases == 1:
        filename += "%s_" % (legend_labels[0])
    else:
        filename += "%s-magcuts_" % (tot_cases)
    filename += "th-r<%s_%s_%s_%s_%s%s%s.png" % (
        theory_lim_mag,
        zbin_tag,
        yr_tag,
        poisson_tag,
        zero_pt_tag,
        ell_tag,
        file_append,
    )
    # save the plot
    plt.savefig("%s/%s/%s" % (out_dir, outdir, filename), format="png", bbox_inches="tight")

    print("\nSaved %s" % filename)
    if show_plot:
        plt.show()
    else:
        plt.close("all")


###############################################################################################################################
def os_bias_overplots_diff_dbs(
    out_dir,
    data_path,
    run_names,
    legend_labels,
    fsky_dict,
    fsky_best,
    mock_data_path,
    theory_lim_mag,
    lim_mag_i,
    ell_min=100,
    ell_max=300,
    lmax=500,
    specified_dith_only=None,
    filters=["u", "g", "r", "i"],
    nside=256,
    pixel_radius=14,
    yr_cutoff=None,
    zbin="0.66<z<1.0",
    poisson_noise=False,
    zero_pt=True,
    plot_interms=False,
    color_dict=None,
    ylim_min=None,
    ylim_max=None,
    show_plot=False,
    suptitle=None,
    file_append=None,
):
    """

    Calculate/plot the OS bias uncertainty and the statistical floor for the specified redshift bin.

    Could vary the dither strategies, but the data should be for the same magnitude cut. The title of
    of each panel in final plot will be "<dither strategy>, and each panel can have
    OS bias uncertainity from many different cadences. Panel legends will specify the redshift bin
    and OpSim output tag.

    Parameters
    -------------------
      * out_dir: str: main directory where the output plots should be saved; a folder named
                      'os_bias_overplots' will be created in the directory, if its not there already.
      * data_path: str: path to the artificialStructure data.
      * run_names: list of str: list for run name tags to identify the output of specified OpSim outputs.
      * legend_labels: list of strings: list of the 'tags' for each case; will be used in the legends. e.g. if
                                          run_names=['enigma1189', 'minion1016'], legend_labels could be
                                          ['enigma_1189', 'minion_1016'].
      * fsky_dict: dict: dictionary of the dictionaries containing the fraction of sky covered for each of the
                         cadences. The keys should match the identifier; fsky_dict[indentifiers[:]] should have
                         the dither strategies as the keys.
      * fsky_best: float: best fsky for the survey to compare everything relative to.
      * mock_data_path: str: path to the mock data to consider
      * theory_lim_mag: float: magnitude cut as the identifer in the filename from Hu.
                               Allowed options: 24.0, 25.6, 27.5
      * lim_mag_i: float: i-band magnitude cut to get the data for.
      * specified_dith_only: list of string: list of the names (strings) of the dither strategies to consider, e.g.
                                           if want to plot only NoDither, specified_dith_only=['NoDither']. If
                                           nothing is specified, all the dither strategies will be considered
                                           (based on the npy files available for the runs). Default: None
      * filters: list of strings: list containing the bands (in strings) to be used to calculate the OS bias
                                  and its error. should contain AT LEAST two bands.
                                  e.g. if filters=['g', 'r'], OS bias (at every ell) will be calculated as the
                                  mean across g and r c_ells, while the bias error (at every ell) will be calculated
                                  as the std. dev. across g and r c_ells.
                                  Default: ['u', 'g', 'r', 'i']
      * nside: int: HEALpix resolution parameter. Default: 256
      * pixel_radius: int: number of pixels to mask along the shallow border. Default: 14
      * yr_cutoff: int: year cut to restrict analysis to only a subset of the survey.
                         Must range from 1 to 9, or None for the full survey analysis (10 yrs).
                         Default: None
      * zbin: str: options include '0.15<z<0.37', '0.37<z<0.66, '0.66<z<1.0', '1.0<z<1.5', '1.5<z<2.0'
                           Default: '0.66<z<1.0'
      * poisson_noise: bool: set to True to consider the case where poisson noise is added to galaxy counts
                                   after border masking (and the incorporation of calibration errors).
                                   Default: False
      * zero_pt: bool: set to True to consider the case where 0pt calibration errors were incorporated.
                          Default: True
      * plot_interms: bool: set to True to plot intermediate plots, e.g. BAO data. Default: False
      * color_dict: dict: color dictionary; keys should be the indentifiers provided. Default: None
                    **** Please note that in-built colors are for only a few indentifiers:
                            minion1016, minion1020, kraken1043 ******
      * ylim_min: float: lower y-lim for the final plots. Defaut: None
      * ylim_max: float: upper y-lim for the final plots. Defaut: None
      * show_plot: bool: set to True if want to display the plot (aside from saving it). Default: False
      * suptitle: str: title to the plot. Default: None
      * file_append: str: optional string to append to the saved plot

    """
    # check to make sure redshift bin is ok.
    allowed_zbins = list(power_law_const_a.keys()) + ["all"]
    if zbin not in allowed_zbins:
        raise ValueError("Invalid redshift bin. Input bin can only be among %s\n" % (allowed_zbins))

    # check to make sure we have at least two bands to calculate the bias uncertainty.
    if len(filters) < 2:
        raise ValueError(
            "Need at least two filters to calculate bias uncertainty. Currently given only: %s\n" % filters
        )

    # all is ok. proceed.
    tot_cases = len(run_names)

    # get the outdir address for each 'case' and each band
    outdir_all = {}
    for i in range(tot_cases):
        outdir = {}
        for band in filters:
            out, yr_tag, zbin_tag, poisson_tag, zero_pt_tag = get_outdir_name(
                band,
                nside,
                pixel_radius,
                yr_cutoff,
                zbin,
                lim_mag_i,
                run_names[i],
                poisson_noise,
                zero_pt,
            )
            outdir[band] = "%s/cls_DeltaByN/" % out
        outdir_all[run_names[i]] = outdir

    # get the cls and calculate the OS bias and error.
    osbias_all, osbias_err_all = {}, {}
    for i in range(tot_cases):
        outdir = outdir_all[run_names[i]]
        c_ells = {}
        for band in filters:
            c_ells[band] = return_cls(
                data_path, outdir[band], band, specified_dith=specified_dith_only
            )  # get the c_ells
        osbias, osbias_err = calc_os_bias_err(c_ells)
        osbias_all[run_names[i]] = osbias
        osbias_err_all[run_names[i]] = osbias_err

    # print stuff
    print(
        "MagCuts: i<%s\nRedshift bin: %s, %s, %s, %s"
        % (lim_mag_i, zbin_tag, yr_tag, poisson_tag, zero_pt_tag)
    )

    # get the data to calculate the statistical floor.
    ell, w_bao_cls, surf_num_density = get_theory_spectra(
        mock_data_path=mock_data_path,
        mag_cut=theory_lim_mag,
        plot_spectra=plot_interms,
        nside=nside,
    )

    ########################################################################################################
    # set the directory
    outdir = "os_bias_overplots"
    if not os.path.exists("%s/%s" % (out_dir, outdir)):
        os.makedirs("%s/%s" % (out_dir, outdir))

    in_built_colors = {}
    in_built_colors["minion1016"] = "r"
    in_built_colors["minion1020"] = "b"
    in_built_colors["kraken1043"] = "g"

    if color_dict is None:
        colors = in_built_colors
    else:
        colors = color_dict

    # figure out how many panels we need
    max_entries = 0
    if specified_dith_only is not None:
        max_entries = len(specified_dith_only)
    else:
        for identifier in run_names:
            max_entries = max(max_entries, len(list(osbias_err_all[identifier].keys())))

    ncols = 2
    if max_entries == 1:
        ncols = 1
    nrows = int(np.ceil(max_entries / ncols))

    # set up the figure
    plt.clf()
    fig, ax = plt.subplots(nrows, ncols)
    fig.subplots_adjust(hspace=0.4)

    diths_to_consider = []
    if specified_dith_only is not None:
        diths_to_consider = specified_dith_only
    else:
        diths_to_consider = colors.keys()

    for i in range(tot_cases):
        # ----------------------------------------------------------------------------------------
        fsky = fsky_dict[run_names[i]]
        osbias_err = osbias_err_all[run_names[i]]
        row, col = 0, 0
        for dith in diths_to_consider:
            # ----------------------------------------------------------------------------------------
            if dith in osbias_err.keys():
                # ----------------------------------------------------------------------------------------
                # look at the appropriate axis.
                if nrows == 1:
                    if ncols == 1:
                        axis = ax
                    else:
                        axis = ax[col]
                else:
                    axis = ax[row, col]
                # ----------------------------------------------------------------------------------------
                # calcuate the floors with and without shot noise
                if zbin not in w_bao_cls:
                    raise ValueError("Invalid redshift bin: %s" % zbin)
                else:
                    # get the floor with shot noise
                    floor_with_eta = get_stat_floor(
                        ell_arr=ell,
                        zbin=zbin,
                        w_bao_cls_zbin=w_bao_cls[zbin],
                        surf_num_density_zbin=surf_num_density[zbin],
                        dither_strategy=dith,
                        fsky=fsky[dith],
                        with_shot_noise=True,
                    )
                    # get the "best" floor for the fom calculation
                    floor_no_eta = get_stat_floor(
                        ell_arr=ell,
                        zbin=zbin,
                        w_bao_cls_zbin=w_bao_cls[zbin],
                        surf_num_density_zbin=surf_num_density[zbin],
                        dither_strategy=dith,
                        fsky=fsky_best,
                        with_shot_noise=False,
                    )
                # calculate the fo_m
                l = np.arange(np.size(osbias_err[dith]))
                fo_m = get_fom(
                    ell_min,
                    ell_max,
                    l,
                    osbias_err[dith],
                    ell,
                    floor_with_eta,
                    floor_no_eta,
                )
                # ----------------------------------------------------------------------------------------
                if i == 0:
                    # plot the floor with shot noise
                    axis.plot(
                        ell,
                        floor_with_eta,
                        color="k",
                        lw=2.0,
                        label="$\Delta$C$_\ell$: $%s$" % zbin,
                    )

                # set up the legend
                add_leg = ""
                if i == 0:
                    add_leg = "$\mathrm{\sigma_{C_{\ell,OS}}}$: "
                else:
                    add_leg = "         "

                # plot the osbias error
                axis.plot(
                    l,
                    osbias_err[dith],
                    color=colors[run_names[i]],
                    label=r"%s%s ; fo_m: %.6f" % (add_leg, legend_labels[i], fo_m),
                )
                # ----------------------------------------------------------------------------------------

                # plot details
                if (ylim_min is not None) & (ylim_max is not None):
                    axis.set_ylim(ylim_min, ylim_max)
                else:
                    ax.set_ylim(0, 0.00001)
                axis.set_title("%s: $i<%s$" % (dith, lim_mag_i))
                axis.set_xlabel(r"$\ell$")
                axis.set_xlim(0, lmax)
                if tot_cases > 4:
                    leg = axis.legend(
                        labelspacing=0.001,
                    )
                else:
                    leg = axis.legend()
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(2.0)
                col += 1
                if col > ncols - 1:
                    col = 0
                    row += 1
    # title to the plot?
    if suptitle is not None:
        plt.suptitle(suptitle, y=1.05)
    # turn off axes on unused panels
    if (max_entries % 2 != 0) and (max_entries > 1):  # i.e. have odd number of diths
        ax[nrows - 1, ncols - 1].axis("off")
    fig.set_size_inches(20, int(nrows * 30 / 7.0))

    # set up the filename
    dith_tag = ""
    if specified_dith_only is not None:
        dith_tag = "%sdith" % max_entries
    date_tag = datetime.date.isoformat(datetime.date.today())
    bias_type_tag = "".join(str(x) for x in filters)
    ell_tag = "_%s<ell<%s" % (ell_min, ell_max)
    if file_append is None:
        file_append = ""

    filename = "%s_OSbiaserr_%s_%s_" % (date_tag, bias_type_tag, dith_tag)
    if tot_cases == 1:
        filename += "%s_" % (run_names[0])
    else:
        filename += "%scadences_" % (tot_cases)
    filename += "th-r<%s_%s_%s_%s_%s%s%s.png" % (
        theory_lim_mag,
        zbin_tag,
        yr_tag,
        poisson_tag,
        zero_pt_tag,
        ell_tag,
        file_append,
    )
    # save the plot
    plt.savefig("%s/%s/%s" % (out_dir, outdir, filename), format="png", bbox_inches="tight")
    print("\nSaved %s" % filename)
    if show_plot:
        plt.show()
    else:
        plt.close("all")
