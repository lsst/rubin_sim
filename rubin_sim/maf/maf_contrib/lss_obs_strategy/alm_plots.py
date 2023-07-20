#####################################################################################################
# Purpose: plot skymaps/cartview plots corresponding to alms with specfied l-range (s).
#
# Humna Awan: humna.awan@rutgers.edu
#####################################################################################################
__all__ = ("alm_plots",)

import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np


def alm_plots(
    path,
    out_dir,
    bundle,
    nside=128,
    lmax=500,
    filterband="i",
    ra_range=[-50, 50],
    dec_range=[-65, 5],
    subsets_to_consider=[[130, 165], [240, 300]],
    show_plots=True,
):
    """
    Plot the skymaps/cartview plots corresponding to alms with specified l-ranges.
    Automatically creates the output directories and saves the plots.

    Parameters
    ----------
    path : `str`
        path to the main directory where output directory is saved
    out_dir : `str`
        name of the main output directory
    bundle : metricBundle
    nside : `int`
        HEALpix resolution parameter. Default: 128
    lmax : `int`
        upper limit on the multipole. Default: 500
    filterband : `str`
        any one of 'u', 'g', 'r', 'i', 'z', 'y'. Default: 'i'
    ra_range : `np.ndarray`
        range of right ascention (in degrees) to consider in cartview plot; only useful when
                              cartview= True. Default: [-50,50]
    dec_range : `np.ndarray`
        range of declination (in degrees) to consider in cartview plot; only useful when
                               cartview= True. Default: [-65,5]
    subsets_to_consider : `np.ndarray`
        l-ranges to consider, e.g. use [[50, 100]] to consider 50<l<100.
                                                Currently built to handle five subsets (= number of colors built in).
                                                Default: [[130,165], [240, 300]]
    show_plots : `bool`
        set to True if want to show figures. Default: True

    """
    # set up the output directory
    out_dir2 = "almAnalysisPlots_%s<RA<%s_%s<Dec<%s" % (
        ra_range[0],
        ra_range[1],
        dec_range[0],
        dec_range[1],
    )
    if not os.path.exists("%s%s/%s" % (path, out_dir, out_dir2)):
        os.makedirs("%s%s/%s" % (path, out_dir, out_dir2))

    out_dir3 = "almSkymaps"
    if not os.path.exists("%s%s/%s/%s" % (path, out_dir, out_dir2, out_dir3)):
        os.makedirs("%s%s/%s/%s" % (path, out_dir, out_dir2, out_dir3))

    out_dir4 = "almCartviewMaps"
    if not os.path.exists("%s%s/%s/%s" % (path, out_dir, out_dir2, out_dir4)):
        os.makedirs("%s%s/%s/%s" % (path, out_dir, out_dir2, out_dir4))

    # ------------------------------------------------------------------------
    # In order to consider the out-of-survey area as with data=0,  assign the masked region of the
    # skymaps the median of the in-survey data, and then subtract the median off the entire survey.
    # Add the median back later. This gets rid of the massive fake monopole and allows reconstructing
    # the full skymap from components.
    survey_median_dict = {}
    survey_std_dict = {}
    for dither in bundle:
        in_survey = np.where(bundle[dither].metricValues.mask == False)[0]
        out_survey = np.where(bundle[dither].metricValues.mask == True)[0]
        bundle[dither].metricValues.mask[out_survey] = False
        # data pixels
        survey_median = np.median(bundle[dither].metricValues.data[in_survey])
        survey_std = np.std(bundle[dither].metricValues.data[in_survey])
        # assign data[outOfSurvey]= medianData[in_survey]
        bundle[dither].metricValues.data[out_survey] = survey_median
        # subtract median off
        bundle[dither].metricValues.data[:] = bundle[dither].metricValues.data[:] - survey_median
        # save median for later use
        survey_median_dict[dither] = survey_median
        survey_std_dict[dither] = survey_std

    # ------------------------------------------------------------------------
    # now find the alms correponding to the map.
    for dither in bundle:
        array = hp.anafast(
            bundle[dither].metricValues.filled(bundle[dither].slicer.badval),
            alm=True,
            lmax=500,
        )
        cl = array[0]
        alm = array[1]
        l = np.arange(len(cl))

        lsubsets = {}
        color_array = ["y", "r", "g", "m", "c"]
        color = {}
        for case in range(len(subsets_to_consider)):
            lsubsets[case] = (l > subsets_to_consider[case][0]) & (l < subsets_to_consider[case][1])
            color[case] = color_array[case]

        # ------------------------------------------------------------------------
        # plot things out
        plt.clf()
        plt.plot(l, (cl * l * (l + 1)) / (2.0 * np.pi), color="b")
        for key in list(lsubsets.keys()):
            plt.plot(
                l[lsubsets[key]],
                (cl[lsubsets[key]] * l[lsubsets[key]] * (l[lsubsets[key]] + 1)) / (2.0 * np.pi),
                color=color[key],
            )
        plt.title(dither)
        plt.xlabel("$\ell$")
        plt.ylabel(r"$\ell(\ell+1)C_\ell/(2\pi)$")
        filename = "cls_%s.png" % (dither)
        plt.savefig(
            "%s%s/%s/%s" % (path, out_dir, out_dir2, filename),
            format="png",
            bbox_inches="tight",
        )

        if show_plots:
            plt.show()
        else:
            plt.close()

        survey_median = survey_median_dict[dither]
        survey_std = survey_std_dict[dither]

        # ------------------------------------------------------------------------
        # plot full-sky-alm plots first
        n_ticks = 5
        color_min = survey_median - 1.5 * survey_std
        color_max = survey_median + 1.5 * survey_std
        increment = (color_max - color_min) / float(n_ticks)
        ticks = np.arange(color_min + increment, color_max, increment)

        # full skymap
        hp.mollview(
            hp.alm2map(alm, nside=nside, lmax=lmax) + survey_median,
            flip="astro",
            rot=(0, 0, 0),
            min=color_min,
            max=color_max,
            title="",
            cbar=False,
        )
        hp.graticule(dpar=20, dmer=20, verbose=False)
        plt.title("Full Map")

        ax = plt.gca()
        im = ax.get_images()[0]

        fig = plt.gcf()
        cbaxes = fig.add_axes([0.1, 0.015, 0.8, 0.04])  # [left, bottom, width, height]
        cb = plt.colorbar(im, orientation="horizontal", format="%.2f", ticks=ticks, cax=cbaxes)
        cb.set_label("$%s$-band Coadded Depth" % filterband)
        filename = "alm_FullMap_%s.png" % (dither)
        plt.savefig(
            "%s%s/%s/%s/%s" % (path, out_dir, out_dir2, out_dir3, filename),
            format="png",
            bbox_inches="tight",
        )

        # full cartview
        hp.cartview(
            hp.alm2map(alm, nside=nside, lmax=lmax) + survey_median,
            lonra=ra_range,
            latra=dec_range,
            flip="astro",
            min=color_min,
            max=color_max,
            title="",
            cbar=False,
        )
        hp.graticule(dpar=20, dmer=20, verbose=False)
        plt.title("Full Map")
        ax = plt.gca()
        im = ax.get_images()[0]
        fig = plt.gcf()
        cbaxes = fig.add_axes([0.1, -0.05, 0.8, 0.04])  # [left, bottom, width, height]
        cb = plt.colorbar(im, orientation="horizontal", format="%.2f", ticks=ticks, cax=cbaxes)
        cb.set_label("$%s$-band Coadded Depth" % filterband)
        filename = "alm_Cartview_FullMap_%s.png" % (dither)
        plt.savefig(
            "%s%s/%s/%s/%s" % (path, out_dir, out_dir2, out_dir4, filename),
            format="png",
            bbox_inches="tight",
        )

        # prepare for the skymaps for l-range subsets
        color_min = survey_median - 0.1 * survey_std
        color_max = survey_median + 0.1 * survey_std
        increment = (color_max - color_min) / float(n_ticks)
        increment = 1.15 * increment
        ticks = np.arange(color_min + increment, color_max, increment)

        # ------------------------------------------------------------------------
        # consider each l-range
        for case in list(lsubsets.keys()):
            index = []
            low_lim = subsets_to_consider[case][0]
            up_lim = subsets_to_consider[case][1]
            for ll in np.arange(low_lim, up_lim + 1):
                for mm in np.arange(0, ll + 1):
                    index.append(hp.Alm.getidx(lmax=lmax, l=ll, m=mm))
            alms1 = alm.copy()
            alms1.fill(0)
            alms1[index] = alm[index]  # an unmasked array

            # plot the skymap
            hp.mollview(
                hp.alm2map(alms1, nside=nside, lmax=lmax) + survey_median,
                flip="astro",
                rot=(0, 0, 0),
                min=color_min,
                max=color_max,
                title="",
                cbar=False,
            )
            hp.graticule(dpar=20, dmer=20, verbose=False)
            plt.title("%s<$\ell$<%s" % (low_lim, up_lim))
            ax = plt.gca()
            im = ax.get_images()[0]
            fig = plt.gcf()
            cbaxes = fig.add_axes([0.1, 0.015, 0.8, 0.04])  # [left, bottom, width, height]
            cb = plt.colorbar(im, orientation="horizontal", format="%.3f", ticks=ticks, cax=cbaxes)
            cb.set_label("$%s$-band Coadded Depth" % filterband)
            filename = "almSkymap_%s<l<%s_%s.png" % (low_lim, up_lim, dither)
            plt.savefig(
                "%s%s/%s/%s/%s" % (path, out_dir, out_dir2, out_dir3, filename),
                format="png",
                bbox_inches="tight",
            )

            # plot cartview
            hp.cartview(
                hp.alm2map(alms1, nside=nside, lmax=lmax) + survey_median,
                lonra=ra_range,
                latra=dec_range,
                flip="astro",
                min=color_min,
                max=color_max,
                title="",
                cbar=False,
            )
            hp.graticule(dpar=20, dmer=20, verbose=False)
            plt.title("%s<$\ell$<%s" % (low_lim, up_lim))

            ax = plt.gca()
            im = ax.get_images()[0]
            fig = plt.gcf()
            cbaxes = fig.add_axes([0.1, -0.05, 0.8, 0.04])  # [left, bottom, width, height]
            cb = plt.colorbar(im, orientation="horizontal", format="%.3f", ticks=ticks, cax=cbaxes)
            cb.set_label("$%s$-band Coadded Depth" % filterband)
            filename = "almCartview_%s<l<%s_%s.png" % (low_lim, up_lim, dither)
            plt.savefig(
                "%s%s/%s/%s/%s" % (path, out_dir, out_dir2, out_dir4, filename),
                format="png",
                bbox_inches="tight",
            )

        if show_plots:
            plt.show()
        else:
            plt.close("all")
