#####################################################################################################
# Purpose: change the values/mask of a metricBundle in the pixels with a certain value/mask.
# Example applicaton: mask the outermost/shallow edge of skymaps.
#
# Humna Awan: humna.awan@rutgers.edu
#####################################################################################################

__all__ = ("masking_algorithm_generalized",)

import copy

import healpy as hp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

import rubin_sim.maf.plots as plots


def masking_algorithm_generalized(
    my_bundles,
    plot_handler,
    data_label,
    nside=128,
    find_value="unmasked",
    relation="=",
    new_value="masked",
    pixel_radius=6,
    return_border_indices=False,
    print_intermediate_info=False,
    plot_intermediate_plots=True,
    print_final_info=True,
    plot_final_plots=True,
    sky_map_color_min=None,
    sky_map_color_max=None,
):
    """
    Assign new_value to all pixels in a skymap within pixel_radius of pixels with value <, >, or = find_value.

    Parameters
    ----------
    my_bundles : `dict` {`rubin_sim.maf.MetricBundles`}
        a dictionary for metricBundles.
    plot_handler :   `rubin_sim.maf.plots.plotHandler.PlotHandler`
    data_label : `str`
        description of the data, i.e. 'numGal'
    nside : `int`
        HEALpix resolution parameter. Default: 128
    find_value : `str`
        if related to mask, must be either 'masked' or 'unmasked'. otherwise, must be a number.
        Default: 'unmasked'
    relation : `str`
        must be '>','=','<'. Default: '='
    new_value : `str`
        if related to mask, must be either 'masked' or 'unmasked'; otherwise, must be a number.
        Default: 'masked'
    pixel_radius : `int`
        number of pixels to consider around a given pixel. Default: 6
    return_border_indices : `bool`
        set to True to return the array of indices of the pixels whose values/mask are changed. Default: False
    print_intermediate_info : `bool`
        set to False if do not want to print intermediate info. Default: True
    plot_intermediate_plots : `bool`
        set to False if do not want to plot intermediate plots. Default: True
    print_final_info : `bool`
        set to False if do not want to print final info, i.e. total pixels changed. Default: True
    plot_final_plots : `bool`
        set to False if do not want to plot the final plots. Default: True
    sky_map_color_min : float
        color_min label value for skymap plot_dict label. Default: None
    sky_map_color_max : float
        color_max label value for skymap plot_dict label. Default: None
    """
    # find pixels such that (pixelValue (relation) find_value) AND their neighbors dont have that (relation) find_value.
    # then assign new_value to all these pixels.
    # relation must be '>','=','<'
    # data indices are the pixels numbers ..
    # ------------------------------------------------------------------------
    # check whether need to mask anything at all
    if pixel_radius == 0:
        print("No masking/changing of the original data.")
        if return_border_indices:
            borders = {}
            for dither in my_bundles:
                borders[dither] = []

            return [my_bundles, borders]
        else:
            return my_bundles
    # ------------------------------------------------------------------------
    # make sure that relation is compatible with find_value
    if (find_value == "masked") | (find_value == "unmasked"):
        if relation != "=":
            print('ERROR: must have relation== "=" if find_value is related to mask.')
            print('Setting:  relation= "="\n')
            relation = "="
    # ------------------------------------------------------------------------
    # translate find_value into what has to be assigned
    find_value_to_consider = find_value
    if find_value.__contains__("mask"):
        if find_value == "masked":
            find_value_to_consider = True
        if find_value == "unmasked":
            find_value_to_consider = False

    # translate new_value into what has to be assigned
    new_value_to_assign = new_value
    if new_value.__contains__("mask"):
        if new_value == "masked":
            new_value_to_assign = True
        if new_value == "unmasked":
            new_value_to_assign = False

    # ------------------------------------------------------------------------
    borders = {}
    for dither in my_bundles:
        total_border_pixel = []
        if print_intermediate_info:
            print("Survey strategy: %s" % dither)

        # find the array to look at.
        if (find_value).__contains__("mask"):
            orig_array = my_bundles[dither].metricValues.mask.copy()  # mask array
        else:
            orig_array = my_bundles[dither].metricValues.data.copy()  # data array

        for r in range(0, pixel_radius):
            border_pixel = []
            temp_copy = copy.deepcopy(my_bundles)
            # ignore the pixels whose neighbors formed the border in previous run
            if r != 0:
                orig_array[total_border_pixel] = new_value_to_assign

            # find the pixels that satisfy the relation with find_value and whose neighbors dont
            for i in range(0, len(orig_array)):
                neighbors_pixels = hp.get_all_neighbours(nside, i)  # i is the pixel number
                for j in neighbors_pixels:
                    condition = None
                    if relation == "<":
                        condition = (orig_array[i] < find_value_to_consider) & (
                            orig_array[j] >= find_value_to_consider
                        )
                    if relation == "=":
                        condition = (orig_array[i] == find_value_to_consider) & (
                            orig_array[j] != find_value_to_consider
                        )
                    if relation == ">":
                        condition = (orig_array[i] > find_value_to_consider) & (
                            orig_array[j] <= find_value_to_consider
                        )
                    if condition == None:
                        raise ValueError("ERROR: invalid relation: %s" % relation)

                    if condition:
                        if j != -1:  # -1 entries correspond to inexistent neighbors
                            border_pixel.append(i)

            border_pixel = np.unique(border_pixel)
            total_border_pixel.extend(border_pixel)

            if print_intermediate_info:
                print("Border pixels from run %s: %s" % (r + 1, len(border_pixel)))
                print("Total pixels so far: %s\n" % len(total_border_pixel))

            # plot found pixels
            if plot_intermediate_plots:
                if new_value.__contains__("mask"):
                    temp_copy[dither].metricValues.mask[:] = new_value_to_assign
                    temp_copy[dither].metricValues.mask[total_border_pixel] = not (new_value_to_assign)
                    temp_copy[dither].metricValues.data[total_border_pixel] = -500
                    plot_dict = {
                        "xlabel": data_label,
                        "title": "%s: %s Round # %s" % (dither, data_label, r + 1),
                        "log_scale": False,
                        "labelsize": 9,
                        "color_min": -550,
                        "color_max": 550,
                        "cmap": cm.jet,
                    }
                else:
                    temp_copy[dither].metricValues.mask[:] = True
                    temp_copy[dither].metricValues.mask[total_border_pixel] = False
                    temp_copy[dither].metricValues.data[total_border_pixel] = new_value_to_assign
                    plot_dict = {
                        "xlabel": data_label,
                        "title": "%s %s Round # %s" % (dither, data_label, r + 1),
                        "log_scale": False,
                        "labelsize": 9,
                        "maxl": 500,
                        "cmap": cm.jet,
                    }
                temp_copy[dither].set_plot_dict(plot_dict)
                temp_copy[dither].set_plot_funcs([plots.HealpixSkyMap(), plots.HealpixPowerSpectrum()])
                temp_copy[dither].plot(plot_handler=plot_handler)
                plt.show()
            # save the found pixels with the appropriate key
            borders[dither] = total_border_pixel

    # ------------------------------------------------------------------------
    # change the original map/array now.
    for dither in my_bundles:
        total_border_pixel = borders[dither]

        if print_final_info:
            print("Survey strategy: %s" % dither)
            print("Total pixels changed: %s\n" % len(total_border_pixel))

        if new_value.__contains__("mask"):
            my_bundles[dither].metricValues.mask[total_border_pixel] = new_value_to_assign
        else:
            my_bundles[dither].metricValues.data[total_border_pixel] = new_value_to_assign

        if plot_final_plots:
            # skymap
            plot_dict = {
                "xlabel": data_label,
                "title": "%s: %s MaskedMap; pixel_radius: %s " % (dither, data_label, pixel_radius),
                "log_scale": False,
                "labelsize": 8,
                "color_min": sky_map_color_min,
                "color_max": sky_map_color_max,
                "cmap": cm.jet,
            }
            my_bundles[dither].set_plot_dict(plot_dict)
            my_bundles[dither].set_plot_funcs([plots.HealpixSkyMap()])
            my_bundles[dither].plot(plot_handler=plot_handler)
            # power spectrum
            plot_dict = {
                "xlabel": data_label,
                "title": "%s: %s MaskedMap; pixel_radius: %s " % (dither, data_label, pixel_radius),
                "log_scale": False,
                "labelsize": 12,
                "maxl": 500,
                "cmap": cm.jet,
            }
            my_bundles[dither].set_plot_dict(plot_dict)
            my_bundles[dither].set_plot_funcs([plots.HealpixPowerSpectrum()])
            my_bundles[dither].plot(plot_handler=plot_handler)
            plt.show()

    if return_border_indices:
        return [my_bundles, borders]
    else:
        return my_bundles
