#####################################################################################################
# Purpose: save data from a metricBundle object as .npz files.

# Humna Awan: humna.awan@rutgers.edu
#####################################################################################################

__all__ = "save_bundle_data_npz_format"


def save_bundle_data_npz_format(path, bundle, base_filename, filter_band):
    """

    Save data in the metricBundle. For each key, a new .npz file is created to
    save the content of the metricBundle object.

    Parameters
    -------------------
      * path: str: path to the directory where the files should be saved
      * bundle: metricBundle object whose contents are to be saved.
      * base_filename: str: basic filename wanted. Final filename would be
                           <base_filename>_<filter_band>_<dither>.npz
      * filter_band: str: filter of the data in the bundle, e.g. 'r'

    """
    # run over keys in the bundle and save the data
    for dither in bundle:
        outfile = "%s_%s_%s.npz" % (base_filename, filter_band, dither)
        bundle[dither].slicer.write_data(
            "%s/%s" % (path, outfile),
            bundle[dither].metricValues,
            metric_name=bundle[dither].metric.name,
            simDataName=bundle[dither].run_name,
            constraint=bundle[dither].constraint,
            metadata=bundle[dither].metadata,
            display_dict=bundle[dither].displayDict,
            plot_dict=bundle[dither].plotDict,
        )
