__all__ = ("DdfDepthElement",)

import copy

import numpy as np
import pandas as pd
import healpy as hp
from rubin_scheduler.utils import angular_separation, ddf_locations, ddf_locations_pre3_5, hpid2_ra_dec

import rubin_sim.splat as splat

from .base_element import BaseReportElement


class DdfDepthElement(BaseReportElement):

    def __init__(self, nside=512, radius=2.1, nside_sne=128, old_coords=False, **kwargs):
        super().__init__(**kwargs)
        self.nside = nside
        self.nside_sne = nside_sne
        self.radius = radius

        self.ddfs = {}
        if old_coords:
            ddfs_rough = ddf_locations_pre3_5()
        else:
            ddfs_rough = ddf_locations()
        for ddf in ddfs_rough:
            self.ddfs[ddf] = {"ra": ddfs_rough[ddf][0], "dec": ddfs_rough[ddf][1]}
        # Combine the Euclid double-field into one - but with two ra/dec values
        self.ddfs["EDFS"] = {
            "ra": [self.ddfs["EDFS_a"]["ra"], self.ddfs["EDFS_b"]["ra"]],
            "dec": [self.ddfs["EDFS_a"]["dec"], self.ddfs["EDFS_b"]["dec"]],
        }
        del self.ddfs["EDFS_a"]
        del self.ddfs["EDFS_b"]
        # Let's include an arbitrary point that should be in the WFD for comparison
        self.ddfs["WFD"] = {"ra": 0, "dec": -20.0}

        
    def compute_elements(self, visits_df=None, visits_array=None, nside=64, bands="ugrizy"):

        ra, dec = hpid2_ra_dec(nside, np.arange(hp.nside2npix(nside)))
        ra_sne, dec_sne = hpid2_ra_dec(self.nside_sne, np.arange(hp.nside2npix(self.nside_sne)))

        stats_to_run = {"mean": np.nanmean, "median": np.nanmedian}
        sl = splat.Slicer(nside=nside)

        result_hp_arrays = []
        result_infos = []
        summary_stats = []

        for ddf in self.ddfs:
            # Define the healpixels to use for this DDF
            if np.size(self.ddfs[ddf]["ra"]) > 1:
                goods = []
                goods_sne = []
                for ddf_ra, ddf_dec in zip(self.ddfs[ddf]["ra"], self.ddfs[ddf]["dec"]):
                    dist = angular_separation(ra, dec, ddf_ra, ddf_dec)
                    goods.append(np.where(dist <= self.radius)[0])
                    dist = angular_separation(ra_sne, dec_sne, ddf_ra, ddf_dec)
                    goods_sne.append(np.where(dist <= self.radius)[0])
                good = np.unique(np.concatenate(goods))
                good_sne = np.unique(np.concatenate(goods_sne))
            else:
                dist = angular_separation(ra, dec, np.mean(self.ddfs[ddf]["ra"]), np.mean(self.ddfs[ddf]["dec"]))
                good = np.where(dist <= self.radius)[0]
                dist = angular_separation(ra_sne, dec_sne, np.mean(self.ddfs[ddf]["ra"]), np.mean(self.ddfs[ddf]["dec"]))
                good_sne = np.where(dist <= self.radius)[0]

            # Things calculated per filter
            for i, bandname in enumerate(bands):
                info = {"data_source": self.run_name}
                info["observations_subset"] = self.subset + "\nband=%s\n%s" % (bandname, ddf)

                metrics = []
                infos = []
                plot_dicts = []
                # Coadd depth
                metrics.append(splat.CoaddM5Metric(bandname))
                temp_info = copy.copy(info)
                temp_info["group"] = "DDF"
                temp_info["subgroup"] = ddf
                infos.append(temp_info)
                plot_dicts.append({})

                # Number of visits
                metrics.append(splat.CountMetric())
                temp_info = copy.copy(info)
                temp_info["group"] = "DDF"
                temp_info["subgroup"] = ddf
                infos.append(temp_info)
                plot_dicts.append({})
                
                # select subset of data
                sub_data = visits_array[np.where(visits_array["band"] == bandname)[0]]
                sl = splat.Slicer(nside=nside, hpindx=good)
                #
                # Run the metics through the slicer
                hp_arrays, infos = sl(sub_data, metrics, info=infos)

                result_hp_arrays.extend(hp_arrays)
                result_infos.extend(infos)
                # For each array we've generated, make a vis, summerize it how we like
                for hp_array, metric, info, plot_dict in zip(hp_arrays, metrics, infos, plot_dicts):
                    pm = splat.PlotGnom(info=info)
                    # Skip if no positive values to plot
                    fig = pm(hp_array, **plot_dict, rot=(self.ddfs[ddf]["ra"], self.ddfs[ddf]["dec"]))
                    self.fig_saver(fig, info=info)
                    # Do whatever stats we want on the hp_array
                    for stat in stats_to_run:
                        summary_stats.append(splat.gen_summary_row(info, stat, stats_to_run[stat](hp_array)))

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)
        return result_hp_arrays, result_infos, summary_dataframe
