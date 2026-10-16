__all__ = ("PerNoteElement",)

import numpy as np
import pandas as pd

import rubin_sim.splat as splat
from .base_element import BaseReportElement


class PerNoteElement(BaseReportElement):

    def compute_elements(self, visits_df=None, visits_array=None):

        summary_stats = []

        # Do some stats over each of the scheduler_note values
        # Could do this as a new column at the start instead.
        note_roots = np.array([val.split(",")[0] for val in visits_array["scheduler_note"]])
        n_obs = np.size(note_roots)
        for note in np.unique(note_roots).tolist():
            info = {"data_source": self.run_name}
            info["group"] = "Per Note"
            info["unit"] = "#"
            info["observations_subset"] = (self.subset + "\nnote start %s" % note).strip()
            count = np.size(np.where(note_roots == note)[0])
            summary_stats.append(splat.gen_summary_row(info, "count", count))
            info["unit"] = "fraction"
            summary_stats.append(splat.gen_summary_row(info, "fraction", count / n_obs))

        # Also do DDFs summed
        note_roots = np.array([val.split(":")[0] for val in visits_array["scheduler_note"]])
        n_obs = np.size(note_roots)
        for note in ["DD"]:
            info = {"data_source": self.run_name}
            info["group"] = "Per Note"
            info["unit"] = "#"
            info["observations_subset"] = (self.subset + "\nnote start %s" % note).strip()
            count = np.size(np.where(note_roots == note)[0])
            summary_stats.append(splat.gen_summary_row(info, "count", count))
            info["unit"] = "fraction"
            summary_stats.append(splat.gen_summary_row(info, "fraction", count / n_obs))

        summary_dataframe = pd.DataFrame.from_dict(summary_stats)
        return [], [], summary_dataframe
