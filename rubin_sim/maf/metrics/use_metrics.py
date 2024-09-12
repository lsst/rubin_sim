__all__ = ("UseMetric",)

import numpy as np

from rubin_sim.maf.metrics.base_metric import BaseMetric


class UseMetric(BaseMetric):  # pylint: disable=too-few-public-methods
    """Metric to classify visits by type of visits"""

    def __init__(self, note_col="scheduler_note", **kwargs):
        self.note_col = note_col
        super().__init__(col=[note_col], metric_dtype="object", **kwargs)

    def run(self, data_slice, slice_point=None):  # pylint: disable=invalid-name
        """Run the metric.

        Parameters
        ----------
        data_slice : `np.ndarray`, (N,)`
        slice_point : `dict`
           Dictionary of slice_point metadata passed to each metric.
           E.g. the ra/dec of the healpix pixel.

        Returns
        -------
        use_name : `str`
            use at each slice_point.
        """
        use_name = None
        visible_bands = ("u", "g", "r")
        notes = data_slice[self.note_col]
        if len(notes.shape) == 0:
            note = notes
        else:
            note = notes[0]
            assert np.all(notes == note)

        note_elems = note.replace(":", ", ").split(", ")
        # XXX--survey note strings should not be hard-coded here.
        if note_elems[0] == "greedy":
            use_name = note_elems[0]
        if note_elems[0] == "DD":
            use_name = note_elems[1]
        if (note_elems[0] == "blob") | (note_elems[0] == "blob_twi"):
            use_name = "wide with only IR"
            for band in visible_bands:
                if band in note_elems[1]:
                    use_name = "wide with u, g, or r"

        assert use_name is not None, f"Unrecognized note: {note}"
        return use_name


# internal functions & classes
