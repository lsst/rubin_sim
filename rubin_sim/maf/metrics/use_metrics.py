# imports
import numpy as np

from rubin_sim.maf.metrics.base_metric import BaseMetric

# constants

__all__ = ["UseMetric"]

# exception classes

# interface functions

# classes

__all__ = ["UseMetric"]


class UseMetric(BaseMetric):  # pylint: disable=too-few-public-methods
    """Metric to classify visits by type of visits"""

    def __init__(self, note_col="note", **kwargs):
        self.note_col = note_col
        super().__init__(col=[note_col], metricDtype="object", **kwargs)

    def run(self, data_slice, slice_point=None):  # pylint: disable=invalid-name
        """Run the metric.

        Parameters
        ----------
        data_slice : numpy.NDarray
           Values passed to metric by the slicer, which the metric will use to calculate
           metric values at each slice_point.
        slice_point : Dict
           Dictionary of slice_point metadata passed to each metric.
           E.g. the ra/dec of the healpix pixel or opsim fieldId.

        Returns
        -------
        str
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
        if note_elems[0] == "greedy":
            use_name = note_elems[0]
        if note_elems[0] == "DD":
            use_name = note_elems[1]
        if note_elems[0] == "blob":
            use_name = "wide with only IR"
            for band in visible_bands:
                if band in note_elems[1]:
                    use_name = "wide with u, g, or r"

        assert use_name is not None, f"Unrecognized note: {note}"
        return use_name


# internal functions & classes
